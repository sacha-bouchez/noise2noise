from pet_simulator import SinogramSimulatorCastor, SinogramSimulator
from pet_recon import CastorPetReconstructor, PetReconstructor
from pytorcher.trainer.pytorch_trainer import PytorchTrainer

from tools.image.castor import read_castor_binary_file, write_binary_file
from tools.image.metrics import PSNR, SSIM

import torch
import numpy as np
import os
import copy

class SingleImageInferencePipeline(PytorchTrainer):

    def __init__(
            self,
            model_name,
            model_version,
            dest_path='./',
            nb_counts=3e6,
            simulator_args={},
            reconstructor_args={},
            seed=42):
        #
        self.dest_path = dest_path
        self.nb_counts = nb_counts
        self.set_seed(seed)
        #
        self.device = self.get_device()
        self.model = self.get_model(model_name, model_version, device=self.device)
        #
        self.is_simulated = False
        # Get simulator and reconstructor
        self.sinogram_simulator = self.get_simulator(**simulator_args)
        self.reconstructor = self.get_reconstructor(**reconstructor_args)

    def get_model(self, model_name, model_version, device):
        """
        Load registered model from MLflow.
        """
        model_uri = f"models:/{model_name}/{model_version}"
        self.model = mlflow.pytorch.load_model(model_uri=model_uri, map_location=device)
        print(f"Loaded model '{model_name}' version {model_version} from MLflow.")
        self.model.eval()
        return self.model
    
    def set_image(self, img_path, img_att_path):
        """
        Set the image path for inference.
        """
        self.img_path = img_path # in kBq/mL
        self.img_att_path = img_att_path # in cm^-1
        #
        self.img = read_castor_binary_file(self.img_path, reader='numpy').squeeze()
        self.img_att = read_castor_binary_file(self.img_att_path, reader='numpy').squeeze()
        self.image_size = self.img.shape
        #
        # Get appropriate acquisition time to reach desired counts on the image
        self.acquisition_time = self.set_acquisition_time(self.nb_counts)

    def set_seed(self, seed):
        """
        Set random seed for reproducibility.
        """
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def set_acquisition_time(self, nb_counts=3e6):
        # Simulate true counts
        _ , _, _, noise_free_prompt = self.sinogram_simulator.get_nfpt(self.img, self.img_att)
        # Get total counts in the noise-free prompt sinogram
        counts = noise_free_prompt.sum()  # in counts
        # Compute acquisition time to reach desired counts (nb_counts)
        half_life = self.sinogram_simulator.half_life
        self.acquisition_time = (nb_counts / counts) * half_life / np.log(2)
        return self.acquisition_time

    def get_simulator(self, **simulator_args):
        self.sinogram_simulator = SinogramSimulator(seed=self.seed, **simulator_args)
        return self.sinogram_simulator

    def get_reconstructor(self, **reconstructor_args):
        self.reconstructor = PetReconstructor(**reconstructor_args)
        return self.reconstructor

    def run_simulation(self):
        """
        Run sinogram simulation for the single image.
        """
        dest_path = os.path.join(self.dest_path)
        self.sinogram_simulator.run(
            img_path=self.img_path,
            img_att_path=self.img_att_path,
            dest_path=dest_path,
            acquisition_time=self.acquisition_time
        )
        # get noisy range from simulated data
        sinogram_data = np.fromfile(os.path.join(dest_path, 'simu', 'simu_pt.s'), dtype='<i2')
        self.is_simulated = True

    def compute_denoiser_bias_and_standard_deviation(self, n_samples=100):
        """Here the estimator is the denoiser only, not the full reconstruction pipeline."""
        print(f"Computing denoiser bias and standard_deviation with {n_samples} samples ...")
        # self.set_seed(seed=self.seed)
        #
        if not self.is_simulated:
            self.run_simulation()
        # read noise-free sinogram
        sinogram_noise_free = read_castor_binary_file(os.path.join(self.dest_path, 'simu', 'simu_nfpt.s.hdr'), reader='numpy')
        sinogram_noise_free = torch.from_numpy(sinogram_noise_free).float().to(self.device)
        if sinogram_noise_free.dim() == 3:
            sinogram_noise_free = sinogram_noise_free.unsqueeze(1)  # add channel dimension
        #
        expectation = torch.zeros_like(sinogram_noise_free)
        expectation_square = torch.zeros_like(sinogram_noise_free)
        #
        with torch.no_grad():
            for _ in range(n_samples):
                # Simulate prompt sinogram with noise
                sinogram_prompt = torch.poisson(sinogram_noise_free)
                # Denoise sinogram
                sinogram_prompt_denoised = self.model(sinogram_prompt)
                # Compute expectation and expectation square
                expectation += sinogram_prompt_denoised
                expectation_square += sinogram_prompt_denoised ** 2
        #
        expectation = expectation.cpu().numpy()
        expectation_square = expectation_square.cpu().numpy()
        sinogram_noise_free = sinogram_noise_free.cpu().numpy()
        #
        expectation /= n_samples
        expectation_square /= n_samples
        expectation = expectation.squeeze()
        expectation_square = expectation_square.squeeze()
        sinogram_noise_free = sinogram_noise_free.squeeze()

        bias = expectation - sinogram_noise_free
        variance = expectation_square - expectation ** 2
        standard_deviation = np.sqrt(variance)
        #
        return  sinogram_noise_free, bias, standard_deviation

    def compute_reconstructor_bias_and_standard_deviation(self, n_samples=10, denoise=True, phantom_threshold=1e-3):
        """Here the estimator is the full reconstruction pipeline."""
        print(f"Computing reconstructor bias and standard_deviation with {n_samples} samples ...")
        self.set_seed(seed=self.seed)
        #
        if not self.is_simulated:
            self.run_simulation()
        # read reference ground truth imahe
        reference = read_castor_binary_file(self.img_path, reader='numpy')
        # read noise-free sinogram
        sinogram_noise_free = read_castor_binary_file(os.path.join(self.dest_path, 'simu', 'simu_nfpt.s.hdr'), reader='numpy')
        sinogram_noise_free = torch.from_numpy(sinogram_noise_free).float()
        #
        expectation = np.zeros_like(reference)
        expectation_square = np.zeros_like(reference)
        #
        with torch.no_grad():
            for i in range(n_samples):
                # Simulate prompt sinogram with noise
                sinogram_prompt = torch.poisson(sinogram_noise_free)
                if denoise:
                    # Denoise sinogram
                    sinogram_prompt = sinogram_prompt.to(self.device)
                    if sinogram_prompt.dim() == 3:
                        sinogram_prompt = sinogram_prompt.unsqueeze(1)  # add channel dimension
                    sinogram_prompt = self.model(sinogram_prompt)
                    sinogram_prompt = sinogram_prompt.squeeze(1)
                #
                sinogram_prompt = sinogram_prompt.cpu().numpy()
                # Overwrite prompt sinogram with denoised sinogram
                self.overwrite_prompt_sinogram(sinogram_prompt)
                # Run reconstruction
                dest_path = os.path.join(self.dest_path, 'recon_denoised_temp')
                self.run_reconstruction(dest_path)
                # Read reconstructed image
                recon_image = read_castor_binary_file(os.path.join(dest_path, 'recon_image.hdr'), reader='numpy')
                # Compute expectation and expectation square
                expectation += recon_image
                expectation_square += recon_image ** 2

        expectation /= n_samples
        expectation_square /= n_samples
        reference = reference.squeeze()
        expectation = expectation.squeeze()
        expectation_square = expectation_square.squeeze()
        # Mask out background
        reference_mask = reference > phantom_threshold
        #
        bias = np.where(reference_mask, 100 * (expectation - reference) / reference, np.nan)
        variance = np.where(reference_mask, 100 * (expectation_square - expectation ** 2) / reference ** 2, np.nan)
        standard_deviation = np.sqrt(variance)
        #
        return reference, bias, standard_deviation

    def overwrite_prompt_sinogram(self, sinogram, scale_factor):
        """
        Overwrite the prompt sinogram file with the given sinogram array.
        """
        write_binary_file(
            file_path=os.path.join(self.dest_path, 'simu', 'simu_pt.s.hdr'),
            metadata={'scale_factor': str(scale_factor)},
            data=sinogram,
            binary_extension=''
        )

    def run_reconstruction(self, dest_path):
        """
        Run reconstruction for the single image sinogram.
        """
        # empty dest_path if exists
        if os.path.exists(dest_path):
            for file in os.listdir(dest_path):
                file_path = os.path.join(dest_path, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
        # run reconstruction
        self.reconstructor.run(
            file_path=os.path.join(self.dest_path, 'simu', 'simu_pt.s.hdr'),
            dest_path=dest_path,
            method='fbp',
            dim=(*self.image_size,1),
            method_kwargs={'filter_name': 'ramp', 'interpolation': 'linear'}
        )

    def run(self):
        """
        Run the full inference pipeline for the single image.
        """

        # Run simulation
        self.run_simulation()

        # Run reconstruction for noisy sinogram
        dest_path_noisy = os.path.join(self.dest_path, 'recon_noisy')
        self.run_reconstruction(dest_path_noisy)

        # Read noisy sinogram
        sinogram_noisy, metadata_noisy = read_castor_binary_file(os.path.join(self.dest_path, 'simu', 'simu_pt.s.hdr'), reader='numpy', return_metadata=True)
        sino_scale_factor = float(metadata_noisy.get('scale_factor', '1.0'))

        # Read noise-free sinogram
        noise_free_sinogram = read_castor_binary_file(os.path.join(self.dest_path, 'simu', 'simu_nfpt.s.hdr'), reader='numpy')

        # Normalize noisy sinogram to [0, 1]
        sinogram_noisy = torch.from_numpy(sinogram_noisy).float()

        # Perform denoising inference
        with torch.no_grad():
            sinogram_noisy = sinogram_noisy.to(self.device)
            if sinogram_noisy.dim() == 3:
                sinogram_noisy = sinogram_noisy.unsqueeze(1)  # add channel dimension
            denoised_sinogram = self.model(sinogram_noisy)
            denoised_sinogram = denoised_sinogram.cpu()

        # Transform it back to numpy array
        denoised_sinogram = denoised_sinogram.squeeze(1).numpy()


        # Overwrite prompt sinogram with denoised sinogram
        self.overwrite_prompt_sinogram(denoised_sinogram, scale_factor=sino_scale_factor)

        # Run reconstruction on denoised sinogram
        dest_path_denoised = os.path.join(self.dest_path, 'recon_denoised')
        self.run_reconstruction(dest_path_denoised)

        # Overwrite prompt sinogram with noise-free sinogram
        self.overwrite_prompt_sinogram(noise_free_sinogram, scale_factor=sino_scale_factor)

        # Run reconstruction on noise-free sinogram
        dest_path_noise_free = os.path.join(self.dest_path, 'recon_noise_free')
        self.run_reconstruction(dest_path_noise_free)

if __name__ == "__main__":

    import mlflow

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    dest_path = f"{os.getenv('WORKSPACE')}/data/brain_web_phantom"

    inference_pipeline = SingleImageInferencePipeline(
        model_name="Noise2Noise_2DPET_Model_val_psnr",
        model_version=4,
        dest_path=dest_path,
        nb_counts=2.5e5,
        simulator_args={
            'n_angles':300,
            'scanner_radius':300,
            'voxel_size_mm':(2.0, 2.0),
            'scatter_component':0.36,
            'random_component':0.50,
            'gaussian_PSF':4,
            'half_life':109.8*60,
        },
        reconstructor_args={},
        seed=42
    )

    inference_pipeline.set_image(
        img_path=f"{os.getenv('WORKSPACE')}/data/brain_web_phantom/object/gt_web_after_scaling.hdr",
        img_att_path=f"{os.getenv('WORKSPACE')}/data/brain_web_phantom/object/attenuat_brain_phantom.hdr"
    )

    inference_pipeline.run()

    # plot PSNR against iterations for noisy and denoised reconstruction
    import matplotlib.pyplot as plt

    # Read ground truth image
    gt_image = read_castor_binary_file(os.path.join(dest_path, 'object', 'gt_web_after_scaling.hdr'), reader='numpy')

    recon_denoised = read_castor_binary_file(os.path.join(dest_path, 'recon_denoised', 'recon_image.hdr'), reader='numpy').squeeze()
    recon_noisy = read_castor_binary_file(os.path.join(dest_path, 'recon_noisy', 'recon_image.hdr'), reader='numpy').squeeze()
    recon_noise_free = read_castor_binary_file(os.path.join(dest_path, 'recon_noise_free', 'recon_image.hdr'), reader='numpy').squeeze()

    PSNR_denoised = PSNR(I=gt_image, K=recon_denoised, mask=gt_image>0)
    PSNR_noisy = PSNR(I=gt_image, K=recon_noisy, mask=gt_image>0)
    PSNR_noise_free = PSNR(I=gt_image, K=recon_noise_free, mask=gt_image>0)

    SSIM_denoised = SSIM(img1=gt_image, img2=recon_denoised, mask=gt_image>0)
    SSIM_noisy = SSIM(img1=gt_image, img2=recon_noisy, mask=gt_image>0)
    SSIM_noise_free = SSIM(img1=gt_image, img2=recon_noise_free, mask=gt_image>0)

    print(f"Denoised Reconstruction - PSNR: {PSNR_denoised:.2f} dB, SSIM: {SSIM_denoised:.4f}")
    print(f"Noise-free Reconstruction - PSNR: {PSNR_noise_free:.2f} dB, SSIM: {SSIM_noise_free:.4f}")
    print(f"Noisy Reconstruction - PSNR: {PSNR_noisy:.2f} dB, SSIM: {SSIM_noisy:.4f}")

    fig, ax = plt.subplots(1,3, figsize=(10,5))
    fig.suptitle(f'Counts: {inference_pipeline.nb_counts}', fontsize=16)
    ax[1].imshow(recon_denoised, cmap='gray_r')
    ax[1].set_title(f'Denoised Reconstruction\nPSNR: {PSNR_denoised:.2f} dB, SSIM: {SSIM_denoised:.4f}')
    ax[1].axis('off')

    ax[0].imshow(recon_noise_free, cmap='gray_r')
    ax[0].set_title(f'Noise-free Reconstruction\nPSNR: {PSNR_noise_free:.2f} dB, SSIM: {SSIM_noise_free:.4f}')
    ax[0].axis('off')

    ax[2].imshow(recon_noisy, cmap='gray_r')
    ax[2].set_title(f'Noisy Reconstruction\nPSNR: {PSNR_noisy:.2f} dB, SSIM: {SSIM_noisy:.4f}')
    ax[2].axis('off')

    plt.show()

    # plt.savefig(os.path.join(dest_path, f'recon_images_final_{int(inference_pipeline.nb_counts)}.png'))

    #
    n_samples = 100
    phantom_threshold = 150  # in kBq/mL

    # init figure
    fig, ax = plt.subplots(3,3, figsize=(15,15))
    fig.suptitle(f'Denoiser and Reconstructor Bias and standard deviation Analysis over {n_samples} Samples and {phantom_threshold} kBq/mL Threshold', fontsize=16)
    plt.setp(ax, xticks=[], yticks=[])
    
    # Sinogram denoiser bias and standard_deviation
    reference_sino, bias, standard_deviation = inference_pipeline.compute_denoiser_bias_and_standard_deviation(n_samples=n_samples)
    ax[0,0].imshow(reference_sino, cmap='gray')
    ax[0,0].set_title('Reference Noise-free Sinogram')
    fig.colorbar(ax[0,0].images[0], ax=ax[0,0])
    ax[0,1].imshow(bias, cmap='magma')
    ax[0,1].set_title('Denoiser Bias (%)')
    fig.colorbar(ax[0,1].images[0], ax=ax[0,1])
    ax[0,2].imshow(standard_deviation, cmap='magma')
    ax[0,2].set_title('Denoiser standard_deviation (%)')
    fig.colorbar(ax[0,2].images[0], ax=ax[0,2])

    # Denoised Reconstructor bias and standard_deviation
    reference_recon, bias_recon, standard_deviation_recon = inference_pipeline.compute_reconstructor_bias_and_standard_deviation(n_samples=n_samples, denoise=True, phantom_threshold=phantom_threshold)
    ax[1,0].imshow(reference_recon, cmap='gray_r')
    ax[1,0].set_title('Activity Concentration (kBq/mL)')
    ax[1, 0].annotate(f'Max: {np.max(reference_recon):.2f} kBq/mL\nMin: {np.min(reference_recon):.2f} kBq/mL', xy=(0.05, 0.95), xycoords='axes fraction', color='black', fontsize=10, verticalalignment='top')
    fig.colorbar(ax[1,0].images[0], ax=ax[1,0])
    ax[1,1].imshow(bias_recon, cmap='magma')
    ax[1,1].set_title('Denoised Reconstructor Bias (%)')
    roi_size = np.sum(~np.isnan(bias_recon))
    scala_bias = (1 / roi_size) * np.sqrt(np.nansum(bias_recon ** 2))
    ax[1, 1].annotate(f'Scalar Relative Bias: {scala_bias:.2f} %', xy=(0.05, 0.95), xycoords='axes fraction', color='black', fontsize=10, verticalalignment='top')
    fig.colorbar(ax[1,1].images[0], ax=ax[1,1])
    ax[1,2].imshow(standard_deviation_recon, cmap='magma')
    ax[1,2].set_title('Denoised Reconstructor standard_deviation (%)')
    scalar_standard_deviation = np.nanmean(standard_deviation_recon)
    ax[1, 2].annotate(f'Scalar Relative standard_deviation: {scalar_standard_deviation:.2f} %', xy=(0.05, 0.95), xycoords='axes fraction', color='black', fontsize=10, verticalalignment='top')
    fig.colorbar(ax[1,2].images[0], ax=ax[1,2])

    # Noisy Reconstructor bias and standard_deviation
    reference_recon_noisy, bias_recon_noisy, standard_deviation_recon_noisy = inference_pipeline.compute_reconstructor_bias_and_standard_deviation(n_samples=n_samples, denoise=False, phantom_threshold=phantom_threshold)
    # remove ax[2,0] axis
    ax[2,0].axis('off')
    # fig.colorbar(ax[2,0].images[0], ax=ax[2,0])
    ax[2,1].imshow(bias_recon_noisy, cmap='magma')
    ax[2,1].set_title('Noisy Reconstructor Bias (%)')
    roi_size_noisy = np.sum(~np.isnan(bias_recon_noisy))
    scala_bias_noisy = (1 / roi_size_noisy) * np.sqrt(np.nansum(bias_recon_noisy ** 2))
    ax[2, 1].annotate(f'Scalar Relative Bias: {scala_bias_noisy:.2f} %', xy=(0.05, 0.95), xycoords='axes fraction', color='black', fontsize=10, verticalalignment='top')
    fig.colorbar(ax[2,1].images[0], ax=ax[2,1])
    ax[2,2].imshow(standard_deviation_recon_noisy, cmap='magma')
    ax[2,2].set_title('Noisy Reconstructor standard_deviation (%)')
    scalar_standard_deviation_noisy = np.nanmean(standard_deviation_recon_noisy)
    ax[2, 2].annotate(f'Scalar Relative standard_deviation: {scalar_standard_deviation_noisy:.2f} %', xy=(0.05, 0.95), xycoords='axes fraction', color='black', fontsize=10, verticalalignment='top')
    fig.colorbar(ax[2,2].images[0], ax=ax[2,2])

    plt.tight_layout()
    plt.savefig(os.path.join(dest_path, 'denoiser_and_reconstructor_bias_standard_deviation.jpg'), dpi=150)

    # PSNRs_denoised = []
    # SSIMs_denoised = []
    # PSNRs_noisy = []
    # SSIMs_noisy = []
    # it = 1
    # while os.path.isfile(os.path.join(dest_path, 'recon_denoised', f'recon_it{it}.hdr')) and \
    #         os.path.isfile(os.path.join(dest_path, 'recon_noisy', f'recon_it{it}.hdr')):

    #     recon_denoised = read_castor_binary_file(os.path.join(dest_path, 'recon_denoised', f'recon_it{it}.hdr'), reader='numpy')
    #     recon_noisy = read_castor_binary_file(os.path.join(dest_path, 'recon_noisy', f'recon_it{it}.hdr'), reader='numpy')

    #     PSNR_denoised = PSNR(I=gt_image, K=recon_denoised)
    #     PSNR_noisy = PSNR(I=gt_image, K=recon_noisy)

    #     SSIM_denoised = SSIM(img1=gt_image, img2=recon_denoised)
    #     SSIM_noisy = SSIM(img1=gt_image, img2=recon_noisy)

    #     PSNRs_denoised.append(PSNR_denoised)
    #     PSNRs_noisy.append(PSNR_noisy)
    #     SSIMs_denoised.append(SSIM_denoised)
    #     SSIMs_noisy.append(SSIM_noisy)

    #     it += 1

    # fig, ax1 = plt.subplots()
    # ax1.plot(range(1, it), PSNRs_denoised, label='Denoised Reconstruction PSNR', color='black')
    # ax1.plot(range(1, it), PSNRs_noisy, label='Noisy Reconstruction PSNR', color=(0/255, 102/255, 204/255))

    # ax1.set_xlabel('Iterations')
    # ax1.set_ylabel('PSNR (dB)')


    # ax2 = ax1.twinx()
    # ax2.plot(range(1, it), SSIMs_denoised, label='Denoised Reconstruction SSIM', linestyle='--', color='black')
    # ax2.plot(range(1, it), SSIMs_noisy, label='Noisy Reconstruction SSIM', linestyle='--', color=(0/255, 102/255, 204/255))
    # ax2.set_xlabel('Iterations')
    # ax2.set_ylabel('SSIM')
    # ax2.set_xlabel('Iterations')
    # plt.suptitle('PSNR and SSIM vs Iterations for Noisy and Denoised Reconstruction')
    # # legend
    # lines_1, labels_1 = ax1.get_legend_handles_labels()
    # lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax1.legend(lines_1 + lines_2, labels_1 + labels_2)
    # ax1.grid()
    # plt.savefig(os.path.join(dest_path, 'psnr_vs_iterations_OSEM.png'))
    
    # def plot_reconstructions_at_it(it):
    #     if isinstance(it, int):
    #         it = (it, it)
    #     fig, ax = plt.subplots(1,2, figsize=(10,5))
    #     recon_denoised_last = np.squeeze(read_castor_binary_file(os.path.join(dest_path, 'recon_denoised', f'recon_it{it[0]}.hdr'), reader='numpy'))
    #     recon_noisy_last = np.squeeze(read_castor_binary_file(os.path.join(dest_path, 'recon_noisy', f'recon_it{it[1]}.hdr'), reader='numpy'))

    #     ax[0].imshow(recon_denoised_last, cmap='gray')
    #     ax[0].se
    # t_title(f'Denoised Reconstruction at iteration ({it[0]})\nPSNR: {PSNRs_denoised[it[0]-1]:.2f} dB, SSIM: {SSIMs_denoised[it[0]-1]:.4f}')
    #     ax[0].axis('off')

    #     ax[1].imshow(recon_noisy_last, cmap='gray')
    #     ax[1].set_title(f'Noisy Reconstruction at iteration ({it[1]})\nPSNR: {PSNRs_noisy[it[1]-1]:.2f} dB, SSIM: {SSIMs_noisy[it[1]-1]:.4f}')
    #     ax[1].axis('off')

    #     plt.savefig(os.path.join(dest_path, f'recon_images_{it}_OSEM.png'))



    # plot_reconstructions_at_it(it-1) # last iteration
    # denoised_optimal_it = int(np.argmax(PSNRs_denoised) + 1)
    # noisy_optimal_it = int(np.argmax(PSNRs_noisy) + 1)
    # plot_reconstructions_at_it((denoised_optimal_it, noisy_optimal_it)) # optimal iteration based on PSNR