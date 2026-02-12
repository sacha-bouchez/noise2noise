from pet_simulator import SinogramSimulator
from pytorcher.trainer.pytorch_trainer import PytorchTrainer

from tools.image.castor import read_castor_binary_file, write_binary_file
from tools.image.metrics import PSNR, SSIM

import torch
import numpy as np
import os
import copy

from noise2noise.model.unet_noise2noise import InferenceUNetNoise2Noise

class SingleImageInferencePipeline(PytorchTrainer):

    def __init__(
            self,
            model_name,
            model_version,
            dest_path='./',
            nb_counts=3e6,
            simulator_args={},
            seed=42):
        #
        self.dest_path = dest_path
        self.nb_counts = nb_counts
        self.set_seed(seed)
        #
        self.model = self.get_model(model_name, model_version)
        self.device = self.model.device
        #
        self.is_simulated = False
        # Get simulator and reconstructor
        self.sinogram_simulator = self.get_simulator(**simulator_args)
        # self.reconstructor = self.get_reconstructor(**reconstructor_args)

    def get_model(self, model_name, model_version):
        self.model = InferenceUNetNoise2Noise(
            model_name=model_name,
            model_version=model_version
        )
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


    
    def compute_bias_and_standard_deviation(self, n_samples=100, mask_threshold=1e-3, noise2noise_only=False, noise2noise=True, split=True):

        if not self.is_simulated:
            self.run_simulation()

        # read reference ground truth imahe
        gth = read_castor_binary_file(self.img_path, reader='numpy')
        # read noise-free prompt sinogram
        nfpt, metadata = read_castor_binary_file(os.path.join(self.dest_path, 'simu', 'simu_nfpt.s.hdr'), reader='numpy', return_metadata=True)
        nfpt_tensor = torch.from_numpy(nfpt).float().to(self.device).unsqueeze(0)  # add batch and channel dimension
        scale = float(metadata.get('scale_factor', '1.0'))
        scale = torch.tensor([scale], dtype=torch.float32).to(self.device)

        if noise2noise_only and self.model.unet_output_domain == 'photon':
            reference = nfpt
        else:
            reference = gth
        
        expectation = np.zeros_like(reference)
        expectation_square = np.zeros_like(reference)

        with torch.no_grad():
            for _ in range(n_samples):
                
                if noise2noise_only:
                    prompt = torch.poisson(nfpt_tensor)
                    x = self.model.noise2noise_module(prompt)
                else:
                    if noise2noise:
                        prompt = torch.poisson(nfpt_tensor)
                        x = self.model.forward(prompt, scale, split=split)
                    else:
                        if split:
                            prompt = torch.poisson(nfpt_tensor / self.model.n_splits)
                        else:
                            prompt = torch.poisson(nfpt_tensor)
                        x = self.model.reconstruction(prompt, scale=scale)
                #
                #
                expectation += x.cpu().numpy().squeeze(0)
                expectation_square += (x.cpu().numpy() ** 2).squeeze(0)

        expectation /= n_samples
        expectation_square /= n_samples

        reference = reference.squeeze()
        expectation = expectation.squeeze()
        expectation_square = expectation_square.squeeze()

        reference_mask = reference > mask_threshold

        bias = np.where(reference_mask, 100 * (expectation - reference) / reference, np.nan)
        variance = np.where(reference_mask, 100 * (expectation_square - expectation ** 2) / reference ** 2, np.nan)
        standard_deviation = np.sqrt(variance)

        return reference, bias, standard_deviation

    def prepare_input(self, file_path):
        """
        Prepare the input sinogram tensor and scale factor tensor from the given file.
        """
        # read sinogram and scale factor
        prompt, metadata = read_castor_binary_file(file_path, reader='numpy', return_metadata=True)
        scale = float(metadata.get('scale_factor', '1.0'))
        #
        # convert to torch tensor
        x = torch.from_numpy(prompt).float().to(self.device).unsqueeze(0)  # add batch dimension
        scale = torch.tensor([scale], dtype=torch.float32).to(self.device)
        return x, scale

    def run_reconstruction(self, file_path, dest_path):
        """
        Just run the recontruction step for the given sinogram file.
        """
        x, scale = self.prepare_input(file_path)
        # run reconstruction
        x_recon = self.model.reconstruction(x, scale=scale)
        x_recon = x_recon.squeeze(0) # remove batch dimension
        #
        write_binary_file(
            file_path=os.path.join(dest_path, 'recon_image.hdr'),
            metadata={'scale_factor': str(scale.item())},
            data=x_recon.cpu().numpy(),
            binary_extension=''
        )
        return x_recon

    def inference(self, file_path, dest_path):
        """
        Perform model inference, including data splitting, reconstruction, and aggregation.
        """
        x, scale = self.prepare_input(file_path)
        # forward pass through the model
        with torch.no_grad():
            x_recon = self.model.forward(x, scale=scale, split=True)
            x_recon = x_recon.squeeze(0)  # remove batch dimension
        #w', f
        write_binary_file(
            file_path=os.path.join(dest_path, 'recon_image.hdr'),
            metadata={'scale_factor': str(scale.item())},
            data=x_recon.cpu().numpy(),
            binary_extension=''
        )
        return x_recon

    def run(self):
        """
        Run the full inference pipeline for the single image.
        """

        # Run simulation
        self.run_simulation()

        # Run reconstruction for noisy sinogram
        dest_path_noisy = os.path.join(self.dest_path, 'recon_noisy')
        prompt_sinogram_path = os.path.join(self.dest_path, 'simu', 'simu_pt.s.hdr')
        self.run_reconstruction(prompt_sinogram_path, dest_path_noisy)

        # Run reconstruction for noise-free prompt sinogram
        dest_path_noise_free = os.path.join(self.dest_path, 'recon_noise_free')
        noise_free_sinogram_path = os.path.join(self.dest_path, 'simu', 'simu_nfpt.s.hdr')
        self.run_reconstruction(noise_free_sinogram_path, dest_path_noise_free)

        # Perform inference
        dest_path_noise2noise = os.path.join(self.dest_path, 'recon_noise2noise')
        self.inference(prompt_sinogram_path, dest_path_noise2noise)


if __name__ == "__main__":

    import mlflow
    import matplotlib.pyplot as plt

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    dest_path = f"{os.getenv('WORKSPACE')}/data/brain_web_phantom"

    inference_pipeline = SingleImageInferencePipeline(
        model_name="Noise2Noise_2DPET_photon_to_photon",
        model_version=1,
        dest_path=dest_path,
        nb_counts=5e5,
        simulator_args={
            'n_angles':300,
            'scanner_radius':300,
            'voxel_size_mm':(2.0, 2.0),
            'scatter_component':0.36,
            'random_component':0.50,
            'gaussian_PSF':4,
            'half_life':109.8*60,
        },
        seed=42
    )

    inference_pipeline.set_image(
        img_path=f"{os.getenv('WORKSPACE')}/data/brain_web_phantom/object/gt_web_after_scaling.hdr",
        img_att_path=f"{os.getenv('WORKSPACE')}/data/brain_web_phantom/object/attenuat_brain_phantom.hdr"
    )
    #
    inference_pipeline.run()

    # Read ground truth image
    gt_image = read_castor_binary_file(os.path.join(dest_path, 'object', 'gt_web_after_scaling.hdr'), reader='numpy')

    recon_noise2noise = read_castor_binary_file(os.path.join(dest_path, 'recon_noise2noise', 'recon_image.hdr'), reader='numpy').squeeze()
    recon_noisy = read_castor_binary_file(os.path.join(dest_path, 'recon_noisy', 'recon_image.hdr'), reader='numpy').squeeze()
    recon_noise_free = read_castor_binary_file(os.path.join(dest_path, 'recon_noise_free', 'recon_image.hdr'), reader='numpy').squeeze()

    PSNR_denoised = PSNR(I=gt_image, K=recon_noise2noise, mask=gt_image>0)
    PSNR_noisy = PSNR(I=gt_image, K=recon_noisy, mask=gt_image>0)
    PSNR_noise_free = PSNR(I=gt_image, K=recon_noise_free, mask=gt_image>0)

    SSIM_denoised = SSIM(img1=gt_image, img2=recon_noise2noise, mask=gt_image>0)
    SSIM_noisy = SSIM(img1=gt_image, img2=recon_noisy, mask=gt_image>0)
    SSIM_noise_free = SSIM(img1=gt_image, img2=recon_noise_free, mask=gt_image>0)

    print(f"Denoised Reconstruction - PSNR: {PSNR_denoised:.2f} dB, SSIM: {SSIM_denoised:.4f}")
    print(f"Noise-free Reconstruction - PSNR: {PSNR_noise_free:.2f} dB, SSIM: {SSIM_noise_free:.4f}")
    print(f"Noisy Reconstruction - PSNR: {PSNR_noisy:.2f} dB, SSIM: {SSIM_noisy:.4f}")

    fig, ax = plt.subplots(1,3, figsize=(10,5))
    fig.suptitle(f'Counts: {inference_pipeline.nb_counts}', fontsize=16)
    ax[1].imshow(recon_noise2noise, cmap='gray_r')
    ax[1].set_title(f'Denoised Reconstruction\nPSNR: {PSNR_denoised:.2f} dB, SSIM: {SSIM_denoised:.4f}')
    ax[1].axis('off')

    ax[0].imshow(recon_noise_free, cmap='gray_r')
    ax[0].set_title(f'Noise-free Reconstruction\nPSNR: {PSNR_noise_free:.2f} dB, SSIM: {SSIM_noise_free:.4f}')
    ax[0].axis('off')

    ax[2].imshow(recon_noisy, cmap='gray_r')
    ax[2].set_title(f'Noisy Reconstruction\nPSNR: {PSNR_noisy:.2f} dB, SSIM: {SSIM_noisy:.4f}')
    ax[2].axis('off')

    plt.savefig(os.path.join(dest_path, f'recon_images_final_{int(inference_pipeline.nb_counts)}.png'))

    # #
    # n_samples = 10
    # mask_threshold = 1  # in kBq/mL

    # # init figure
    # fig, ax = plt.subplots(3,3, figsize=(15,15))
    # fig.suptitle(f'Bias and standard deviation Analysis over {n_samples} Samples and {mask_threshold} kBq/mL Threshold', fontsize=16)
    # plt.setp(ax, xticks=[], yticks=[])

    # # Noise2Noise only
    # nfpt, bias, standard_deviation = inference_pipeline.compute_bias_and_standard_deviation(n_samples=n_samples, mask_threshold=mask_threshold, noise2noise_only=True)
    # ax[0, 0].imshow(nfpt, cmap='gray')
    # ax[0, 0].set_title('Noise-Free Prompt Sinogram')
    # fig.colorbar(ax[0,0].images[0], ax=ax[0,0])
    # ax[0,1].imshow(bias, cmap='magma')
    # ax[0,1].set_title('N2N Bias (%)')
    # roi_size = np.sum(~np.isnan(bias))
    # scala_bias = (1 / roi_size) * np.sqrt(np.nansum(bias ** 2))
    # ax[0, 1].annotate(f'Scalar Relative Bias: {scala_bias:.2f} %', xy=(0.05, 0.95), xycoords='axes fraction', color='white', fontsize=10, verticalalignment='top')
    # fig.colorbar(ax[0,1].images[0], ax=ax[0,1])
    # ax[0,2].imshow(standard_deviation, cmap='magma')
    # ax[0,2].set_title('N2N standard_deviation (%)')
    # scalar_standard_deviation = np.nanmean(standard_deviation)
    # ax[0, 2].annotate(f'Scalar Relative standard_deviation: {scalar_standard_deviation:.2f} %', xy=(0.05, 0.95), xycoords='axes fraction', color='white', fontsize=10, verticalalignment='top')
    # fig.colorbar(ax[0,2].images[0], ax=ax[0,2])

    # # With noise2noise
    # gth, bias, standard_deviation = inference_pipeline.compute_bias_and_standard_deviation(n_samples=n_samples, mask_threshold=mask_threshold, noise2noise=True)
    # ax[1, 0].imshow(gth, cmap='gray_r')
    # ax[1, 0].set_title('Activity Concentration (kBq/mL)')
    # ax[1, 0].annotate(f'Max: {np.max(gth):.2f} kBq/mL\nMin: {np.min(gth):.2f} kBq/mL', xy=(0.05, 0.95), xycoords='axes fraction', color='black', fontsize=10, verticalalignment='top')
    # fig.colorbar(ax[1,0].images[0], ax=ax[1,0])
    # ax[1,1].imshow(bias, cmap='magma')
    # ax[1,1].set_title('N2N + Reconstructor Bias (%)')
    # roi_size = np.sum(~np.isnan(bias))
    # scala_bias = (1 / roi_size) * np.sqrt(np.nansum(bias ** 2))
    # ax[1, 1].annotate(f'Scalar Relative Bias: {scala_bias:.2f} %', xy=(0.05, 0.95), xycoords='axes fraction', color='black', fontsize=10, verticalalignment='top')
    # fig.colorbar(ax[1,1].images[0], ax=ax[1,1])
    # ax[1,2].imshow(standard_deviation, cmap='magma')
    # ax[1,2].set_title('N2N + Reconstructor standard_deviation (%)')
    # scalar_standard_deviation = np.nanmean(standard_deviation)
    # ax[1, 2].annotate(f'Scalar Relative standard_deviation: {scalar_standard_deviation:.2f} %', xy=(0.05, 0.95), xycoords='axes fraction', color='black', fontsize=10, verticalalignment='top')
    # fig.colorbar(ax[1,2].images[0], ax=ax[1,2])

    # # Without noise2noise
    # gth_noisy, bias_noisy, standard_deviation_noisy = inference_pipeline.compute_bias_and_standard_deviation(n_samples=n_samples, mask_threshold=mask_threshold, noise2noise=False)
    # # remove ax[1,0] axis
    # ax[2,0].axis('off')
    # # fig.colorbar(ax[1,0].images[0], ax=ax[1,0])
    # ax[2,1].imshow(bias_noisy, cmap='magma')
    # ax[2,1].set_title('Reconstructor Bias (%)')
    # roi_size_noisy = np.sum(~np.isnan(bias_noisy))
    # scala_bias_noisy = (1 / roi_size_noisy) * np.sqrt(np.nansum(bias_noisy ** 2))
    # ax[2, 1].annotate(f'Scalar Relative Bias: {scala_bias_noisy:.2f} %', xy=(0.05, 0.95), xycoords='axes fraction', color='black', fontsize=10, verticalalignment='top')
    # fig.colorbar(ax[2,1].images[0], ax=ax[2,1])
    # ax[2,2].imshow(standard_deviation_noisy, cmap='magma')
    # ax[2,2].set_title('Reconstructor standard_deviation (%)')
    # scalar_standard_deviation_noisy = np.nanmean(standard_deviation_noisy)
    # ax[2, 2].annotate(f'Scalar Relative standard_deviation: {scalar_standard_deviation_noisy:.2f} %', xy=(0.05, 0.95), xycoords='axes fraction', color='black', fontsize=10, verticalalignment='top')
    # fig.colorbar(ax[2,2].images[0], ax=ax[2,2])

    # plt.tight_layout()
    # plt.savefig(os.path.join(dest_path, 'denoiser_and_reconstructor_bias_standard_deviation.jpg'), dpi=150)