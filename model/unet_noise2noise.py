from pytorcher.models import UNet
from pytorcher.trainer import PytorchTrainer

from pytorcher.utils import PetForwardRadon, deepinv_iradon as iradon
from noise2noise.utils.adjoint import backward_pet_radon

import torch
from torch import nn

import ast
import mlflow

from math import sqrt
from pytorcher.utils import tensor_hash

class UNetNoise2NoisePET(UNet):

    """
    U-Net model for Noise2Noise.
    If outputs are in the photon/Poisson domain using 'mse_anscombe' loss, apply the rescale stage at inference time.
    """

    def __init__(self, *args,
                 input_domain='image', output_domain='image',
                 physics='backward_pet_radon', 
                 sinogram_size=(300, 300),
                 geometry={},
                 image_size=(160,160), **kwargs):
        super().__init__(*args, **kwargs)
        self.input_domain = input_domain
        self.output_domain = output_domain
        assert physics in [None, 'backward_pet_radon'], "Currently only 'backward_pet_radon' physics is supported for UNetNoise2NoisePET. Future versions may include more complex physics models such as the pseudo-inverse."
        self.physics = physics
        if self.input_domain == 'image' and self.output_domain == 'image':
            self.image_size = image_size
            self.sinogram_size = None
        elif self.input_domain == 'photon' and self.output_domain == 'photon':
            self.sinogram_size = sinogram_size
            self.image_size = None
        elif self.input_domain == 'photon' and self.output_domain == 'image':
            self.sinogram_size = sinogram_size
            self.image_size = image_size
        else:
            raise ValueError(f"Unsupported domain combination: input_domain={input_domain}, output_domain={output_domain}. Supported combinations are: photon to image, image to image, photon to photon.")
        #
        self.n_angles = geometry.get('n_angles', 300)
        self.scanner_radius = geometry.get('scanner_radius_mm', 300)
        self.gaussian_PSF = geometry.get('gaussian_PSF_fwhm_mm', 4.0)
        self.voxel_size_mm = geometry.get('voxel_size_mm', 2.0)
        #
        self.get_pet_forward_operator()

    def get_pet_forward_operator(self):
        if self.physics == 'backward_pet_radon':
            self.forward_pet_radon_operator = PetForwardRadon(
                n_angles=self.n_angles ,
                scanner_radius_mm=self.scanner_radius,
                gaussian_PSF_fwhm_mm=self.gaussian_PSF,
                voxel_size_mm=self.voxel_size_mm
            )
        else:
            self.forward_pet_radon_operator = None

    def adjoint(self, y, attenuation_map=None, scale=None):
        #
        if self.input_domain == 'photon' and self.output_domain == 'image' and \
            self.physics is not None and isinstance(self.image_size, tuple): # Ensure that we only apply the pre-inverse when the output is an image and the input is a sinogram, and that the output size is specified (i.e. not 'same')
            # Photon to image models need pre-inverse to map sinogram back to image domain.
            # This ensures that skip connections are consistent.
            if self.physics == 'backward_pet_radon':
                with torch.enable_grad():
                    At_y = backward_pet_radon(
                        y,
                        attenuation_map=attenuation_map,
                        scale=scale,
                        image_size=self.image_size,
                        forward_pet_radon_operator=self.forward_pet_radon_operator
                    )
            else:
                raise ValueError(f"Physics model '{self.physics}' not recognized for UNetNoise2NoisePET.")
            #
            return At_y
        else:
            # For other cases, the adjoint is just the identity function (i.e. no pre-inverse needed).
            return y

    def forward(self, x, attenuation_map=None, scale=None):
        """
        :param x: either a batch of sinograms (B, C, H, W) if input_domain is 'photon', or a batch of images if input_domain is 'image'.
                  Even if input_domain is 'photon', one can provide a pre-computed adjoint image as input to save time during inference and training.
        :param x_domain: 'photon' or 'image', only needed if the domain of x is different from self.input_domain and we need to apply the adjoint operator. If None, it will be inferred from the shape of x and self.input_domain.
        :param attenuation_map: (B, C, H, W) attenuation map to be used for the adjoint operator. If None, no attenuation will be applied.
        :param scale: (B,) scale factor to be applied to sinogram before reconstruction. This is typically acquisition_time * np.log(2) / half_life, but can be set to 1 if the input sinogram has already been scaled accordingly. If None, no scaling will be applied.
        """
        #
        if self.input_domain == 'photon' and self.output_domain == 'image' and \
            self.physics is not None and isinstance(self.image_size, tuple) and \
            (x.shape[-2], x.shape[-1]) == self.sinogram_size: 
                # Ensure that we only apply the pre-inverse when the input is a sinogram
                x = self.adjoint(x, attenuation_map=attenuation_map, scale=scale)
        # NOTE if normalize_input is True, this is done in the UNet forward method, so we don't need to do it here again.
        #
        output = super().forward(x)
        # Anscombe inverse transform to convert back to original Poisson scale
        if hasattr(self, 'loss_type') and self.loss_type == 'mse_anscombe' and not self.training:
            output = ( (output / 2) ** 2 ) - (3 / 8)
            output = torch.clamp(output, min=0.0)
        #
        return output

class UnetNoise2NoisePETCommons:

    """
    Common methods for UNet Noise2Noise models.
    """

    def reconstruction(self, *x, scale=None, **kwargs):
        if scale is not None:
            x = [ xx / scale[i] for i, xx in enumerate(x) ]  # list of (B, C, H, W)
        n_splits = len(x)
        batch_size = x[0].shape[0]
        x = torch.stack(x, dim=0) # (n_sinos, B, C, H, W)
        #
        x = x.view(n_splits * batch_size, x.shape[2], x.shape[3], x.shape[4])  # (n_sinos*B, C, H, W)

        # Compute out_size base on scanner radius and voxel size to ensure that the reconstructed image fits within the circular FOV
        out_size = int(self.scanner_radius * sqrt(2) / self.voxel_size_mm)
        x_recon = iradon(
            torch.transpose(x, -2, -1), circle=True, out_size=out_size, **kwargs
        )  # (n_sinos*B, C, H, W)
        # Crop to original image size
        pad_x = (x_recon.shape[-2] - self.image_size[0]) // 2
        pad_y = (x_recon.shape[-1] - self.image_size[1]) // 2
        x_recon = x_recon[:, :, pad_x:x_recon.shape[-2]-pad_x, pad_y:x_recon.shape[-1]-pad_y] # (n_sinos*B, C, H, W)
        #
        x_recon = x_recon.view(n_splits, batch_size, x_recon.shape[1], x_recon.shape[2], x_recon.shape[3])  # (n_sinos, B, C, H, W)
        x_recon = torch.mean(x_recon, dim=0)  # (B, C, H, W)
        return x_recon


    def split_prompt(self, prompt, mode='multinomial', consistent=True, seed=None):
        """
        Split prompt sinogram into n_splits sinograms with multinomial statistics.

        :param prompt: (B, C, H, W) sinogram batch (non-negative integer counts)
        :param consistent: if True, each sample gets its own deterministic seed
        :param seed: global seed used only when consistent=False
        :return: list of n_splits tensors, each (B, C, H, W)
        """
        B = prompt.shape[0]
        device = prompt.device

        if mode != 'multinomial':
            raise ValueError(f'Unknown split mode: {mode}')

        # Create per-sample generators
        generators = []

        if consistent:
            # Deterministic seed per sample based on its content
            for b in range(B):
                g = torch.Generator(device=device)
                s = tensor_hash(prompt[b], format='int') % (2**63 - 1)
                g.manual_seed(s)
                generators.append(g)
        else:
            # One shared generator (e.g. inference-time reproducibility)
            g = torch.Generator(device=device)
            if seed is not None:
                g.manual_seed(seed)
            generators = [g] * B

        # Multinomial splitting via sequential binomials
        remaining = prompt.clone()
        splits = []

        for i in range(self.n_splits - 1):
            p = 1.0 / (self.n_splits - i)

            split_i = torch.empty_like(prompt)

            # We loop over batch for RNG correctness, but each draw is fully vectorized over (C,H,W)
            for b in range(B):
                split_i[b] = torch.binomial(
                    count=remaining[b],
                    prob=torch.full_like(remaining[b], p, dtype=torch.float32),
                    generator=generators[b]
                )

            splits.append(split_i)
            remaining = remaining - split_i

        splits.append(remaining)  # last split gets leftovers

        return splits
    
class InferenceUNetNoise2Noise(nn.Module, UnetNoise2NoisePETCommons, PytorchTrainer):

    """
    Inference U-Net model for Noise2Noise.
    During inference, splits the input and aggregates the outputs according to the specified domains.
    """

    def __init__(self, model_name, model_version, **kwargs):

        #
        super().__init__(**kwargs)
        
        #
        self.device = self.get_device()
        
        # Load Noise2Noise module from MLflow
        self.noise2noise_module = self.get_noise2noise_module(model_name, model_version)
        run_id = mlflow.MlflowClient().get_model_version(model_name, model_version).run_id
        self.get_run_parameters(run_id)

    def get_noise2noise_module(self, model_name, model_version):
        """
        Load registered model from MLflow.
        """
        model_uri = f"models:/{model_name}/{model_version}"
        self.noise2noise_module = mlflow.pytorch.load_model(model_uri=model_uri, map_location=self.device)
        print(f"Loaded model '{model_name}' version {model_version} from MLflow.")
        self.noise2noise_module.eval()
        return self.noise2noise_module
    
    def get_run_parameters(self, run_id):
        """
        Retrieve model parameters from MLflow run.
        """
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        params = run.data.params
        # Extract relevant parameters
        self.unet_input_domain = params.get('unet_input_domain', 'image')
        self.unet_output_domain = params.get('unet_output_domain', 'image')
        self.reconstruction_algorithm = params.get('reconstruction_algorithm', 'fbp')
        self.reconstruction_config = {'filter': True} # For now, we hardcode the reconstruction config to ensure that the pre-inverse and post-inverse are consistent. Future versions may include more flexible handling of reconstruction configs.
        self.n_splits = int(params.get('n_splits', 2))
        self.image_size = ast.literal_eval(params.get('image_size', '(160, 160)'))
        self.scanner_radius = int(params.get('scanner_radius', 300))
        self.voxel_size_mm = float(params.get('voxel_size_mm', 2.0))
        print(f"Model parameters: input_domain={self.unet_input_domain}, output_domain={self.unet_output_domain}, reconstruction_algorithm={self.reconstruction_algorithm}, n_splits={self.n_splits}, image_size={self.image_size}")

    def forward(self, x, scale, seed=None, monte_carlo_steps=1, split=False):
        """
        Forward pass through the Noise2Noise U-Net model with input splitting and output aggregation.
        The splitting process has some randomness; set seed for reproducibility.
        Use monte_carlo_steps > 1 for multiple stochastic passes and average the results.
        :param x: (B, C, H, W) input sinogram tensor
        :param scale: (B,) scale factor to be applied to sinogram before reconstruction
        :param seed: random seed for splitting
        :param monte_carlo_steps: number of stochastic passes to average
        :return: (B, C, H, W) output image tensor
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Stack scale accordingly
        if split:
            scale = (scale / self.n_splits).repeat(self.n_splits) # Dividing the number of counts by n_splits is equivalent to dividing scale factor by n_splits

        outputs = torch.zeros((x.shape[0], x.shape[1], self.image_size[0], self.image_size[1]), device=self.device)
        for i in range(monte_carlo_steps):
            if monte_carlo_steps > 1:
                print(f"Monte Carlo step {i+1}/{monte_carlo_steps}")
            # Split input sinogram
            if split:
                splits = self.split_prompt(x, mode='multinomial', consistent=False)  # list of (B, C, H, W)
            else:
                splits = [x]

            # Concatenate splits along batch dimension
            splits = torch.cat(splits, dim=0)  # (B * n_splits, C, H, W)


            # Apply reconstruction if needed
            if self.unet_input_domain == 'image':
                splits = self.reconstruction(splits, scale=scale, **self.reconstruction_config)  # (B * n_splits, C, H, W)

            # Denoise
            splits_denoised = self.noise2noise_module(splits)

            # Expand splits to have (n_splits, B, C, H, W)
            splits_denoised = torch.chunk(splits_denoised, self.n_splits, dim=0)  # list of (B, C, H, W)

            # Apply reconstruction if needed and average outputs
            if self.unet_output_domain == 'photon':
                output = self.reconstruction(*splits_denoised, scale=scale, **self.reconstruction_config) # (B, C, H, W)
            else:
                splits_denoised = torch.stack(splits_denoised, dim=0)  # (n_splits, B, C, H, W)
                output = torch.mean(splits_denoised, dim=0)  # (B, C, H, W)

            outputs += output

            # torch.cuda.empty_cache()
            # time.sleep(5)

        # Average over monte carlo steps
        outputs = outputs / monte_carlo_steps # (B, C, H, W)
        return outputs
    
if __name__ == '__main__':

    unet_noise2noise_pet = UNetNoise2NoisePET(
        n_channels=1,
        n_classes=1,
        global_conv=32,
        n_levels=4,
        bilinear=True,
        conv_layer_type='Conv2d',
        residual=True,
        normalize_input=True,
        input_domain='photon',
        output_domain='image',
        physics='backward_pet_radon',
        image_size=(160, 160)
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    unet_noise2noise_pet.eval().to(device)

    from tools.image.castor import read_castor_binary_file
    import os

    dest_path = f"{os.getenv('WORKSPACE')}/data/brain_web_phantom"
    sino = read_castor_binary_file(os.path.join(dest_path, 'simu', 'simu_pt.s.hdr'), reader='numpy')
    sino = torch.from_numpy(sino).unsqueeze(0).float().to(device) # shape (1, 1, H, W)
    attenuation_map = read_castor_binary_file(os.path.join(dest_path, 'object', 'attenuat_brain_phantom.hdr'), reader='numpy')
    attenuation_map = torch.from_numpy(attenuation_map).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = unet_noise2noise_pet(sino, attenuation_map=attenuation_map, scale=torch.tensor(0.02, device=device))
        print(f"Output shape: {output.shape}")

    from matplotlib import pyplot as plt

    plt.imshow(output.cpu().squeeze(), cmap='gray_r')
    plt.show()