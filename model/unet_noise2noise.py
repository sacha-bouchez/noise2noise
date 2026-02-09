from pytorcher.models import UNet
from pytorcher.trainer import PytorchTrainer

from pytorcher.utils import deepinv_iradon as iradon, pet_forward_radon

import torch
from torch import nn

import ast
import mlflow

from math import sqrt

class UNetNoise2NoisePET(UNet):

    """
    U-Net model for Noise2Noise.
    If outputs are in the photon/Poisson domain using 'mse_anscombe' loss, apply the rescale stage at inference time.
    """

    def __init__(self, *args, input_domain='image', output_domain='image', physics='backward_radon', **kwargs):
        super().__init__(*args, **kwargs)
        self.input_domain = input_domain
        self.output_domain = output_domain
        assert physics in ['backward_radon'], "Currently only 'backward_radon' physics is supported for UNetNoise2NoisePET. Future versions may include more complex physics models such as the pseudo-inverse."
        self.physics = physics

    def forward(self, x, attenuation_map=None, scale=None, physics_kwargs={}):
        #
        if self.input_domain == 'photon' and self.output_domain == 'image' and self.physics is not None and isinstance(self.output_size, tuple):
            # Photon to image models need pre-inverse to map sinogram back to image domain.
            # This ensures that skip connections are consistent.
            if self.physics == 'backward_radon':
                with torch.enable_grad():
                    # Create dummy image to be projected.
                    # The forward operator is linear, so the gradient will be the same regardless of the values in the dummy image.
                    x0 = torch.zeros(x.shape[0], x.shape[1], self.output_size[0], self.output_size[1], device=x.device, requires_grad=True)  # (B, C, H, W)
                    if attenuation_map is None:
                        print("Warning: No attenuation map provided for pre-inverse. Assuming no attenuation.")
                    if scale is None:
                        print("Warning: No scale provided for pre-inverse. Assuming scale of 1. Results may need to be rescaled accordingly.")
                    Ax0 = pet_forward_radon(
                        x0,
                        attenuation_map=attenuation_map,
                        scale=scale,
                        n_angles=self.n_angles if hasattr(self, 'n_angles') else physics_kwargs.get('n_angles', 300),
                        scanner_radius_mm=self.scanner_radius if hasattr(self, 'scanner_radius') else physics_kwargs.get('scanner_radius_mm', 300),
                        gaussian_PSF_fwhm_mm=self.gaussian_PSF if hasattr(self, 'gaussian_PSF') else physics_kwargs.get('gaussian_PSF_fwhm_mm', 4.0),
                        voxel_size_mm=self.voxel_size_mm if hasattr(self, 'voxel_size_mm') else physics_kwargs.get('voxel_size_mm', 2.0)
                        )  # (B, C, H, W)
                    loss = (Ax0 * x).sum()
                    loss.backward()
                    x = x0.grad  # (B, C, H, W)
            else:
                raise ValueError(f"Physics model '{self.physics}' not recognized for UNetNoise2NoisePET.")
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

    def split_prompt(self, prompt, mode='multinomial'):
        """
        Split prompt sinogram into n_splits sinograms with multinomial statistics.
        
        :param prompt: (B, C, H, W) sinogram tensor
        :return: list of n_splits sinogram tensors, each of shape (B, C, H, W)
        """
        if mode == 'multinomial':
            splits = []
            remaining = prompt
            for i in range(self.n_splits - 1):
                split_i = torch.distributions.Binomial(total_count=remaining, probs=1/(self.n_splits - i)).sample()
                splits.append(split_i)
                remaining = remaining - split_i
            # Final split gets the remaining counts
            splits.append(remaining)
        else:
            raise ValueError(f'Unknown split mode: {mode}')
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
        self.reconstruction_config = ast.literal_eval(params.get('reconstruction_config', '{}'))
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
                splits = self.split_prompt(x, mode='multinomial')  # list of (B, C, H, W)
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
        physics='backward_radon',
        output_size=(160, 160)
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