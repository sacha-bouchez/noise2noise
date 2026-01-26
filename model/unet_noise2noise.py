from pytorcher.models import UNet
from pytorcher.trainer import PytorchTrainer

from pytorcher.utils import iradon as iradon_torch

import torch
from torch import nn

import ast
import mlflow

class UNetNoise2Noise(UNet):

    """
    U-Net model for Noise2Noise.
    If outputs are in the photon/Poisson domain using 'mse_anscombe' loss, apply the rescale stage at inference time.
    """

    def forward(self, x):
        output = super().forward(x)
        # Anscombe inverse transform to convert back to original Poisson scale
        if hasattr(self, 'loss_type') and self.loss_type == 'mse_anscombe' and not self.training:
            output = ( (output / 2) ** 2 ) - (3 / 8)
            output = torch.clamp(output, min=0.0)
        #
        return output

class UnetNoise2NoiseCommons:

    """
    Common methods for UNet Noise2Noise models.
    """

    def reconstruction(self, *x, scale=None, **kwargs):
        """
        Apply reconstruction algorithm to batches of sinograms x's.
        
        :param x: (B, C, H, W) sinogram tensor
        :param scale: (B,) scale factor to be applied to sinogram before reconstruction
        :return: (B, C, H, W) reconstructed image tensor
        """
        batch_size = x[0].shape[0]
        out = []
        # divide by scale factor if provided
        # Have to divide each batch sample separately as scale factors may differ
        if scale is not None:
            x = [ xx / scale.view(-1, 1, 1, 1) for xx in x ]  # list of (B, C, H, W)
        # reconstruct each batch sample            
        for i in range(batch_size):
            # jointly reconstruct i-th batch sample(s) (multiple sinograms may be provided)
            batch_i_sinos = [ xx.select(0, i) for xx in x ]  # list of (C, H, W)
            if self.reconstruction_algorithm.lower() == 'fbp':
                recon = [ iradon_torch(torch.transpose(sino.squeeze(0), 0, 1), circle=False, output_size=max(self.image_size), **kwargs).unsqueeze(0) for sino in batch_i_sinos ]  # list of (C, H, W)
                recon = torch.stack(recon, dim=0)  # (n_sinos, C, H, W)
                recon = torch.mean(recon, dim=0)  # (C, H, W)
            else:
                raise NotImplementedError(f'Reconstruction algorithm {self.reconstruction_algorithm} not implemented yet.')
            out.append(recon)
        out = torch.stack(out, dim=0)  # (B, C, H, W)
        # reshape to (B, C, H, W)
        out = out.view(-1, out.shape[1], out.shape[-2], out.shape[-1])
        return out

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
    
class InferenceUNetNoise2Noise(nn.Module, UnetNoise2NoiseCommons, PytorchTrainer):

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