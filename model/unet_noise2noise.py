from pytorcher.models import UNet
from pytorcher.trainer import PytorchTrainer

from pytorcher.utils import PetForwardRadon
from pytorcher.utils import iradon
from noise2noise.utils.adjoint import backward_pet_radon

import torch
from torch import nn

import ast
import mlflow

import hashlib


from pytorcher.utils import tensor_hash

class UnetNoise2NoisePETCommons:

    """
    Common methods for UNet Noise2Noise models.
    """

    def init_pet_forward_operator(self, n_angles=None, scanner_radius_mm=None, gaussian_PSF_fwhm_mm=None, voxel_size_mm=None):
        if self.physics == 'backward_pet_radon':
            self.forward_pet_radon_operator = {}
            geometry = {
                'n_angles':self.n_angles if n_angles is None else n_angles,
                'scanner_radius_mm':self.scanner_radius if scanner_radius_mm is None else scanner_radius_mm,
                'gaussian_PSF_fwhm_mm':self.gaussian_PSF if gaussian_PSF_fwhm_mm is None else gaussian_PSF_fwhm_mm,
                'voxel_size_mm':self.voxel_size_mm if voxel_size_mm is None else voxel_size_mm
            }
            self.forward_pet_radon_operator[hashlib.md5(str(geometry).encode()).hexdigest()] = PetForwardRadon(**geometry)
        else:
            self.forward_pet_radon_operator = None
        
    def get_pet_forward_operator(self, n_angles=None, scanner_radius_mm=None, gaussian_PSF_fwhm_mm=None, voxel_size_mm=None):
        if self.physics == 'backward_pet_radon':
            if not hasattr(self, 'forward_pet_radon_operator'):
                self.init_pet_forward_operator()
            geometry = {
                'n_angles':self.n_angles if n_angles is None else n_angles,
                'scanner_radius_mm':self.scanner_radius if scanner_radius_mm is None else scanner_radius_mm,
                'gaussian_PSF_fwhm_mm':self.gaussian_PSF if gaussian_PSF_fwhm_mm is None else gaussian_PSF_fwhm_mm,
                'voxel_size_mm':self.voxel_size_mm if voxel_size_mm is None else voxel_size_mm
            }
            geometry_hash = hashlib.md5(str(geometry).encode()).hexdigest()
            if geometry_hash not in self.forward_pet_radon_operator:
                self.init_pet_forward_operator(n_angles=n_angles, scanner_radius_mm=scanner_radius_mm, gaussian_PSF_fwhm_mm=gaussian_PSF_fwhm_mm, voxel_size_mm=voxel_size_mm)
            return self.forward_pet_radon_operator[geometry_hash]
        else:
            return None

    def reconstruction(self, *x, scale=None, **kwargs):
        if scale is not None:
            x = [ xx / scale[i] for i, xx in enumerate(x) ]  # list of (B, C, H, W)
        n_splits = len(x)
        batch_size = x[0].shape[0]
        x = torch.stack(x, dim=0) # (n_sinos, B, C, H, W)
        #
        theta = torch.linspace(0, torch.pi, self.n_angles, device=x.device)
        x_recon = iradon(x, theta=theta, output_size=self.image_size[-1], filter='ramp', circle=False)  # (n_sinos*B, C, H, W)
        #
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
    
    def forward_inference(self, x, scale, attenuation_map=None, seed=None, monte_carlo_steps=1, split=True):
        """
        Forward pass through the Noise2Noise U-Net model with input splitting and output aggregation.
        The splitting process has some randomness; set seed for reproducibility.
        Use monte_carlo_steps > 1 for multiple stochastic passes and average the results.
        :param x: (B, C, H, W) input sinogram tensor
        :param attenuation_map: (B, 1, H, W) attenuation map tensor, used for adjoint computation.
        :param scale: (B,) scale factor to be applied to sinogram before reconstruction.
        :param seed: random seed for splitting
        :param monte_carlo_steps: number of stochastic passes to average
        :return: (B, C, H, W) output image tensor
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Stack scale accordingly
        if split:
            scale = (scale / self.n_splits).repeat(self.n_splits) # Dividing the number of counts by n_splits is equivalent to dividing scale factor by n_splits

        outputs = torch.zeros((x.shape[0], x.shape[1], self.image_size[0], self.image_size[1]), device=x.device) # (B, C, H, W)
        for i in range(monte_carlo_steps):
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
            splits_denoised = self.forward(splits, attenuation_map=attenuation_map, scale=scale)  # (B * n_splits, C, H, W)

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

class UNetNoise2NoisePET(UNet, UnetNoise2NoisePETCommons):

    """
    U-Net model for Noise2Noise.
    If outputs are in the photon/Poisson domain using 'mse_anscombe' loss, apply the rescale stage at inference time.
    """

    def __init__(self, *args,
                 input_domain='image', output_domain='image',
                 physics='backward_pet_radon', 
                 physics_mode='pre_inverse',
                 sinogram_size=(300, 300),
                 geometry={},
                 reconstruction_type='fbp',
                 reconstruction_config={},
                 image_size=(160,160),
                 n_splits=2,
                 **kwargs):
        #
        self.input_domain = input_domain
        self.output_domain = output_domain
        assert physics in [None, 'backward_pet_radon'], "Currently only 'backward_pet_radon' physics is supported for UNetNoise2NoisePET. Future versions may include more complex physics models such as the pseudo-inverse."
        self.physics = physics
        self.physics_mode = physics_mode
        assert self.physics_mode in ['pre_inverse', 'skip_connection'], "Currently only 'pre_inverse' and 'skip_connection' physics modes are supported for UNetNoise2NoisePET."
        #
        self.n_angles = geometry.get('n_angles', 300)
        self.scanner_radius = geometry.get('scanner_radius_mm', 300)
        self.gaussian_PSF = geometry.get('gaussian_PSF_fwhm_mm', 4.0)
        self.voxel_size_mm = geometry.get('voxel_size_mm', 2.0)
        #
        #
        self.image_size = image_size
        self.sinogram_size = sinogram_size
        self.n_splits = n_splits
        self.reconstruction_type = reconstruction_type
        self.reconstruction_config = reconstruction_config
        assert self.reconstruction_type.lower() in ['fbp'], "Currently only FBP is supported."
        #
        # must call nn.Module init before assigning any nn.Module attributes such as done in init_pet_forward_operator and get_reconstruction_operator
        super(UNetNoise2NoisePET, self).__init__(*args, **kwargs)
        #
        if self.input_domain == 'photon' and self.output_domain == 'image' and not hasattr(self, 'forward_pet_radon_operator'):
            self.init_pet_forward_operator()


    def adjoint(self, y, image_size=None, voxel_size_mm=None, attenuation_map=None, scale=None):
        #
        if self.physics == 'backward_pet_radon':
            if voxel_size_mm is None:
                voxel_size_mm = self.voxel_size_mm
            # Select appropriate operator as image size and voxel_size may vary and the number of angular bins may be subsampled.
            pet_forward_operator = self.get_pet_forward_operator(
                n_angles=y.shape[-1],
                voxel_size_mm=voxel_size_mm
            )
            At_y = backward_pet_radon(
                y,
                attenuation_map=attenuation_map,
                scale=scale,
                image_size=image_size if image_size is not None else self.image_size,
                voxel_size_mm=voxel_size_mm if voxel_size_mm is not None else self.voxel_size_mm,
                forward_pet_radon_operator=pet_forward_operator,
            )
        else:
            raise ValueError(f"Physics model '{self.physics}' not recognized for UNetNoise2NoisePET.")
        #
        return At_y

    def compute_skip_connection(self, x, attenuation_map=None, scale=None):
        if self.input_domain == 'photon' and self.output_domain == 'image' and self.physics_mode == 'skip_connection':
            # We compute the adjoint in the skip connection
            size_ratio = ( x.shape[-2] / self.sinogram_size[0], x.shape[-1] / self.sinogram_size[1] )
            image_size = ( int(self.image_size[0] * size_ratio[0]), int(self.image_size[1] * size_ratio[1]) )
            # Resize attenuation map if needed
            if attenuation_map is not None and (attenuation_map.shape[-2], attenuation_map.shape[-1]) != image_size:
                attenuation_map = torch.nn.functional.interpolate(attenuation_map, size=image_size, mode='bilinear', align_corners=False)
            # Add channels to attenuation map if needed
            n_channels_x = x.shape[1]
            if attenuation_map is not None and attenuation_map.shape[1] != n_channels_x:
                attenuation_map = attenuation_map.repeat(1, n_channels_x, 1, 1)
            # Rescale voxel size if needed
            if size_ratio != (1.0, 1.0):
                voxel_size_mm = (self.voxel_size_mm / size_ratio[0], self.voxel_size_mm / size_ratio[1])
            else:
                voxel_size_mm = self.voxel_size_mm
            x = self.adjoint(x, image_size=image_size, voxel_size_mm=voxel_size_mm, attenuation_map=attenuation_map, scale=scale)
            return x
        else:
            return super().compute_skip_connection(x)

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
        self.physics_mode == 'pre_inverse':
            # Ensure that we only apply the pre-inverse when the input is a sinogram
            x = self.adjoint(x, image_size=self.image_size, attenuation_map=attenuation_map, scale=scale)
        #
        output = super().forward(x, attenuation_map=attenuation_map, scale=scale)
        # 
        # Anscombe inverse transform to convert back to original Poisson scale
        if hasattr(self, 'loss_type') and self.loss_type == 'mse_anscombe' and not self.training:
            output = ( (output / 2) ** 2 ) - (3 / 8)
            output = torch.clamp(output, min=0.0)
        #
        if self.output_domain == 'image' and (output.shape[-2], output.shape[-1]) != self.image_size:
            output = torch.nn.functional.interpolate(output, size=self.image_size, mode='bilinear', align_corners=False)
        return output
    
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
        self.model = self.get_model(model_name, model_version)
        run_id = mlflow.MlflowClient().get_model_version(model_name, model_version).run_id
        self.get_run_parameters(run_id)

    def get_model(self, model_name, model_version):
        """
        Load registered model from MLflow.
        """
        model_uri = f"models:/{model_name}/{model_version}"
        self.model = mlflow.pytorch.load_model(model_uri=model_uri, map_location=self.device)
        print(f"Loaded model '{model_name}' version {model_version} from MLflow.")
        self.model.eval()
        return self.model
    
    def get_run_parameters(self, run_id):
        """
        Retrieve model parameters from MLflow run.
        """
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        params = run.data.params
        # Extract relevant parameters
        for param_key, param_value in params.items():
            try:
                param_value = ast.literal_eval(param_value)
            except:
                param_value = param_value
            self.__dict__.update({param_key: param_value})
        print(f"Model parameters: input_domain={self.unet_input_domain}, output_domain={self.unet_output_domain}, reconstruction_algorithm={self.reconstruction_type}, n_splits={self.n_splits}, image_size={self.image_size}")

    
if __name__ == '__main__':

    unet_noise2noise_pet = InferenceUNetNoise2Noise(
        model_name="Noise2Noise_2DPET_image_to_image_N2N_val_im_psnr",
        model_version=3,
    )

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    unet_noise2noise_pet.eval().to(device)

    from tools.image.castor import read_castor_binary_file
    import os

    dest_path = f"{os.getenv('WORKSPACE')}/data/brain_web_phantom"
    sino, meta = read_castor_binary_file(os.path.join(dest_path, 'simu', 'simu_pt.s.hdr'), reader='numpy', return_metadata=True)
    scale = float(meta['scale_factor'])
    sino = torch.from_numpy(sino).unsqueeze(0).float().to(device) # shape (1, 1, H, W)
    # sino = sino.transpose(-2, -1) # shape (1, A, D)
    # attenuation_map = read_castor_binary_file(os.path.join(dest_path, 'object', 'attenuat_brain_phantom.hdr'), reader='numpy')
    # attenuation_map = torch.from_numpy(attenuation_map).unsqueeze(0).float().to(device)
    attenuation_map=None

    with torch.no_grad():
        output = unet_noise2noise_pet(sino, attenuation_map=attenuation_map, scale=torch.tensor(scale, device=device))
        print(f"Output shape: {output.shape}")

    from matplotlib import pyplot as plt

    plt.imshow(output.cpu().squeeze(), cmap='gray_r')
    plt.show()