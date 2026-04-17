from pytorcher.models import UNet

from pytorcher.utils import PetForwardRadon

import torch
from torch import nn

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

    def reconstruction(self, *y, scale=None, corr=None, filter='ramp', **kwargs):

        pet_forward_operator = self.get_pet_forward_operator()
        #
        if corr is not None:
            y = [ torch.clamp(yy - corr, min=0) for yy in y ]  # list of (B, C, H, W)
        #
        y = torch.stack(y, dim=0) # (n_sinos, B, C, H, W)
        #
        x_recon = pet_forward_operator.radon_backward(y.view(-1, y.shape[2], y.shape[3], y.shape[4]), filter_name=filter, scale=scale, output_size=self.image_size[-1]) # (n_sinos*B, C, H, W)
        #
        x_recon = x_recon.view(y.shape[0], y.shape[1], x_recon.shape[1], x_recon.shape[2], x_recon.shape[3]) # (n_sinos, B, C, H, W)
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
    
    def forward_inference(self, x, scale, attenuation_map=None, corr=None, seed=None, mask=None, monte_carlo_steps=1, split=True):
        """
        Forward pass through the Noise2Noise U-Net model with input splitting and output aggregation.
        The splitting process has some randomness; set seed for reproducibility.
        Use monte_carlo_steps > 1 for multiple stochastic passes and average the results.
        :param x: (B, C, H, W) input sinogram tensor
        :param attenuation_map: (B, 1, H, W) attenuation map tensor, used for adjoint computation.
        :param scale: (B,) scale factor to be applied to sinogram before reconstruction.
        :param seed: random seed for splitting
        :param mask: (B, 1, H, W) mask tensor to apply to the output
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
                splits = self.reconstruction(splits, scale=scale, corr=corr)  # (B * n_splits, C, H, W)

            # Denoise
            splits_denoised = self.forward(splits, attenuation_map=attenuation_map, scale=scale, mask=mask)  # (B * n_splits, C, H, W)

            # Expand splits to have (n_splits, B, C, H, W)
            splits_denoised = torch.chunk(splits_denoised, self.n_splits, dim=0)  # list of (B, C, H, W)

            # Apply reconstruction if needed and average outputs
            if self.unet_output_domain == 'photon':
                output = self.reconstruction(*splits_denoised, scale=scale, corr=corr) # (B, C, H, W)
            else:
                splits_denoised = torch.stack(splits_denoised, dim=0)  # (n_splits, B, C, H, W)
                output = torch.mean(splits_denoised, dim=0)  # (B, C, H, W)

            outputs += output

            # torch.cuda.empty_cache()
            # time.sleep(5)

        # Average over monte carlo steps
        outputs = outputs / monte_carlo_steps # (B, C, H, W)
        return outputs
    
    def del_unpickable_attributes(self):
        # Remove attributes that cannot be pickled (e.g. for MLFlow model registering)
        if hasattr(self, 'forward_pet_radon_operator'):
            del self.forward_pet_radon_operator

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
            At_y = self.reconstruction(y, scale=scale, corr=corr, filter='ramp')
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

    def forward(self, x, attenuation_map=None, scale=None, mask=None, corr=None):
        """
        :param x: either a batch of sinograms (B, C, H, W) if input_domain is 'photon', or a batch of images if input_domain is 'image'.
                  Even if input_domain is 'photon', one can provide a pre-computed adjoint image as input to save time during inference and training.
        :param x_domain: 'photon' or 'image', only needed if the domain of x is different from self.input_domain and we need to apply the adjoint operator. If None, it will be inferred from the shape of x and self.input_domain.
        :param attenuation_map: (B, C, H, W) attenuation map to be used for the adjoint operator. If None, no attenuation will be applied.
        :param scale: (B,) scale factor to be applied to sinogram before reconstruction. This is typically acquisition_time * np.log(2) / half_life, but can be set to 1 if the input sinogram has already been scaled accordingly. If None, no scaling will be applied.
        """
        # x may be stacked splits, attenuation_map may need to be repeated
        if attenuation_map is not None and attenuation_map.shape[0] != x.shape[0]:
            attenuation_map = torch.repeat_interleave(attenuation_map, repeats=self.n_splits, dim=0)
        if self.input_domain == 'photon' and self.output_domain == 'image' and \
           self.physics is not None and isinstance(self.image_size, tuple) and \
           self.physics_mode == 'pre_inverse':
            x = self.adjoint(x, image_size=self.image_size, attenuation_map=attenuation_map, scale=scale, corr=corr)
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
        #
        if mask is not None:
            output = output * mask
        return output