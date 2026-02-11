import torch
from pytorcher.utils.forward import PetForwardRadon

def backward_pet_radon(
          y,
          attenuation_map=None,
          scale=None,
          image_size=(160, 160),
          forward_pet_radon_operator=None
):
    """
    Docstring for backward_pet_radon
    
    :param y: Sinogram tensor of shape (B, C, H, W) to be backprojected.
    :param attenuation_map: (B, C, H, W) attenuation map to be used for the adjoint operator. If None, no attenuation will be applied.
    :param scale: (B,) scale factor to be applied to sinogram before reconstruction. This is typically acquisition_time * np.log(2) / half_life, but can be set to 1 if the input sinogram has already been scaled accordingly. If None, no scaling will be applied.
    :param image_size: (H, W) size of the output image. This is needed to ensure that the backprojection is done correctly with respect to the geometry of the scanner and the input sinogram.
    :param n_angles: Number of projection angles in the sinogram.
    :param scanner_radius_mm: Radius of the scanner in millimeters.
    :param gaussian_PSF_fwhm_mm: Full width at half maximum of the Gaussian point spread function in millimeters.
    :param voxel_size_mm: Size of the image voxels in millimeters.
    """
    assert isinstance(forward_pet_radon_operator, PetForwardRadon), f"PetForwardRadon forward operator must be given, not {type(forward_pet_radon_operator)}."
    with torch.enable_grad():
        # Create dummy image to be projected.
        # The forward operator is linear, so the gradient will be the same regardless of the values in the dummy image.
        x0 = torch.zeros(y.shape[0], y.shape[1], image_size[0], image_size[1], device=y.device, requires_grad=True)  # (B, C, H, W)
        if attenuation_map is None:
            print("Warning: No attenuation map provided for pre-inverse. Assuming no attenuation.")
        if scale is None:
            print("Warning: No scale provided for pre-inverse. Assuming scale of 1. Results may need to be rescaled accordingly.")
        Ax0 = forward_pet_radon_operator.forward(
            x0,
            attenuation_map=attenuation_map,
            scale=scale
            )  # (B, C, H, W)
        loss = (Ax0 * y).sum()
        loss.backward()
        At_y = x0.grad  # (B, C, H, W)
    return At_y.detach() 