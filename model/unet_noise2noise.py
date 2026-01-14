from pytorcher.models import UNet
import torch

class UNetNoise2Noise(UNet):

    """
    U-Net model for Noise2Noise.
    Outputs are in the photon/Poisson domain, hence the rescale stage when using 'mse_anscombe' loss at inference time.
    """

    def forward(self, x):
        output = super().forward(x)
        # Anscombe inverse transform to convert back to original Poisson scale
        if hasattr(self, 'loss_type') and self.loss_type == 'mse_anscombe' and not self.training:
            output = ( (output / 2) ** 2 ) - (3 / 8)
            output = torch.clamp(output, min=0.0)
        #
        return output