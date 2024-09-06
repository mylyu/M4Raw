import torch
from torch import nn
import segmentation_models_pytorch as smp

class SMPUNET(nn.Module):
    """
    PyTorch implementation of a U-Net model using the ResNet50 backbone from
    segmentation_models_pytorch for denoising tasks.
    """

    def __init__(
        self,
        args,
        in_chans: int = 1,
        out_chans: int = 1
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
        """
        super().__init__()

        # Create the U-Net model with a ResNet50 backbone
        self.model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights=None,  # No pre-trained weights
            in_channels=in_chans,
            classes=out_chans,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.model(image)
