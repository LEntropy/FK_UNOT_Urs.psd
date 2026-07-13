"""VGG19-based style feature extractor.

Same architectural choice as Gatys et al. (2015) neural style transfer and,
by extension, Glaze's style-cloaking approach: VGG's convolutional features
correlate well with perceptual "style" when summarized as Gram matrices,
because each Gram entry captures how strongly pairs of filter responses
co-activate across the image, independent of spatial layout.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights

# Layer names follow the classic Gatys et al. convention.
STYLE_LAYERS = {
    "0": "relu1_1",
    "5": "relu2_1",
    "10": "relu3_1",
    "19": "relu4_1",
    "28": "relu5_1",
}

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class StyleFeatureExtractor(nn.Module):
    """Wraps VGG19's conv stack and returns Gram matrices at STYLE_LAYERS."""

    def __init__(self, device: torch.device):
        super().__init__()
        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        for param in vgg.parameters():
            param.requires_grad_(False)
        # In-place ReLU overwrites the activation storage a Gram matrix view
        # (see _gram_matrix) may still need for backprop at an earlier layer.
        # Standard fix (also used in PyTorch's own style-transfer tutorial):
        # force every ReLU to be out-of-place.
        for module in vgg.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
        self.vgg = vgg
        self.device = device
        self.mean = IMAGENET_MEAN.to(device)
        self.std = IMAGENET_STD.to(device)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected in [0, 1], NCHW.
        return (x - self.mean) / self.std

    def gram_matrices(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns {layer_name: gram_matrix} for the style layers."""
        grams: dict[str, torch.Tensor] = {}
        h = self._normalize(x)
        for name, layer in self.vgg._modules.items():
            h = layer(h)
            if name in STYLE_LAYERS:
                grams[STYLE_LAYERS[name]] = _gram_matrix(h)
        return grams


def _gram_matrix(feature_map: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feature_map.shape
    assert b == 1, "batch size 1 only (PoC keeps this simple)"
    flat = feature_map.view(c, h * w)
    gram = flat @ flat.t()
    return gram / (c * h * w)  # normalize so magnitude doesn't scale with feature map size
