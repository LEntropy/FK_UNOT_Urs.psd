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


class ConceptFeatureExtractor(nn.Module):
    """Wraps CLIP's image encoder for concept_misalign.py -- the joint
    image-text embedding space is what actually correlates with a
    downstream model's caption-to-visual-feature association during
    fine-tuning (PHASE4_SCOPING.md §1), unlike StyleFeatureExtractor's
    VGG19 Gram matrices, which have no notion of text at all.

    ViT-B-32/openai (the original CLIP release) is the smallest,
    best-documented checkpoint open_clip ships -- picked for the same
    "smallest thing that proves the mechanism" reasoning as VGG19 above,
    not for best embedding quality. Requires network access to fetch the
    pretrained checkpoint on first use (same one-time cost as
    torchvision's VGG19_Weights.DEFAULT above).
    """

    def __init__(self, device: torch.device):
        super().__init__()
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
        model = model.to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.model = model
        self.device = device
        # open_clip's own preprocessing normalizes with CLIP's own
        # mean/std, not ImageNet's -- pulled from the transform it built,
        # not hardcoded here, so this always matches whatever checkpoint
        # was actually loaded.
        normalize = next(t for t in preprocess.transforms if hasattr(t, "mean"))
        self.mean = torch.tensor(normalize.mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(normalize.std).view(1, 3, 1, 1).to(device)
        # CLIP's own input resolution (fixed by the checkpoint's patch
        # embedding), independent of whatever `size` a caller's image
        # tensor was loaded at -- resized here so callers can keep using
        # style_cloak.py's load_image_tensor at this project's own
        # (256, validated) resolution.
        self.input_resolution = model.visual.image_size[0] if hasattr(model.visual, "image_size") else 224

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected in [0, 1], NCHW, at any resolution.
        x = torch.nn.functional.interpolate(
            x, size=(self.input_resolution, self.input_resolution), mode="bicubic", align_corners=False
        )
        return (x - self.mean) / self.std

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the (unnormalized -- concept_loss's cosine similarity
        handles that) CLIP image embedding for a 1x3xHxW [0,1] tensor."""
        return self.model.encode_image(self._normalize(x))
