"""ControlNet conditioning for Test Lab's LoRA-retraining preview samples.

WHY: score_protection() (protection_score.py) generates preview images from
pure random noise (t2i) using each trained LoRA + the artwork's own
upload-time tags. Two t2i samples from unrelated random seeds can land on
completely different compositions/poses, which makes it hard for a user to
visually judge "did protection actually change what got learned" -- the
composition difference swamps the signal. Conditioning BOTH the baseline
and protected generation on the SAME ControlNet edge map (taken from the
true original image) pins composition/pose across both samples, so the
only thing that visibly differs between baseline and protected previews is
what the protection mechanism actually changed.

Canny edges are computed with PIL only (ImageFilter.FIND_EDGES + a
threshold), not opencv -- this project has no opencv dependency anywhere
else, and a rough structural edge map is all ControlNet needs here; this
isn't a computer-vision task that needs cv2.Canny's exact algorithm.
"""

import numpy as np
import torch
from PIL import Image, ImageFilter


def compute_edge_control_image(original_path: str, size: int) -> Image.Image:
    """Canny-style binary edge map, resized to `size`x`size`, RGB (the shape
    every ControlNetModel expects for controlnet_cond)."""
    img = Image.open(original_path).convert("L").resize((size, size), Image.LANCZOS)
    edges = img.filter(ImageFilter.FIND_EDGES)
    edges = edges.point(lambda p: 255 if p > 40 else 0)
    return edges.convert("RGB")


def control_image_to_tensor(control_image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    arr = np.array(control_image).astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=dtype)


def load_controlnet_sd15(device: torch.device, dtype: torch.dtype):
    from diffusers import ControlNetModel

    cn = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=dtype)
    return cn.to(device).eval()


def load_controlnet_sdxl(device: torch.device, dtype: torch.dtype):
    from diffusers import ControlNetModel

    cn = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=dtype)
    return cn.to(device).eval()


@torch.no_grad()
def controlnet_residuals_sd15(
    controlnet, latents_input: torch.Tensor, t, encoder_hidden_states: torch.Tensor,
    control_image_tensor: torch.Tensor, conditioning_scale: float = 0.8,
):
    return controlnet(
        latents_input, t,
        encoder_hidden_states=encoder_hidden_states,
        controlnet_cond=control_image_tensor,
        conditioning_scale=conditioning_scale,
        return_dict=False,
    )


@torch.no_grad()
def controlnet_residuals_sdxl(
    controlnet, latents_input: torch.Tensor, t, encoder_hidden_states: torch.Tensor,
    added_cond_kwargs: dict, control_image_tensor: torch.Tensor, conditioning_scale: float = 0.8,
):
    return controlnet(
        latents_input, t,
        encoder_hidden_states=encoder_hidden_states,
        added_cond_kwargs=added_cond_kwargs,
        controlnet_cond=control_image_tensor,
        conditioning_scale=conditioning_scale,
        return_dict=False,
    )
