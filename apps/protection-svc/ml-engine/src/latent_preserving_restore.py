"""Free visual-quality recovery: makes a protected image look closer to the
original WITHOUT weakening its protection at all, by exploiting the fact
that the VAE is many-to-one.

THE IDEA (2026-08-08). A LoRA trainer never sees pixels. It sees
`vae.encode(x).latent_dist` -- that is the entire interface between an
image and the training process (see aspl_attack.py's denoising_loss and
every real trainer this project's threat model is built on). Two
consequences:

  1. If two images encode to the SAME latent distribution, they are
     *literally indistinguishable* to training. Whatever protection one
     has, the other has exactly, not approximately.
  2. The VAE is lossy and many-to-one -- an 8x spatial downsample into a
     4-channel code. So for any protected image there is a whole
     manifold of images sharing its latent, differing only in detail the
     encoder throws away.

Therefore: search that manifold for the member that looks most like the
ORIGINAL. Any visible damage that lives in the encoder's null space
(texture bleed, moire, colour fringing that carries no latent signal --
in other words, damage that was never buying us any protection) gets
removed for free. Damage that does carry latent signal is pinned in place
by the constraint and survives untouched.

    x_final = argmin_x  perceptual_distance(x, x_original)
              subject to  encode(x) == encode(x_protected)

implemented as a penalty rather than a hard constraint, with the latent
term weighted high enough that it is effectively hard.

WHY BOTH mean AND logvar ARE PINNED: `latent_dist` is a diagonal Gaussian
and trainers call `.sample()`, so the *variance* is part of what training
sees, not just the mean. Matching only the mean would let this step
quietly undo a variance-based attack (see vae_uncertainty_attack.py,
whose entire mechanism lives in logvar). Both are constrained.

WHAT THIS IS NOT: it does not add protection, and it cannot rescue an
attack that damaged the image in latent-carrying ways -- if a perturbation
genuinely needs to be visible to work, this will preserve that visibility.
It only removes the part that was costing visual quality while
contributing nothing. That is the honest scope: free lunch where a free
lunch exists, and no claim beyond it.

Perceptual distance uses model.py's existing VGG19 stack (no new
dependency; LPIPS would be the textbook choice and is a natural upgrade
later) at feature level rather than Gram level -- Gram matrices are
deliberately spatially invariant, which is exactly wrong here: we want
this pixel to look like THAT pixel, not "same texture statistics
somewhere in the image".

STATUS: new, 2026-08-08, unvalidated. The claim "protection is exactly
preserved" is only as true as the latent penalty is tight -- that is
measurable (report the final latent error) but has not yet been checked
against a real train-and-score run.
"""

import argparse

import torch
import torch.nn.functional as F

from style_cloak import load_image_tensor, save_tensor_image

# Early/mid VGG19 conv layers -- spatially localized detail, which is what
# "looks like the original in this spot" actually means. Deliberately not
# model.py's STYLE_LAYERS (those feed Gram matrices, which discard exactly
# the spatial correspondence this needs).
PERCEPTUAL_LAYERS = {"3": "relu1_2", "8": "relu2_2", "17": "relu3_4"}


class _PerceptualFeatures(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        from model import IMAGENET_MEAN, IMAGENET_STD
        from torchvision import models
        from torchvision.models import VGG19_Weights

        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        for m in vgg.modules():
            if isinstance(m, torch.nn.ReLU):
                m.inplace = False
        self.vgg = vgg
        self.mean = IMAGENET_MEAN.to(device)
        self.std = IMAGENET_STD.to(device)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = (x - self.mean) / self.std
        feats = []
        for name, layer in self.vgg._modules.items():
            h = layer(h)
            if name in PERCEPTUAL_LAYERS:
                feats.append(h)
            if name == max(PERCEPTUAL_LAYERS, key=int):
                break
        return feats


def perceptual_distance(extractor: _PerceptualFeatures, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    fa, fb = extractor(a), extractor(b)
    return sum(F.mse_loss(x, y) for x, y in zip(fa, fb)) / len(fa)


def latent_preserving_restore(
    original_path: str,
    protected_path: str,
    checkpoint_path: str,
    output_path: str,
    size: int = 512,
    steps: int = 300,
    lr: float = 0.005,
    latent_weight: float = 1000.0,
    pixel_weight: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> dict:
    from vae_uncertainty_attack import load_vae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = load_vae(checkpoint_path, device, dtype)
    extractor = _PerceptualFeatures(device)

    x_orig = load_image_tensor(original_path, size, device)
    x_prot = load_image_tensor(protected_path, size, device)

    with torch.no_grad():
        target = vae.encode(x_prot.to(dtype) * 2 - 1).latent_dist
        z_target = target.mean.detach().clone()
        logvar_target = target.logvar.detach().clone()
        start_perceptual = perceptual_distance(extractor, x_prot, x_orig).item()

    # Starts from the protected image, not the original: this is a repair
    # of x_prot, and starting at x_orig would begin with a huge latent
    # error the optimizer would have to climb back out of.
    x = x_prot.clone().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)

    trace = []
    for i in range(steps):
        opt.zero_grad()
        x_clamped = x.clamp(0, 1)
        post = vae.encode(x_clamped.to(dtype) * 2 - 1).latent_dist
        latent_err = F.mse_loss(post.mean, z_target) + F.mse_loss(post.logvar, logvar_target)
        percep = perceptual_distance(extractor, x_clamped, x_orig)
        # Small direct pixel term alongside the perceptual one: VGG
        # features are invariant to some low-frequency colour shifts that
        # a human absolutely does notice on a flat background.
        pixel = F.mse_loss(x_clamped, x_orig)
        loss = latent_weight * latent_err + percep + pixel_weight * pixel
        loss.backward()
        opt.step()
        if i % 25 == 0 or i == steps - 1:
            trace.append(
                {"step": i, "latent_err": float(latent_err.item()), "perceptual": float(percep.item())}
            )
            print(
                f"  step {i:3d}  latent_err={latent_err.item():.6e}  perceptual={percep.item():.6f}",
                flush=True,
            )

    x_final = x.detach().clamp(0, 1)
    save_tensor_image(x_final, output_path)

    with torch.no_grad():
        final_post = vae.encode(x_final.to(dtype) * 2 - 1).latent_dist
        final_latent_err = (
            F.mse_loss(final_post.mean, z_target) + F.mse_loss(final_post.logvar, logvar_target)
        ).item()
        final_perceptual = perceptual_distance(extractor, x_final, x_orig).item()

    result = {
        "perceptual_before": start_perceptual,
        "perceptual_after": final_perceptual,
        "perceptual_improvement": start_perceptual - final_perceptual,
        # The honesty number: how far the restored image's latent drifted
        # from the protected image's. Near zero means "training cannot
        # tell these apart, so the protection is intact"; large means this
        # step gave some protection back and the claim does not hold.
        "final_latent_error": final_latent_err,
        "trace": trace,
    }
    print(f"[latent_preserving_restore] wrote {output_path}", flush=True)
    print(
        f"  perceptual {start_perceptual:.6f} -> {final_perceptual:.6f} "
        f"({100 * (start_perceptual - final_perceptual) / max(start_perceptual, 1e-9):.1f}% closer to original), "
        f"latent_err={final_latent_err:.6e}",
        flush=True,
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--protected", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--latent-weight", type=float, default=1000.0)
    parser.add_argument("--pixel-weight", type=float, default=1.0)
    args = parser.parse_args()

    latent_preserving_restore(
        original_path=args.original,
        protected_path=args.protected,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        size=args.size,
        steps=args.steps,
        lr=args.lr,
        latent_weight=args.latent_weight,
        pixel_weight=args.pixel_weight,
    )
