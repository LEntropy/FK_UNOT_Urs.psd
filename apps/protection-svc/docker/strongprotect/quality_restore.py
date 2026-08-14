"""Recovers visual quality from a protected image without giving back
protection -- second attempt, after latent_preserving_restore.py's naive
penalty formulation measurably made things WORSE (perceptual 3.083 ->
3.195 on a real illustration, 2026-08-08).

WHY THE FIRST ATTEMPT FAILED, since the fix depends on it: it minimized
`latent_weight * latent_error + perceptual_distance` from x_protected.
At the starting point latent_error is ~0 by construction, so that term
contributed 1000 * 1.86e-5 = 0.019 against a perceptual term of 3.08 --
a 160x imbalance. The optimizer immediately left the latent constraint
(error spiked to 0.37 by step 25) and then spent its entire remaining
budget climbing back, ending worse on BOTH terms than where it started.
A soft penalty cannot express "this constraint is nearly hard" when the
objective it competes with is orders of magnitude larger at the start.

THE PRINCIPLE IS STILL SOUND. A LoRA trainer only ever sees
`vae.encode(x).latent_dist`; the VAE is a lossy 8x spatial downsample,
so a large family of images share any given latent. Somewhere in that
family is one that looks much more like the original. The first attempt
searched for it badly; this one searches for it in ways that are
structurally incapable of wandering off the constraint.

FOUR CANDIDATES, MEASURED AGAINST A HONEST CONTROL:

  1. linear (CONTROL, not a real proposal): x = x_orig + t*(x_prot -
     x_orig). Blends naively toward the original. This obviously trades
     protection for quality -- that is the point. It defines the
     "free lunch" baseline: any real method must deliver MORE perceptual
     gain per unit of latent drift than this, or it is just an
     obfuscated way of weakening the attack.
  2. highfreq: keep x_prot's low frequencies, take x_orig's high
     frequencies. Rationale: the VAE downsamples 8x, so detail finer
     than ~8px is structurally under-represented in the latent -- it is
     where "free" visual budget should live. Moire/texture-bleed
     artifacts are exactly that band.
  3. chroma: keep x_prot's luma (Y), restore x_orig's chroma (Cb/Cr).
     Rationale: human chroma acuity is far below luma acuity, so colour
     fringing ("번짐") is visually expensive and, being a smooth
     low-amplitude signal, cheap in latent terms.
  4. projected: proper constrained descent. Alternates a small
     perceptual step toward x_orig with an explicit projection back onto
     the latent constraint (inner loop minimizing latent error alone),
     and -- unlike attempt one -- keeps a running BEST checkpoint under a
     latent-error budget, so it can never return something worse than
     what it was given.

Every candidate reports the same two numbers, so they are directly
comparable and the control is meaningful:
  - latent_drift: MSE of (mean, logvar) against x_prot's. 0 means
    training literally cannot distinguish the result from the protected
    image -- protection fully intact.
  - perceptual_gain: LPIPS(x_prot, x_orig) - LPIPS(x_result, x_orig).
    Positive means closer to the original than the protected image was.

WHAT THIS CANNOT DO: it cannot remove damage that is genuinely carrying
the protection. If an attack's visible artifact IS its mechanism, every
method here will correctly refuse to remove it (as a near-zero
perceptual_gain at near-zero latent_drift). That is a real possible
outcome and is reported as such, not tuned around.

STATUS: new 2026-08-08, unvalidated. latent_drift is a proxy -- it says
training sees the same latent, which is a strong argument that
protection is preserved, but the only proof is a real train-and-score
run on the restored image. Do that before believing any number here.
"""

import argparse
import json

import torch
import torch.nn.functional as F

from style_cloak import load_image_tensor, save_tensor_image


# Maximum latent drift a candidate may introduce and still be considered.
# Quality restoration must not cost protection -- that is a hard
# requirement, so this is a gate, not a weight in a tradeoff.
#
# CORRECTED (2026-08-08) from an initial guess of 1e-4, which was wrong by
# four to five orders of magnitude -- real perturbations that are visually
# negligible already sit at drift ~0.3-4, so nothing but an exact no-op
# could ever have passed that gate. The real number was measured, not
# guessed: experiments/quality_restore_validation/drift_calibration.py
# swept a linear blend from the protected image toward the original and
# trained+scored SD1.5 at each point (n=1, single real illustration,
# 2026-08-08):
#
#   drift    0        0.8     4.0     16.5    52.0    88.4 (=original)
#   delta   +0.114   +0.124  +0.128  +0.142  +0.089   +0.005
#
# Protection doesn't just survive up to drift~16 -- delta is HIGHER there
# than at drift=0. It only starts collapsing past drift~50, and is
# essentially gone by the point a candidate has drifted as far as the
# unprotected original itself (drift~88 here). 20 sits with real margin
# below the drop-off (~50) and above the "still fine, possibly better"
# region the calibration found -- not at some symbolic round number.
#
# This is still a proxy, still from one image, still n=1: it tells you a
# candidate near this drift is very unlikely to have lost its effect on
# THIS mechanism against THIS image, not that it is proven safe in
# general. Re-run the calibration sweep (or a small n) before trusting
# this gate for a materially different attack or image distribution.
DRIFT_GATE = 20.0


def _gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(3 * sigma))
    coords = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(coords**2) / (2 * sigma**2))
    k = k / k.sum()
    return k


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur, reflect-padded so edges don't darken."""
    k = _gaussian_kernel(sigma, x.device)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    kh = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    kv = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, kh, groups=c)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    x = F.conv2d(x, kv, groups=c)
    return x


# ITU-R BT.601, the same basis rust-core's watermark.rs uses for its own
# luma extraction -- kept identical so "luma" means one thing project-wide.
_RGB2Y = torch.tensor([0.299, 0.587, 0.114])


def to_ycbcr(x: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = (b - y) * 0.564 + 0.5
    cr = (r - y) * 0.713 + 0.5
    return torch.cat([y, cb, cr], dim=1)


def from_ycbcr(x: torch.Tensor) -> torch.Tensor:
    y, cb, cr = x[:, 0:1], x[:, 1:2] - 0.5, x[:, 2:3] - 0.5
    r = y + 1.403 * cr
    b = y + 1.773 * cb
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return torch.cat([r, g, b], dim=1).clamp(0, 1)


class LatentProbe:
    """Encodes with the VAE and reports drift from a fixed target
    distribution. Both mean and logvar are compared: trainers call
    latent_dist.sample(), so the variance is part of what training sees
    (and is the entire mechanism of vae_uncertainty_attack.py)."""

    def __init__(self, vae, dtype: torch.dtype):
        self.vae = vae
        self.dtype = dtype
        self.z_target = None
        self.logvar_target = None

    @torch.no_grad()
    def set_target(self, x: torch.Tensor) -> None:
        post = self.vae.encode(x.to(self.dtype) * 2 - 1).latent_dist
        self.z_target = post.mean.detach().clone()
        self.logvar_target = post.logvar.detach().clone()

    def drift(self, x: torch.Tensor) -> torch.Tensor:
        post = self.vae.encode(x.to(self.dtype) * 2 - 1).latent_dist
        return F.mse_loss(post.mean, self.z_target) + F.mse_loss(post.logvar, self.logvar_target)

    @torch.no_grad()
    def drift_value(self, x: torch.Tensor) -> float:
        return float(self.drift(x).item())


class Perceptual:
    """LPIPS if available (the right metric -- learned, calibrated against
    human judgements), falling back to unit-normalized VGG19 feature
    distance, which is LPIPS minus the learned per-channel weights. The
    normalization matters: raw VGG MSE (what the failed first attempt
    used) is dominated by whichever layer happens to have the largest
    activations, not by what looks worst."""

    def __init__(self, device: torch.device):
        self.device = device
        self.lpips = None
        try:
            import lpips as _lpips

            self.lpips = _lpips.LPIPS(net="alex").to(device).eval()
            for p in self.lpips.parameters():
                p.requires_grad_(False)
            print("[quality_restore] perceptual metric: LPIPS(alex)", flush=True)
        except Exception as exc:  # noqa: BLE001 -- fallback is a real metric, not a failure
            print(f"[quality_restore] LPIPS unavailable ({exc}) -- using normalized VGG19 features", flush=True)
            self._init_vgg()

    def _init_vgg(self) -> None:
        from model import IMAGENET_MEAN, IMAGENET_STD
        from torchvision import models
        from torchvision.models import VGG19_Weights

        vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(self.device).eval()
        for p in vgg.parameters():
            p.requires_grad_(False)
        for m in vgg.modules():
            if isinstance(m, torch.nn.ReLU):
                m.inplace = False
        self.vgg = vgg
        self.mean = IMAGENET_MEAN.to(self.device)
        self.std = IMAGENET_STD.to(self.device)
        self.layers = {"3", "8", "17"}

    def _vgg_feats(self, x: torch.Tensor) -> list[torch.Tensor]:
        h = (x - self.mean) / self.std
        out = []
        for name, layer in self.vgg._modules.items():
            h = layer(h)
            if name in self.layers:
                # Unit-normalize across channels, exactly what LPIPS does
                # before its learned weighting -- puts every layer on the
                # same scale so none of them dominates by magnitude alone.
                out.append(h / (h.norm(dim=1, keepdim=True) + 1e-8))
            if name == "17":
                break
        return out

    def __call__(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.lpips is not None:
            return self.lpips(a * 2 - 1, b * 2 - 1).mean()
        fa, fb = self._vgg_feats(a), self._vgg_feats(b)
        return sum(F.mse_loss(x, y) for x, y in zip(fa, fb)) / len(fa)

    def value(self, a: torch.Tensor, b: torch.Tensor) -> float:
        with torch.no_grad():
            return float(self(a, b).item())


def restore_linear(x_orig, x_prot, t: float, **_):
    """CONTROL. Not a proposal -- the honest baseline every real method
    must beat on gain-per-unit-drift."""
    return (x_orig + t * (x_prot - x_orig)).clamp(0, 1)


def restore_highfreq(x_orig, x_prot, alpha: float, sigma: float = 2.0, **_):
    """Protected low frequencies + original high frequencies."""
    low_prot = gaussian_blur(x_prot, sigma)
    high_orig = x_orig - gaussian_blur(x_orig, sigma)
    high_prot = x_prot - low_prot
    return (low_prot + (1 - alpha) * high_prot + alpha * high_orig).clamp(0, 1)


def restore_chroma(x_orig, x_prot, alpha: float, **_):
    """Protected luma + original chroma (blended by alpha)."""
    ycc_p, ycc_o = to_ycbcr(x_prot), to_ycbcr(x_orig)
    merged = torch.cat(
        [ycc_p[:, 0:1], (1 - alpha) * ycc_p[:, 1:3] + alpha * ycc_o[:, 1:3]],
        dim=1,
    )
    return from_ycbcr(merged)


def restore_vae_residual(x_orig, x_prot, vae, dtype, alpha: float = 1.0, **_):
    """RESIDUAL TRANSPLANT -- the mechanism built specifically for this
    architecture rather than borrowed from generic image processing.

    `x - decode(encode(x))` is not an approximation of what the VAE
    discards; it IS what this VAE discards from this image, measured
    directly. Everything else in this module guesses at the null space
    ("high frequencies", "chroma"); this reads it off.

    Two pieces:
      clean = decode(encode(x_prot))
          The protected latent rendered by the decoder itself. The
          decoder can only emit points on its own output manifold, so
          adversarial pixel noise, moire and colour fringing -- none of
          which the decoder has any way to express -- are gone. What
          survives is exactly the part of the attack the latent carries,
          i.e. the part that was doing the protecting.
      detail = x_orig - decode(encode(x_orig))
          The original's own VAE residual: the fine structure the encoder
          throws away for THIS image. Adding it should barely move the
          latent, because it is by construction the component the encoder
          is blind to.

      result = clean + alpha * detail

    So the output carries the protected image's latent (protection) with
    the original's real detail (quality), and the artifacts belong to
    neither and are dropped. The residual is only first-order
    null-space (encode is nonlinear, so re-adding it does perturb the
    latent slightly) -- which is why latent_drift is measured rather
    than assumed.
    """
    with torch.no_grad():
        clean = vae.decode(vae.encode(x_prot.to(dtype) * 2 - 1).latent_dist.mode()).sample
        clean = (clean / 2 + 0.5).clamp(0, 1)
        orig_rt = vae.decode(vae.encode(x_orig.to(dtype) * 2 - 1).latent_dist.mode()).sample
        orig_rt = (orig_rt / 2 + 0.5).clamp(0, 1)
        detail = x_orig - orig_rt
        return (clean + alpha * detail).clamp(0, 1)


def restore_adaptive(x_orig, x_prot, vae, dtype, strength: float = 1.0, block: int = 8, **_):
    """SPATIALLY ADAPTIVE RESTORATION -- restores each region toward the
    original in inverse proportion to how much that region actually
    matters to the latent.

    Every other candidate here applies one global alpha, which is the
    wrong shape for the problem: artifacts are not uniform. Moire and
    banding concentrate in smooth areas (sky, flat gradients) -- exactly
    where a human notices them most AND where the latent has the least
    real content to lose. Detailed regions are the opposite. A single
    global blend has to price both at the same rate and therefore gets
    both wrong.

    Sensitivity is measured, not assumed: d||encode(x)||^2/dx, pooled to
    the VAE's own 8x downsampling grid (so a "block" here is exactly one
    latent cell's receptive footprint, not an arbitrary tile). Normalized
    to [0,1] per image, then each block is blended toward the original by
    (1 - sensitivity) * strength -- insensitive blocks get restored
    nearly fully, sensitive ones are left alone.
    """
    xv = x_prot.clone().requires_grad_(True)
    latent = vae.encode(xv.to(dtype) * 2 - 1).latent_dist.mode()
    (grad,) = torch.autograd.grad((latent**2).sum(), xv)

    with torch.no_grad():
        sens = grad.abs().mean(dim=1, keepdim=True)
        sens = F.avg_pool2d(sens, block)
        lo, hi = sens.amin(), sens.amax()
        sens = (sens - lo) / (hi - lo + 1e-8)
        # Nearest-neighbour upsample: the map is defined per latent cell,
        # and smooth interpolation would blur the boundary between "this
        # cell carries protection" and "this one does not".
        weight = F.interpolate(1.0 - sens, size=x_prot.shape[-2:], mode="nearest") * strength
        return (x_prot + weight * (x_orig - x_prot)).clamp(0, 1)


def restore_nullspace_cg(
    x_orig,
    x_prot,
    vae,
    dtype,
    cg_iters: int = 20,
    damping: float = 1e-3,
    scale: float = 1.0,
    **_,
):
    """EXACT FIRST-ORDER NULL-SPACE PROJECTION.

    The other candidates approximate "move toward the original without
    changing the latent". This solves it, to first order:

        min_d ||d - (x_orig - x_prot)||^2   s.t.  J d = 0

    where J is the encoder Jacobian at x_prot. The solution is the target
    direction with its latent-visible component removed:

        d* = t - J^T (J J^T + lambda I)^{-1} J t,   t = x_orig - x_prot

    J is far too large to form (786k x 16k here), so (J J^T) is applied
    as an operator via one VJP + one JVP per conjugate-gradient
    iteration, and never materialized. Damping keeps the solve stable
    where the encoder is near-singular, which is most of it -- that
    near-singularity is precisely the free budget this whole module is
    trying to spend.

    This is the ceiling for any linear method: no first-order-correct
    step can move further toward the original at zero latent cost. It
    exists mainly as a reference point -- if a cheap heuristic like the
    residual transplant lands close to this, the heuristic is good
    enough to ship; if it does not, this says how much is being left
    behind.

    JVP IMPLEMENTATION NOTE (2026-08-08): the obvious approach,
    `torch.func.jvp` (forward-mode AD), fails on this VAE --
    `_scaled_dot_product_efficient_attention` (the fused attention kernel
    diffusers' AutoencoderKL uses in its mid-block) has no forward-AD
    rule implemented in this PyTorch version (NotImplementedError, hit
    live on the first attempt). Forward-mode AD is not actually needed
    here, though: J v (Jacobian-vector product) can be computed with two
    ordinary REVERSE-mode passes instead, via the standard double-
    backward trick -- introduce a dummy cotangent `u` on the output,
    backprop once to get `J^T u` as a function of `u`, then backprop
    THAT (differentiating through the graph a second time, hence
    `create_graph=True`) with respect to `u` in direction `v`. This uses
    only `torch.autograd.grad`, which every op here (including the fused
    attention kernel) already supports in reverse mode.
    """

    def encode_mean(x):
        return vae.encode(x.to(dtype) * 2 - 1).latent_dist.mode()

    def Jt_mv(u, xv=None):  # transpose: latent-space -> pixel-space
        xv = xv if xv is not None else x_prot.clone().requires_grad_(True)
        latent = encode_mean(xv)
        (g,) = torch.autograd.grad(latent, xv, grad_outputs=u, create_graph=u.requires_grad)
        return g

    def J_mv(v):  # Jacobian-vector product: pixel-space -> latent-space, via double-backward
        xv = x_prot.clone().requires_grad_(True)
        latent = encode_mean(xv)
        dummy = torch.zeros_like(latent, requires_grad=True)
        (vjp,) = torch.autograd.grad(latent, xv, grad_outputs=dummy, create_graph=True)
        (out,) = torch.autograd.grad(vjp, dummy, grad_outputs=v)
        return out

    t = (x_orig - x_prot).detach()
    b = J_mv(t)

    # Conjugate gradient on (J J^T + damping I) y = b
    y = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = (r * r).sum()
    for i in range(cg_iters):
        Ap = J_mv(Jt_mv(p)) + damping * p
        denom = (p * Ap).sum()
        if float(denom.abs().item()) < 1e-20:
            break
        alpha = rs / denom
        y = y + alpha * p
        r = r - alpha * Ap
        rs_new = (r * r).sum()
        if float(rs_new.item()) < 1e-16:
            break
        p = r + (rs_new / rs) * p
        rs = rs_new

    # Jt_mv must run OUTSIDE no_grad -- it builds its own fresh graph
    # internally (encode_mean(xv) needs a grad_fn to differentiate
    # through), which an enclosing no_grad would suppress regardless of
    # xv.requires_grad_(True). Detach only the final result.
    d_star = t - Jt_mv(y)
    return (x_prot + scale * d_star).detach().clamp(0, 1)


def restore_projected(
    x_orig,
    x_prot,
    probe: LatentProbe,
    percep: Perceptual,
    outer_steps: int = 60,
    inner_steps: int = 8,
    step_size: float = 0.002,
    drift_budget: float = DRIFT_GATE,
    **_,
):
    """Alternating projected descent, with a best-so-far guard.

    One outer step = one small move toward the original in perceptual
    terms, then an inner loop that pushes latent drift back under budget.
    Unlike a soft penalty, the constraint is restored explicitly every
    iteration instead of merely being priced, so the optimizer cannot
    wander off it the way the first attempt did.

    Returns the best checkpoint seen, defined as lowest perceptual
    distance among iterates that satisfied the drift budget -- initialized
    to x_prot itself, so this function can never return something worse
    than what it was handed.
    """
    x = x_prot.clone()
    best_x = x_prot.clone()
    best_percep = percep.value(x_prot, x_orig)
    start_percep = best_percep

    for outer in range(outer_steps):
        xv = x.clone().requires_grad_(True)
        loss = percep(xv, x_orig)
        (grad,) = torch.autograd.grad(loss, xv)
        with torch.no_grad():
            x = (x - step_size * grad.sign()).clamp(0, 1)

        for _ in range(inner_steps):
            xv = x.clone().requires_grad_(True)
            d = probe.drift(xv)
            if float(d.item()) <= drift_budget:
                break
            (grad,) = torch.autograd.grad(d, xv)
            gn = grad.norm() + 1e-12
            with torch.no_grad():
                x = (x - step_size * grad / gn * grad.numel() ** 0.5 * 0.5).clamp(0, 1)

        d_now = probe.drift_value(x)
        p_now = percep.value(x, x_orig)
        if d_now <= drift_budget and p_now < best_percep:
            best_percep, best_x = p_now, x.clone()
        if outer % 10 == 0:
            print(f"    outer {outer:3d}  drift={d_now:.3e}  percep={p_now:.5f}  best={best_percep:.5f}", flush=True)

    print(f"    projected: perceptual {start_percep:.5f} -> {best_percep:.5f}", flush=True)
    return best_x


def run_all(
    original_path: str,
    protected_path: str,
    checkpoint_path: str,
    out_prefix: str,
    size: int = 512,
    dtype: torch.dtype = torch.float32,
) -> dict:
    from vae_uncertainty_attack import load_vae

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # eager_attention=True: restore_nullspace_cg double-differentiates
    # through this VAE, which the fused SDPA kernel cannot do (see
    # load_vae's own doc) -- every other candidate here only needs a
    # single backward or none at all, so paying eager attention's small
    # overhead everywhere is simpler than juggling two VAE instances.
    vae = load_vae(checkpoint_path, device, dtype, eager_attention=True)
    probe = LatentProbe(vae, dtype)
    percep = Perceptual(device)

    x_orig = load_image_tensor(original_path, size, device)
    x_prot = load_image_tensor(protected_path, size, device)
    probe.set_target(x_prot)

    base_percep = percep.value(x_prot, x_orig)
    print(f"[quality_restore] baseline: perceptual(protected, original) = {base_percep:.5f}", flush=True)

    candidates = []
    for t in (0.9, 0.75, 0.5):
        candidates.append((f"control_linear_t{t}", restore_linear(x_orig, x_prot, t)))
    for a in (0.25, 0.5, 0.75, 1.0):
        candidates.append((f"highfreq_a{a}", restore_highfreq(x_orig, x_prot, a)))
    for a in (0.5, 1.0):
        candidates.append((f"chroma_a{a}", restore_chroma(x_orig, x_prot, a)))

    # Architecture-specific mechanisms (built for this VAE, not borrowed
    # from generic image processing) -- see each function's own doc.
    for a in (0.5, 1.0):
        candidates.append((f"vae_residual_a{a}", restore_vae_residual(x_orig, x_prot, vae, dtype, a)))
    for s in (0.5, 1.0):
        candidates.append((f"adaptive_s{s}", restore_adaptive(x_orig, x_prot, vae, dtype, s)))

    print("[quality_restore] running null-space CG projection ...", flush=True)
    try:
        for s in (0.5, 1.0):
            candidates.append((f"nullspace_cg_s{s}", restore_nullspace_cg(x_orig, x_prot, vae, dtype, scale=s)))
    except Exception as exc:  # noqa: BLE001 -- one candidate failing shouldn't lose the others' measurements
        print(f"  nullspace_cg failed ({type(exc).__name__}: {exc}) -- skipping", flush=True)

    print("[quality_restore] running projected descent ...", flush=True)
    candidates.append(("projected", restore_projected(x_orig, x_prot, probe, percep)))

    results = {}
    for name, x in candidates:
        drift = probe.drift_value(x)
        p = percep.value(x, x_orig)
        gain = base_percep - p
        # Gain per unit drift -- the only number that separates a real
        # null-space exploit from simply blending protection away. The
        # control's value is the bar to beat.
        efficiency = gain / drift if drift > 1e-9 else float("inf")
        results[name] = {
            "latent_drift": drift,
            "perceptual": p,
            "perceptual_gain": gain,
            "gain_per_drift": efficiency,
            # Hard requirement, not a preference: quality work must not
            # cost protection. A candidate over budget is disqualified
            # outright regardless of how good it looks -- reported, so
            # the tradeoff stays visible, but never selectable.
            "passes_drift_gate": drift <= DRIFT_GATE,
        }
        path = f"{out_prefix}_{name}.png"
        save_tensor_image(x, path)
        flag = "OK " if drift <= DRIFT_GATE else "REJECT"
        print(
            f"  [{flag}] {name:22s} drift={drift:.3e}  percep={p:.5f}  gain={gain:+.5f}  gain/drift={efficiency:.3e}",
            flush=True,
        )

    eligible = {
        k: v for k, v in results.items()
        if v["passes_drift_gate"] and v["perceptual_gain"] > 0 and not k.startswith("control_")
    }
    winner = max(eligible, key=lambda k: eligible[k]["perceptual_gain"]) if eligible else None
    print(
        f"\n[quality_restore] winner under drift gate {DRIFT_GATE:.1e}: {winner or 'NONE -- no candidate improved quality at acceptable latent cost'}",
        flush=True,
    )
    if winner:
        print(
            "[quality_restore] REMINDER: latent_drift is a proxy. Near-zero drift argues strongly that "
            "training cannot tell the images apart, but only a real train-and-score run on the restored "
            "image proves protection survived. Do that before shipping this.",
            flush=True,
        )

    return {
        "baseline_perceptual": base_percep,
        "drift_gate": DRIFT_GATE,
        "winner": winner,
        "candidates": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--protected", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    out = run_all(
        original_path=args.original,
        protected_path=args.protected,
        checkpoint_path=args.checkpoint,
        out_prefix=args.out_prefix,
        size=args.size,
    )
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
