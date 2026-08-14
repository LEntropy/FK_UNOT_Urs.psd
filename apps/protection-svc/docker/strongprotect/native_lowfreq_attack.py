"""Native-resolution protection whose perturbation is BAND-LIMITED BY
CONSTRUCTION, instead of hoping it survives resizing (EOT) or repairing
it after the fact (resolution_restore.py). The third and most direct fix
in today's chain of native-resolution attempts.

THE PROGRESSION, all measured for real, all on the same image (SD1.5, n=1):

  1. Attack at 512, reconstruct native resolution from the small
     protected image + the true original's own high-frequency detail
     (resolution_restore.py):            +0.1594 -> +0.0808, then
     -> +0.0519 after fixing the resize-kernel mismatch that caused it.
  2. Attack directly at 1920x1080, one fixed pipeline resize in the loss
     (naive native):                     +0.1632 -> -0.0214 (reversed).
  3. Same, but the loss is evaluated through a RANDOM resize each step
     (EOT over target size + kernel + antialias) -- native_eot_attack.py:
     +0.0438, then +0.0809 with more steps/views/epsilon, then plateaued
     around +0.076-0.081 with further tuning. Real, but capped well
     below (1)'s original ceiling, and epsilon had to go high enough
     (0.045-0.05) to start showing visible blotching -- exactly the
     complaint that prompted this module.

WHY EOT PLATEAUS: it is a probabilistic argument -- "robust in
expectation over sampled resizes" -- applied to a perturbation that is
still, at heart, whatever PGD's per-pixel grad.sign() wants it to be:
maximally high-frequency in native coordinates. Every one of those
resizes is a low-pass filter. A genuinely high-frequency native
perturbation has real energy sitting above EVERY plausible target
Nyquist limit, and that energy is destroyed on every sample, not
sometimes -- EOT can only average over WHICH remaining low-frequency
residue happens to survive best, it cannot stop the destruction from
happening in the first place. Averaging over lossy channels still loses
signal; it does not recover it.

THE FIX: stop the destruction from being possible. Parametrize delta
itself at a LOW resolution (matching the smallest size training
realistically uses, e.g. 512 on the long edge) and upsample it smoothly
(bicubic) to native before adding it to the true original. A
band-limited signal is close to a fixed point of downsample-then-
upsample for ANY reasonable interpolation kernel -- there is no
high-frequency content for different kernels to disagree about, so this
does not merely survive resizing better on average, it is close to
INVARIANT to which exact resize a downstream trainer happens to use.
EOT is kept anyway (see below) as a residual safety margin, not as the
mechanism doing the work anymore.

CONSEQUENCE, not a separate design choice: a signal with no content above
the 512-equivalent frequency band cannot look like pixel grain at native
resolution -- pixel grain IS high-frequency content, by definition. The
same property that fixes the resize-robustness problem also fixes the
"epsilon budget looks blotchy" complaint this module was built to answer,
because both complaints have the same root cause (raw per-pixel PGD
noise) and this removes that root rather than masking it per-symptom the
way hue_lock/structure_align did for the fixed-512 attack.

Because the optimizer's real degrees of freedom are now the same
512-equivalent grid the original, strongest attack used
(vae_uncertainty_attack.py, delta +0.1714 with hue_lock+structure_align+
perceptual_mask), this should have a real shot at recovering close to
that ceiling -- something EOT's fight against active destruction could
not do even with more steps.

hue_lock is kept, computed directly in the low-resolution parameter
space (project delta_param, not delta_native -- projecting after
upsampling would just get partially undone by the next bicubic step
anyway). structure_align is DROPPED here: it existed to manufacture
spatial smoothness by hand for a per-pixel-noise attack; a low-frequency
parametrization has that smoothness built in structurally and does not
need it re-imposed.

STATUS: new 2026-08-09, n=1 SD1.5-only. The reasoning above is the same
kind of clean argument attempt (2) and the first resolution_restore.py
version also had -- both were wrong in ways only a real train+score run
caught, so this WAS re-checked for real rather than trusted on
reasoning alone:

  epsilon=0.03, steps=150, eot_samples=2, recon_weight=0 (logvar only):
      delta +0.0639, visually clean, no grid artifact.
  epsilon=0.05, steps=250, eot_samples=4, recon_weight=0:
      delta +0.0852 -- already beats native_eot_attack.py's best
      (+0.0809, same epsilon) at better visual quality.
  epsilon=0.05, steps=300, eot_samples=1, recon_weight=0.5 (the
  reconstruction term re-enabled -- cheap here since the loss is always
  evaluated on the letterboxed EOT VIEW, at most 1024px, never the full
  native canvas; recon_weight=1.0 needs the full VAE decode per view,
  which is what OOM'd eot_samples>1, so eot_samples had to drop to 1 to
  fit it at all):
      delta +0.0958.
  recon_weight=1.0 (same eot_samples=1, steps=350): delta +0.0433 --
      WORSE. 0.5 is a real optimum, not "more is better".
  epsilon=0.05, steps=500, eot_samples=1, recon_weight=0.5:
      delta +0.1098 -- crossed this project's 0.1 target, native
      1920x1080, no reconstruction step, visible artifact is a faint
      low-frequency grid on flat regions (curtains/walls), not pixel
      grain. This is the current best native-resolution configuration.

  epsilon=0.05, steps=500, eot_samples=1, recon_weight=0.5,
  param_smooth_sigma=1.0:
      delta +0.1371 -- current DELTA record, still the CLI default. The
      grid artifact this sigma introduced turned out, once the
      perturbation itself was finally inspected directly rather than
      judged by eye on the composited image, to be a coarse 20-40px-scale
      blob texture landing in the band human contrast sensitivity peaks
      in -- not the wallpaper print earlier notes blamed it on (see
      compute_spectral_mask's docstring for the retraction).

  Fourteen follow-up experiments across three independent axes -- WHERE
  budget goes (6 spatial/spectral masks: default/loosened/none/
  irregularity/colour-diversity/visibility-weighted-mask), HOW the step is
  computed (4 optimizer variants: Adam alone/+smooth/higher-lr/hybrid-
  polish), and WHAT the loss directly penalises (visibility_loss_weight,
  2 points) -- were run to fix that artifact without losing delta:

  visibility_loss_weight=0.00005 (mean native-band FFT power subtracted
  straight into the ascended loss, see visibility_band_energy):
      delta +0.1054 (-23% vs the record) but the problem 20-40px band
      dropped from 47.6% to 3.7% of the perturbation's spectral energy --
      a 12.8x reduction, by far the largest quality win of any of the
      fourteen variants tried. This is the best QUALITY/DELTA trade-off
      found this session, even though it is not a new delta record --
      worth treating as the practical default if visible artifact matters
      more than the last ~0.03 of delta, and worth a finer sweep (e.g.
      0.00002, 0.00001) next session to see whether delta can be pulled
      closer to +0.1371 while keeping most of this suppression.

  All 14 follow-ups, and their honest failure/partial-success reasoning,
  are in the lora-protection-research memory, not restated here in full.

Still n=1, one image, one seed -- the usual n=30-plus-replication bar
has not been cleared and this is not near production.
"""

import argparse
import random

import torch
import torch.nn.functional as F

from native_eot_attack import _EOT_MODES, _EOT_SIZES, differentiable_letterbox, load_native, save_native
from style_cloak import compute_perceptual_mask


def compute_irregularity_mask(x: torch.Tensor, low: float, high: float, radius: int = 4) -> torch.Tensor:
    """Per-pixel epsilon multiplier, like style_cloak's compute_perceptual_mask
    -- but graded by local IRREGULARITY, not just local gradient magnitude.

    WHY (2026-08-09): compute_perceptual_mask boosts budget wherever the
    Sobel gradient is strong -- "textured", by its own definition. That
    definition can't tell a real artist's messy brushwork apart from a
    perfectly regular repeating print (wallpaper, curtain fabric): both
    have strong local gradients. Measured for real on this project's own
    test image, redistributing budget toward a regular print made the
    result look WORSE (delta_softmask/delta_nomask both visibly spread
    the pattern further, even onto skin, versus the default mask), not
    better -- because a periodic deviation from an already-periodic
    pattern beats against it (moire), which is exactly the kind of
    structured, eye-catching artifact a human notices fastest. Random
    texture hides noise; regular texture can amplify its visibility.

    The fix: grade "safe to hide noise here" by how NON-uniform the local
    gradient strength itself is, not just how strong it is. A repeating
    print has gradient magnitude that barely varies from one repeat to
    the next (low local coefficient of variation); real organic texture
    (foliage, hair, fabric creases, brushwork) has gradient magnitude
    that varies a lot cell-to-cell even at similar mean strength (high
    local CV). This multiplies the existing gradient-magnitude signal by
    that irregularity factor, so a region only gets boosted when it is
    BOTH textured AND non-repeating.
    """
    import torch.nn.functional as _F

    gray = x.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
    gx = _F.conv2d(_F.pad(gray, (1, 1, 1, 1), mode="replicate"), sobel_x)
    gy = _F.conv2d(_F.pad(gray, (1, 1, 1, 1), mode="replicate"), sobel_y)
    edge_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)

    k = 2 * radius + 1
    box = torch.ones(1, 1, k, k, device=x.device) / (k * k)

    def local_mean(t):
        return _F.conv2d(_F.pad(t, (radius,) * 4, mode="reflect"), box)

    mean_e = local_mean(edge_mag)
    var_e = (local_mean(edge_mag**2) - mean_e**2).clamp(min=0)
    std_e = torch.sqrt(var_e + 1e-8)
    irregularity = std_e / (mean_e + 1e-3)

    signal = edge_mag * irregularity
    signal = signal.expand(-1, 3, -1, -1)
    lo, hi = signal.amin(), signal.amax()
    norm = (signal - lo) / (hi - lo + 1e-8)
    return low + norm * (high - low)


def compute_color_diversity_mask(x: torch.Tensor, low: float, high: float, radius: int = 6) -> torch.Tensor:
    """Per-pixel epsilon multiplier graded by local HUE DIVERSITY, not
    gradient strength or its regularity.

    WHY (2026-08-09): compute_irregularity_mask still judges "safe" by
    the STRUCTURE of gradients (uniform vs bursty), which also failed to
    beat the plain gradient mask in a real train+score check. A
    different signal entirely: a repeating print is very often built
    from a small palette (2-3 hues repeating), even when its gradient
    magnitude is high and locally irregular from aliasing/antialiasing
    at print edges. Real organic texture (foliage, hair, brushwork,
    fabric folds) usually spans a genuinely wider range of hues at
    similar spatial scale, because it is not built from a repeated
    tile. Judging by hue variety rather than any property of the
    gradient sidesteps the whole "is this edge structured or not"
    question this module's other two mask variants got stuck on.

    Reuses the same per-pixel colour-direction math hue_lock already
    computes (unit vector in RGB, brightness divided out) -- here it
    measures how much each pixel's own colour direction DISAGREES with
    its neighbourhood's average colour direction, instead of projecting
    the perturbation onto it. A pixel matching its neighbourhood's
    average hue closely (low disagreement) is part of a locally
    uniform-hue region -- likely a flat surface OR a low-palette
    repeating print, both cases this project's own pilots showed are bad
    places to add visible budget. High disagreement -- many different
    hues fighting for the same neighbourhood's average -- means real
    colour variety, judged safer to hide budget in.
    """
    import torch.nn.functional as _F

    u = x / (x.norm(dim=1, keepdim=True) + 1e-6)
    k = 2 * radius + 1
    box = torch.ones(3, 1, k, k, device=x.device) / (k * k)
    local_avg = _F.conv2d(_F.pad(u, (radius,) * 4, mode="reflect"), box, groups=3)
    local_avg = local_avg / (local_avg.norm(dim=1, keepdim=True) + 1e-6)

    agreement = (u * local_avg).sum(dim=1, keepdim=True)  # in [-1, 1], 1 = same hue as neighbourhood
    diversity = (1 - agreement).clamp(min=0).expand(-1, 3, -1, -1)
    lo, hi = diversity.amin(), diversity.amax()
    norm = (diversity - lo) / (hi - lo + 1e-8)
    return low + norm * (high - low)


def compute_visibility_mask(
    ph: int,
    pw: int,
    native_h: int,
    native_w: int,
    gamma: float,
    pixels_per_degree: float = 48.0,
    max_boost: float = 6.0,
    protect_lowfreq: float = 0.006,
    device=None,
    dtype=torch.float32,
) -> tuple[torch.Tensor, dict]:
    """Reshape the perturbation's SPECTRUM away from the band the human eye
    sees best and into the band the VAE actually reads.

    HOW THIS WAS FOUND (2026-08-09). Seven attempts to remove this
    artefact had failed: five spatial masks (default/loosened/none/
    irregularity/colour-diversity), a spectral notch, and an aliasing
    theory. All seven were reasoned from a guess about the CAUSE. The
    eighth step was to stop guessing and measure the perturbation itself
    -- delta = protected - original -- which had never once been looked at
    directly. Two things fell out immediately, on the best run to date
    (v_smooth1, delta +0.1371):

      * Its energy by native spatial frequency:
            >100px   3.5%
            40-100px  21.2%
            20-40px   47.6%   <-- half of everything, in one octave
            11-20px   21.8%
            7.5-11px   4.3%
            <7.5px     1.6%
      * Amplified 20x it is plainly a COARSE MOTTLED BLOB TEXTURE at
        roughly 20-40px across. Not pixel grain. Not interference with any
        print.

    That 20-40px band is, for an image viewed at any normal distance,
    almost exactly where human contrast sensitivity PEAKS. The
    perturbation was putting half its energy in the single most visible
    place available to it. That is the artefact -- not moire, not the
    wallpaper, not budget placement.

    Worse, this was self-inflicted and recent: param_smooth_sigma (the
    change that lifted delta +0.1098 -> +0.1371) blurs the step in
    parameter space, which strips the fine end and pushes energy DOWN into
    exactly this band -- measured, 27.4% -> 47.6% in 20-40px between
    v_lf_recon5 and v_smooth1. It bought delta and traded away visibility,
    and the "grid" complaint that persisted after it is precisely the
    coarse blobs it created.

    THE OPENING: the visible band and the useful band are not the same
    band. The VAE encoder this attack maximises uncertainty in works on
    8x8 patches, so its latent responds most to structure around 8-16px --
    while the eye's sensitivity is already falling off there. Nothing
    forces the perturbation to sit at 20-40px; PGD put it there because
    nothing told it not to. So weight the step in the Fourier domain by
    the INVERSE of a contrast-sensitivity curve: attenuate where the eye
    is sharp, leave alone where it is not. Energy relocates toward the
    finer end, which costs visibility little and should cost the attack
    nothing -- possibly less than nothing.

    Two limits are respected by construction. Energy is never pushed above
    the parameter grid's own band (there is nothing up there to push into
    -- that is this whole module's premise, and attempts 1-3 in the module
    docstring are what happens when you ignore it). And the very lowest
    frequencies are attenuated too, not boosted, because 1/CSF rises again
    at DC and an unchecked boost there is a global colour cast -- its own,
    worse, visible artefact.

    `pixels_per_degree` is the one soft assumption: it converts cycles/px
    to the cycles/degree the CSF model is defined in, and it depends on
    viewing distance, which is not knowable here. 48 px/deg corresponds to
    a 1920px image filling roughly 40 degrees of view. It is exposed
    rather than buried precisely because it is an assumption, and `gamma`
    (0 = off) is what should actually be tuned against real measurements.
    """
    fy = torch.fft.fftfreq(ph, device=device, dtype=dtype).view(-1, 1)
    fx = torch.fft.rfftfreq(pw, device=device, dtype=dtype).view(1, -1)
    # Parameter-grid frequency -> native frequency: the grid is a downscaled
    # copy, so one cycle per param-pixel is one cycle per (native/param)
    # native pixels. Visibility is a property of the final native image.
    scale = ph / native_h
    radial_param = torch.sqrt(fy**2 + fx**2)
    radial_native = radial_param * scale

    f_cpd = radial_native * pixels_per_degree
    a = 0.114 * f_cpd
    # Mannos-Sakrison contrast sensitivity.
    csf = 2.6 * (0.0192 + a) * torch.exp(-(a**1.1))

    weight = (csf.amax() / csf.clamp(min=1e-6)) ** gamma
    weight = weight.clamp(max=max_boost)
    weight = weight / weight.amax()
    # Kill the DC end rather than letting 1/CSF boost it into a colour cast.
    weight = weight * (radial_param > protect_lowfreq).to(dtype)

    stats = {}
    for lo, hi, name in (
        (0.0, 0.010, "<0.010"),
        (0.010, 0.025, "0.010-0.025"),
        (0.025, 0.050, "0.025-0.050"),
        (0.050, 0.0889, "0.050-0.089"),
        (0.0889, 0.1333, "0.089-0.133"),
    ):
        sel = (radial_native >= lo) & (radial_native < hi)
        stats[name] = float(weight[sel].mean()) if bool(sel.any()) else 0.0
    return weight, stats


def visibility_band_energy(delta_native: torch.Tensor, low_cyc: float, high_cyc: float) -> torch.Tensor:
    """Differentiable power of `delta_native` inside a native-cycles-per-
    pixel band -- the LOSS-TERM counterpart to compute_visibility_mask's
    fixed 1/CSF gradient-step mask.

    WHY A LOSS TERM AND NOT ANOTHER MASK (2026-08-09, third axis this
    session after masking (6 variants, all failed) and optimizer-rule
    (4 variants, all failed to beat +0.1371)): every masking approach --
    spatial or spectral, including compute_visibility_mask -- multiplies
    the gradient STEP by a fixed weight decided in advance, before the
    optimizer has any say in whether that frequency was actually worth
    attacking. A frequency that happens to matter a lot for VAE
    uncertainty gets suppressed exactly as hard as one that does not, and
    the +0.1371 delta lost is not recoverable -- the attack never gets a
    chance to argue for it. Folding the SAME visible-band penalty into the
    loss instead lets gradient competition decide per step: if the
    uncertainty gradient in that band is large, it can still outweigh the
    penalty gradient and win; if it is small, the penalty wins and that
    energy shrinks. This is strictly more information than a fixed mask
    has access to, because it sees the actual competing gradient, not a
    static heuristic about where textures usually are.

    Computed on `delta_native` -- the full-resolution accumulated
    perturbation actually added to the image, not delta_param -- because
    the visible-band definition (native pixels-per-cycle) is a property of
    what a VIEWER sees, which is native resolution, matching
    analyze_delta.py's own band definitions used to diagnose this
    artefact in the first place. Cheap relative to the VAE forward passes
    already dominating each step: one rfft2 over the native canvas."""
    g = delta_native.mean(dim=1, keepdim=True)
    power = torch.fft.rfft2(g).abs() ** 2
    h, w = g.shape[-2:]
    fy = torch.fft.fftfreq(h, device=g.device, dtype=g.dtype).view(-1, 1)
    fx = torch.fft.rfftfreq(w, device=g.device, dtype=g.dtype).view(1, -1)
    radial = torch.sqrt(fy**2 + fx**2)
    band = ((radial >= low_cyc) & (radial < high_cyc)).to(power.dtype)
    # mean, not sum: a native rfft2 has ~1M bins, so a raw sum lives on a
    # completely different scale than the ~O(10) VAE uncertainty loss it
    # is added to -- caught locally before any GPU run via a 3-step
    # dry-run where vis_band_energy hit 3e7 and swallowed the entire loss.
    n_bins = band.sum().clamp(min=1.0)
    return (power * band).sum() / n_bins


def compute_spectral_mask(
    x: torch.Tensor,
    notch: bool = True,
    notch_strength: float = 1.0,
    notch_bandwidth: int = 2,
    notch_z: float = 3.0,
    envelope_alpha: float = 0.0,
    bg_radius: int = 6,
    protect_lowfreq: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Notch the perturbation out of any sharp periodic peaks in the
    original's spectrum, and/or shape it to the original's smooth spectral
    envelope.

    STATUS -- PREMISE REFUTED, KEPT ONLY AS A TOOL (2026-08-09). This was
    built to kill a moire, on the theory that the artwork had a repeating
    print whose frequency the perturbation was beating against. Measured
    on this project's own test image before spending any GPU time, that
    theory did not survive:

      * No radial peak anywhere in the native spectrum clears z=1.5 -- the
        artwork's spectrum is smooth and broadband, with no narrowband
        print to notch. At the default notch_z=3.0 this function finds
        ZERO peaks and is a silent no-op.
      * Only 0.30% of non-DC energy sits above the parameter grid's
        Nyquist, so there is almost nothing to alias back either, and a
        stronger prefilter moves the top-band energy only 1.147% ->
        0.978%.

    The real cause turned out to be where the perturbation puts its OWN
    energy, not any interaction with the artwork -- see
    compute_visibility_mask, which is the mechanism that followed from
    actually measuring it. This function stays because the machinery is
    sound and an image that genuinely does carry a strong periodic print
    (halftone, screen-door, woven fabric photographed head-on) would be a
    real use for it -- but it is off by default and notch_z would have to
    drop to ~2.5 to fire at all here.

    Returns (step_mask, notch_hard, n_peaks). `notch_hard` is the binary
    peak mask alone: multiplying by it is an exact orthogonal projection,
    hence idempotent and safe to re-apply after the spatial clamp, which
    is nonlinear and would otherwise leak broadband energy back into the
    notched bins -- the same reason hue_lock is re-projected there. The
    envelope term is a reweighting, not a projection, so it stays in the
    step path only.
    """
    gray = x.mean(dim=1, keepdim=True)
    ph, pw = gray.shape[-2:]
    spec = torch.fft.rfft2(gray)
    logp = torch.log(spec.abs() ** 2 + 1e-12)

    k = 2 * bg_radius + 1
    box = torch.ones(1, 1, k, k, device=x.device, dtype=logp.dtype) / (k * k)
    bg = F.conv2d(F.pad(logp, (bg_radius,) * 4, mode="reflect"), box)

    fy = torch.fft.fftfreq(ph, device=x.device, dtype=logp.dtype).view(-1, 1)
    fx = torch.fft.rfftfreq(pw, device=x.device, dtype=logp.dtype).view(1, -1)
    radial = torch.sqrt(fy**2 + fx**2)

    notch_hard = torch.ones_like(bg)
    n_peaks = 0
    if notch:
        resid = logp - bg
        peaks = (resid > notch_z * resid.std()).to(logp.dtype)
        peaks = peaks * (radial > protect_lowfreq).to(logp.dtype)
        if notch_bandwidth > 0:
            kk = 2 * notch_bandwidth + 1
            peaks = F.max_pool2d(peaks, kk, stride=1, padding=notch_bandwidth)
        n_peaks = int(peaks.sum().item())
        notch_hard = 1.0 - peaks

    step_mask = 1.0 - notch_strength * (1.0 - notch_hard)
    if envelope_alpha > 0:
        env = torch.exp(0.5 * bg)  # amplitude envelope = sqrt(power)
        env = env / (env.amax() + 1e-12)
        step_mask = step_mask * env**envelope_alpha

    return step_mask, notch_hard, n_peaks


from vae_uncertainty_attack import load_vae, vae_uncertainty_loss


def native_lowfreq_attack(
    original_path: str,
    checkpoint_path: str,
    output_path: str,
    param_size: int = 512,
    epsilon: float = 0.03,
    steps: int = 150,
    step_size: float | None = None,
    eot_samples: int = 2,
    logvar_weight: float = 1.0,
    recon_weight: float = 0.0,
    hue_lock: bool = True,
    hue_lock_radius: int = 2,
    perceptual_mask: bool = True,
    mask_low: float = 0.3,
    mask_high: float = 1.7,
    irregularity_mask: bool = False,
    color_diversity_mask: bool = False,
    param_smooth_sigma: float = 0.0,
    spectral_notch: bool = False,
    notch_strength: float = 1.0,
    notch_bandwidth: int = 2,
    notch_z: float = 3.0,
    spectral_envelope: float = 0.0,
    visibility_gamma: float = 0.0,
    visibility_ppd: float = 48.0,
    optimizer: str = "sign",
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_lr: float | None = None,
    adam_polish_steps: int = 0,
    visibility_loss_weight: float = 0.0,
    visibility_loss_low: float = 0.025,
    visibility_loss_high: float = 0.050,
    eot_sizes: tuple[int, ...] | None = None,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
) -> dict:
    """param_smooth_sigma>0 (2026-08-09 addition) fixes a visible artifact
    the first version of this module had: band-limiting relative to NATIVE
    resolution (the whole point of this module) does not stop delta_param
    from being checkerboard-noisy AT ITS OWN param_size resolution --
    grad.sign() still updates each of its cells independently. Bicubic
    upsampling spreads that cell-scale oscillation over ~(native/param_size)
    native pixels without removing it, and on an image with its own
    repeating print (wallpaper, curtain fabric) that residual periodicity
    can beat against the artwork's own pattern -- a faint moire/grid,
    exactly what showed up in the first real result (delta +0.1098).

    The fix is the same principle as vae_uncertainty_attack.py's
    smooth_sigma, just applied one level down: blur the gradient STEP in
    PARAMETER space (not the native-resolution image) before it
    accumulates into delta_param. Only the step is blurred, not
    delta_param itself, for the same non-idempotence reason documented in
    vae_uncertainty_attack.py -- re-blurring an already-accumulated signal
    every iteration compounds across all `steps`, smearing far past
    param_smooth_sigma; blurring only the fresh increment keeps total
    smoothing bounded regardless of step count.

    `optimizer="adam"` (2026-08-09, new axis after SIX masking variants --
    spatial (default/loosened/none/irregularity/colour-diversity) and
    spectral (visibility_gamma) -- all failed to beat the +0.1371
    param_smooth_sigma baseline). Every one of those six changed WHERE or
    at WHICH FREQUENCY the step could land -- a fixed multiplicative mask
    applied on top of the same underlying step. None of them touched HOW
    that step itself is computed, and the step's own construction is where
    this module's own docstring already located the root cause: plain PGD
    takes `grad.sign()` every iteration, which keeps only the DIRECTION of
    each parameter cell's gradient and throws away its magnitude, and does
    so independently per cell. Two adjacent cells with strongly correlated
    real gradients (as any smooth loss landscape over a natural image
    produces) can still get opposite-magnitude, sign-only steps if their
    raw gradients merely differ in relative size -- that is what makes the
    accumulated signal checkerboard-noisy at the param grid's own
    resolution in the first place, the very thing param_smooth_sigma has
    to blur back out after the fact.

    Adam removes the reason for that noise instead of cleaning it up
    afterward: it accumulates the RAW gradient (first moment, `m`) and its
    squared magnitude (second moment, `v`) across steps, so cells with
    small or noisy gradients naturally take small, damped steps while
    cells with large, consistent gradients take confident ones -- the
    per-cell relationship between adjacent gradients that sign() discards
    is preserved in exactly the situations where it matters. No blur or
    mask is layered on top when this path is used; the update is smooth
    because its INPUT is no longer artificially decorrelated, not because
    an extra step scrubbed the artifact out afterward. param_smooth_sigma
    and every mask_fn above remain compatible if the user explicitly
    re-enables them, but they are not needed to explain or motivate this
    path -- this replaces the mechanism that CREATED the artifact, not one
    more attempt to hide it after creation.

    `adam_lr` defaults to `step_size` (or `epsilon/4` if that too is
    unset) -- same magnitude PGD was already using per step, so any delta
    beats sign-PGD by using its gradient information better at an equal
    step budget, not simply by taking bigger steps.

    MEASURED (2026-08-09, n=1): Adam alone (`--optimizer adam`,
    param_smooth_sigma=0) scored delta +0.1109 -- above the first
    0.1-crossing milestone (+0.1098) but below the sign+smooth record
    (+0.1371). Stacking param_smooth_sigma on top of Adam made it WORSE
    (+0.0944) -- that blur exists specifically to undo sign()'s
    per-cell-independent noise, and Adam's step never had that noise to
    undo, so the blur only throws away real gradient information.
    Doubling adam_lr (0.03 vs the 0.0125 default) did not move delta
    (+0.1081, within n=1 noise of the default). Across all three variants
    the spectral shift away from the visible 20-40px band and into the
    VAE-relevant 11-20px/7.5-11px bands was consistent and in the
    direction visibility_gamma had tried and failed to achieve by
    masking -- so this axis is real for QUALITY, just not (yet, at n=1)
    for raw delta. `adam_polish_steps` (below) exists to try capturing
    both: sign-PGD's proven strength for the bulk of the budget, Adam's
    proven spectral cleanliness for a short finishing pass.

    `adam_polish_steps>0` runs sign-PGD (with param_smooth_sigma if set)
    for `steps - adam_polish_steps` iterations to reach the same
    optimization depth the +0.1371 record reached, THEN switches to Adam
    for the final `adam_polish_steps` -- warm-started from the converged
    delta_param, with its own fresh moment buffers (m, v start at zero at
    the switch, not before it, so the polish phase is a clean Adam run
    over a good starting point, not a continuation of irrelevant early
    statistics). The hope is that Adam's smoother, gradient-magnitude-
    aware steps can reshape the FINAL few percent of the perturbation's
    energy toward less visible frequencies without undoing the strength
    sign-PGD already banked -- untested at time of writing, see the
    module's docstring changelog / lora-protection-research memory for
    whether it actually worked.

    MEASURED: adam_polish_steps=100 (of 500) scored +0.1125, barely above
    Adam-alone and still well below +0.1371, and its spectrum turned out
    nearly IDENTICAL to pure sign+smooth (the 20-40px band barely moved,
    47.6% -> 46.7%) -- the hypothesis failed. By the time the polish phase
    starts, most of delta_param's cells are already clamped at
    +/-epsilon_map from the first 400 sign-PGD steps, so a late-arriving
    Adam phase has almost no room left to reshape anything; it can only
    nudge already-saturated cells. Warm-starting from a converged,
    boundary-saturated solution does not work the way warm-starting an
    unconverged one would -- this axis (mixing optimizers across a single
    run) is a dead end for this mechanism specifically because of that
    saturation, not a general property of hybrids.

    `visibility_loss_weight` (2026-08-09, THIRD axis this session, after
    masking (6 variants, all failed) and optimizer choice (4 variants, all
    failed) both plateaued): every previous quality attempt fixed a
    weight for each frequency/location BEFORE the optimizer saw the
    gradient -- a static bet about which bins deserve less budget,
    decided in advance. This instead folds `visibility_band_energy(...)`
    (a differentiable readout of how much power the CURRENT delta_native
    carries inside the visible 20-40px-equivalent band, see that
    function's own docstring) directly into the loss the optimizer already
    ascends, so the trade-off is settled by actual gradient competition
    every single step: a bin the uncertainty objective genuinely needs can
    still win against the penalty; a bin it does not need loses budget
    without ever needing an outside heuristic to say so. This is strictly
    more informed than any fixed mask, because it is the only mechanism
    this session that lets the ATTACK ITSELF answer "is this bin worth the
    visibility cost", rather than a human-authored heuristic answering on
    its behalf.

    `eot_sizes` (2026-08-09, added for the SDXL crossover): defaults to
    None, which keeps native_eot_attack.py's _EOT_SIZES = (512, 512, 640,
    768, 1024) -- weighted toward 512 because every experiment through
    this point in the module's history targeted SD1.5, whose real LoRA
    trainers resize to 512. load_vae/vae_uncertainty_loss never
    hardcode an architecture (they only touch vae.encode/.decode and the
    posterior's own logvar/mode, no scaling_factor assumption, no UNet
    involvement -- see vae_uncertainty_attack.py's own module docstring:
    "SD1.5, SDXL, SD3, Flux all use a KL-VAE of this same family"), so
    pointing checkpoint_path at an SDXL single-file checkpoint should work
    mechanically without touching that code at all. What DOES need to
    change for SDXL is which resizes the EOT loop simulates: a real SDXL
    LoRA trainer resizes to ~1024, not ~512, so leaving _EOT_SIZES at its
    SD1.5-weighted default would train this attack's robustness against
    the wrong downstream operator for that architecture. Pass an
    SDXL-appropriate tuple, e.g. (1024, 1024, 896, 1152, 1280), to fix
    that; param_size should also move to 1024 to match (already exposed,
    no code change needed there)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(seed)
    eot_sizes = eot_sizes if eot_sizes is not None else _EOT_SIZES
    vae = load_vae(checkpoint_path, device, dtype)

    x0 = load_native(original_path, device)
    h, w = x0.shape[-2:]

    # The optimizer's real degrees of freedom: a grid at param_size on the
    # long edge, same scale as this project's original validated attack.
    scale = param_size / max(h, w)
    ph, pw = max(1, round(h * scale)), max(1, round(w * scale))
    delta_param = torch.zeros((1, 3, ph, pw), device=device, dtype=x0.dtype, requires_grad=True)
    step_size = step_size if step_size is not None else epsilon / 4

    def upsample_delta(d: torch.Tensor) -> torch.Tensor:
        # Always bicubic + antialias: this ONE upsample is fully under our
        # control (unlike the downstream downsample, which is not), so it
        # is fixed and smooth on purpose -- the entire point is that delta
        # never has content above this band, regardless of what any
        # downstream trainer's resize does.
        return F.interpolate(d, size=(h, w), mode="bicubic", align_corners=False, antialias=True)

    # Low-res original, for constraints computed natively in parameter
    # space -- keeps hue_lock's colour direction consistent with what
    # delta_param actually controls, rather than a native-resolution map
    # that the upsample would then partially misalign.
    x0_low = F.interpolate(x0, size=(ph, pw), mode="bicubic", align_corners=False, antialias=True)

    c_hat = None
    if hue_lock:
        r = hue_lock_radius
        k = 2 * r + 1
        kernel = torch.ones(3, 1, k, k, device=device, dtype=x0.dtype) / (k * k)
        local = F.conv2d(F.pad(x0_low, (r,) * 4, mode="reflect"), kernel, groups=3)
        c_hat = local / (local.norm(dim=1, keepdim=True) + 1e-6)

    def hue_project(d):
        if c_hat is None:
            return d
        return (d * c_hat).sum(dim=1, keepdim=True) * c_hat

    blur_kernel = None
    if param_smooth_sigma > 0:
        radius = max(1, int(3 * param_smooth_sigma))
        coords = torch.arange(-radius, radius + 1, device=device, dtype=x0.dtype)
        g = torch.exp(-(coords**2) / (2 * param_smooth_sigma**2))
        g = g / g.sum()
        blur_kernel = (g, radius)

    def param_blur(d):
        if blur_kernel is None:
            return d
        g, r = blur_kernel
        kh = g.view(1, 1, 1, -1).expand(3, 1, 1, -1)
        kv = g.view(1, 1, -1, 1).expand(3, 1, -1, 1)
        d = F.pad(d, (r, r, 0, 0), mode="reflect")
        d = F.conv2d(d, kh, groups=3)
        d = F.pad(d, (0, 0, r, r), mode="reflect")
        d = F.conv2d(d, kv, groups=3)
        return d

    spec_step_mask = None
    spec_notch_hard = None
    if spectral_notch or spectral_envelope > 0:
        spec_step_mask, spec_notch_hard, n_peaks = compute_spectral_mask(
            x0_low,
            notch=spectral_notch,
            notch_strength=notch_strength,
            notch_bandwidth=notch_bandwidth,
            notch_z=notch_z,
            envelope_alpha=spectral_envelope,
        )
        total_bins = spec_notch_hard.numel()
        print(
            f"[native_lowfreq_attack] spectral: notched {n_peaks}/{total_bins} bins "
            f"({100.0 * n_peaks / total_bins:.2f}% of the spectrum), envelope_alpha={spectral_envelope}",
            flush=True,
        )

    if visibility_gamma > 0:
        # Multiplies into the same step_mask as the notch (both are
        # multiplicative Fourier weights on the fresh gradient step, so they
        # compose for free) -- this is the mechanism that actually follows
        # from measuring v_smooth1's own spectrum, see
        # compute_visibility_mask's docstring for the full derivation.
        vis_weight, vis_stats = compute_visibility_mask(
            ph, pw, h, w, gamma=visibility_gamma, pixels_per_degree=visibility_ppd,
            device=device, dtype=x0.dtype,
        )
        spec_step_mask = vis_weight if spec_step_mask is None else spec_step_mask * vis_weight
        print(
            "[native_lowfreq_attack] visibility weight by native band (1.0=full step kept): "
            + ", ".join(f"{k}={v:.3f}" for k, v in vis_stats.items()),
            flush=True,
        )

    def spectral_project(d, mask):
        if mask is None:
            return d
        return torch.fft.irfft2(torch.fft.rfft2(d) * mask, s=(ph, pw))

    def project_step(d):
        return hue_project(spectral_project(param_blur(d), spec_step_mask))

    epsilon_map = epsilon
    if perceptual_mask:
        # Computed on the low-res view -- texture that only exists above
        # this band is invisible to delta_param anyway, so the mask should
        # judge "texture" at the resolution the attack actually operates in.
        if color_diversity_mask:
            mask_fn = compute_color_diversity_mask
        elif irregularity_mask:
            mask_fn = compute_irregularity_mask
        else:
            mask_fn = compute_perceptual_mask
        epsilon_map = mask_fn(x0_low, mask_low, mask_high) * epsilon

    lr = adam_lr if adam_lr is not None else step_size
    polish_start = max(0, steps - adam_polish_steps) if adam_polish_steps > 0 else steps
    m = v = None  # lazily created at the sign->adam switch, not before

    losses = []
    for i in range(steps):
        # optimizer="adam" forces Adam from step 0; adam_polish_steps>0
        # instead runs sign-PGD up to polish_start (banking the same
        # strength the +0.1371 record reached) and only switches to Adam
        # for the tail -- see the docstring above for why the switch, not
        # a blend, and why moment buffers reset exactly at that point.
        use_adam = optimizer == "adam" or i >= polish_start
        if use_adam and m is None:
            m = torch.zeros_like(delta_param)
            v = torch.zeros_like(delta_param)
            polish_t0 = i

        delta_native = upsample_delta(delta_param)
        x_adv = (x0 + delta_native).clamp(0, 1)

        total = torch.zeros((), device=device)
        views = []
        for _ in range(eot_samples):
            size = rng.choice(eot_sizes)
            mode, aa = rng.choice(_EOT_MODES)
            views.append((size, mode, aa))
            train_view = differentiable_letterbox(x_adv, size, mode, aa)
            total = total + vae_uncertainty_loss(vae, train_view, dtype, logvar_weight, recon_weight)
        loss = total / eot_samples

        vis_penalty = None
        if visibility_loss_weight > 0:
            # Subtracted, not masked in: the attack ASCENDS `loss`, so
            # subtracting a term here makes ascent actively work against
            # growing it -- gradient competition between the uncertainty
            # objective and this penalty, resolved per step, per
            # frequency bin, not decided in advance by a fixed mask.
            vis_penalty = visibility_band_energy(delta_native, visibility_loss_low, visibility_loss_high)
            loss = loss - visibility_loss_weight * vis_penalty

        (grad,) = torch.autograd.grad(loss, delta_param)
        with torch.no_grad():
            if use_adam:
                # Ascent, not descent: this attack MAXIMISES the VAE
                # uncertainty loss (loss is already negated -- see the
                # step printouts' large negative values), so the update
                # follows +grad exactly the way sign-PGD used +grad.sign().
                t = i - polish_t0 + 1
                m.mul_(adam_beta1).add_(grad, alpha=1 - adam_beta1)
                v.mul_(adam_beta2).addcmul_(grad, grad, value=1 - adam_beta2)
                m_hat = m / (1 - adam_beta1**t)
                v_hat = v / (1 - adam_beta2**t)
                raw_step = m_hat / (v_hat.sqrt() + 1e-8)
                delta_param += lr * project_step(raw_step)
            else:
                delta_param += step_size * project_step(grad.sign())
            delta_param.clamp_(min=-epsilon_map, max=epsilon_map)
            # The clamp is nonlinear, so it leaks broadband energy back into
            # the notched bins -- re-project, same reason hue_lock is
            # re-projected here. Only the binary notch goes in this path
            # (an exact orthogonal projection, idempotent); the envelope
            # term is a reweighting and would compound across steps.
            # hue_project stays last, keeping the validated ordering: the
            # two projections are onto different subspaces and do not
            # commute, so whichever runs last is the one held exactly.
            delta_param.copy_(hue_project(spectral_project(delta_param, spec_notch_hard)))

        if i % 10 == 0 or i == steps - 1:
            losses.append(float(loss.item()))
            vis_str = f"  vis_band_energy={vis_penalty.item():.6f}" if vis_penalty is not None else ""
            print(f"  step {i:3d}  loss={loss.item():.6f}  views={views}{vis_str}", flush=True)

    with torch.no_grad():
        x_final = (x0 + upsample_delta(delta_param)).clamp(0, 1)
    save_native(x_final, output_path)
    print(f"[native_lowfreq_attack] wrote {output_path} ({w}x{h}, native, param grid {pw}x{ph})", flush=True)
    return {"loss_trace": losses, "final_loss": losses[-1] if losses else None, "native_size": (w, h), "param_size": (pw, ph)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--param-size", type=int, default=512)
    parser.add_argument("--hue-lock-radius", type=int, default=2)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--eot-samples", type=int, default=1)
    parser.add_argument("--logvar-weight", type=float, default=1.0)
    parser.add_argument("--recon-weight", type=float, default=0.5)
    parser.add_argument("--no-hue-lock", action="store_true")
    parser.add_argument("--no-perceptual-mask", action="store_true")
    parser.add_argument("--mask-low", type=float, default=0.3)
    parser.add_argument("--mask-high", type=float, default=1.7)
    parser.add_argument("--irregularity-mask", action="store_true")
    parser.add_argument("--color-diversity-mask", action="store_true")
    parser.add_argument("--param-smooth-sigma", type=float, default=0.0)
    parser.add_argument("--spectral-notch", action="store_true")
    parser.add_argument("--notch-strength", type=float, default=1.0)
    parser.add_argument("--notch-bandwidth", type=int, default=2)
    parser.add_argument("--notch-z", type=float, default=3.0)
    parser.add_argument("--spectral-envelope", type=float, default=0.0)
    parser.add_argument("--visibility-gamma", type=float, default=0.0)
    parser.add_argument("--visibility-ppd", type=float, default=48.0)
    parser.add_argument("--optimizer", choices=["sign", "adam"], default="sign")
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-lr", type=float, default=None)
    parser.add_argument("--adam-polish-steps", type=int, default=0)
    parser.add_argument("--visibility-loss-weight", type=float, default=0.0)
    parser.add_argument("--visibility-loss-low", type=float, default=0.025)
    parser.add_argument("--visibility-loss-high", type=float, default=0.050)
    parser.add_argument(
        "--eot-sizes", type=str, default=None,
        help="comma-separated int sizes, e.g. '1024,1024,896,1152,1280' for SDXL "
             "(default: native_eot_attack.py's SD1.5-weighted (512,512,640,768,1024))",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    eot_sizes = tuple(int(s) for s in args.eot_sizes.split(",")) if args.eot_sizes else None

    native_lowfreq_attack(
        original_path=args.original,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        param_size=args.param_size,
        hue_lock_radius=args.hue_lock_radius,
        epsilon=args.epsilon,
        steps=args.steps,
        step_size=args.step_size,
        eot_samples=args.eot_samples,
        logvar_weight=args.logvar_weight,
        recon_weight=args.recon_weight,
        hue_lock=not args.no_hue_lock,
        perceptual_mask=not args.no_perceptual_mask,
        mask_low=args.mask_low,
        mask_high=args.mask_high,
        irregularity_mask=args.irregularity_mask,
        color_diversity_mask=args.color_diversity_mask,
        param_smooth_sigma=args.param_smooth_sigma,
        spectral_notch=args.spectral_notch,
        notch_strength=args.notch_strength,
        notch_bandwidth=args.notch_bandwidth,
        notch_z=args.notch_z,
        spectral_envelope=args.spectral_envelope,
        visibility_gamma=args.visibility_gamma,
        visibility_ppd=args.visibility_ppd,
        optimizer=args.optimizer,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_lr=args.adam_lr,
        adam_polish_steps=args.adam_polish_steps,
        visibility_loss_weight=args.visibility_loss_weight,
        visibility_loss_low=args.visibility_loss_low,
        visibility_loss_high=args.visibility_loss_high,
        eot_sizes=eot_sizes,
        seed=args.seed,
    )
