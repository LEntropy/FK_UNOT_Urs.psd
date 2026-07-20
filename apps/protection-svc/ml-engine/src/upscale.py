"""Restores resolution after cloak()'s epsilon-bounded perturbation, which
only ever runs at a small, validated processing size (default 256 -- see
style_cloak.py's `size` param doc, "presets/EOT/robustness numbers were
validated at 256x256 specifically"). Real per-artwork uploads are almost
never 256px, so without this step every published image was capped at
that size regardless of what was actually uploaded -- a real, reported
problem (asset-service can't generate its larger delivery-gateway variants,
public_preview_1280/2048, from a source that's structurally never bigger
than 256px on its long edge).

Uses a real learned super-resolution model (EDSR, via the `super-image`
package) rather than a naive resize, so the restored resolution has
actual reconstructed detail instead of just blur.
"""

from PIL import Image

_model_cache: dict[int, object] = {}


def _get_model(scale: int):
    from super_image import EdsrModel

    if scale not in _model_cache:
        _model_cache[scale] = EdsrModel.from_pretrained("eugenesiow/edsr-base", scale=scale)
    return _model_cache[scale]


def upscale_to_size(input_path: str, output_path: str, target_width: int, target_height: int) -> bool:
    """Upscales input_path toward (target_width, target_height), writing the
    result to output_path (may be the same path as input_path). Returns
    True if a real SR model was used, False if it fell back to a plain
    resize (already at/above target size, or the model was unavailable) --
    callers should log this rather than silently claiming SR happened when
    it didn't.
    """
    img = Image.open(input_path).convert("RGB")
    src_w, src_h = img.size

    if src_w >= target_width and src_h >= target_height:
        # Nothing to upscale -- e.g. the original upload was already small.
        # A plain resize down to the exact target is a real, honest
        # operation here, not a degraded fallback.
        img.resize((target_width, target_height), Image.LANCZOS).save(output_path)
        return False

    try:
        from super_image import ImageLoader

        needed_scale = max(target_width / src_w, target_height / src_h)
        # super-image's pretrained EDSR only ships fixed integer scales
        # (2x/3x/4x) -- pick the smallest that covers what's needed, then
        # do one final precise resize to hit the exact target dimensions.
        # Chaining multiple SR passes for a very large gap would compound
        # quality further, but adds real latency this PoC doesn't need yet.
        model_scale = next((s for s in (2, 3, 4) if s >= needed_scale), 4)
        model = _get_model(model_scale)
        inputs = ImageLoader.load_image(img)
        preds = model(inputs)
        ImageLoader.save_image(preds, output_path)
        Image.open(output_path).resize((target_width, target_height), Image.LANCZOS).save(output_path)
        return True
    except Exception as exc:  # noqa: BLE001 -- a real (if lower-res) image beats a crashed upload
        print(f"[upscale] SR model unavailable ({exc}), falling back to LANCZOS resize", flush=True)
        img.resize((target_width, target_height), Image.LANCZOS).save(output_path)
        return False
