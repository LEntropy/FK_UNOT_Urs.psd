"""Content-based prompt generation -- replaces trusting a caller-supplied
title as the training/attack prompt.

WHY THIS EXISTS (2026-08-08): every validated n=30 experiment in this
project used a hand-written prompt that actually described the image's
content ("anime original character, illustration", etc.). Production
never did that -- orchestrate.py's strong_protection call and asset-
service's score-protection job both pass the artist's own artwork TITLE
as the prompt. A title doesn't have to describe what's in the image at
all ("무제", a pun, a series name). When it doesn't, the baseline LoRA
(trained on the ORIGINAL image under that prompt) can't actually learn
to reproduce the image, so its similarity score collapses toward the
CLIP floor (~0.55-0.60, roughly what two unrelated same-genre images
score) regardless of any attack. A protected-image delta measured
against a broken baseline is not a measurement of the attack -- it's
noise. This was caught live: a real deployed artwork scored baseline=0.57
and the protected condition came back with a NEGATIVE delta.

TAGGER MODEL, CORRECTED (2026-08-08): the first version of this module
used BLIP (natural-language captions: "a girl sitting on a motorcycle in
a field"). That was a real improvement over a title, but a targeted
diagnostic on this project's own checkpoints (SD1.5 anime-tuned, SDXL
Illustrious-XL) showed BLIP's natural-language style STILL left baseline
badly degraded (0.536) -- switching to a hand-written Danbooru-tag-style
prompt on the exact same pipeline jumped baseline to 0.6385 and delta
from +0.007 (noise) to +0.111 (real signal). The checkpoints this
project protects are anime-illustration models, and virtually every
public one -- including community fine-tunes like Illustrious-XL -- is
trained on Danbooru-tag captions, not natural-language sentences. A
prompt style the base model has never seen conditioned on during ITS OWN
training is a bad prompt for a LoRA built on top of it, no matter how
accurately it describes the image in English.

This module now uses WD14 (SmilingWolf/wd-vit-tagger-v3, the ONNX
tagger this exact ecosystem standardizes on -- it's what kohya_ss's own
tagging script, jointly with a dozen other anime-LoRA toolchains, wraps),
producing genuine Danbooru-style tags ("1girl, motorcycle, rainbow,
outdoors, ...") instead of a natural-language sentence. This is also the
more honest threat model: a real scraper training on stolen anime art
overwhelmingly uses WD14 or a Danbooru-tag captioner, not BLIP -- BLIP is
the general-photography-domain choice, not this domain's.

Falls back to BLIP if WD14 can't load (network/model-hub issue) --
degraded but still real content-conditioning, strictly better than no
fallback whatsoever letting a caller's title back in by omission.
"""

import functools

_WD14_REPO = "SmilingWolf/wd-vit-tagger-v3"
_WD14_MODEL_FILE = "model.onnx"
_WD14_TAGS_FILE = "selected_tags.csv"
_WD14_IMAGE_SIZE = 448
_WD14_GENERAL_THRESHOLD = 0.35
_WD14_CHARACTER_THRESHOLD = 0.85
_WD14_MAX_TAGS = 24

_BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"


@functools.lru_cache(maxsize=1)
def _load_wd14():
    import csv

    import numpy as np  # noqa: F401 -- imported to fail fast here if missing, not just inside caption_image
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(_WD14_REPO, _WD14_MODEL_FILE)
    tags_path = hf_hub_download(_WD14_REPO, _WD14_TAGS_FILE)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(model_path, providers=providers)

    names, categories = [], []
    with open(tags_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["name"])
            categories.append(int(row["category"]))  # 0=general, 4=character, 9=rating

    return session, names, categories


def _wd14_tags(image_path: str) -> str:
    import numpy as np
    from PIL import Image

    session, names, categories = _load_wd14()

    image = Image.open(image_path).convert("RGB").resize((_WD14_IMAGE_SIZE, _WD14_IMAGE_SIZE))
    arr = np.asarray(image, dtype=np.float32)[:, :, ::-1]  # RGB -> BGR, this tagger's expected channel order
    arr = np.expand_dims(arr, 0)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    probs = session.run([output_name], {input_name: arr})[0][0]

    general, character = [], []
    for name, category, prob in zip(names, categories, probs):
        if category == 9:  # rating (safe/questionable/explicit) -- not descriptive content
            continue
        if category == 4 and prob >= _WD14_CHARACTER_THRESHOLD:
            character.append((prob, name))
        elif category != 4 and prob >= _WD14_GENERAL_THRESHOLD:
            general.append((prob, name))

    general.sort(reverse=True)
    character.sort(reverse=True)
    tags = [name for _, name in character] + [name for _, name in general]
    tags = tags[:_WD14_MAX_TAGS]
    return ", ".join(t.replace("_", " ") for t in tags)


@functools.lru_cache(maxsize=1)
def _load_blip():
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(_BLIP_MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(_BLIP_MODEL_ID).to(device).eval()
    return processor, model, device


def _blip_caption(image_path: str) -> str:
    import torch
    from PIL import Image

    processor, model, device = _load_blip()
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(output_ids[0], skip_special_tokens=True).strip()


def caption_image(image_path: str) -> str:
    """Generates a Danbooru-tag-style prompt for the image at image_path --
    the caption style this project's anime-tuned checkpoints were
    themselves trained on, not a natural-language description (see this
    module's own doc for the real measured difference: baseline delta
    +0.007 with a natural-language caption vs +0.111 with tag style, same
    image, same pipeline). Falls back to BLIP if WD14 fails to load."""
    try:
        tags = _wd14_tags(image_path)
        if tags:
            return tags
    except Exception as exc:  # noqa: BLE001 -- degrade to BLIP rather than let a caller's title back in
        print(f"[caption_image] WD14 tagger unavailable ({type(exc).__name__}: {exc}) -- falling back to BLIP", flush=True)

    try:
        caption = _blip_caption(image_path)
        if caption:
            return caption
    except Exception as exc:  # noqa: BLE001 -- last resort below is still content-neutral, never the caller's title
        print(f"[caption_image] BLIP unavailable ({type(exc).__name__}: {exc}) -- using generic fallback", flush=True)

    return "1girl, illustration, anime style"
