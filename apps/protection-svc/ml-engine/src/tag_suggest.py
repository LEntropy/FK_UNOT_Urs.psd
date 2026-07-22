"""CLIP zero-shot tag suggestion for the upload-preview feature (PixAI-style
image-to-tag preview: suggest descriptive tags for a user to review/edit
before publishing, not a fixed classifier's final answer).

Reuses the exact same open_clip checkpoint (ViT-B-32/openai)
concept_misalign.py already loads for Concept Misalignment -- zero new
model downloads or pip dependencies. Mechanism: embed the uploaded image
once, embed every tag string in TAG_VOCABULARY once (cached across calls,
since text embeddings don't depend on the image), rank by cosine
similarity. Same joint image-text space concept_misalign.py's
targeted-decoy mechanism already exploits, just used here for description
instead of attack.

Deliberately a *fixed vocabulary* zero-shot classifier, not a free-form
captioning model (BLIP/BLIP-2 etc.) -- see PROJECT context: a captioning
model would need a new checkpoint download and meaningfully more VRAM on
top of the CLIP-ensemble/VGG19 load style_cloak.py's own comments already
document as tight on this project's 8GB GPU. A curated tag list can't
describe everything a free-form caption could, but it's a real, fast,
zero-extra-footprint starting point a user edits anyway -- the ranking
only ever needs to be "plausible enough to save typing," not exhaustive.
"""

import torch
import torch.nn.functional as F

from model import ConceptFeatureExtractor
from style_cloak import load_image_tensor

# Curated, not exhaustive -- grouped by facet in source for readability,
# returned as one flat ranked list (simplest UI: a single suggested-tags
# chip row the user can remove from or add to).
TAG_VOCABULARY: list[str] = [
    # style / medium
    "oil painting", "watercolor", "digital art", "pencil sketch", "ink drawing",
    "acrylic painting", "pastel", "charcoal drawing", "gouache", "pixel art",
    "anime style", "manga style", "cartoon", "photorealistic", "impressionism",
    "expressionism", "surrealism", "cubism", "abstract art", "minimalism",
    "art nouveau", "art deco", "baroque", "renaissance", "ukiyo-e",
    "pop art", "concept art", "3D render", "vector art", "line art",
    # subject
    "portrait", "self portrait", "landscape", "seascape", "cityscape",
    "still life", "figure study", "animal", "bird", "cat", "dog",
    "horse", "fantasy creature", "dragon", "architecture", "interior scene",
    "flowers", "plants", "food", "vehicle", "abstract composition",
    "group scene", "crowd", "battle scene", "mythological scene",
    # mood / lighting / color
    "vibrant colors", "monochrome", "black and white", "pastel colors",
    "dark and moody", "warm tones", "cool tones", "high contrast",
    "soft lighting", "dramatic lighting", "night scene", "golden hour",
    "stormy", "serene", "melancholic", "whimsical", "eerie", "cheerful",
    # composition
    "close-up", "wide shot", "symmetrical composition", "dynamic pose",
    "action scene", "minimal background", "detailed background",
]

# Cached module-level, same pattern as style_cloak.py's get_clip_ensemble --
# the CLIP model and every tag's text embedding are identical across calls,
# only the image changes, so both are computed once per process and reused.
_extractor: ConceptFeatureExtractor | None = None
_tag_embeds: torch.Tensor | None = None
_tokenizer = None


def _get_extractor(device: torch.device) -> ConceptFeatureExtractor:
    global _extractor
    if _extractor is None:
        _extractor = ConceptFeatureExtractor(device)
    return _extractor


def _get_tag_embeds(device: torch.device) -> torch.Tensor:
    global _tag_embeds, _tokenizer
    if _tag_embeds is None:
        import open_clip

        extractor = _get_extractor(device)
        if _tokenizer is None:
            _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokens = _tokenizer(TAG_VOCABULARY).to(device)
        with torch.no_grad():
            _tag_embeds = extractor.model.encode_text(tokens)
    return _tag_embeds


def suggest_tags(image_path: str, top_k: int = 10, size: int = 256) -> list[dict]:
    """Returns up to top_k tags from TAG_VOCABULARY ranked by CLIP cosine
    similarity to the given image, as [{"tag": str, "score": float}, ...]
    sorted highest-first. A single CLIP forward pass plus a
    cosine-similarity ranking against an already-cached text-embedding
    matrix -- fast enough to run synchronously in an HTTP request, unlike
    cloak()'s multi-second-to-minutes optimization.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = _get_extractor(device)
    tag_embeds = _get_tag_embeds(device)

    image = load_image_tensor(image_path, size, device)
    with torch.no_grad():
        image_embed = extractor.embed(image)

    similarities = F.cosine_similarity(image_embed, tag_embeds)
    top_indices = torch.argsort(similarities, descending=True)[:top_k]

    return [
        {"tag": TAG_VOCABULARY[i], "score": round(similarities[i].item(), 4)}
        for i in top_indices.tolist()
    ]
