"""Google Cloud Vision Web Detection wrapper -- the "역이미지 검색" (reverse
image search) leg of PROJECT_DESIGN.md §3-7. Optional by design: without
GOOGLE_APPLICATION_CREDENTIALS configured, callers get an empty match list
rather than a hard failure, so /scan still returns pHash + watermark
findings. This mirrors the project's established pattern of degrading
gracefully and documenting the gap (see GPU/C2PA/KMS notes elsewhere)
rather than making an entire endpoint depend on a paid external API key.
"""

import os


def vision_configured() -> bool:
    return bool(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))


def web_detect_matching_urls(image_path: str) -> list[str]:
    """Returns candidate URLs the Vision API considers full/partial matches
    for the given image. Returns [] if Vision isn't configured -- callers
    should check vision_configured() first if they want to distinguish
    "configured but zero matches" from "not configured".
    """
    if not vision_configured():
        return []

    # Imported lazily so environments without the google-cloud-vision
    # package (or without credentials) don't fail at module import time --
    # only when a scan actually tries to use it.
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    with open(image_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)
    response = client.web_detection(image=image)
    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    web = response.web_detection
    urls: list[str] = []
    for page in web.pages_with_matching_images:
        urls.append(page.url)
    for img in web.full_matching_images:
        urls.append(img.url)
    for img in web.partial_matching_images:
        urls.append(img.url)
    # dedupe, preserve order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped
