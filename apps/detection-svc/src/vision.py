"""Google Cloud Vision Web Detection wrapper -- the "역이미지 검색" (reverse
image search) leg of PROJECT_DESIGN.md §3-7. Optional by design: without
GOOGLE_VISION_API_KEY configured, callers get an empty match list rather
than a hard failure, so /scan still returns pHash + watermark findings.
This mirrors the project's established pattern of degrading gracefully
and documenting the gap (see GPU/C2PA/KMS notes elsewhere) rather than
making an entire endpoint depend on a paid external API key.

Uses the plain REST API with a simple API key rather than the
google-cloud-vision SDK (which requires Application Default Credentials --
typically a service-account JSON key). Many GCP orgs now enforce the
iam.disableServiceAccountKeyCreation organization policy by default
("Secure by Default"), which blocks minting that key file entirely and
has no simple per-project override. The REST API accepts a plain API key
via `?key=`, which isn't subject to that constraint and needs zero
credential files to mount into a container.
"""

import os

import httpx

VISION_ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"


def vision_configured() -> bool:
    return bool(os.environ.get("GOOGLE_VISION_API_KEY"))


def web_detect_matching_urls(image_path: str) -> list[str]:
    """Returns candidate URLs the Vision API considers full/partial matches
    for the given image. Returns [] if Vision isn't configured -- callers
    should check vision_configured() first if they want to distinguish
    "configured but zero matches" from "not configured".
    """
    api_key = os.environ.get("GOOGLE_VISION_API_KEY")
    if not api_key:
        return []

    import base64

    with open(image_path, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode("ascii")

    payload = {
        "requests": [
            {
                "image": {"content": content_b64},
                "features": [{"type": "WEB_DETECTION"}],
            }
        ]
    }

    resp = httpx.post(VISION_ENDPOINT, params={"key": api_key}, json=payload, timeout=30.0)
    resp.raise_for_status()
    result = resp.json()["responses"][0]

    if "error" in result:
        raise RuntimeError(f"Vision API error: {result['error'].get('message')}")

    web = result.get("webDetection", {})
    urls: list[str] = []
    for page in web.get("pagesWithMatchingImages", []):
        if "url" in page:
            urls.append(page["url"])
    for img in web.get("fullMatchingImages", []):
        if "url" in img:
            urls.append(img["url"])
    for img in web.get("partialMatchingImages", []):
        if "url" in img:
            urls.append(img["url"])

    # dedupe, preserve order
    seen = set()
    deduped = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped
