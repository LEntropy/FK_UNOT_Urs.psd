"""Per-URL evidence capture: downloads the candidate image, fetches HTTP
headers, and takes a best-effort screenshot via Playwright. Screenshot
capture is explicitly best-effort -- if Chromium isn't installed (e.g. a
minimal CI/dev environment that skipped `playwright install`), evidence
capture still produces headers + the downloaded image; only the
screenshot field comes back None. A single missing browser binary
shouldn't block evidence collection for the two things that don't need it.
"""

import time
from pathlib import Path

import httpx


class CapturedEvidence:
    def __init__(self, image_path: str | None, headers: dict, screenshot_path: str | None, captured_at: float):
        self.image_path = image_path
        self.headers = headers
        self.screenshot_path = screenshot_path
        self.captured_at = captured_at


async def capture(url: str, out_dir: Path) -> CapturedEvidence:
    out_dir.mkdir(parents=True, exist_ok=True)
    captured_at = time.time()

    headers: dict = {}
    image_path: str | None = None
    async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
        try:
            resp = await client.get(url)
            headers = dict(resp.headers)
            content_type = resp.headers.get("content-type", "")
            if content_type.startswith("image/"):
                image_path = str(out_dir / "candidate_image")
                Path(image_path).write_bytes(resp.content)
        except httpx.HTTPError as exc:
            headers = {"_fetch_error": str(exc)}

    screenshot_path = await _try_screenshot(url, out_dir)

    return CapturedEvidence(image_path, headers, screenshot_path, captured_at)


async def _try_screenshot(url: str, out_dir: Path) -> str | None:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return None

    screenshot_path = str(out_dir / "screenshot.png")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url, timeout=15_000)
            await page.screenshot(path=screenshot_path)
            await browser.close()
        return screenshot_path
    except Exception:
        # Chromium not installed, page unreachable, timeout, etc. -- evidence
        # capture continues without a screenshot rather than failing the case.
        return None
