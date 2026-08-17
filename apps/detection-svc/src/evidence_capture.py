"""Per-URL evidence capture: downloads the candidate image(s), fetches HTTP
headers/redirects, and takes a best-effort screenshot via Playwright.
Screenshot capture is explicitly best-effort -- if Chromium isn't installed
(e.g. a minimal CI/dev environment that skipped `playwright install`),
evidence capture still produces headers + the downloaded image; only the
screenshot field comes back None. A single missing browser binary
shouldn't block evidence collection for the two things that don't need it.

SSRF-safe by construction (_validate_public_url): every URL this module
fetches -- the candidate itself, each redirect hop, every extracted
<img>/og:image/JSON/rendered-image URL -- is resolved and checked against
private/loopback/link-local/reserved ranges before any request goes out,
since every one of those URLs ultimately comes from an external source
(Vision API results or a caller-submitted report), not this service's own
config.
"""

import ipaddress
import json
import socket
import time
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from pathlib import Path

import httpx


class CapturedEvidence:
    def __init__(self, image_path: str | None, headers: dict, screenshot_path: str | None, captured_at: float,
                 access_status: str = "UNKNOWN", final_url: str | None = None,
                 image_paths: list[str] | None = None):
        self.image_path = image_path
        self.image_paths = image_paths or ([image_path] if image_path else [])
        self.headers = headers
        self.screenshot_path = screenshot_path
        self.captured_at = captured_at
        self.access_status = access_status
        self.final_url = final_url


MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_REDIRECTS = 5

# Real bug, found live: httpx's default User-Agent ("python-httpx/x.x.x")
# gets a real site's basic bot-filtering to reject the request outright --
# confirmed against imgur specifically, which returns 429 for the default
# UA and 200 for this one, same URL, nothing else different. Every fetch
# in this module needs a browser-shaped UA or evidence capture silently
# can't even download the candidate image, regardless of how well the
# pHash/watermark matching itself works downstream.
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


class _ImageFinder(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "meta" and values.get("property") in {"og:image", "twitter:image"} and values.get("content"):
            self.urls.insert(0, values["content"])
        elif tag == "img" and values.get("src"):
            self.urls.append(values["src"])


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("only public http/https URLs are accepted")
    for result in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80)):
        address = ipaddress.ip_address(result[4][0])
        if not address.is_global:
            raise ValueError("private, loopback, link-local, and reserved destinations are blocked")


def _auth_headers(url: str, bearer_token: str | None, allowed_hosts: set[str]) -> dict[str, str]:
    hostname = (urlparse(url).hostname or "").lower()
    if bearer_token and hostname in allowed_hosts:
        return {"Authorization": f"Bearer {bearer_token}"}
    return {}


def _looks_like_login(url: str, html: str = "") -> bool:
    path = urlparse(url).path.lower()
    sample = html[:20_000].lower()
    return any(part in path for part in ("/login", "/signin", "/sign-in")) or any(
        marker in sample for marker in ('type="password"', "로그인", "sign in")
    )


async def capture(url: str, out_dir: Path, *, bearer_token: str | None = None,
                  auth_hosts: set[str] | None = None) -> CapturedEvidence:
    _validate_public_url(url)
    out_dir.mkdir(parents=True, exist_ok=True)
    captured_at = time.time()

    headers: dict = {}
    image_path: str | None = None
    image_paths: list[str] = []
    access_status = "UNKNOWN"
    final_url = url
    allowed_hosts = {host.strip().lower() for host in (auth_hosts or set()) if host.strip()}
    async with httpx.AsyncClient(
        timeout=15.0, follow_redirects=False, headers={"User-Agent": _BROWSER_USER_AGENT}
    ) as client:
        try:
            current_url = url
            redirects = []
            for _ in range(MAX_REDIRECTS + 1):
                _validate_public_url(current_url)
                resp = await client.get(current_url, headers=_auth_headers(current_url, bearer_token, allowed_hosts))
                if resp.is_redirect:
                    next_url = urljoin(current_url, resp.headers["location"])
                    redirects.append({"status": resp.status_code, "from": current_url, "to": next_url})
                    current_url = next_url
                    continue
                break
            else:
                raise ValueError("too many redirects")
            final_url = str(resp.url)
            if resp.status_code == 401:
                access_status = "AUTH_REQUIRED"
            elif resp.status_code == 403:
                access_status = "ACCESS_DENIED"
            elif resp.status_code == 404:
                access_status = "NOT_FOUND"
            resp.raise_for_status()
            headers = dict(resp.headers)
            if len(resp.content) > MAX_DOWNLOAD_BYTES:
                raise ValueError("response exceeds the 25 MiB evidence limit")
            (out_dir / "headers.json").write_text(json.dumps(headers, indent=2), encoding="utf-8")
            (out_dir / "source_url.txt").write_text(str(resp.url), encoding="utf-8")
            (out_dir / "redirects.json").write_text(json.dumps(redirects, indent=2), encoding="utf-8")
            content_type = resp.headers.get("content-type", "")
            if content_type.startswith("image/"):
                access_status = "IMAGE_ACCESSIBLE"
                # Real bug, found live: rust-core's `detect` (and phash_match's
                # own PIL.Image.open before it) both resolve the image
                # decoder from the file's *extension*, not just its content
                # -- rust-core's `image::open()` failed with "Unsupported
                # ... Format(Unknown)" against a real, valid PNG saved
                # extension-less as plain "candidate_image", even though the
                # file's own magic bytes were unambiguous. Map the real
                # content-type to a real extension so every downstream
                # extension-dependent tool (rust-core's watermark detector,
                # anything else that shells out to a CLI expecting a real
                # file) can actually open what gets saved here.
                ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}.get(
                    content_type.split(";")[0].strip(), ""
                )
                image_path = str(out_dir / f"candidate_image{ext}")
                Path(image_path).write_bytes(resp.content)
                image_paths.append(image_path)
            elif content_type.startswith("text/html"):
                html = resp.text
                (out_dir / "page.html").write_text(html, encoding="utf-8")
                if _looks_like_login(final_url, html):
                    access_status = "AUTH_REQUIRED"
                else:
                    access_status = "HTML_NO_IMAGE"
                finder = _ImageFinder()
                finder.feed(html)
                for candidate in finder.urls[:10]:
                    image_url = urljoin(str(resp.url), candidate)
                    try:
                        _validate_public_url(image_url)
                        image_resp = await client.get(
                            image_url, headers=_auth_headers(image_url, bearer_token, allowed_hosts)
                        )
                        image_resp.raise_for_status()
                        image_type = image_resp.headers.get("content-type", "").split(";")[0].strip()
                        if not image_type.startswith("image/") or len(image_resp.content) > MAX_DOWNLOAD_BYTES:
                            continue
                        ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}.get(image_type, "")
                        if not ext:
                            continue
                        image_path = str(out_dir / f"candidate_image{ext}")
                        Path(image_path).write_bytes(image_resp.content)
                        image_paths.append(image_path)
                        (out_dir / "candidate_url.txt").write_text(str(image_resp.url), encoding="utf-8")
                        access_status = "IMAGE_EXTRACTED"
                        break
                    except (httpx.HTTPError, ValueError):
                        continue
            elif content_type.startswith("application/json"):
                access_status = "JSON_NO_IMAGE_URL"
                try:
                    payload = resp.json()
                    image_url = next((payload.get(key) for key in ("url", "imageUrl", "image_url")
                                      if isinstance(payload, dict) and payload.get(key)), None)
                    if image_url:
                        image_url = urljoin(final_url, image_url)
                        _validate_public_url(image_url)
                        image_resp = await client.get(
                            image_url, headers=_auth_headers(image_url, bearer_token, allowed_hosts)
                        )
                        image_resp.raise_for_status()
                        image_type = image_resp.headers.get("content-type", "").split(";")[0].strip()
                        ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/gif": ".gif"}.get(image_type)
                        if ext and len(image_resp.content) <= MAX_DOWNLOAD_BYTES:
                            image_path = str(out_dir / f"candidate_image{ext}")
                            Path(image_path).write_bytes(image_resp.content)
                            image_paths.append(image_path)
                            (out_dir / "candidate_url.txt").write_text(str(image_resp.url), encoding="utf-8")
                            access_status = "IMAGE_RESOLVED_FROM_JSON"
                except (ValueError, httpx.HTTPError):
                    pass
        except httpx.TimeoutException as exc:
            access_status = "TIMEOUT"
            headers = {"_fetch_error": str(exc)}
        except httpx.HTTPError as exc:
            if access_status == "UNKNOWN":
                access_status = "FETCH_ERROR"
            headers = {"_fetch_error": str(exc)}

    headers["_access_status"] = access_status
    headers["_final_url"] = final_url
    (out_dir / "access.json").write_text(
        json.dumps({"status": access_status, "finalUrl": final_url}, indent=2), encoding="utf-8"
    )

    screenshot_path, rendered_urls, rendered_access = await _try_screenshot(url, out_dir)
    if rendered_access == "AUTH_REQUIRED":
        access_status = "AUTH_REQUIRED"

    # SPAs often ship an empty HTML shell and add artwork images only after
    # JavaScript runs. Download every rendered/network image, rather than
    # treating the first logo or social-preview image as the whole page.
    if rendered_urls:
        async with httpx.AsyncClient(
            timeout=15.0, follow_redirects=True, headers={"User-Agent": _BROWSER_USER_AGENT}
        ) as client:
            seen = set()
            for index, image_url in enumerate(rendered_urls[:50], start=1):
                if image_url in seen or image_url.startswith(("data:", "blob:")):
                    continue
                seen.add(image_url)
                try:
                    _validate_public_url(image_url)
                    response = await client.get(
                        image_url, headers=_auth_headers(image_url, bearer_token, allowed_hosts)
                    )
                    response.raise_for_status()
                    image_type = response.headers.get("content-type", "").split(";")[0].strip()
                    ext = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp",
                           "image/gif": ".gif"}.get(image_type)
                    if not ext or len(response.content) > MAX_DOWNLOAD_BYTES:
                        continue
                    rendered_path = out_dir / f"candidate_image_{index}{ext}"
                    rendered_path.write_bytes(response.content)
                    image_paths.append(str(rendered_path))
                except (httpx.HTTPError, ValueError):
                    continue
        if image_paths:
            image_path = image_paths[0]
            access_status = "RENDERED_IMAGES_EXTRACTED"
            (out_dir / "rendered_image_urls.json").write_text(
                json.dumps(rendered_urls, indent=2), encoding="utf-8"
            )

    # Preserve order while removing duplicates (the initial HTML image may
    # also be observed by Playwright's network listener).
    image_paths = list(dict.fromkeys(image_paths))
    return CapturedEvidence(image_path, headers, screenshot_path, captured_at, access_status, final_url, image_paths)


async def _try_screenshot(url: str, out_dir: Path) -> tuple[str | None, list[str], str | None]:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return None, [], None

    screenshot_path = str(out_dir / "screenshot.png")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            network_images: list[str] = []

            def record_response(response) -> None:
                if response.request.resource_type == "image":
                    network_images.append(response.url)

            page.on("response", record_response)
            await page.goto(url, timeout=15_000)
            await page.wait_for_timeout(1500)
            rendered_access = "AUTH_REQUIRED" if (
                _looks_like_login(page.url, await page.content())
                or await page.locator('input[type="password"]').count() > 0
            ) else None
            dom_images = await page.eval_on_selector_all(
                "img",
                "els => els.flatMap(e => [e.currentSrc, e.src, ...(e.srcset || '').split(',').map(x => x.trim().split(/\\s+/)[0])]).filter(Boolean)",
            )
            background_images = await page.evaluate(
                """() => [...document.querySelectorAll('*')].flatMap(e => {
                    const value = getComputedStyle(e).backgroundImage;
                    return [...value.matchAll(/url\\([\"']?(.*?)[\"']?\\)/g)].map(m => new URL(m[1], location.href).href);
                })"""
            )
            await page.screenshot(path=screenshot_path)
            await browser.close()
        return screenshot_path, list(dict.fromkeys(network_images + dom_images + background_images)), rendered_access
    except Exception:
        # Chromium not installed, page unreachable, timeout, etc. -- evidence
        # capture continues without a screenshot rather than failing the case.
        return None, [], None
