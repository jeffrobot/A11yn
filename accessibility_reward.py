from pathlib import Path
from playwright.sync_api import Route, sync_playwright
from typing import List, Tuple
import os
import re
import sys
import traceback
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
AXE_SCRIPT_PATH = Path(os.environ.get("A11YN_AXE_SCRIPT_PATH", str(SCRIPT_DIR / "axe.min.js")))
HTML_RENDER_TIMEOUT_MS = int(os.environ.get("A11YN_HTML_RENDER_TIMEOUT_MS", "10000"))
BLOCK_EXTERNAL_REQUESTS = os.environ.get("A11YN_BLOCK_EXTERNAL_REQUESTS", "1") != "0"
CHROMIUM_EXECUTABLE_PATH = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
TAILWIND_CSS_URL = os.environ.get(
    "A11YN_TAILWIND_CSS_URL",
    "https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css",
)
LOCAL_TAILWIND_CSS_PATH = Path(
    os.environ.get(
        "A11YN_LOCAL_TAILWIND_CSS_PATH",
        str(SCRIPT_DIR / "vendor" / "tailwind-2.2.19.min.css"),
    )
)
ALLOWED_EXTERNAL_HOSTS = tuple(
    host.strip().lower()
    for host in os.environ.get("A11YN_ALLOWED_EXTERNAL_HOSTS", "").split(",")
    if host.strip()
)
IMPACT_WEIGHTS = {
    "minor": 0.1,
    "moderate": 0.2,
    "serious": 0.3,
    "critical": 0.4,
}


def load_axe_script() -> str:
    """Load the vendored axe-core JavaScript so Playwright can run accessibility checks."""
    with open(AXE_SCRIPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


LOCAL_ASSET_MAP = {}
if LOCAL_TAILWIND_CSS_PATH.exists():
    LOCAL_ASSET_MAP[TAILWIND_CSS_URL] = (LOCAL_TAILWIND_CSS_PATH, "text/css")

CANONICAL_TAILWIND_LINK_TAG = f'<link href="{TAILWIND_CSS_URL}" rel="stylesheet">'
TAILWIND_LINK_RE = re.compile(
    r"""<link\b[^>]*href\s*=\s*["'][^"']*tailwind[^"']*["'][^>]*>""",
    re.IGNORECASE,
)
TAILWIND_SCRIPT_RE = re.compile(
    r"""<script\b[^>]*src\s*=\s*["']https?://cdn\.tailwindcss\.com[^"']*["'][^>]*>\s*</script>""",
    re.IGNORECASE | re.DOTALL,
)


def _canonicalize_tailwind_link(html: str) -> str:
    """Replace any Tailwind include with the single stylesheet URL expected by the router."""
    html = TAILWIND_LINK_RE.sub("", html)
    html = TAILWIND_SCRIPT_RE.sub("", html)

    if re.search(r"</head\s*>", html, re.IGNORECASE):
        return re.sub(
            r"</head\s*>",
            f"    {CANONICAL_TAILWIND_LINK_TAG}\n</head>",
            html,
            count=1,
            flags=re.IGNORECASE,
        )

    if re.search(r"<head\b[^>]*>", html, re.IGNORECASE):
        return re.sub(
            r"(<head\b[^>]*>)",
            rf"\1\n    {CANONICAL_TAILWIND_LINK_TAG}",
            html,
            count=1,
            flags=re.IGNORECASE,
        )

    if re.search(r"<html\b[^>]*>", html, re.IGNORECASE):
        return re.sub(
            r"(<html\b[^>]*>)",
            rf"\1\n<head>\n    {CANONICAL_TAILWIND_LINK_TAG}\n</head>",
            html,
            count=1,
            flags=re.IGNORECASE,
        )

    if re.search(r"<body\b[^>]*>", html, re.IGNORECASE):
        return re.sub(
            r"(<body\b[^>]*>)",
            rf"<head>\n    {CANONICAL_TAILWIND_LINK_TAG}\n</head>\n\1",
            html,
            count=1,
            flags=re.IGNORECASE,
        )

    return f"<head>\n    {CANONICAL_TAILWIND_LINK_TAG}\n</head>\n{html}"


def extract_html_from_completion(completions: List[str]) -> List[str]:
    """Extract raw HTML from model responses and normalize the Tailwind include."""
    html_list = []
    for text in completions:
        answer_match = re.search(r"<answer>.*?```html(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if answer_match:
            html_code = answer_match.group(1).strip()
            html_list.append(_canonicalize_tailwind_link(html_code))
        else:
            html_list.append(_canonicalize_tailwind_link(text.strip()))
    return html_list


def _is_allowed_external_host(hostname: str) -> bool:
    """Return True when a requested hostname is explicitly allowlisted."""
    hostname = hostname.lower()
    return any(
        hostname == allowed_host or hostname.endswith(f".{allowed_host}")
        for allowed_host in ALLOWED_EXTERNAL_HOSTS
    )


def _handle_page_route(route: Route) -> None:
    """Serve vendored assets locally and block non-allowlisted external requests."""
    url = route.request.url
    local_asset = LOCAL_ASSET_MAP.get(url)
    if local_asset is not None:
        local_path, content_type = local_asset
        route.fulfill(path=str(local_path), content_type=content_type)
        return

    parsed_url = urlparse(url)
    scheme = parsed_url.scheme.lower()
    hostname = (parsed_url.hostname or "").lower()
    if (
        BLOCK_EXTERNAL_REQUESTS
        and scheme in {"http", "https"}
        and not _is_allowed_external_host(hostname)
    ):
        route.abort()
        return
    route.continue_()


def _browser_launch_kwargs() -> dict:
    """Build the Chromium launch options used for reward evaluation."""
    launch_kwargs = {"headless": True, "args": ["--no-sandbox"]}
    if CHROMIUM_EXECUTABLE_PATH:
        launch_kwargs["executable_path"] = CHROMIUM_EXECUTABLE_PATH
    return launch_kwargs


def _compute_score(result: dict, dom_node_count: int) -> float:
    """Compute the final reward as 1 - weighted_violations / DOM_size."""
    violations = result.get("violations", [])
    weighted_violations = sum(
        IMPACT_WEIGHTS.get(v.get("impact", ""), 0.0) * len(v.get("nodes", []))
        for v in violations
        if v.get("impact", "") in IMPACT_WEIGHTS
    )
    return 1.0 - (weighted_violations / max(dom_node_count, 1))


def _html_debug_preview(html: str, max_length: int = 200) -> str:
    """Create a short one-line HTML preview for error logs."""
    compact_html = re.sub(r"\s+", " ", html).strip()
    if len(compact_html) <= max_length:
        return compact_html
    return f"{compact_html[:max_length]}..."


def _evaluate_html_on_page(page, html: str, axe_script: str) -> Tuple[dict, int]:
    """Render one HTML document, run axe, and return the axe result plus DOM size."""
    page.set_default_timeout(HTML_RENDER_TIMEOUT_MS)
    page.set_default_navigation_timeout(HTML_RENDER_TIMEOUT_MS)

    # Let allowlisted CDN resources apply before the accessibility scan.
    page.set_content(html, wait_until="load", timeout=HTML_RENDER_TIMEOUT_MS)
    page.wait_for_load_state("networkidle", timeout=HTML_RENDER_TIMEOUT_MS)
    page.evaluate("() => document.fonts ? document.fonts.ready : Promise.resolve()")

    page.evaluate(axe_script)
    result = page.evaluate(
        """() => axe.run(document, {
            resultTypes: ['violations'],
            reporter: 'v2'
        })"""
    )
    dom_node_count = page.evaluate(
        """() => {
            const root = document.body || document.documentElement;
            if (!root) return 0;
            return root.querySelectorAll('*').length + 1;
        }"""
    )
    return result, dom_node_count


def score_htmls_with_reused_browser(
    html_list: List[str],
    axe_script: str = None,
) -> List[float]:
    """Score a batch of HTML documents while reusing one browser session for efficiency."""
    axe_script = axe_script or load_axe_script()
    scores: List[float] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(**_browser_launch_kwargs())
        context = browser.new_context()
        context.route("**/*", _handle_page_route)
        page = context.new_page()
        try:
            for i, html in enumerate(html_list):
                try:
                    result, dom_node_count = _evaluate_html_on_page(page, html, axe_script)
                    scores.append(_compute_score(result, dom_node_count))
                except Exception as e:
                    print(
                        f"[axe_violation_reward_func] Error on HTML {i}: "
                        f"{type(e).__name__}: {e}",
                        file=sys.stderr,
                    )
                    print(
                        f"[axe_violation_reward_func] HTML {i} length={len(html)} "
                        f"preview={_html_debug_preview(html)!r}",
                        file=sys.stderr,
                    )
                    traceback.print_exc(file=sys.stderr)
                    scores.append(0.0)
        finally:
            context.close()
            browser.close()
    return scores


def score_completions_with_axe(
    completions: List[str],
    axe_script: str = None,
) -> List[float]:
    """Convert model completions to HTML and score each one with axe."""
    html_completions = extract_html_from_completion(completions)
    return score_htmls_with_reused_browser(
        html_completions,
        axe_script=axe_script,
    )


def axe_violation_reward_func(
    completions: List[str],
    **_kwargs,
) -> List[float]:
    """TRL reward hook that scores completions using the accessibility metric."""
    axe_script = load_axe_script()
    return score_completions_with_axe(completions, axe_script=axe_script)
