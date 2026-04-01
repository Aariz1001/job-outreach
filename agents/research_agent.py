"""
Research Agent — scrapes a company's website and extracts a structured
profile (what they do, tech signals, how the CV overlaps).

Results are cached in data/research_cache/<slug>.json.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from openai import OpenAI

from utils.config import DATA_DIR, GENERATION_MODEL, OPENROUTER_API_KEY

CACHE_DIR = DATA_DIR / "research_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    )
}

CV_SUMMARY = """
Mohammad Aariz Waqas — AI Engineer (University of Glasgow, Computing Science).
Core focus: AI agent systems and workflows — autonomous agents, multi-agent orchestration,
production RAG pipelines, agentic reasoning, tool-use, LangChain/CrewAI/LlamaIndex/OpenRouter.
Also built: WhatsApp AI agent (GCP/Twilio/Gemini), multi-tenant AI platforms,
Flutter/Firebase production app, sub-200ms intent classification for live AI systems.
Looking for: roles building AI agents, agent workflows, or applied AI systems — UK-based or remote.
"""


def _slug(company_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", company_name.lower()).strip("-")


def _fetch_text(url: str, max_chars: int = 8000) -> str:
    """Fetch visible text from a URL, following one redirect."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        resp.raise_for_status()
    except Exception as e:
        return f"[fetch error: {e}]"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text[:max_chars]


def _also_fetch_about(base_url: str) -> str:
    """Try /about and /about-us pages for extra context."""
    parsed = urlparse(base_url)
    root = f"{parsed.scheme}://{parsed.netloc}"
    for path in ["/about", "/about-us", "/company"]:
        extra = _fetch_text(urljoin(root, path), max_chars=3000)
        if len(extra) > 200 and "fetch error" not in extra:
            return extra
    return ""


def research_company(company: str, website: str, focus: str) -> dict:
    """Return a structured research dict for a company.  Uses cache if available."""
    cache_file = CACHE_DIR / f"{_slug(company)}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    homepage_text = _fetch_text(website)
    about_text = _also_fetch_about(website)
    combined = f"Homepage:\n{homepage_text}\n\nAbout page:\n{about_text}"

    prompt = f"""You are a research assistant helping an AI engineer (Aariz) find the best outreach targets.

Company: {company}
Website: {website}
Known focus: {focus}

Scraped website content:
---
{combined[:6000]}
---

Candidate CV summary:
{CV_SUMMARY}

Return ONLY valid JSON (no markdown) with these keys:
{{
  "company": "...",
  "one_liner": "One sentence about what this company does",
  "tech_signals": ["list", "of", "tech", "keywords", "found"],
  "product_stage": "e.g. production / beta / stealth",
  "fit_reason": "2-3 sentences: why Aariz's background specifically aligns",
  "talking_points": ["specific thing 1 to mention in outreach", "specific thing 2"],
  "contact_roles": ["Founder", "CTO", "AI Engineer"]
}}"""

    resp = _client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        extra_body={"reasoning": {"enabled": True}},
    )

    raw = resp.choices[0].message.content.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {
            "company": company,
            "one_liner": focus,
            "tech_signals": [],
            "product_stage": "unknown",
            "fit_reason": raw,
            "talking_points": [],
            "contact_roles": ["Founder"],
        }

    result["website"] = website
    cache_file.write_text(json.dumps(result, indent=2))
    time.sleep(1)  # polite rate limiting
    return result


if __name__ == "__main__":
    import sys
    from rich.console import Console
    from rich.pretty import Pretty

    console = Console()
    company = sys.argv[1] if len(sys.argv) > 1 else "Vapi AI"
    website = sys.argv[2] if len(sys.argv) > 2 else "https://vapi.ai"
    focus   = sys.argv[3] if len(sys.argv) > 3 else "Voice AI infrastructure"

    result = research_company(company, website, focus)
    console.print(Pretty(result))
