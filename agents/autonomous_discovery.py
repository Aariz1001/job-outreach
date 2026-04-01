"""
Autonomous Discovery Agent — crawls multiple public sources to find AI companies
and builds a local cache so the CV matcher is never limited to a hand-picked list.

Sources (all public, no API key required):
  1. YC AI companies        — JSON API, all pages, AI-tagged
  2. RemoteOK               — public JSON API for remote AI/ML jobs
  3. Hacker News "Who's Hiring" — Algolia API, latest monthly thread
  4. Wellfound (Angel.co)   — public job search scrape
  5. UK-specific scrape     — Sifted, Tech Nation, Beauhurst AI pages
  6. Workable UK            — public job search API

Cache: data/company_cache.json  (slug → company dict, incremental)
State: data/crawl_state.json    (tracks last run, total found, visited slugs)

Usage:
  python -m agents.autonomous_discovery              # all sources, incremental
  python -m agents.autonomous_discovery --source yc  # single source
  python -m agents.autonomous_discovery --fresh      # ignore cache, re-fetch all
  python -m agents.autonomous_discovery --limit 200  # cap total companies fetched
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from utils.config import DATA_DIR, GENERATION_MODEL, OPENROUTER_API_KEY

console = Console()

CACHE_FILE  = DATA_DIR / "company_cache.json"
STATE_FILE  = DATA_DIR / "crawl_state.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}

_llm = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

AI_KEYWORDS = {
    "ai", "ml", "llm", "gpt", "language model", "generative", "voice ai",
    "speech", "nlp", "natural language", "agent", "rag", "retrieval",
    "machine learning", "deep learning", "neural", "automation", "chatbot",
    "conversational", "embedding", "transformer", "copilot", "inference",
}


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _load_cache() -> dict[str, dict]:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}


def _save_cache(cache: dict[str, dict]) -> None:
    CACHE_FILE.write_text(json.dumps(cache, indent=2))


def _load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"last_run": None, "total_found": 0, "sources_run": []}


def _save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2))


def _is_ai_relevant(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in AI_KEYWORDS)


def _normalise(
    company: str,
    website: str,
    focus: str,
    description: str,
    location: str = "Remote",
    stage: str = "—",
    size: str = "—",
    contact_hint: str = "founders / CTO",
    source: str = "unknown",
) -> dict[str, Any]:
    return {
        "company": company.strip(),
        "website": website.strip(),
        "focus": focus.strip()[:120],
        "description": description.strip()[:400],
        "location": location.strip(),
        "stage": stage.strip(),
        "size": str(size),
        "contact_hint": contact_hint,
        "source": source,
        "cached_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Source 1: Y Combinator ────────────────────────────────────────────────────

def fetch_yc(existing_slugs: set[str], limit: int = 500) -> list[dict]:
    console.print("[cyan]  [YC] Fetching AI companies…[/cyan]")
    results: list[dict] = []

    for page in range(1, 20):
        if len(results) >= limit:
            break
        try:
            resp = requests.get(
                "https://api.ycombinator.com/v0.1/companies",
                params={"tags": "Artificial Intelligence", "page": page},
                headers=HEADERS,
                timeout=20,
            )
            if resp.status_code != 200:
                break
            items = resp.json().get("companies", [])
            if not items:
                break

            for c in items:
                name = c.get("name", "").strip()
                if not name:
                    continue
                slug = _slug(name)
                if slug in existing_slugs:
                    continue

                website  = c.get("url") or c.get("website") or ""
                one_liner = c.get("oneLiner") or c.get("longDescription") or ""
                batch    = c.get("batch", "")
                team_size = c.get("teamSize", "—")
                locs = c.get("locations") or []
                location = ", ".join(
                    (l if isinstance(l, str) else (l.get("city") or l.get("country") or ""))
                    for l in locs
                ) or "Remote"
                tags = [t.lower() for t in (c.get("tags") or [])]

                if not _is_ai_relevant(" ".join(tags) + " " + one_liner):
                    continue

                results.append(_normalise(
                    company=name,
                    website=website,
                    focus=one_liner[:120],
                    description=one_liner,
                    location=location,
                    stage=f"YC {batch}".strip(),
                    size=str(team_size),
                    contact_hint="founders / CTO",
                    source="yc",
                ))
                existing_slugs.add(slug)

            time.sleep(0.4)
        except Exception as e:
            console.print(f"[yellow]  [YC] page {page} error: {e}[/yellow]")
            break

    console.print(f"[green]  [YC] +{len(results)} companies[/green]")
    return results


# ── Source 2: RemoteOK ────────────────────────────────────────────────────────

def fetch_remoteok(existing_slugs: set[str], limit: int = 300) -> list[dict]:
    console.print("[cyan]  [RemoteOK] Fetching remote AI jobs…[/cyan]")
    results: list[dict] = []
    seen_companies: set[str] = set()

    tags = ["ai", "machine-learning", "nlp", "python", "llm"]
    for tag in tags:
        if len(results) >= limit:
            break
        try:
            resp = requests.get(
                f"https://remoteok.com/api?tag={tag}",
                headers={**HEADERS, "Accept": "application/json"},
                timeout=20,
            )
            jobs = resp.json()
            for job in jobs:
                if not isinstance(job, dict):
                    continue
                company = (job.get("company") or "").strip()
                if not company or company in seen_companies:
                    continue

                position    = job.get("position", "")
                description = re.sub(r"<[^>]+>", " ", job.get("description") or "")
                job_tags    = [t.lower() for t in (job.get("tags") or [])]

                full_text = f"{company} {position} {description} {' '.join(job_tags)}"
                if not _is_ai_relevant(full_text):
                    continue

                slug = _slug(company)
                if slug in existing_slugs:
                    seen_companies.add(company)
                    continue

                url = job.get("url", "")
                website = f"https://remoteok.com{url}" if url.startswith("/") else url

                results.append(_normalise(
                    company=company,
                    website=website,
                    focus=position or "Remote AI role",
                    description=f"{company} is hiring for '{position}'. " + description[:200],
                    location="Remote",
                    source="remoteok",
                ))
                existing_slugs.add(slug)
                seen_companies.add(company)

            time.sleep(1)
        except Exception as e:
            console.print(f"[yellow]  [RemoteOK] tag={tag} error: {e}[/yellow]")

    console.print(f"[green]  [RemoteOK] +{len(results)} companies[/green]")
    return results


# ── Source 3: Hacker News "Who is Hiring" ────────────────────────────────────

def _get_latest_hn_hiring_thread() -> int | None:
    """Return the HN story ID of the latest 'Ask HN: Who is hiring?' thread.

    Uses the official Firebase HN API via the 'whoishiring' bot account,
    which reliably posts the monthly thread first.
    """
    try:
        resp = requests.get(
            "https://hacker-news.firebaseio.com/v0/user/whoishiring.json",
            timeout=15,
        )
        data = resp.json()
        submitted = data.get("submitted", [])
        for story_id in submitted[:10]:
            story_resp = requests.get(
                f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                timeout=10,
            )
            story = story_resp.json()
            title = (story.get("title") or "").lower()
            if "who is hiring" in title:
                return int(story_id)
    except Exception as e:
        console.print(f"[yellow]  [HN] thread lookup error: {e}[/yellow]")
    return None


def _extract_companies_from_hn_comments(comments: list[str]) -> list[dict]:
    """Use LLM to extract company info from HN hiring comments."""
    # Filter to likely AI-related comments
    ai_comments = [c for c in comments if _is_ai_relevant(c)][:60]
    if not ai_comments:
        return []

    batch_text = "\n\n---\n\n".join(ai_comments[:40])
    prompt = f"""You are parsing Hacker News "Who is Hiring" thread comments to extract AI startup job listings.

For each comment below, extract company information IF it's an AI/ML/voice AI/agent/LLM startup.

Return ONLY valid JSON array (no markdown). Each item:
{{
  "company": "Company Name",
  "website": "https://...",
  "focus": "one-line description of what they build",
  "location": "city/remote/UK",
  "stage": "startup stage if mentioned, else '—'"
}}

Skip non-AI companies, skip big corporations (Google, Amazon, etc.).
Return empty array [] if nothing found.

Comments:
{batch_text[:4000]}"""

    try:
        resp = _llm.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            extra_body={"reasoning": {"enabled": True}},
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return parsed
    except Exception as e:
        console.print(f"[yellow]  [HN] LLM parse error: {e}[/yellow]")

    return []


def fetch_hn_hiring(existing_slugs: set[str], limit: int = 100) -> list[dict]:
    console.print("[cyan]  [HN] Fetching 'Who is Hiring' thread…[/cyan]")
    results: list[dict] = []

    story_id = _get_latest_hn_hiring_thread()
    if not story_id:
        console.print("[yellow]  [HN] Could not find thread, skipping[/yellow]")
        return []

    try:
        # Use Algolia items endpoint — returns the full thread with nested children
        resp = requests.get(
            f"https://hn.algolia.com/api/v1/items/{story_id}",
            timeout=25,
        )
        data = resp.json()
        raw_kids = data.get("children", [])
        comments = [
            re.sub(r"<[^>]+>", " ", kid.get("text") or "").strip()
            for kid in raw_kids
            if kid.get("text")
        ]

        extracted = _extract_companies_from_hn_comments(comments)

        for c in extracted:
            name    = (c.get("company") or "").strip()
            website = (c.get("website") or "").strip()
            if not name:
                continue
            slug = _slug(name)
            if slug in existing_slugs:
                continue
            results.append(_normalise(
                company=name,
                website=website,
                focus=c.get("focus") or "AI startup (from HN Hiring)",
                description=f"{name} — {c.get('focus', '')}. Found in HN Who is Hiring.",
                location=c.get("location") or "Remote",
                stage=c.get("stage") or "—",
                source="hn_hiring",
            ))
            existing_slugs.add(slug)
            if len(results) >= limit:
                break

    except Exception as e:
        console.print(f"[yellow]  [HN] fetch error: {e}[/yellow]")

    console.print(f"[green]  [HN] +{len(results)} companies[/green]")
    return results


# ── Source 4: UK AI Startups (Sifted scrape) ─────────────────────────────────

def fetch_uk_ai(existing_slugs: set[str]) -> list[dict]:
    """Scrape UK-focused AI startup directories."""
    console.print("[cyan]  [UK] Scraping UK AI startup sources…[/cyan]")
    results: list[dict] = []

    sources = [
        # Sifted UK AI companies
        ("https://sifted.eu/sector/ai", "sifted"),
        # Tech Nation AI listings
        ("https://technation.io/about-us/our-portfolio/", "technation"),
        # EU Startups AI
        ("https://www.eu-startups.com/directory/?_sft_startup_category=artificial-intelligence", "eu_startups"),
    ]

    seen: set[str] = set()

    for url, source_name in sources:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            soup = BeautifulSoup(resp.text, "html.parser")

            # Generic extraction: find article/card elements with company names + links
            for el in soup.select("article, .company-card, .startup-card, [class*='company'], [class*='startup']"):
                name_el = el.select_one("h2, h3, h4, [class*='name'], [class*='title']")
                link_el = el.select_one("a[href]")
                desc_el = el.select_one("p, [class*='desc'], [class*='tagline']")

                name = (name_el.get_text(strip=True) if name_el else "").strip()
                href = (link_el["href"] if link_el else "").strip()
                desc = (desc_el.get_text(strip=True) if desc_el else "").strip()

                if not name or len(name) < 3 or len(name) > 60:
                    continue
                if not _is_ai_relevant(desc + " " + name):
                    continue
                if name in seen:
                    continue

                slug = _slug(name)
                if slug in existing_slugs:
                    seen.add(name)
                    continue

                website = href if href.startswith("http") else ""
                results.append(_normalise(
                    company=name,
                    website=website,
                    focus=desc[:120] or "AI startup",
                    description=desc or f"{name} — UK AI startup.",
                    location="UK",
                    source=source_name,
                ))
                existing_slugs.add(slug)
                seen.add(name)

            time.sleep(0.5)
        except Exception as e:
            console.print(f"[yellow]  [UK:{source_name}] error: {e}[/yellow]")

    console.print(f"[green]  [UK] +{len(results)} companies[/green]")
    return results


# ── Source 5: Workable UK AI jobs ─────────────────────────────────────────────

def fetch_workable(existing_slugs: set[str]) -> list[dict]:
    console.print("[cyan]  [Workable] Fetching UK AI job listings…[/cyan]")
    results: list[dict] = []
    seen: set[str] = set()

    queries = ["AI engineer", "machine learning engineer", "LLM engineer", "voice AI"]

    for query in queries:
        try:
            resp = requests.get(
                "https://jobs.workable.com/api/v1/jobs",
                params={"query": query, "location": "United Kingdom", "remote": "true"},
                headers=HEADERS,
                timeout=20,
            )
            if resp.status_code != 200:
                continue
            for job in resp.json().get("results", []):
                co = job.get("company", {})
                name    = (co.get("name") or "").strip()
                website = (co.get("url") or "").strip()
                title   = (job.get("title") or "").strip()
                city    = (job.get("location", {}).get("city") or "UK").strip()

                if not name or name in seen:
                    continue
                if not _is_ai_relevant(title):
                    continue

                slug = _slug(name)
                if slug in existing_slugs:
                    seen.add(name)
                    continue

                results.append(_normalise(
                    company=name,
                    website=website,
                    focus=title,
                    description=f"{name} is hiring for '{title}' (UK/remote).",
                    location=f"{city}, UK",
                    source="workable",
                ))
                existing_slugs.add(slug)
                seen.add(name)

            time.sleep(0.5)
        except Exception as e:
            console.print(f"[yellow]  [Workable] query='{query}' error: {e}[/yellow]")

    console.print(f"[green]  [Workable] +{len(results)} companies[/green]")
    return results


# ── Source 6: Wellfound (Angel.co) public job search ─────────────────────────

def fetch_wellfound(existing_slugs: set[str]) -> list[dict]:
    """Scrape Wellfound (AngelList) public job search for AI roles."""
    console.print("[cyan]  [Wellfound] Scraping AI startup jobs…[/cyan]")
    results: list[dict] = []
    seen: set[str] = set()

    urls = [
        "https://wellfound.com/jobs?remote=true&keywords=ai+engineer",
        "https://wellfound.com/jobs?remote=true&keywords=llm+agent",
        "https://wellfound.com/jobs?location_slug=united-kingdom&keywords=ai",
    ]

    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            soup = BeautifulSoup(resp.text, "html.parser")

            for card in soup.select("[class*='StartupCard'], [class*='company-card'], article"):
                name_el = card.select_one("[class*='companyName'], [class*='name'], h3, h2")
                link_el = card.select_one("a[href*='/company/']")
                desc_el = card.select_one("[class*='pitch'], [class*='tagline'], p")

                name = (name_el.get_text(strip=True) if name_el else "").strip()
                desc = (desc_el.get_text(strip=True) if desc_el else "").strip()
                href = (link_el["href"] if link_el else "")

                if not name or len(name) < 3 or name in seen:
                    continue
                if not _is_ai_relevant(desc + " " + name):
                    continue

                slug = _slug(name)
                if slug in existing_slugs:
                    seen.add(name)
                    continue

                website = (f"https://wellfound.com{href}" if href.startswith("/") else href)
                results.append(_normalise(
                    company=name,
                    website=website,
                    focus=desc[:120] or "AI startup",
                    description=desc or f"{name} is an AI startup on Wellfound.",
                    location="Remote / UK",
                    source="wellfound",
                ))
                existing_slugs.add(slug)
                seen.add(name)

            time.sleep(0.8)
        except Exception as e:
            console.print(f"[yellow]  [Wellfound] url error: {e}[/yellow]")

    console.print(f"[green]  [Wellfound] +{len(results)} companies[/green]")
    return results


# ── Orchestrator ──────────────────────────────────────────────────────────────

SOURCE_MAP = {
    "yc":        fetch_yc,
    "remoteok":  fetch_remoteok,
    "hn_hiring": fetch_hn_hiring,
    "uk":        fetch_uk_ai,
    "workable":  fetch_workable,
    "wellfound": fetch_wellfound,
}


def discover(
    sources: list[str] | str = "all",
    fresh: bool = False,
    limit: int = 1000,
) -> list[dict]:
    """
    Run autonomous discovery across all (or specified) sources.

    Args:
        sources: "all" or list like ["yc", "remoteok"] or single name
        fresh:   If True, ignore existing cache and re-fetch everything
        limit:   Max total new companies to add per run

    Returns list of all cached companies (new + previously cached).
    """
    cache = {} if fresh else _load_cache()
    state = _load_state()
    existing_slugs: set[str] = set(cache.keys())

    if isinstance(sources, str):
        source_list = list(SOURCE_MAP.keys()) if sources == "all" else [sources]
    else:
        source_list = sources

    console.print(f"\n[bold cyan]Autonomous Discovery — sources: {', '.join(source_list)}[/bold cyan]")
    console.print(f"[dim]Currently cached: {len(cache)} companies[/dim]\n")

    total_new = 0

    for src in source_list:
        if src not in SOURCE_MAP:
            console.print(f"[red]Unknown source: {src}[/red]")
            continue
        if total_new >= limit:
            console.print(f"[yellow]Limit of {limit} new companies reached, stopping.[/yellow]")
            break

        fetcher = SOURCE_MAP[src]
        # Pass limit for sources that support it
        try:
            if src in ("yc", "remoteok", "hn_hiring"):
                new_companies = fetcher(existing_slugs, limit=limit - total_new)
            else:
                new_companies = fetcher(existing_slugs)
        except Exception as e:
            console.print(f"[red]Source {src} failed: {e}[/red]")
            continue

        for c in new_companies:
            slug = _slug(c["company"])
            cache[slug] = c
        total_new += len(new_companies)

    _save_cache(cache)

    state["last_run"] = datetime.now(timezone.utc).isoformat()
    state["total_found"] = len(cache)
    state["sources_run"] = source_list
    _save_state(state)

    all_companies = list(cache.values())
    console.print(
        f"\n[bold green]Discovery complete — "
        f"{total_new} new companies added, "
        f"{len(all_companies)} total in cache[/bold green]"
    )
    console.print(f"[dim]Cache: {CACHE_FILE}[/dim]")
    return all_companies


def load_cached_companies() -> list[dict]:
    """Load all cached companies for use in cv_matcher."""
    return list(_load_cache().values())


def cache_stats() -> dict:
    cache = _load_cache()
    state = _load_state()
    by_source: dict[str, int] = {}
    for c in cache.values():
        src = c.get("source", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
    return {
        "total": len(cache),
        "by_source": by_source,
        "last_run": state.get("last_run"),
    }


if __name__ == "__main__":
    import argparse
    from rich.pretty import Pretty

    parser = argparse.ArgumentParser(description="Autonomous AI Company Discovery")
    parser.add_argument("--source", default="all", help="Source(s): all, yc, remoteok, hn_hiring, uk, workable, wellfound")
    parser.add_argument("--fresh",  action="store_true", help="Ignore cache, re-fetch all")
    parser.add_argument("--limit",  type=int, default=1000, help="Max new companies per run")
    parser.add_argument("--stats",  action="store_true", help="Show cache stats only")
    args = parser.parse_args()

    if args.stats:
        stats = cache_stats()
        console.print(Pretty(stats))
    else:
        sources = args.source.split(",") if "," in args.source else args.source
        discover(sources=sources, fresh=args.fresh, limit=args.limit)
