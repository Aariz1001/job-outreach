"""
Discovery Agent — dynamically finds AI companies from public sources so you're
not limited to a hand-coded seed list.

Sources:
  1. YC AI companies (public JSON API)
  2. Otta.com job listings (UK-focused job board) — scrapes AI/ML roles
  3. WorksHub (remote AI roles)

Run standalone:
  python -m agents.discovery_agent               # scrape all sources
  python -m agents.discovery_agent --source yc   # YC only
  python -m agents.discovery_agent --source otta # Otta only

Outputs to data/discovered_companies.json and can be fed into cv_matcher.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from rich.console import Console

from utils.config import DATA_DIR

console = Console()
DISCOVERED_FILE = DATA_DIR / "discovered_companies.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}

# Tags we care about (YC uses these)
AI_TAGS = {
    "artificial intelligence", "machine learning", "nlp", "voice ai",
    "generative ai", "llm", "conversational ai", "ai assistant",
    "ai automation", "chatbot", "speech recognition", "ai agents",
    "developer tools", "b2b", "saas",
}

UK_REGIONS = {
    "london", "uk", "united kingdom", "england", "scotland",
    "glasgow", "edinburgh", "manchester", "cambridge", "bristol",
    "remote", "anywhere",
}


# ── Source 1: YC companies ────────────────────────────────────────────────────

def _scrape_yc_ai() -> list[dict[str, Any]]:
    """Pull AI-tagged companies from YC's public API (batches W21-present)."""
    console.print("[cyan]Fetching YC AI companies…[/cyan]")

    url = "https://api.ycombinator.com/v0.1/companies"
    params = {
        "tags": "Artificial Intelligence",
        "page": 1,
    }
    companies = []

    for page in range(1, 8):  # max 7 pages ≈ 350 companies
        params["page"] = page
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if resp.status_code != 200:
                break
            data = resp.json()
            items = data.get("companies", [])
            if not items:
                break

            for c in items:
                name = c.get("name", "")
                website = c.get("url", "") or c.get("website", "")
                description = c.get("oneLiner", "") or c.get("longDescription", "")
                batch = c.get("batch", "")
                tags = [t.lower() for t in (c.get("tags") or [])]
                team_size = c.get("teamSize", "")
                locations = [
                    (loc.get("city") or loc.get("country") or "").lower()
                    for loc in (c.get("locations") or [])
                ]

                if not name or not website:
                    continue

                # Filter: must have ≥1 AI tag or AI keyword in description
                desc_lower = (description or "").lower()
                has_ai = any(t in AI_TAGS for t in tags) or any(
                    kw in desc_lower for kw in [
                        "voice ai", "llm", "language model", "agent",
                        "generative", "rag", "speech", "nlp", "gpt"
                    ]
                )
                if not has_ai:
                    continue

                # Prefer UK/remote but don't exclude global
                loc_str = ", ".join(
                    loc.get("city", "") or loc.get("country", "")
                    for loc in (c.get("locations") or [])
                ) or "Remote"

                companies.append({
                    "company": name,
                    "website": website,
                    "focus": description[:120] if description else "AI startup",
                    "stage": f"YC {batch}" if batch else "YC",
                    "size": str(team_size) if team_size else "—",
                    "contact_hint": "founders / CTO",
                    "location": loc_str,
                    "description": description or f"{name} is a YC-backed AI startup.",
                    "source": "yc",
                })

            time.sleep(0.5)
        except Exception as e:
            console.print(f"[red]YC page {page} error: {e}[/red]")
            break

    console.print(f"[green]YC: found {len(companies)} AI companies[/green]")
    return companies


# ── Source 2: Otta / Welcome to the Jungle (UK AI jobs) ──────────────────────

def _scrape_otta_ai() -> list[dict[str, Any]]:
    """
    Scrape AI/ML job listings from Otta (UK-focused job board).
    Extracts unique companies posting AI roles.
    """
    console.print("[cyan]Fetching Otta AI job listings (UK)…[/cyan]")

    # Otta's public search — returns company/job cards
    base_url = "https://app.otta.com/jobs/search"
    companies_seen: dict[str, dict] = {}

    # Try their API endpoint (used by the SPA)
    api_endpoints = [
        "https://api.otta.com/graphql",  # GraphQL — skip if blocked
    ]

    # Fallback: scrape the HTML search results for AI roles
    search_url = "https://app.otta.com/jobs/search?query=AI+engineer+UK"
    try:
        resp = requests.get(search_url, headers=HEADERS, timeout=20)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Extract company names/links from job cards (structure may vary)
        for card in soup.select("[class*='company'], [class*='job-card'], article"):
            name_el = card.select_one("[class*='company-name'], h2, h3")
            link_el = card.select_one("a[href]")
            if name_el and name_el.get_text(strip=True):
                name = name_el.get_text(strip=True)
                href = link_el["href"] if link_el else ""
                if name and name not in companies_seen:
                    companies_seen[name] = {
                        "company": name,
                        "website": href if href.startswith("http") else f"https://otta.com{href}",
                        "focus": "AI startup (from Otta job listing)",
                        "stage": "—",
                        "size": "—",
                        "contact_hint": "engineering / founders",
                        "location": "UK / Remote",
                        "description": f"{name} is hiring AI engineers (found via Otta UK).",
                        "source": "otta",
                    }
    except Exception as e:
        console.print(f"[yellow]Otta scrape partial: {e}[/yellow]")

    results = list(companies_seen.values())
    console.print(f"[green]Otta: found {len(results)} companies[/green]")
    return results


# ── Source 3: RemoteOK AI jobs ────────────────────────────────────────────────

def _scrape_remoteok_ai() -> list[dict[str, Any]]:
    """Scrape RemoteOK for remote AI/ML engineer roles — extract unique companies."""
    console.print("[cyan]Fetching RemoteOK AI jobs…[/cyan]")
    companies_seen: dict[str, dict] = {}

    try:
        # RemoteOK has a public JSON API
        resp = requests.get(
            "https://remoteok.com/api?tag=ai",
            headers={**HEADERS, "Accept": "application/json"},
            timeout=20,
        )
        jobs = resp.json()

        for job in jobs:
            if not isinstance(job, dict):
                continue
            company = job.get("company", "")
            url = job.get("url", "")
            position = job.get("position", "")
            description = job.get("description", "")
            tags = [t.lower() for t in (job.get("tags") or [])]

            if not company:
                continue

            # Must be AI-related
            ai_kws = ["ai", "ml", "llm", "agent", "voice", "nlp", "gpt", "gemini"]
            is_ai = any(kw in (position + description).lower() for kw in ai_kws) or \
                    any(t in ai_kws for t in tags)
            if not is_ai:
                continue

            if company not in companies_seen:
                # Try to get their real website from job URL
                website = job.get("company_logo", "")
                companies_seen[company] = {
                    "company": company,
                    "website": f"https://remoteok.com{url}" if url.startswith("/") else url,
                    "focus": position or "AI / ML role",
                    "stage": "—",
                    "size": "—",
                    "contact_hint": "engineering / founders",
                    "location": "Remote",
                    "description": (
                        f"{company} is hiring for '{position}'. "
                        f"{description[:200].strip() if description else ''}"
                    ),
                    "source": "remoteok",
                }

        time.sleep(1)
    except Exception as e:
        console.print(f"[yellow]RemoteOK error: {e}[/yellow]")

    results = list(companies_seen.values())
    console.print(f"[green]RemoteOK: found {len(results)} companies[/green]")
    return results


# ── Source 4: WorksHub (remote AI) ───────────────────────────────────────────

def _scrape_workable_ai_uk() -> list[dict[str, Any]]:
    """
    Scrape AI engineer jobs from the UK via Workable public job board search.
    """
    console.print("[cyan]Fetching Workable UK AI jobs…[/cyan]")
    companies_seen: dict[str, dict] = {}

    try:
        resp = requests.get(
            "https://jobs.workable.com/api/v1/jobs?query=ai+engineer&location=United+Kingdom&remote=true",
            headers=HEADERS,
            timeout=20,
        )
        data = resp.json()
        for job in data.get("results", []):
            company = job.get("company", {}).get("name", "")
            website = job.get("company", {}).get("url", "")
            title = job.get("title", "")
            location = job.get("location", {}).get("city", "") or "UK"

            if not company:
                continue

            ai_kws = ["ai", "ml", "llm", "agent", "voice", "nlp", "machine learning"]
            if not any(kw in title.lower() for kw in ai_kws):
                continue

            if company not in companies_seen:
                companies_seen[company] = {
                    "company": company,
                    "website": website or "",
                    "focus": title,
                    "stage": "—",
                    "size": "—",
                    "contact_hint": "engineering / founders",
                    "location": f"{location}, UK",
                    "description": f"{company} is hiring for '{title}' (UK/remote).",
                    "source": "workable",
                }
    except Exception as e:
        console.print(f"[yellow]Workable scrape partial: {e}[/yellow]")

    results = list(companies_seen.values())
    console.print(f"[green]Workable: found {len(results)} companies[/green]")
    return results


# ── Deduplication ─────────────────────────────────────────────────────────────

def _dedup(companies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for c in companies:
        key = re.sub(r"[^a-z0-9]", "", c["company"].lower())
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def discover(source: str = "all") -> list[dict[str, Any]]:
    """
    Fetch companies from the specified source.

    source: "all" | "yc" | "otta" | "remoteok" | "workable"
    """
    results: list[dict[str, Any]] = []

    if source in ("all", "yc"):
        results += _scrape_yc_ai()
    if source in ("all", "otta"):
        results += _scrape_otta_ai()
    if source in ("all", "remoteok"):
        results += _scrape_remoteok_ai()
    if source in ("all", "workable"):
        results += _scrape_workable_ai_uk()

    results = _dedup(results)

    DISCOVERED_FILE.write_text(json.dumps(results, indent=2))
    console.print(
        f"\n[bold green]Total discovered: {len(results)} companies → {DISCOVERED_FILE}[/bold green]"
    )
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Discover AI companies from public sources")
    parser.add_argument(
        "--source",
        default="all",
        choices=["all", "yc", "otta", "remoteok", "workable"],
        help="Which source to scrape",
    )
    args = parser.parse_args()
    discover(source=args.source)
