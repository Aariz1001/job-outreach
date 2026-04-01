"""
Agency Hunter — autonomously discovers UK tech/AI recruitment agencies,
scores each one for relevance, then generates a personalised registration
email + Gmail compose link for every agency above the score threshold.

How it works
────────────
1. Seed search queries are fired at DuckDuckGo HTML (no API key needed).
2. Each result URL is scraped for agency name, contact email, specialism,
   and whether they have Glasgow/Scotland or remote-UK coverage.
3. An LLM call scores + enriches each agency and produces a tailored
   registration email using the candidate profile in utils/config.
4. Results are cached in data/agency_cache.json (incremental like company_cache).
5. The CLI command prints a ranked table and a Gmail link per agency.

Run standalone:
  python -m agents.agency_hunter
  python -m agents.agency_hunter --fresh        # re-fetch everything
  python -m agents.agency_hunter --topk 20      # show more results
  python -m agents.agency_hunter --min-score 6  # lower threshold
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote_plus, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from rich.console import Console
from rich.table import Table
from rich.rule import Rule

from utils.config import DATA_DIR, GENERATION_MODEL, OPENROUTER_API_KEY
from utils.gmail import gmail_compose_link

console = Console()

AGENCY_CACHE_FILE = DATA_DIR / "agency_cache.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xhtml+xml;q=0.9,*/*;q=0.8",
}

_llm = OpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")

# ── Candidate profile (used in email drafts) ──────────────────────────────────

CANDIDATE = {
    "name": "Aariz",
    "full_name": "Mohammad Aariz Waqas",
    "email": "m.aariz.shah@gmail.com",
    "location": "Glasgow, UK",
    "role": "AI Engineer",
    "github": "github.com/Aariz1001/job-outreach",
    "target_roles": "AI Engineer / ML Engineer / AI Software Engineer",
    "target_level": "Junior–Mid (1-3 years)",
    "target_stage": "Seed / Series A startup, or innovative scale-up",
    "target_location": "Glasgow-based or remote UK",
    "salary_range": "£40,000–£60,000",
    "not_interested_in": "large enterprise, consultancy, outsourcing firms",
    "cv_summary": (
        "University of Glasgow, Computing Science. "
        "Production AI systems: LLM agent pipelines, RAG, multi-agent orchestration "
        "(LangChain/CrewAI/LlamaIndex/OpenRouter), WhatsApp AI agent on GCP/Twilio/Gemini, "
        "sub-200ms intent classification for live voice systems, multi-tenant AI platforms. "
        "Python, GCP, Firebase, Flutter. "
        "Built an autonomous job outreach tool (github.com/Aariz1001/job-outreach) that crawls "
        "120+ companies, scores them by CV fit, and generates tailored emails — used it to find you."
    ),
}

# ── Search queries — deliberately varied to reach niche/underground agencies ──

SEARCH_QUERIES: list[str] = [
    # Glasgow-specific
    "AI engineer jobs Glasgow recruitment agency",
    "machine learning jobs Glasgow Scotland recruiter",
    "tech recruitment agency Glasgow Scotland AI",
    "AI startup jobs glasgow recruitment consultant",
    "python developer jobs glasgow recruitment agency 2025",
    # UK-wide AI specialists
    "AI machine learning recruitment agency UK specialist",
    "LLM engineer jobs UK recruitment",
    "generative AI jobs UK recruitment agency",
    "AI engineer jobs UK startup recruitment",
    "data science AI recruitment agency UK niche",
    # Niche / less-known angles
    "site:linkedin.com/company recruiter AI machine learning Glasgow",
    "\"AI recruitment\" OR \"ML recruitment\" agency UK independent boutique",
    "\"AI engineer\" placement agency UK early career",
    "python AI engineer jobs uk boutique recruitment",
    "nlp engineer jobs uk recruitment specialist agency",
    "\"language model\" engineer jobs uk recruitment",
    # Job boards that list agencies
    "AI engineer jobs Glasgow site:cv-library.co.uk recruiter",
    "AI engineer jobs Glasgow site:reed.co.uk agency",
    "AI engineer jobs remote UK site:totaljobs.com agency",
    # Scottish ecosystem
    "Techscaler Scotland AI jobs recruitment",
    "ScotlandIS tech jobs AI recruitment",
    "CodeBase Edinburgh Glasgow AI engineer jobs recruiting",
    # LinkedIn company page searches
    "site:linkedin.com/company \"ai recruitment\" OR \"machine learning recruitment\" UK",
    # Underdog angles
    "\"AI placement\" agency UK submit CV",
    "\"technology recruitment\" Scotland boutique AI",
    "emerging tech recruiter UK AI python engineer",
]

# Known domains to skip (job boards, not agencies)
SKIP_DOMAINS = {
    "indeed.com", "linkedin.com", "glassdoor.com", "totaljobs.com",
    "reed.co.uk", "cv-library.co.uk", "monster.co.uk", "jobsite.co.uk",
    "cwjobs.co.uk", "technojobs.co.uk", "github.com", "stackoverflow.com",
    "google.com", "bing.com", "bbc.co.uk", "theguardian.com",
    "twitter.com", "x.com", "facebook.com", "youtube.com",
    "ycombinator.com", "wellfound.com", "angellist.com",
}

# Seed list of known agencies so we never miss the obvious ones
SEED_AGENCIES = [
    {"name": "Cathcart Technology", "website": "https://cathcart.io",
     "specialism": "Tech/AI, Scotland", "location_coverage": "Glasgow/Scotland/Remote UK"},
    {"name": "Harnham", "website": "https://harnham.com",
     "specialism": "Data & AI specialist, global", "location_coverage": "UK-wide/Remote"},
    {"name": "Eden Scott", "website": "https://edenscott.com",
     "specialism": "Scottish generalist + Tech", "location_coverage": "Glasgow/Edinburgh"},
    {"name": "Understanding Recruitment", "website": "https://understandingrecruitment.com",
     "specialism": "ML & AI, Python, backend engineering", "location_coverage": "UK-wide/Remote"},
    {"name": "Empiric", "website": "https://empiric.com",
     "specialism": "Data & AI, Cloud, Glasgow office", "location_coverage": "UK-wide"},
    {"name": "Lorien", "website": "https://lorienglobal.com",
     "specialism": "Tech recruitment UK, Scotland office", "location_coverage": "UK-wide"},
    {"name": "Nigel Frank", "website": "https://nigelfrank.com",
     "specialism": "Microsoft + emerging tech UK", "location_coverage": "UK-wide"},
    {"name": "La Fosse", "website": "https://lafosse.com",
     "specialism": "Tech + AI leadership UK", "location_coverage": "UK-wide"},
    {"name": "Tenth Revolution Group", "website": "https://tenthrevolution.com",
     "specialism": "Tech & data UK, boutique", "location_coverage": "UK-wide"},
    {"name": "Broster Buchanan", "website": "https://brosterbuchanan.com",
     "specialism": "Scotland tech + data", "location_coverage": "Scotland/Remote"},
    {"name": "CGI Talent", "website": "https://cgitalent.co.uk",
     "specialism": "AI/data UK boutique", "location_coverage": "UK-wide"},
    {"name": "Burns Sheehan", "website": "https://burnssheehan.co.uk",
     "specialism": "Tech startup recruitment UK", "location_coverage": "UK-wide/Remote"},
    {"name": "FRG Technology Consulting", "website": "https://frgconsulting.com",
     "specialism": "Data Science & ML UK", "location_coverage": "UK-wide"},
    {"name": "Inspired Search", "website": "https://inspiredsearch.co.uk",
     "specialism": "Scotland tech recruitment", "location_coverage": "Scotland"},
    {"name": "Venn Group", "website": "https://venngroup.com",
     "specialism": "Public sector + tech UK", "location_coverage": "Scotland/UK"},
]


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _load_agency_cache() -> dict[str, dict]:
    if AGENCY_CACHE_FILE.exists():
        return json.loads(AGENCY_CACHE_FILE.read_text())
    return {}


def _save_agency_cache(cache: dict[str, dict]) -> None:
    AGENCY_CACHE_FILE.write_text(json.dumps(cache, indent=2))


# ── Web helpers ───────────────────────────────────────────────────────────────

def _fetch_text(url: str, max_chars: int = 6000) -> str:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "noscript"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return re.sub(r"\s+", " ", text)[:max_chars]
    except Exception:
        return ""


def _extract_emails(text: str) -> list[str]:
    """Pull email addresses from raw text."""
    found = re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
    # Filter out image/asset URLs that got picked up
    return [e for e in found if not any(
        e.endswith(ext) for ext in [".png", ".jpg", ".gif", ".svg", ".webp"]
    )]


def _extract_links_from_ddg(query: str, max_results: int = 8) -> list[str]:
    """
    Scrape DuckDuckGo HTML results for a query.
    Returns a list of result URLs (not DDG redirect URLs).
    """
    try:
        resp = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={**HEADERS, "Referer": "https://duckduckgo.com/"},
            timeout=20,
        )
        soup = BeautifulSoup(resp.text, "html.parser")
        urls: list[str] = []

        for a in soup.select("a.result__url, a[href*='uddg=']"):
            href = a.get("href", "")
            # DDG wraps results in redirect — extract the real URL
            if "uddg=" in href:
                import urllib.parse
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                real = parsed.get("uddg", [""])[0]
                if real:
                    href = real
            # Normalise
            if href.startswith("//"):
                href = "https:" + href
            if not href.startswith("http"):
                continue

            domain = urlparse(href).netloc.lstrip("www.")
            if any(skip in domain for skip in SKIP_DOMAINS):
                continue

            if href not in urls:
                urls.append(href)
            if len(urls) >= max_results:
                break

        return urls
    except Exception as e:
        console.print(f"[yellow]  [DDG] search error: {e}[/yellow]")
        return []


# ── LLM analysis ──────────────────────────────────────────────────────────────

def _llm_analyse_agency(name: str, website: str, page_text: str, known_specialism: str = "") -> dict:
    """
    Use LLM to extract structured info from the agency page and score
    relevance for the candidate profile.
    """
    prompt = f"""You are helping a job seeker (Aariz) identify the best recruitment agencies.

Candidate profile:
- Role sought: {CANDIDATE["target_roles"]}
- Level: {CANDIDATE["target_level"]}
- Location: {CANDIDATE["location"]} — wants Glasgow-based or remote UK roles
- Skills: LLM agents, RAG, LangChain, CrewAI, Python, GCP, multi-agent systems
- Target companies: AI startups at seed–Series A
- NOT interested in: {CANDIDATE["not_interested_in"]}
- Salary: {CANDIDATE["salary_range"]}

Agency being evaluated:
- Name: {name}
- Website: {website}
- Known specialism hint: {known_specialism or "unknown"}
- Page content:
{page_text[:4000]}

Return ONLY valid JSON (no markdown fences) with these exact keys:
{{
  "name": "Official agency name",
  "website": "{website}",
  "specialism": "One sentence: what types of roles they fill",
  "location_coverage": "Glasgow / Scotland / UK-wide / Remote UK etc",
  "has_ai_ml_desk": true or false,
  "contact_email": "best registration/candidate email if found, else empty string",
  "contact_page": "URL of their candidate upload/register page if found, else empty string",
  "score": 0-10,
  "score_reason": "One sentence why this score",
  "is_agency": true or false
}}

Score guide:
10 = AI/ML specialist with Glasgow or remote UK coverage
8-9 = strong tech recruiter with Glasgow/Scotland or UK-wide, have placed AI roles
6-7 = general tech recruiter, UK-wide, likely see AI roles
4-5 = generalist with some tech, limited AI focus
1-3 = unlikely to help (wrong specialism, no UK base, pure enterprise/consultancy)
0 = not a recruitment agency at all

Set is_agency to false if the URL is a job board, employer, news site, or anything other than a recruitment/staffing agency."""

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
        result = json.loads(raw)
        result["cached_at"] = datetime.now(timezone.utc).isoformat()
        return result
    except Exception as e:
        console.print(f"[yellow]  [LLM] analyse error for {name}: {e}[/yellow]")
        return {}


def _llm_draft_email(agency: dict) -> str:
    """Generate a personalised registration email for a specific agency."""
    prompt = f"""Write a short, direct candidate registration email from {CANDIDATE["full_name"]} to the recruitment agency below.

Agency: {agency.get("name")}
Agency specialism: {agency.get("specialism")}
Agency coverage: {agency.get("location_coverage")}

Candidate profile:
{CANDIDATE["cv_summary"]}

Looking for: {CANDIDATE["target_roles"]} at {CANDIDATE["target_stage"]}, 
location: {CANDIDATE["target_location"]}, salary: {CANDIDATE["salary_range"]}.
NOT interested in: {CANDIDATE["not_interested_in"]}.

Rules:
- Plain text only. No markdown.
- Subject line on the first line, formatted as: Subject: [subject here]
- Leave one blank line, then write the email body.
- Maximum 150 words in the body.
- Do not use "I am excited to", "I am passionate about", "leverage", "harness", "streamline", "delve", "landscape", "synergies".
- Open with what the candidate does and has built, not with enthusiasm.
- Mention the GitHub project (github.com/Aariz1001/job-outreach) as proof of work — it found this agency.
- End with: name, email, phone placeholder [PHONE], and a note that CV is attached.
- Tone: confident, direct, brief. The email should make the agency want to call."""

    try:
        resp = _llm.chat.completions.create(
            model=GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        raw = resp.choices[0].message.content.strip()
        # Strip any accidental markdown
        raw = re.sub(r"\*\*(.+?)\*\*", r"\1", raw)
        raw = re.sub(r"\*(.+?)\*", r"\1", raw)
        return raw
    except Exception as e:
        console.print(f"[yellow]  [LLM] email draft error: {e}[/yellow]")
        return ""


# ── Discovery pipeline ────────────────────────────────────────────────────────

def _discover_from_search(existing_slugs: set[str], fresh: bool = False) -> list[dict]:
    """Fire search queries and scrape each result URL."""
    discovered: list[dict] = []

    cache = {} if fresh else _load_agency_cache()

    with console.status("[cyan]Running search queries…[/cyan]"):
        seen_urls: set[str] = set()

        for i, query in enumerate(SEARCH_QUERIES):
            console.print(f"  [dim][{i+1}/{len(SEARCH_QUERIES)}] {query[:70]}[/dim]")
            urls = _extract_links_from_ddg(query, max_results=6)

            for url in urls:
                domain = urlparse(url).netloc.lstrip("www.")
                slug = _slug(domain)

                if slug in existing_slugs or slug in seen_urls:
                    continue
                seen_urls.add(slug)

                # Fetch page
                text = _fetch_text(url)
                if len(text) < 200:
                    continue

                # Use domain as placeholder name until LLM corrects it
                placeholder_name = domain.split(".")[0].title()

                result = _llm_analyse_agency(
                    name=placeholder_name,
                    website=url,
                    page_text=text,
                )

                if not result or not result.get("is_agency"):
                    continue

                score = result.get("score", 0)
                if score < 3:
                    continue

                result["source"] = "search"
                slug_final = _slug(result.get("name", placeholder_name))
                cache[slug_final] = result
                existing_slugs.add(slug_final)
                discovered.append(result)

            # Be polite to DDG
            time.sleep(1.5)

    return discovered


def _process_seed_agencies(existing_slugs: set[str], cache: dict, fresh: bool = False) -> list[dict]:
    """Score the hardcoded seed agencies if not already cached."""
    results: list[dict] = []

    for seed in SEED_AGENCIES:
        slug = _slug(seed["name"])
        if slug in existing_slugs and not fresh:
            # Already have it — load from cache
            if slug in cache:
                results.append(cache[slug])
            continue

        console.print(f"  [dim]Seeding: {seed['name']}[/dim]")
        text = _fetch_text(seed["website"])
        result = _llm_analyse_agency(
            name=seed["name"],
            website=seed["website"],
            page_text=text,
            known_specialism=seed["specialism"],
        )
        if result:
            result["source"] = "seed"
            cache[slug] = result
            existing_slugs.add(slug)
            results.append(result)
        time.sleep(0.8)

    return results


# ── Main entry point ──────────────────────────────────────────────────────────

def run_agency_hunter(
    fresh: bool = False,
    topk: int = 15,
    min_score: int = 6,
    draft_emails: bool = True,
    recipient_email: str = "",
) -> list[dict]:
    """
    Full pipeline:
      1. Score seed agencies
      2. Search + scrape for more
      3. Rank by score
      4. Draft personalised email per agency
      5. Print results table
    """
    console.print(Rule("[bold cyan]Agency Hunter[/bold cyan]"))

    cache = {} if fresh else _load_agency_cache()
    existing_slugs: set[str] = set(cache.keys())

    # Step 1 — seed agencies
    console.print(Rule("Processing seed agencies", style="dim"))
    seed_results = _process_seed_agencies(existing_slugs, cache, fresh=fresh)
    _save_agency_cache(cache)
    console.print(f"[green]Seed agencies: {len(seed_results)} processed[/green]")

    # Step 2 — search discovery
    console.print(Rule("Search discovery", style="dim"))
    search_results = _discover_from_search(existing_slugs, fresh=fresh)
    # Reload cache (seed step mutated it) and merge
    cache = _load_agency_cache()
    for r in search_results:
        slug = _slug(r.get("name", ""))
        if slug:
            cache[slug] = r
    _save_agency_cache(cache)
    console.print(f"[green]Search discovered: {len(search_results)} new agencies[/green]")

    # Step 3 — rank
    all_agencies = list(cache.values())
    ranked = sorted(
        [a for a in all_agencies if a.get("is_agency") and a.get("score", 0) >= min_score],
        key=lambda x: x.get("score", 0),
        reverse=True,
    )[:topk]

    # Step 4 — print table
    console.print(Rule("Ranked Agencies", style="bold green"))
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("#", width=3)
    table.add_column("Agency", min_width=22)
    table.add_column("Score", width=6, justify="center")
    table.add_column("Specialism", min_width=28)
    table.add_column("Coverage", min_width=20)
    table.add_column("Contact", min_width=24)

    for i, a in enumerate(ranked, 1):
        score = a.get("score", 0)
        score_str = f"[green]{score}[/green]" if score >= 8 else (
            f"[yellow]{score}[/yellow]" if score >= 6 else f"[red]{score}[/red]"
        )
        contact = a.get("contact_email") or a.get("contact_page") or a.get("website", "")
        table.add_row(
            str(i),
            a.get("name", "?"),
            score_str,
            (a.get("specialism") or "")[:40],
            (a.get("location_coverage") or "")[:25],
            contact[:35],
        )
    console.print(table)

    # Step 5 — draft emails
    if draft_emails:
        console.print(Rule("Email Drafts + Gmail Links", style="bold cyan"))
        for a in ranked:
            name = a.get("name", "Agency")
            console.print(Rule(f"[bold]{name}[/bold]", style="dim"))

            draft = _llm_draft_email(a)
            if not draft:
                continue

            # Split subject/body
            lines = draft.strip().splitlines()
            subject = ""
            body_lines: list[str] = []
            body_started = False
            for line in lines:
                if not body_started and line.lower().startswith("subject:"):
                    subject = line.split(":", 1)[1].strip()
                    body_started = True
                elif body_started:
                    body_lines.append(line)
            body = "\n".join(body_lines).strip()

            console.print(f"[bold]Subject:[/bold] {subject}")
            console.print()
            console.print(body)
            console.print()

            to = recipient_email or a.get("contact_email", "")
            link = gmail_compose_link(subject=subject, body=body, to=to)
            console.print(f"[bold cyan]Gmail:[/bold cyan] {link[:120]}…")
            console.print()

            time.sleep(0.5)  # avoid rate limiting between LLM calls

    console.print(Rule("Done", style="bold green"))
    console.print(
        f"[green]{len(ranked)} agencies at score ≥{min_score}. "
        f"Full cache: {len(cache)} entries → {AGENCY_CACHE_FILE}[/green]"
    )
    return ranked


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agency Hunter — find UK AI recruitment agencies")
    parser.add_argument("--fresh",      action="store_true", help="Ignore cache, re-fetch everything")
    parser.add_argument("--topk",       type=int, default=15, help="Max agencies to show")
    parser.add_argument("--min-score",  type=int, default=6,  help="Minimum LLM score (0-10)")
    parser.add_argument("--no-emails",  action="store_true",  help="Skip email drafting")
    parser.add_argument("--to",         default="",           help="Pre-fill Gmail links with this email")
    args = parser.parse_args()

    run_agency_hunter(
        fresh=args.fresh,
        topk=args.topk,
        min_score=args.min_score,
        draft_emails=not args.no_emails,
        recipient_email=args.to,
    )
