#!/usr/bin/env python3
"""
Job-Outreach CLI — AI-powered outreach pipeline.

Commands:
  discover      Autonomously crawl YC, RemoteOK, HN Hiring, UK tech, Workable, Wellfound
  shortlist     Score all cached + bootstrap companies against your CV, print ranked matches
  cache         Show discovery cache statistics
  research      Scrape and analyse a specific company
  draft         Generate a personalised LinkedIn/email message (email includes Gmail compose link)
  pipeline      Research + draft both channels in one shot (includes Gmail compose link)
  add           Log a sent outreach to the tracker
  followups     Show (and optionally mark) pending follow-ups
  status        Print tracker stats
  agencies      Discover UK AI recruitment agencies, score them, draft registration emails

Examples:
  python main.py discover                               # crawl all sources
  python main.py discover --source yc,hn_hiring         # specific sources
  python main.py discover --fresh                       # ignore cache, re-fetch all
  python main.py shortlist --topk 20
  python main.py cache
  python main.py research "Vapi AI" https://vapi.ai
  python main.py draft "Jordan" "Founder" "Vapi AI" https://vapi.ai --channel email --to jordan@vapi.ai
  python main.py pipeline "Retell AI" https://retellai.com
  python main.py add "Jordan" "Founder" "Vapi AI" linkedin
  python main.py followups --mark
  python main.py status
  python main.py agencies                               # discover + rank agencies + draft emails
  python main.py agencies --fresh --topk 25 --no-emails
"""
import json
import re
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

console = Console()


# ── helpers ──────────────────────────────────────────────────────────────────

def _print_message(msg: str, title: str) -> None:
    console.print(Panel(msg, title=title, border_style="cyan"))


def _show_gmail_link(msg: str, company: str, recipient_email: str = "") -> None:
    """Parse an email draft and print a clickable Gmail compose link."""
    from utils.gmail import parse_email_draft, gmail_compose_link

    subject, body = parse_email_draft(msg)
    link = gmail_compose_link(subject=subject, body=body, to=recipient_email)
    preview = link[:120] + ("…" if len(link) > 120 else "")
    console.print(
        Panel(
            f"[bold]Subject:[/bold] {subject or '(none parsed)'}\n\n"
            f"[bold]Open in Gmail (pre-filled draft):[/bold]\n"
            f"[link={link}][cyan underline]{preview}[/cyan underline][/link]",
            title=f"[bold green]Gmail Draft → {company}[/bold green]",
            border_style="green",
        )
    )
    # Raw URL in case the terminal doesn't support hyperlinks
    console.print(f"\n[dim]Full URL:[/dim]\n[green]{link}[/green]\n")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ── CLI ───────────────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Job-Outreach: AI-powered outreach pipeline."""


@cli.command()
@click.option(
    "--source", default="all",
    help="Comma-separated sources: all | yc | remoteok | hn_hiring | uk | workable | wellfound",
    show_default=True,
)
@click.option("--fresh", is_flag=True, help="Ignore existing cache and re-fetch everything.")
@click.option("--limit", default=1000, show_default=True, help="Max new companies per run.")
def discover(source: str, fresh: bool, limit: int):
    """Autonomously crawl public sources to build a limitless company cache.

    Sources: YC AI companies, RemoteOK remote jobs, Hacker News 'Who is Hiring',
    UK tech scrapes (Sifted, EU-Startups), Workable UK, Wellfound.

    Results accumulate in data/company_cache.json (incremental by default).
    Run `shortlist` afterwards to score them against your CV.
    """
    from agents.autonomous_discovery import discover as _disc

    console.print(Rule("[bold cyan]Autonomous Discovery[/bold cyan]"))
    sources = [s.strip() for s in source.split(",")] if "," in source else source
    _disc(sources=sources, fresh=fresh, limit=limit)
    console.print("\n[green]Run `python main.py shortlist` to score everything against your CV.[/green]")


@cli.command()
@click.option("--topk", default=15, show_default=True, help="Number of top companies to show.")
@click.option("--source", default=None, help="Filter by discovery source (e.g. yc, hn_hiring).")
def shortlist(topk: int, source: str | None):
    """Score all cached + bootstrap companies against your CV and print ranked shortlist.

    Automatically merges the autonomous discovery cache with the bootstrap seed list.
    Run `discover` first to populate the cache with hundreds of companies.
    """
    from agents.cv_matcher import score_companies, print_shortlist, BOOTSTRAP_LIST
    from agents.autonomous_discovery import load_cached_companies, cache_stats

    console.print(Rule("[bold cyan]CV ↔ Company Matcher[/bold cyan]"))

    stats = cache_stats()
    if stats["total"] == 0:
        console.print("[yellow]Discovery cache is empty — running quick YC discovery first…[/yellow]")
        from agents.autonomous_discovery import discover as _disc
        _disc(sources="yc", limit=200)

    cached = load_cached_companies()
    if source:
        cached = [c for c in cached if c.get("source") == source]
        console.print(f"[dim]Filtered to source='{source}': {len(cached)} companies[/dim]")

    seen: set[str] = {_slug(c["company"]) for c in cached}
    extra = [c for c in BOOTSTRAP_LIST if _slug(c["company"]) not in seen]
    companies = cached + extra

    stats = cache_stats()
    console.print(
        f"[dim]Pool: {len(companies)} companies "
        f"({stats['total']} cached + {len(extra)} bootstrap)[/dim]"
    )
    console.print(f"[dim]Sources: {stats['by_source']}[/dim]\n")

    ranked = score_companies(companies=companies, topk=topk)
    print_shortlist(ranked)

    out = Path("data/shortlist.json")
    out.write_text(json.dumps(ranked, indent=2))
    console.print(f"\n[green]Saved → {out}[/green]")


@cli.command()
def cache():
    """Show autonomous discovery cache statistics."""
    from agents.autonomous_discovery import cache_stats
    from rich.table import Table

    stats = cache_stats()
    console.print(Rule("[bold cyan]Discovery Cache Stats[/bold cyan]"))

    t = Table(show_header=False)
    t.add_column("Key", style="bold")
    t.add_column("Value", style="cyan")
    t.add_row("Total companies", str(stats["total"]))
    t.add_row("Last run", str(stats.get("last_run") or "never"))
    console.print(t)

    if stats["by_source"]:
        t2 = Table(title="By Source", show_header=True)
        t2.add_column("Source", style="bold")
        t2.add_column("Count", style="green")
        for src, cnt in sorted(stats["by_source"].items(), key=lambda x: -x[1]):
            t2.add_row(src, str(cnt))
        console.print(t2)


@cli.command()
@click.argument("company")
@click.argument("website")
@click.option("--focus", default="AI startup", show_default=True)
def research(company: str, website: str, focus: str):
    """Scrape and analyse COMPANY at WEBSITE, extract fit + talking points."""
    from agents.research_agent import research_company
    from rich.pretty import Pretty

    console.print(Rule(f"[bold cyan]Research: {company}[/bold cyan]"))
    result = research_company(company, website, focus)
    console.print(Pretty(result))


@cli.command()
@click.argument("person_name")
@click.argument("person_role")
@click.argument("company")
@click.argument("website")
@click.option("--focus", default="AI startup")
@click.option(
    "--channel", default="both",
    type=click.Choice(["linkedin", "email", "both"]),
    show_default=True,
    help="Which channel(s) to draft for.",
)
@click.option("--to", "recipient_email", default="", help="Recipient email address for the Gmail compose link.")
def draft(
    person_name: str,
    person_role: str,
    company: str,
    website: str,
    focus: str,
    channel: str,
    recipient_email: str,
):
    """Draft personalised outreach for PERSON_NAME (PERSON_ROLE) at COMPANY.

    With --channel email or both, also prints a Gmail compose link so you
    can open the pre-filled draft directly in your browser.
    """
    from agents.research_agent import research_company
    from agents.message_agent import generate_message

    console.print(Rule(f"[bold cyan]Drafting for {person_name} @ {company}[/bold cyan]"))

    with console.status("Researching company…"):
        res = research_company(company, website, focus)

    channels = ["linkedin", "email"] if channel == "both" else [channel]
    for ch in channels:
        with console.status(f"Generating {ch} message…"):
            msg = generate_message(person_name, person_role, company, res, channel=ch)
        _print_message(msg, title=f"{ch.upper()} → {person_name} @ {company}")
        if ch == "email":
            _show_gmail_link(msg, company, recipient_email)


@cli.command()
@click.argument("company")
@click.argument("website")
@click.option("--focus", default="AI startup")
@click.option("--person", default="there", help="First name of the target contact.")
@click.option("--person-role", default="Founder", help="Their role (Founder, CTO, Head of AI…).")
@click.option("--to", "recipient_email", default="", help="Recipient email — pre-fills the Gmail link.")
@click.option("--skip-cv", is_flag=True, help="Skip CV specialisation (quick mode).")
def outreach(
    company: str,
    website: str,
    focus: str,
    person: str,
    person_role: str,
    recipient_email: str,
    skip_cv: bool,
):
    """Full outreach: scrape company → specialise CV → humanised email + Gmail link.

    The main command for targeted applications. Runs three steps:\n
      1. Scrapes and analyses the company website (cached per company).\n
      2. Creates a tailored 1-page DOCX CV in specialised_cvs/<company>/.\n
      3. Drafts a humanised email and generates a pre-filled Gmail compose link.\n

    Example:\n
      python main.py outreach "PolyAI" https://poly.ai --person Tom --person-role CTO --to tom@poly.ai
    """
    from agents.research_agent import research_company
    from agents.message_agent import generate_message
    from agents.cv_specialiser import specialise_cv
    from rich.pretty import Pretty
    from rich.table import Table

    console.print(Rule(f"[bold cyan]Outreach: {company}[/bold cyan]"))

    # Step 1 — Research
    with console.status("Scraping company website…"):
        res = research_company(company, website, focus)
    console.print(Rule("Research"))
    console.print(Pretty(res))

    # Step 2 — Specialise CV
    docx_path, pdf_path = None, None
    if not skip_cv:
        console.print(Rule("CV Specialisation"))
        with console.status("Tailoring CV for this company…"):
            docx_path, pdf_path = specialise_cv(company, res)
        console.print(f"\n[bold green]Specialised DOCX:[/bold green] {docx_path}")
        if pdf_path:
            console.print(f"[bold green]PDF for email:[/bold green] {pdf_path}")
        else:
            console.print("[yellow]PDF not converted — install LibreOffice: brew install --cask libreoffice[/yellow]")

    # Step 3 — Generate humanised email
    console.print(Rule("Email Draft"))
    with console.status("Drafting humanised email…"):
        email_msg = generate_message(person, person_role, company, res, channel="email")
    _print_message(email_msg, title=f"Email → {person} @ {company}")
    _show_gmail_link(email_msg, company, recipient_email)

    # Summary
    console.print(Rule("Summary"))
    t = Table(show_header=False, box=None)
    t.add_column(style="bold dim", width=22)
    t.add_column(style="cyan")
    t.add_row("Company",          company)
    t.add_row("Website",          website)
    t.add_row("Contact",          f"{person} ({person_role})")
    t.add_row("Recipient email",  recipient_email or "(not set — add with --to EMAIL)")
    t.add_row("Specialised DOCX", str(docx_path) if docx_path else "skipped (--skip-cv)")
    if pdf_path:
        t.add_row("PDF — attach manually", str(pdf_path))
    else:
        t.add_row("PDF", "not converted (brew install --cask libreoffice)")
    console.print(t)
    if pdf_path:
        console.print(f"\n[bold yellow]Attach to email:[/bold yellow] {pdf_path}")
        console.print("[dim]Gmail can't auto-attach via URL — drag the PDF into your compose window.[/dim]")
    console.print("\n[dim]Run `python main.py add` to log once sent.[/dim]\n")


@cli.command()
@click.argument("company")
@click.argument("website")
@click.argument("focus", default="AI startup")
@click.option("--to", "recipient_email", default="", help="Recipient email for Gmail compose link.")
def pipeline(company: str, website: str, focus: str, recipient_email: str):
    """Full pipeline in one shot: research COMPANY then draft both LinkedIn and email.

    The email draft is presented with a clickable Gmail compose link.
    """
    from agents.research_agent import research_company
    from agents.message_agent import generate_message
    from rich.pretty import Pretty

    console.print(Rule(f"[bold cyan]Pipeline: {company}[/bold cyan]"))

    with console.status("Researching…"):
        res = research_company(company, website, focus)

    console.print(Rule("Research"))
    console.print(Pretty(res))

    contact_roles = res.get("contact_roles", ["Founder"])
    person_role   = contact_roles[0] if contact_roles else "Founder"

    console.print(Rule("LinkedIn Draft"))
    with console.status("Generating LinkedIn message…"):
        li_msg = generate_message("there", person_role, company, res, channel="linkedin")
    _print_message(li_msg, title=f"LinkedIn → {company}")

    console.print(Rule("Email Draft + Gmail Link"))
    with console.status("Generating email…"):
        email_msg = generate_message("there", person_role, company, res, channel="email")
    _print_message(email_msg, title=f"Email → {company}")
    _show_gmail_link(email_msg, company, recipient_email)


@cli.command()
@click.argument("name")
@click.argument("role")
@click.argument("company")
@click.argument("channel", default="linkedin")
@click.option("--linkedin", default="", help="LinkedIn profile URL.")
@click.option("--email",    default="", help="Email address.")
@click.option("--message",  default="", help="The message you sent (paste it).")
@click.option("--notes",    default="")
def add(name: str, role: str, company: str, channel: str,
        linkedin: str, email: str, message: str, notes: str):
    """Log a sent outreach to the tracker."""
    from agents.tracker import add_record

    if not message:
        console.print("[yellow]Paste your message, then press Enter twice:[/yellow]")
        lines = []
        while True:
            try:
                line = input()
                if line == "":
                    break
                lines.append(line)
            except EOFError:
                break
        message = "\n".join(lines)

    rec = add_record(
        name=name, role=role, company=company, channel=channel,
        message=message, linkedin=linkedin, email=email, notes=notes,
    )
    console.print(f"[green]✓ Logged[/green] {name} @ {company} (ID: {rec['id'][:8]})")


@cli.command()
@click.option("--mark", is_flag=True, help="Mark generated follow-ups as sent.")
@click.option("--days", default=4, show_default=True, help="Days before follow-up is due.")
def followups(mark: bool, days: int):
    """Show pending follow-ups (optionally mark them as sent with --mark)."""
    from agents.followup_agent import run
    run(mark=mark, days=days)


@cli.command()
def status():
    """Print outreach tracker statistics and full record list."""
    from agents.tracker import stats, get_all
    from rich.table import Table

    s = stats()
    table = Table(title="Outreach Tracker Stats", show_header=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")
    for k, v in s.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)

    records = get_all()
    if records:
        t2 = Table(title="All Records", show_lines=True)
        t2.add_column("ID",      width=8)
        t2.add_column("Name",    min_width=12)
        t2.add_column("Company", min_width=16)
        t2.add_column("Channel", width=10)
        t2.add_column("Sent",    width=12)
        t2.add_column("Replied", width=8)
        t2.add_column("FU Sent", width=8)
        for r in records:
            t2.add_row(
                r["id"][:8],
                r["name"],
                r["company"],
                r["channel"],
                r["sent_at"][:10],
                "[green]yes[/green]" if r["replied"] else "[red]no[/red]",
                "[green]yes[/green]" if r["follow_up_sent"] else "[red]no[/red]",
            )
        console.print(t2)


@cli.command()
@click.option("--fresh",      is_flag=True,  help="Ignore cache, re-fetch all agencies.")
@click.option("--topk",       default=15,    show_default=True, help="Max agencies to show.")
@click.option("--min-score",  default=6,     show_default=True, help="Minimum LLM relevance score (0-10).")
@click.option("--no-emails",  is_flag=True,  help="Skip email drafting (faster, table only).")
@click.option("--to",         "recipient_email", default="", help="Pre-fill Gmail compose links with this address.")
def agencies(fresh: bool, topk: int, min_score: int, no_emails: bool, recipient_email: str):
    """Discover UK AI recruitment agencies, score by relevance, draft registration emails.

    Fires 25+ search queries at DuckDuckGo, scrapes each result, uses an LLM to score
    relevance for your profile (AI engineer, Glasgow/remote UK), then generates a
    personalised registration email and Gmail compose link for every agency above the
    score threshold.

    Results are cached in data/agency_cache.json — subsequent runs are fast.

    Examples:\n
      python main.py agencies                           # full run, score ≥6\n
      python main.py agencies --fresh --topk 25        # re-fetch + show more\n
      python main.py agencies --no-emails              # just the ranked table\n
      python main.py agencies --min-score 8            # only top-tier agencies
    """
    from agents.agency_hunter import run_agency_hunter

    console.print(Rule("[bold cyan]Agency Hunter[/bold cyan]"))
    run_agency_hunter(
        fresh=fresh,
        topk=topk,
        min_score=min_score,
        draft_emails=not no_emails,
        recipient_email=recipient_email,
    )


if __name__ == "__main__":
    cli()
