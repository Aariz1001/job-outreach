"""
CV Matcher — scores companies (from autonomous discovery cache + seed bootstrap)
against your CV embedding and returns a ranked shortlist.

Primary source: data/company_cache.json  (populated by autonomous_discovery.py)
Fallback:       BOOTSTRAP_LIST below      (always included, ~10 high-signal targets)

Usage:
    python -m agents.cv_matcher            # scores all cached + bootstrap
    python -m agents.cv_matcher --topk 15  # show top 15
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from utils.config import CV_EMBED_FILE, DATA_DIR
from utils.embeddings import cosine_similarity, embed_text
from utils.pdf_utils import extract_cv_text

console = Console()

CV_PDF = Path(__file__).parent.parent / "Mohammad_Aariz_Waqas_AI_Engineer.pdf"

# ------------------------------------------------------------------
# BOOTSTRAP — always scored; high-confidence targets that should
# always appear regardless of what the crawler finds.
# This is intentionally small — the real list comes from the cache.
# ------------------------------------------------------------------
BOOTSTRAP_LIST: list[dict[str, Any]] = [
    # Same entries, just renamed variable
    {
        "company": "PolyAI",
        "website": "https://poly.ai",
        "focus": "Voice AI customer service agents, conversational AI",
        "stage": "Series C",
        "size": "100-300",
        "contact_hint": "engineering leads / ML engineers",
        "location": "London, UK (remote-friendly)",
        "description": (
            "PolyAI builds voice assistants for enterprise customer service. "
            "They deploy production voice agents for large call centres, "
            "with real-time NLU, interruption handling, and multi-turn dialogue."
        ),
    },
    {
        "company": "Speechmatics",
        "website": "https://speechmatics.com",
        "focus": "Speech recognition, real-time transcription, voice AI APIs",
        "stage": "Series B",
        "size": "100-200",
        "contact_hint": "engineering / product engineers",
        "location": "Cambridge, UK (remote-friendly)",
        "description": (
            "Speechmatics develops world-class speech recognition technology "
            "including real-time transcription APIs, speaker identification, "
            "and voice AI integrations used in enterprise products."
        ),
    },
    {
        "company": "Cleo",
        "website": "https://web.meetcleo.com",
        "focus": "AI financial assistant, LLM-powered chatbot for finance",
        "stage": "Series C",
        "size": "150-300",
        "contact_hint": "AI / ML engineers",
        "location": "London, UK (remote-friendly)",
        "description": (
            "Cleo builds an AI-powered personal finance assistant that uses "
            "LLMs and conversational agents to help users manage money, "
            "predict spending, and automate financial workflows."
        ),
    },
    {
        "company": "Faculty AI",
        "website": "https://faculty.ai",
        "focus": "Applied AI consulting and AI product development",
        "stage": "Series B",
        "size": "100-250",
        "contact_hint": "engineering leads",
        "location": "London, UK",
        "description": (
            "Faculty AI builds and deploys applied AI systems for enterprise "
            "and government clients, specialising in production ML pipelines, "
            "decision intelligence, and AI automation workflows."
        ),
    },
    {
        "company": "Wayve",
        "website": "https://wayve.ai",
        "focus": "Autonomous driving AI, embodied AI, real-time inference",
        "stage": "Series C",
        "size": "200-500",
        "contact_hint": "AI / software engineers",
        "location": "London, UK",
        "description": (
            "Wayve develops end-to-end learned autonomous driving systems "
            "using large-scale AI models and real-time deployment pipelines "
            "for self-driving vehicles."
        ),
    },
    {
        "company": "Synthesia",
        "website": "https://synthesia.io",
        "focus": "AI video generation, synthetic media, LLM pipelines",
        "stage": "Series D",
        "size": "200-400",
        "contact_hint": "AI / backend engineers",
        "location": "London, UK (remote)",
        "description": (
            "Synthesia builds AI video generation tools that let users create "
            "professional videos with AI avatars, integrating LLMs, TTS, and "
            "video synthesis into enterprise content workflows."
        ),
    },
    {
        "company": "Phoebe AI",
        "website": "https://phoebe.ai",
        "focus": "AI voice agents for healthcare and care coordination",
        "stage": "Seed",
        "size": "5-20",
        "contact_hint": "founders",
        "location": "UK (remote)",
        "description": (
            "Phoebe AI builds voice AI agents that handle care coordination "
            "calls, patient follow-ups, and health admin automation with "
            "real-time voice pipelines."
        ),
    },
    {
        "company": "Cogito",
        "website": "https://cogitocorp.com",
        "focus": "Real-time voice AI coaching for call centres",
        "stage": "Series C",
        "size": "100-300",
        "contact_hint": "engineering",
        "location": "Remote / UK",
        "description": (
            "Cogito delivers real-time AI that analyses live phone calls and "
            "provides live coaching signals to agents, using voice and "
            "speech intelligence pipelines."
        ),
    },
    {
        "company": "Tortoise Media / Tortoise Intelligence",
        "website": "https://www.tortoisemedia.com",
        "focus": "AI-powered journalism and audio intelligence",
        "stage": "Series A",
        "size": "50-150",
        "contact_hint": "tech leads",
        "location": "London, UK",
        "description": (
            "Tortoise Media uses AI for editorial intelligence, audio "
            "processing, and automated insight generation across long-form "
            "journalism and podcast productions."
        ),
    },
    {
        "company": "Quantexa",
        "website": "https://quantexa.com",
        "focus": "Decision intelligence, entity resolution, knowledge graphs + AI",
        "stage": "Series E",
        "size": "500-1000",
        "contact_hint": "engineering / AI teams",
        "location": "London, UK (remote)",
        "description": (
            "Quantexa builds decision intelligence platforms using graph AI, "
            "entity resolution, and LLMs to detect fraud, money laundering, "
            "and support enterprise risk workflows."
        ),
    },
    {
        "company": "Adzuna",
        "website": "https://adzuna.com",
        "focus": "AI-powered job search, NLP, ranking and matching",
        "stage": "Series C",
        "size": "100-200",
        "contact_hint": "ML / AI engineers",
        "location": "London, UK (remote-friendly)",
        "description": (
            "Adzuna uses AI and NLP to power job search, salary prediction, "
            "and candidate matching at scale — deploying LLMs and RAG "
            "pipelines across job data."
        ),
    },
    {
        "company": "Orbit AI",
        "website": "https://orbitai.co.uk",
        "focus": "AI automation and workflow agents for SMEs, UK",
        "stage": "Seed",
        "size": "5-20",
        "contact_hint": "founders",
        "location": "UK (remote)",
        "description": (
            "Orbit AI builds AI workflow automation agents for UK small "
            "businesses, combining LLMs with business process automation "
            "and API integrations."
        ),
    },
    {
        "company": "Kognitive",
        "website": "https://kognitive.ai",
        "focus": "AI voice and conversational automation for telecoms",
        "stage": "Seed/Series A",
        "size": "20-60",
        "contact_hint": "CTO / founders",
        "location": "UK (remote)",
        "description": (
            "Kognitive builds voice AI and conversational automation "
            "products for telecoms and business communication platforms."
        ),
    },
    {
        "company": "Foundry AI (Scotland)",
        "website": "https://foundryai.co.uk",
        "focus": "Applied AI for Scottish businesses, AI system integration",
        "stage": "Seed",
        "size": "5-20",
        "contact_hint": "founders",
        "location": "Scotland, UK",
        "description": (
            "Foundry AI delivers applied AI integration and system design "
            "for Scottish enterprises, building AI automation and "
            "decision-support tools."
        ),
    },
    {
        "company": "Competitive Capabilities International (CCI)",
        "website": "https://www.ccint.com",
        "focus": "AI-powered contact centre and voice automation, UK",
        "stage": "Growth",
        "size": "200-500",
        "contact_hint": "AI / product engineers",
        "location": "UK / Remote",
        "description": (
            "CCI builds AI-powered contact centre technology including "
            "voice automation, real-time agent assist, and conversational "
            "AI for enterprise customer service."
        ),
    },
    {
        "company": "Automated Analytics",
        "website": "https://automatedanalytics.com",
        "focus": "AI call analytics, voice intelligence, marketing attribution",
        "stage": "Series A",
        "size": "50-100",
        "contact_hint": "engineering",
        "location": "UK (remote)",
        "description": (
            "Automated Analytics uses AI and voice intelligence to analyse "
            "inbound calls, extract intent, and attribute revenue to "
            "marketing channels for businesses."
        ),
    },
    # ── Global high-signal (voice AI / agents) ───────────────────
    {
        "company": "ElevenLabs",
        "website": "https://elevenlabs.io",
        "focus": "Voice AI, text-to-speech, real-time voice cloning",
        "stage": "Series C",
        "size": "100-300",
        "contact_hint": "ML engineers / founders",
        "location": "Remote (global)",
        "description": (
            "ElevenLabs builds state-of-the-art voice AI including "
            "real-time multilingual speech synthesis, voice cloning, and "
            "conversational voice agents."
        ),
    },
    {
        "company": "Bland AI",
        "website": "https://bland.ai",
        "focus": "Autonomous phone call AI agents",
        "stage": "Seed",
        "size": "10-50",
        "contact_hint": "founders",
        "location": "Remote",
        "description": (
            "Bland AI builds AI phone agents that can autonomously conduct "
            "phone calls, handle customer interactions, and integrate with "
            "backend workflows."
        ),
    },
    {
        "company": "Gleen AI",
        "website": "https://gleen.ai",
        "focus": "Hallucination-free RAG for enterprise support",
        "stage": "Seed",
        "size": "5-20",
        "contact_hint": "founders",
        "location": "Remote",
        "description": (
            "Gleen AI builds high-accuracy RAG pipelines for customer support "
            "and enterprise knowledge bases, with a focus on factuality "
            "and retrieval quality."
        ),
    },
    {
        "company": "Retell AI",
        "website": "https://retellai.com",
        "focus": "Voice AI agent platform for customer calls",
        "stage": "Seed/Series A",
        "size": "10-50",
        "contact_hint": "CTO / founders",
        "location": "Remote",
        "description": (
            "Retell AI provides infrastructure for building, testing and "
            "deploying voice AI agents, with real-time latency, interruption "
            "handling, and telephony integrations."
        ),
    },
    {
        "company": "Vapi AI",
        "website": "https://vapi.ai",
        "focus": "Voice AI infrastructure for developers",
        "stage": "Seed",
        "size": "10-30",
        "contact_hint": "founders / engineers",
        "location": "Remote",
        "description": (
            "Vapi provides a developer platform for building real-time "
            "voice AI pipelines — transcription, LLM, TTS chained with "
            "telephony and sub-200ms latency."
        ),
    },
    {
        "company": "Synthflow AI",
        "website": "https://synthflow.ai",
        "focus": "No-code voice AI agent builder",
        "stage": "Seed",
        "size": "10-50",
        "contact_hint": "founders / product engineers",
        "location": "Remote",
        "description": (
            "Synthflow AI lets businesses build voice agents without code, "
            "focusing on outbound/inbound call automation and CRM integration."
        ),
    },
    {
        "company": "Thoughtful AI",
        "website": "https://thoughtful.ai",
        "focus": "Healthcare automation agents using AI",
        "stage": "Series A",
        "size": "50-200",
        "contact_hint": "engineering leads",
        "location": "Remote",
        "description": (
            "Thoughtful AI automates healthcare revenue cycle workflows "
            "with AI agents — verification, billing, prior auth — deployed "
            "in live healthcare environments."
        ),
    },
    {
        "company": "Ema",
        "website": "https://ema.co",
        "focus": "Universal AI employee / multi-agent platform",
        "stage": "Series A",
        "size": "50-100",
        "contact_hint": "CTO / AI engineers",
        "location": "Remote",
        "description": (
            "Ema builds a multi-agent AI platform that acts as a universal "
            "AI employee for enterprises, orchestrating agent workflows "
            "across business functions."
        ),
    },
    {
        "company": "Sierra AI",
        "website": "https://sierra.ai",
        "focus": "Conversational AI platform for customer service",
        "stage": "Series B",
        "size": "50-200",
        "contact_hint": "engineering",
        "location": "Remote",
        "description": (
            "Sierra builds conversational AI agents for customer-facing "
            "support — empathetic, multi-turn dialogue, deployed in "
            "live enterprise environments."
        ),
    },
    {
        "company": "Letta (formerly MemGPT)",
        "website": "https://letta.com",
        "focus": "Stateful LLM agent infrastructure",
        "stage": "Seed",
        "size": "10-30",
        "contact_hint": "founders / research engineers",
        "location": "Remote",
        "description": (
            "Letta (MemGPT) builds infrastructure for LLM agents with "
            "long-term memory, persistent state, and tool-use, targeting "
            "production agent deployments."
        ),
    },
    {
        "company": "AgentOps",
        "website": "https://agentops.ai",
        "focus": "Observability and eval for AI agents",
        "stage": "Seed",
        "size": "5-20",
        "contact_hint": "founders",
        "location": "Remote",
        "description": (
            "AgentOps provides monitoring, replay debugging, and evaluation "
            "tooling for AI agent workflows in production environments."
        ),
    },
    {
        "company": "Beam AI",
        "website": "https://beam.ai",
        "focus": "Agentic process automation",
        "stage": "Seed/Series A",
        "size": "20-60",
        "contact_hint": "founders / engineers",
        "location": "Remote",
        "description": (
            "Beam AI builds intelligent process automation using AI agents "
            "that can execute complex multi-step business workflows."
        ),
    },
    {
        "company": "Observe AI",
        "website": "https://observeai.com",
        "focus": "Real-time voice AI for contact centres",
        "stage": "Series C",
        "size": "200-500",
        "contact_hint": "engineering",
        "location": "Remote",
        "description": (
            "Observe AI analyses live agent calls with real-time AI assistance, "
            "intent detection, and post-call intelligence for contact centres."
        ),
    },
    {
        "company": "Lindy AI",
        "website": "https://lindy.ai",
        "focus": "AI employee / automation agent platform",
        "stage": "Seed",
        "size": "10-40",
        "contact_hint": "founders",
        "location": "Remote",
        "description": (
            "Lindy builds AI employees that automate knowledge work — "
            "email, scheduling, research, CRM updates — using LLM-powered "
            "agent workflows."
        ),
    },
    {
        "company": "Relevance AI",
        "website": "https://relevanceai.com",
        "focus": "No-code AI agent and tool builder",
        "stage": "Series A",
        "size": "30-80",
        "contact_hint": "engineering leads",
        "location": "Remote",
        "description": (
            "Relevance AI provides a platform to build, deploy, and manage "
            "AI agents and automated workflows without deep ML expertise."
        ),
    },
    {
        "company": "Hamming AI",
        "website": "https://hamming.ai",
        "focus": "Voice AI testing and red-teaming",
        "stage": "Seed",
        "size": "5-20",
        "contact_hint": "founders",
        "location": "Remote",
        "description": (
            "Hamming AI automates testing and evaluation of voice AI agents "
            "through simulated calls, edge case detection, and regression "
            "testing pipelines."
        ),
    },
    {
        "company": "Cogna",
        "website": "https://cogna.co",
        "focus": "AI-powered customer engagement agents",
        "stage": "Early",
        "size": "10-50",
        "contact_hint": "founders",
        "location": "Remote",
        "description": (
            "Cogna builds AI agents for customer engagement, combining "
            "messaging, voice, and automation to handle inbound/outbound "
            "business communication."
        ),
    },
    {
        "company": "Kore.ai",
        "website": "https://kore.ai",
        "focus": "Enterprise conversational AI and agent platform",
        "stage": "Series C",
        "size": "200-1000",
        "contact_hint": "engineering leads",
        "location": "Remote / UK offices",
        "description": (
            "Kore.ai provides enterprise-grade conversational AI and virtual "
            "assistant platforms with RAG, agent orchestration, and "
            "omnichannel deployment."
        ),
    },
    {
        "company": "Fixie AI",
        "website": "https://fixie.ai",
        "focus": "Conversational AI voice agents",
        "stage": "Seed",
        "size": "10-30",
        "contact_hint": "founders / engineers",
        "location": "Remote",
        "description": (
            "Fixie builds real-time voice AI infrastructure: "
            "ultra-low-latency speech, interruption handling, "
            "and conversational agent pipelines."
        ),
    },
    {
        "company": "Floatbot",
        "website": "https://floatbot.ai",
        "focus": "Voicebot and chatbot automation",
        "stage": "Series A",
        "size": "50-150",
        "contact_hint": "engineering",
        "location": "Remote",
        "description": (
            "Floatbot provides AI-powered voice bots and chatbots with "
            "real-time STT/TTS pipelines and backend workflow integrations "
            "for customer service."
        ),
    },
]


def _get_cv_embedding() -> list[float]:
    """Return cached CV embedding, computing if absent."""
    if CV_EMBED_FILE.exists():
        return json.loads(CV_EMBED_FILE.read_text())["values"]

    console.print("[cyan]Computing CV embedding…[/cyan]")
    cv_text = extract_cv_text(CV_PDF)
    vec = embed_text(cv_text)
    CV_EMBED_FILE.write_text(json.dumps({"values": vec}))
    return vec


def _load_all_companies() -> list[dict[str, Any]]:
    """
    Merge companies from:
      1. autonomous discovery cache (primary, potentially 400+ companies)
      2. BOOTSTRAP_LIST (always included, high-confidence targets)
    Deduplicates by slug.
    """
    from agents.autonomous_discovery import load_cached_companies
    import re

    def slug(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")

    seen: set[str] = set()
    merged: list[dict[str, Any]] = []

    # Cache first (has more detail from live scraping)
    cached = load_cached_companies()
    for c in cached:
        s = slug(c["company"])
        if s not in seen:
            seen.add(s)
            merged.append(c)

    # Bootstrap always included
    for c in BOOTSTRAP_LIST:
        s = slug(c["company"])
        if s not in seen:
            seen.add(s)
            merged.append(c)

    return merged


def score_companies(
    companies: list[dict[str, Any]] | None = None,
    topk: int = 15,
) -> list[dict[str, Any]]:
    """
    Score companies against the CV embedding and return ranked results.
    If `companies` is None, loads from autonomous discovery cache + bootstrap.
    """
    if companies is None:
        companies = _load_all_companies()

    cv_vec = _get_cv_embedding()

    console.print(f"[cyan]Scoring {len(companies)} companies…[/cyan]")
    scored = []
    for c in companies:
        profile_text = (
            f"{c['company']}. Focus: {c.get('focus', '')}. {c.get('description', '')}"
        )
        comp_vec = embed_text(profile_text)
        score = cosine_similarity(cv_vec, comp_vec)
        scored.append({**c, "match_score": round(score, 4)})

    scored.sort(key=lambda x: x["match_score"], reverse=True)
    return scored[:topk]


def print_shortlist(ranked: list[dict[str, Any]]) -> None:
    table = Table(title="CV Match Shortlist", show_lines=True)
    table.add_column("#", style="bold", width=3)
    table.add_column("Company", style="bold cyan", min_width=18)
    table.add_column("Focus", min_width=28)
    table.add_column("Location", min_width=18)
    table.add_column("Stage", width=12)
    table.add_column("Score", style="bold green", width=7)
    table.add_column("Contact", width=20)

    for i, c in enumerate(ranked, 1):
        table.add_row(
            str(i),
            c["company"],
            c["focus"],
            c.get("location", "—"),
            c.get("stage", "—"),
            str(c["match_score"]),
            c.get("contact_hint", "—"),
        )
    console.print(table)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CV → Company Matcher")
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    results = score_companies(topk=args.topk)
    print_shortlist(results)

    out = DATA_DIR / "shortlist.json"
    out.write_text(json.dumps(results, indent=2))
    console.print(f"\n[green]Saved to {out}[/green]")
