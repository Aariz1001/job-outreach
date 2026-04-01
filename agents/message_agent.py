"""
Message Generation Agent — produces a highly personalised, short outreach
message for a specific person at a specific company, grounded in research.

The humanize-writing skill (installed at .agents/skills/humanize-writing/SKILL.md)
is loaded at runtime and injected as the system prompt so the LLM acts as a
human-writing editor BEFORE it generates the message.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from openai import OpenAI

from utils.config import GENERATION_MODEL, OPENROUTER_API_KEY

_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

_SKILL_PATH = Path(__file__).parent.parent / ".agents" / "skills" / "humanize-writing" / "SKILL.md"


@lru_cache(maxsize=1)
def _load_humanize_skill() -> str:
    """Load the humanize-writing SKILL.md as a system prompt (cached after first read)."""
    if _SKILL_PATH.exists():
        # Strip the YAML frontmatter (--- ... ---) — models don't need it
        raw = _SKILL_PATH.read_text(encoding="utf-8")
        if raw.startswith("---"):
            end = raw.find("---", 3)
            if end != -1:
                raw = raw[end + 3:].lstrip()
        return raw.strip()
    # Fallback minimal rules if file missing
    return (
        "You are an expert human-writing editor. Remove all AI writing patterns: "
        "no significance inflation, no AI vocabulary (delve/leverage/landscape), "
        "no hedging, no filler openers, no sycophantic tone, vary sentence rhythm."
    )


# Focus: agents, agent workflows, RAG pipelines — NOT voice AI first
CV_HIGHLIGHTS = """
Mohammad Aariz Waqas — AI Engineer (University of Glasgow, Computing Science)

Core focus — AI agents and agent workflows:
- AgentForge: autonomous CLI coding agent framework with planning, reflection, tool orchestration
- Production RAG pipelines: retrieval accuracy 50% → 85%, 3x faster lookups
- Multi-agent systems using LangChain, CrewAI, LlamaIndex, OpenRouter
- WhatsApp AI agent on GCP/Twilio/Gemini — live business workflow automation
- Multi-tenant AI platforms supporting real client operations
- Reduced AI costs 40-50% through prompt optimisation + workflow redesign
- FlexiDiet: production Flutter/Firebase AI app on Google Play
- Tech: Python, GCP, Firebase, LangChain, CrewAI, LlamaIndex, OpenRouter, Gemini
"""


def generate_message(
    person_name: str,
    person_role: str,
    company: str,
    research: dict,
    channel: str = "linkedin",  # "linkedin" | "email"
) -> str:
    """
    Generate a personalised outreach message.

    Args:
        person_name:  First name (or full name) of the target.
        person_role:  Their role (e.g. "Founder", "CTO").
        company:      Company name.
        research:     Dict from research_agent.research_company().
        channel:      "linkedin" (≤300 chars body) or "email" (longer).
    """
    one_liner    = research.get("one_liner", "")
    fit_reason   = research.get("fit_reason", "")
    talking_pts  = research.get("talking_points", [])
    tech_signals = research.get("tech_signals", [])

    talking_str = "\n".join(f"- {t}" for t in talking_pts[:2])
    tech_str    = ", ".join(tech_signals[:6])

    if channel == "linkedin":
        format_instruction = (
            "Write a LinkedIn connection request note (≤300 characters) "
            "OR a short LinkedIn DM (≤500 characters). Be direct, human, no fluff. "
            "Do NOT use bullet points or numbered lists."
        )
    else:
        format_instruction = (
            "Write a cold email. Subject line first, exactly like: Subject: <your subject here>\n"
            "Body: 3-4 short paragraphs, no fluff. Include a clear CTA at the end.\n"
            "CRITICAL: Plain text only. NO markdown formatting whatsoever — no **bold**, "
            "no *italic*, no # headers, no bullet points, no backticks."
        )

    prompt = f"""You are writing outreach for Mohammad Aariz, an AI engineer looking for a role at an AI startup.

{format_instruction}

Target:
- Name: {person_name}
- Role: {person_role} at {company}
- Company one-liner: {one_liner}
- Their tech signals: {tech_str}

Specific things to reference about {company}:
{talking_str if talking_str else '- ' + one_liner}

Why Aariz fits:
{fit_reason}

Aariz's key highlights (pick 1-2 most relevant to this company):
{CV_HIGHLIGHTS}

WRITING RULES — these are non-negotiable (based on humanize-writing guidelines):
- Sound like a real person wrote this, not a language model. Vary sentence length: mix short punchy lines with longer ones.
- Lead with something specific about {company} — not yourself. Show you actually know the product.
- Be direct. No hedging ("it's worth noting", "I thought I'd reach out").
- No filler openers: never "I hope this finds you well", "I came across your profile", "I wanted to reach out".
- No significance inflation: avoid "pivotal", "groundbreaking", "transformative", "revolutionise", "seamless", "innovative", "game-changer".
- No AI vocabulary: avoid "delve", "leverage", "navigate (metaphorical)", "landscape", "paradigm", "synergy", "ecosystem (non-tech)", "harness", "realm", "embark".
- No transitions like "Moreover", "Furthermore", "Additionally", "That being said".
- No -ing padding: avoid tacking on "highlighting that...", "showcasing...", "underscoring..." phrases.
- No generic sign-offs: avoid "excited to connect", "looking forward to hearing from you", "would love to explore opportunities".
- End with one low-friction, specific ask: a 15-minute call, a demo link, a direct question. Not "open to opportunities".
- Max 2 sentences about Aariz's own work. The message is primarily about THEM, not Aariz.
- ONLY reference technologies Aariz actually has experience with. Never mention languages or tools he hasn't used (e.g. COBOL, Fortran, Java, Rust, C++, MATLAB, Scala, etc.).
- Plain text only — no **bold**, no *italic*, no markdown of any kind.

Write the message now:"""

    humanize_system = _load_humanize_skill()

    resp = _client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": humanize_system},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.7,
        extra_body={"reasoning": {"enabled": True}},
    )

    raw = resp.choices[0].message.content.strip()
    # Strip any markdown bold/italic the model may have added despite instructions
    import re as _re
    raw = _re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", raw)   # **x** / *x* / ***x***
    raw = _re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", raw)       # __x__ / _x_
    return raw


def generate_followup(
    person_name: str,
    company: str,
    original_message: str,
    days_since: int = 4,
) -> str:
    """Generate a brief follow-up message."""
    prompt = f"""Write a short follow-up message from Mohammad Aariz to {person_name} at {company}.

Context:
- Original message was sent {days_since} days ago (no reply yet).
- Original message: "{original_message[:300]}..."

Rules:
- 2-3 sentences max.
- Reference something specific — a demo, a repo, a build — not just "following up".
- One concrete offer (demo, repo link, quick call).
- Human tone: no "just circling back", no "I wanted to follow up", no "touching base".
- Short. Direct. Specific.

Write the follow-up now:"""

    resp = _client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        extra_body={"reasoning": {"enabled": True}},
    )

    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    from rich.console import Console
    console = Console()

    # Quick smoke test
    sample_research = {
        "one_liner": "Vapi builds real-time voice AI infrastructure for developers.",
        "tech_signals": ["voice AI", "real-time", "telephony", "LLM", "TTS", "STT"],
        "fit_reason": (
            "Aariz built VoxGuard, a real-time intent classification system at <200ms "
            "integrated with Gemini Live and OpenAI Realtime — directly aligned with "
            "Vapi's core infrastructure."
        ),
        "talking_points": [
            "Vapi's interruption-handling architecture",
            "sub-200ms latency pipelines for live voice",
        ],
    }

    msg = generate_message("Jordan", "Founder", "Vapi AI", sample_research, channel="linkedin")
    console.rule("[bold]LinkedIn Message[/bold]")
    console.print(msg)

    email = generate_message("Jordan", "Founder", "Vapi AI", sample_research, channel="email")
    console.rule("[bold]Email[/bold]")
    console.print(email)
