# Job Outreach

An autonomous AI pipeline that crawls job boards and company directories, scores opportunities against your CV, researches each target, surgically tailors your CV as a DOCX/PDF, and writes humanised cold emails — all from a single CLI command.

Built with Python, LangChain, CrewAI, OpenRouter (Grok 4), and Gemini text embeddings.

---

## What it does

| Step | Command | What happens |
|------|---------|--------------|
| **Discover** | `discover` | Crawls YC, Wellfound, HN Who's Hiring, RemoteOK, Sifted, and more |
| **Shortlist** | `shortlist` | Embeds your CV and every company description, ranks by cosine similarity |
| **Research** | `research` | Scrapes the company website and extracts fit signals and talking points |
| **Outreach** | `outreach` | Tailors your CV for the company, drafts a humanised email, generates a Gmail compose link |
| **Draft** | `draft` | Standalone email/LinkedIn draft for a named contact |
| **Follow-up** | `followups` | Surfaces contacts due a follow-up after N days |
| **Status** | `status` | Shows full outreach tracker |

All LLM calls go through OpenRouter. CV embedding uses the Gemini embedding API. The pipeline runs locally — nothing is sent to a third-party service other than the API calls.

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/job-outreach.git
cd job-outreach
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add your keys

```bash
cp .env.example .env
# edit .env — add your OpenRouter key, Gemini key, and email
```

### 3. Add your CV

Place your CV at the project root:
- `YourName_AI_Engineer.docx` — the DOCX the specialiser will edit
- `YourName_AI_Engineer.pdf` — used by the CV matcher for embeddings

Then update the two references in `agents/cv_matcher.py` and `agents/cv_specialiser.py` to point to your filenames.

### 4. Discover and run

```bash
# Crawl public sources and build a company cache
python main.py discover

# Score everything against your CV
python main.py shortlist --topk 15

# Full outreach for a specific company
python main.py outreach "Company Name" https://company.com --person "Founder Name" --person-role "Founder" --to founder@company.com
```

---

## CLI reference

```
python main.py discover        [--source all|yc|hn_hiring|uk|wellfound] [--fresh] [--limit N]
python main.py shortlist       [--topk N] [--source FILTER]
python main.py cache
python main.py research        COMPANY WEBSITE [--focus TEXT]
python main.py outreach        COMPANY WEBSITE [--person NAME] [--person-role ROLE] [--to EMAIL] [--skip-cv]
python main.py draft           PERSON_NAME PERSON_ROLE COMPANY WEBSITE [--channel linkedin|email|both] [--to EMAIL]
python main.py pipeline        COMPANY WEBSITE
python main.py add             NAME ROLE COMPANY CHANNEL [--email ADDR] [--message TEXT]
python main.py followups       [--days N] [--mark]
python main.py status
```

---

## Project layout

```
.
├── agents/
│   ├── autonomous_discovery.py   # multi-source company crawler
│   ├── cv_matcher.py             # cosine-similarity CV ↔ company scorer
│   ├── cv_specialiser.py         # DOCX surgical tailoring + PDF export
│   ├── message_agent.py          # humanised email / LinkedIn drafter
│   ├── research_agent.py         # company scraper and fit analyser
│   ├── tracker.py                # SQLite-backed outreach log
│   └── followup_agent.py         # follow-up reminder engine
├── utils/
│   ├── config.py                 # central config (reads .env)
│   ├── embeddings.py             # Gemini embedding wrapper
│   ├── gmail.py                  # Gmail compose URL builder
│   └── pdf_utils.py              # LibreOffice DOCX→PDF helper
├── data/                         # gitignored runtime data
│   └── .gitkeep
├── specialised_cvs/              # gitignored output directory
├── .env.example                  # copy to .env and fill in keys
├── main.py                       # Click CLI entry point
└── requirements.txt
```

---

## Requirements

- Python 3.11+
- [OpenRouter](https://openrouter.ai) API key (LLM calls — Grok 4 by default, swap in `utils/config.py`)
- [Google AI Studio](https://aistudio.google.com) API key (text embeddings)
- LibreOffice — for DOCX → PDF conversion (`brew install --cask libreoffice` on macOS)

---

## Security

- API keys are read from `.env` only. No keys are hardcoded.
- `.env` is in `.gitignore`. Your CV, outreach logs, and all generated files are also excluded.
- This repo contains no personal data. Add your own CV and keys locally.
