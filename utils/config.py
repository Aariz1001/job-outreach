"""Centralised config — reads from .env in project root."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (works regardless of cwd)
load_dotenv(Path(__file__).parent.parent / ".env")

OPENROUTER_API_KEY: str = os.environ["OPENROUTER_API_KEY"]
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
YOUR_EMAIL: str = os.environ.get("YOUR_EMAIL", "")

EMBEDDING_MODEL = "models/gemini-embedding-2-preview"
GENERATION_MODEL = "x-ai/grok-4.20"   # via OpenRouter

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TRACKER_FILE    = DATA_DIR / "tracker.json"
LEADS_FILE      = DATA_DIR / "leads.json"
CV_EMBED_FILE   = DATA_DIR / "cv_embedding.json"
COMPANY_CACHE   = DATA_DIR / "company_cache.json"
CRAWL_STATE     = DATA_DIR / "crawl_state.json"
