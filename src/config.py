"""Central paths and settings.

All paths are resolved from the repository root so that modules work no
matter which directory the app or a script is launched from.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
CLASSICAL_MODELS_DIR = MODELS_DIR / "classical"
RESULTS_DIR = BASE_DIR / "results"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Local FEVER Wikipedia index (large, never committed). Can be relocated
# with the EAFND_WIKI_INDEX environment variable.
WIKI_INDEX_PATH = Path(
    os.getenv("EAFND_WIKI_INDEX", str(PROCESSED_DIR / "wiki_pages.sqlite"))
)

# LLM response caches (small, local, ignored by git).
LLM_FEATURE_CACHE = PROCESSED_DIR / "llm_features.sqlite"
EVENT_PARSE_CACHE = PROCESSED_DIR / "event_parse_cache.sqlite"
EVIDENCE_CACHE = PROCESSED_DIR / "evidence_verification_cache.sqlite"

DEFAULT_LLM_MODEL = "openai/gpt-oss-20b"

# Input limits for the interactive application.
MIN_INPUT_CHARS = 20
MIN_INPUT_WORDS = 4
MAX_INPUT_CHARS = 20000
MAX_HEADLINE_CHARS = 400

# Text sent to the LLM is truncated to bound latency and token cost.
LLM_TEXT_CHAR_LIMIT = 6000

SQLITE_TIMEOUT_SECONDS = 10


def llm_model():
    """Model name, overridable with GROQ_MODEL."""
    return os.getenv("GROQ_MODEL", DEFAULT_LLM_MODEL)
