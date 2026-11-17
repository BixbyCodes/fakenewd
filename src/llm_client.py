"""Shared Groq client, JSON parsing and safe error reporting.

The API key is read only from the environment (or a local .env file). It is
never logged, returned, or included in user-facing error messages.
"""

import json
import logging
import os
from contextlib import closing
import sqlite3

from dotenv import load_dotenv

from src.config import BASE_DIR, SQLITE_TIMEOUT_SECONDS

logger = logging.getLogger("eafnd.llm")

load_dotenv(BASE_DIR / ".env")

REQUEST_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "45"))
MAX_RETRIES = 1


class LLMUnavailable(RuntimeError):
    """Raised when the LLM cannot be used. The message is safe to display."""


def api_key_available():
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def get_client():
    """Create a Groq client with a timeout, or raise LLMUnavailable."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise LLMUnavailable(
            "GROQ_API_KEY is not configured, so LLM-based components are disabled."
        )
    try:
        from groq import Groq
    except ImportError as exc:  # pragma: no cover - dependency problem
        raise LLMUnavailable("The groq package is not installed.") from exc

    return Groq(api_key=key, timeout=REQUEST_TIMEOUT_SECONDS, max_retries=MAX_RETRIES)


def describe_error(exc):
    """Map an exception to a short message that is safe to show to users."""
    if isinstance(exc, LLMUnavailable):
        return str(exc)

    name = type(exc).__name__
    messages = {
        "AuthenticationError": "The LLM service rejected the API key.",
        "PermissionDeniedError": "The LLM service denied access for this API key.",
        "RateLimitError": "The LLM service rate limit was reached. Try again shortly.",
        "APITimeoutError": "The LLM service did not respond in time.",
        "APIConnectionError": "Could not connect to the LLM service.",
        "InternalServerError": "The LLM service returned a server error.",
        "BadRequestError": "The LLM service rejected the request.",
        "NotFoundError": "The configured LLM model was not found.",
    }
    if name in messages:
        return messages[name]
    if isinstance(exc, (ValueError, json.JSONDecodeError)):
        return "The LLM returned a response that could not be validated."
    return "The LLM component failed unexpectedly."


def parse_json_object(content):
    """Decode the first JSON object in an LLM response."""
    if not content or not content.strip():
        raise ValueError("The LLM returned an empty response.")
    start = content.find("{")
    if start < 0:
        raise ValueError("The LLM response contains no JSON object.")
    data, _ = json.JSONDecoder().raw_decode(content[start:])
    if not isinstance(data, dict):
        raise ValueError("The LLM response is not a JSON object.")
    return data


def sqlite_connect(path):
    """Open a SQLite connection with a lock timeout. Use with closing()."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path), timeout=SQLITE_TIMEOUT_SECONDS)


class JsonCache:
    """Tiny key/value cache for validated LLM JSON results."""

    def __init__(self, path, table):
        self.path = path
        self.table = table

    def _ensure(self, connection):
        connection.execute(
            "CREATE TABLE IF NOT EXISTS {} ("
            "cache_key TEXT PRIMARY KEY, result_json TEXT NOT NULL)".format(self.table)
        )

    def get(self, key):
        try:
            with closing(sqlite_connect(self.path)) as connection:
                self._ensure(connection)
                row = connection.execute(
                    "SELECT result_json FROM {} WHERE cache_key = ?".format(self.table),
                    (key,),
                ).fetchone()
            return json.loads(row[0]) if row else None
        except (sqlite3.Error, OSError, ValueError) as exc:
            logger.warning("Cache read failed (%s): %s", self.table, type(exc).__name__)
            return None

    def set(self, key, value):
        try:
            with closing(sqlite_connect(self.path)) as connection:
                self._ensure(connection)
                connection.execute(
                    "INSERT OR REPLACE INTO {} VALUES (?, ?)".format(self.table),
                    (key, json.dumps(value, ensure_ascii=False)),
                )
                connection.commit()
        except (sqlite3.Error, OSError) as exc:
            # A cache failure must never break the analysis.
            logger.warning("Cache write failed (%s): %s", self.table, type(exc).__name__)
