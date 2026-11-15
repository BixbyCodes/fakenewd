"""Input validation and normalization for EAFND."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any
from urllib.parse import urlparse
import json
import re
import unicodedata


def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace while preserving punctuation."""
    if not isinstance(value, str):
        raise TypeError("Text fields must be strings.")

    value = unicodedata.normalize("NFKC", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[\t\f\v ]+", " ", value)
    value = re.sub(r" *\n *", "\n", value)
    value = re.sub(r"\n{3,}", "\n\n", value)

    return value.strip()


def validate_source_url(url: str | None) -> str | None:
    """Validate an optional HTTP or HTTPS source URL."""
    if url is None or not url.strip():
        return None

    candidate = url.strip()
    parsed = urlparse(candidate)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("source_url must be a valid http:// or https:// URL.")

    return candidate


@dataclass
class ArticleRecord:
    headline: str
    body: str
    source_url: str | None = None
    language: str = "en"
    publisher: str | None = None
    published_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    record_id: str = ""
    ingested_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_article_record(
    headline: str,
    body: str,
    source_url: str | None = None,
    language: str = "en",
    publisher: str | None = None,
    published_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArticleRecord:

    clean_headline = normalize_text(headline)
    clean_body = normalize_text(body)

    if not clean_headline:
        raise ValueError("headline cannot be empty.")

    if not clean_body:
        raise ValueError("body cannot be empty.")

    clean_url = validate_source_url(source_url)
    clean_language = normalize_text(language).lower()

    if not clean_language:
        raise ValueError("language cannot be empty.")

    identity_payload = {
        "headline": clean_headline,
        "body": clean_body,
        "source_url": clean_url,
        "language": clean_language,
    }

    canonical = json.dumps(
        identity_payload,
        sort_keys=True,
        ensure_ascii=False
    )

    record_id = sha256(canonical.encode("utf-8")).hexdigest()

    return ArticleRecord(
        headline=clean_headline,
        body=clean_body,
        source_url=clean_url,
        language=clean_language,
        publisher=normalize_text(publisher) if publisher else None,
        published_at=published_at,
        metadata=metadata or {},
        record_id=record_id,
        ingested_at=datetime.now(timezone.utc).isoformat(),
    )
