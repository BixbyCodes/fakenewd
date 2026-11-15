"""Run the first EAFND ingestion demonstration."""

import json
from src.ingestion import create_article_record


def main() -> None:
    sample = create_article_record(
        headline="  Example headline: a claim to investigate  ",
        body=(
            "This is demonstration text, not a real news report. "
            "The next project stages will retrieve evidence before making a decision."
        ),
        source_url="https://example.org/sample-story",
        language="en",
        publisher="Example Publisher",
        metadata={"demo": True, "stage": "ingestion"},
    )
    print(json.dumps(sample.to_dict(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
