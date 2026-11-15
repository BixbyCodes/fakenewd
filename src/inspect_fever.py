"""Inspect a local FEVER JSONL file without loading it all into memory."""

import argparse
import json
from src.fever_loader import inspect_fever, iter_fever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to FEVER train/dev JSONL file")
    parser.add_argument("--sample", type=int, default=1000)
    args = parser.parse_args()

    print("Dataset summary:")
    print(json.dumps(inspect_fever(args.path, args.sample), indent=2))

    print("\nFirst 3 normalized records:")
    for row in iter_fever(args.path, limit=3):
        print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
