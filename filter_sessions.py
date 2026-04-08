"""Extract turns for a specific language from sessions.jsonl.

Usage:
    python filter_sessions.py sessions.jsonl japanese
    python filter_sessions.py sessions.jsonl korean -o korean_only.jsonl
    python filter_sessions.py sessions.jsonl thai --user-id user_004
"""

import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser(description="Filter sessions.jsonl by language")
    p.add_argument("input", help="Path to sessions.jsonl")
    p.add_argument("language", help="Language to extract (japanese, korean, thai)")
    p.add_argument("-o", "--output", help="Output file (default: stdout)")
    p.add_argument("--user-id", help="Further filter by user_id (e.g. user_000)")
    args = p.parse_args()

    out = open(args.output, "w", encoding="utf-8") if args.output else sys.stdout
    count = 0

    with open(args.input, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry["language"] != args.language:
                continue
            if args.user_id and entry["user_id"] != args.user_id:
                continue
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            count += 1

    if args.output:
        out.close()
        print(f"Wrote {count} turns to {args.output}", file=sys.stderr)
    else:
        print(f"\n# {count} turns extracted", file=sys.stderr)


if __name__ == "__main__":
    main()
