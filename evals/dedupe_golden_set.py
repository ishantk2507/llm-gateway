"""One-shot hygiene: collapse duplicate prompts in golden_set.jsonl.

Keeps the FIRST occurrence; fails loudly if duplicates carry conflicting
labels — that's not duplication, that's contradiction, and no eval number
is trustworthy until it's resolved by hand against the rubric.
"""

import json
import sys
from pathlib import Path

PATH = Path(__file__).resolve().parent / "golden_set.jsonl"


def main() -> None:
    rows = [
        json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    seen: dict[str, dict] = {}
    kept: list[dict] = []
    conflicts: list[tuple[int, str, str, str]] = []
    for lineno, row in enumerate(rows, start=1):
        prior = seen.get(row["prompt"])
        if prior is None:
            seen[row["prompt"]] = row
            kept.append(row)
        elif prior["expected_tier"] != row["expected_tier"]:
            conflicts.append(
                (lineno, row["prompt"][:60], prior["expected_tier"], row["expected_tier"])
            )

    if conflicts:
        print(
            f"{len(conflicts)} CONFLICTING labels among duplicates — resolve by the rubric:",
            file=sys.stderr,
        )
        for lineno, prompt, a, b in conflicts:
            print(f"  line {lineno}: {prompt!r}: {a} vs {b}", file=sys.stderr)
        raise SystemExit(1)

    PATH.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n", encoding="utf-8"
    )
    print(f"kept {len(kept)} unique prompts (removed {len(rows) - len(kept)} duplicates)")


if __name__ == "__main__":
    main()
