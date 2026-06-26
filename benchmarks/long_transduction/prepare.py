# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare long_transduction benchmark dataset.

Five prompt variants per difficulty tier:

  Arithmetic chain summing (3 variants per max_operands × N_SAMPLES):
    - "unnumbered_streaming_sum" : plain expressions in order
    - "streaming_sum"            : "[N]<expr>" in order
    - "shuffled_streaming_sum"   : "[N]<expr>" shuffled in input

  Per-line UUID sort (2 variants per uuids_per_line × N_SAMPLES):
    - "streaming_uuid_sort"          : "[N](u),(u),..." in order; model sorts
                                       UUIDs within each line by hex order.
    - "shuffled_streaming_uuid_sort" : same but line order is shuffled in input.

Each row carries a `type` field so the resource server selects the right
parser. Sum variants share an `expressions` payload; uuid_sort variants share
a `uuid_lines` payload.

Usage:
    python prepare.py
    python prepare.py --force   # regenerate even if output already exists
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "long_transduction.jsonl"

# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_UNNUMBERED_STREAMING_SUM = """You are a calculator. You will be given a long sequence of simple arithmetic expressions to evaluate.
Your task is to output each expression and the result of evaluating the expression.

See the example below.

```
Input:

5+6
2+4-1
4+8-3+2

Output:

5+6=11
2+4-1=5
4+8-3+2=11
```

The real sequence will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output an answer to all expressions.
Do not add whitespace.
Do not change the format.

Here is the real sequence.

Input:

{input}"""

PROMPT_STREAMING_SUM = """You are a calculator. You will be given a long sequence of simple arithmetic expressions to evaluate.
Each expression is preceded by a numeric index in brackets like [1], [2], [3], ...
Your task is to output each expression with its index and the result of evaluating the expression.

See the example below.

```
Input:

[1]5+6
[2]2+4-1
[3]4+8-3+2

Output:

[1]5+6=11
[2]2+4-1=5
[3]4+8-3+2=11
```

The real sequence will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output an answer to all expressions.
Do not add whitespace.
Do not change the format.

Here is the real sequence.

Input:

{input}"""


PROMPT_SHUFFLED_STREAMING_SUM = """You are a calculator. You will be given a long sequence of simple arithmetic expressions to evaluate.
Each expression is preceded by a numeric index in brackets like [1], [2], [3], ...
The input expressions are SHUFFLED — they appear in arbitrary order, not in numerical order.
Your task is to output each expression with its index and the result, IN ASCENDING ORDER OF INDEX, starting at [1].

See the example below.

```
Input:

[2]2+4-1
[3]4+8-3+2
[1]5+6

Output:

[1]5+6=11
[2]2+4-1=5
[3]4+8-3+2=11
```


The real sequence will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output an answer to all expressions.
Do not add whitespace.
Do not change the format.

Here is the real sequence.

Input:

{input}"""

PROMPT_STREAMING_UUID_SORT = """You will be given a long list of numbered lines. Each line has the form:

[N]hex,hex,hex,...

Where each token is an 8-character lowercase hex string (for example: a1b2c3d4).
Your task is to output each line with its index and the same hex tokens sorted in ASCENDING LEXICOGRAPHIC ORDER (compare them as plain strings).

See the example below.

```
Input:

[1]c0a8e1d2,a1b2c3d4,b1c2d3e4
[2]f0e1d2c3,01234567

Output:

[1]a1b2c3d4,b1c2d3e4,c0a8e1d2
[2]01234567,f0e1d2c3
```

The real sequence will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output every line.
Do not add whitespace.
Do not change the format.

Here is the real sequence.

Input:

{input}"""


PROMPT_SHUFFLED_STREAMING_UUID_SORT = """You will be given a long list of numbered lines. Each line has the form:

[N]hex,hex,hex,...

Where each token is an 8-character lowercase hex string (for example: a1b2c3d4).
The input lines are SHUFFLED — they appear in arbitrary order, not in numerical order.
Your task is to output each line with its index and the same hex tokens sorted in ASCENDING LEXICOGRAPHIC ORDER (compare them as plain strings), AND emit the lines themselves in ASCENDING ORDER OF [N] starting at [1].


See the example below.

```
Input:

[2]f0e1d2c3,01234567
[1]c0a8e1d2,a1b2c3d4,b1c2d3e4

Output:

[1]a1b2c3d4,b1c2d3e4,c0a8e1d2
[2]01234567,f0e1d2c3
```


The real sequence will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output every line.
Do not add whitespace.
Do not change the format.

Here is the real sequence.

Input:

{input}"""

PROMPT_UNNUMBERED_UUID_SORT = """You will be given a long list of lines. Each line has the form:

hex,hex,hex,...

Where each token is an 8-character lowercase hex string (for example: a1b2c3d4).
Your task is to output each line with the same hex tokens sorted in ASCENDING LEXICOGRAPHIC ORDER (compare them as plain strings).

See the example below:

```
Input:

c0a8e1d2,a1b2c3d4,b1c2d3e4
f0e1d2c3,01234567

Output:

a1b2c3d4,b1c2d3e4,c0a8e1d2
01234567,f0e1d2c3
```

The real sequence will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output every line.
Do not add whitespace.
Do not change the format.

Here is the real sequence.

Input:

{input}"""

PROMPT_CSV_PERMUTATION = """You will be given a CSV table and reordering instructions.

The CSV has row headers [R0], [R1], [R2], ... and column headers [C0], [C1], [C2], ...
The "New row order" lists original row headers in the order they should appear in the output.
The "New col order" lists original column headers in the order they should appear in the output.

See the example below.

```
Input:

,[C0],[C1],[C2]
[R0],a,b,c
[R1],d,e,f
[R2],g,h,i

Reordering Instructions:
New row order: [R2],[R0],[R1]
New col order: [C1],[C0],[C2]

Output:

,[C1],[C0],[C2]
[R2],h,g,i
[R0],b,a,c
[R1],e,d,f
```

The real CSV will be much longer than the example.
Do not think.
Do not ask any questions.
Do not stop until you output every line.
Do not add whitespace.
Do not change the format.

Here is the real CSV.

Input:

{input}

Reordering Instructions:
New row order: {row_order}
New col order: {col_order}"""


PROMPT_TEMPLATES = {
    "streaming_sum":                PROMPT_STREAMING_SUM,
    "shuffled_streaming_sum":       PROMPT_SHUFFLED_STREAMING_SUM,
    "unnumbered_streaming_sum":     PROMPT_UNNUMBERED_STREAMING_SUM,
    "streaming_uuid_sort":          PROMPT_STREAMING_UUID_SORT,
    "shuffled_streaming_uuid_sort": PROMPT_SHUFFLED_STREAMING_UUID_SORT,
    "unnumbered_uuid_sort":         PROMPT_UNNUMBERED_UUID_SORT,
    "csv_permutation":              PROMPT_CSV_PERMUTATION,
}

SUM_TYPES = ["unnumbered_streaming_sum",
             "streaming_sum", "shuffled_streaming_sum"]
UUID_SORT_TYPES = ["unnumbered_uuid_sort",
                   "streaming_uuid_sort", "shuffled_streaming_uuid_sort"]
NUMBERED_TYPES = {
    "streaming_sum",
    "shuffled_streaming_sum",
    "streaming_uuid_sort",
    "shuffled_streaming_uuid_sort",
}
PERM_FRACTIONS = [0.2, 0.4, 0.8, 1.0]

TARGET_TOKENS_LIST = [2048, 4096, 8192, 16384, 32768, 65536]
N_SAMPLES = 5
MAX_OPERANDS_RANGE = [2, 4, 8, 16]


def _get_encoder():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


# ─────────────────────────────────────────────────────────────────────────────
# Arithmetic chain generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_expression(max_operands: int) -> tuple[str, int]:
    n = random.randint(2, max_operands)
    operands = [random.randint(0, 9) for _ in range(n)]
    operators = [random.choice(["+", "-"]) for _ in range(n - 1)]
    parts = [str(operands[0])]
    for op, operand in zip(operators, operands[1:]):
        parts.append(op + str(operand))
    expr = "".join(parts)
    result = operands[0]
    for op, operand in zip(operators, operands[1:]):
        result = result + operand if op == "+" else result - operand
    return expr, result


def _generate_expressions(max_operands: int, enc, target_tokens: int) -> tuple[list[dict], int]:
    """Generate as many expressions as fit a target_tokens-budget prompt.

    Budget is computed against the longest sum-prompt header (shuffled).
    """
    longest_header = max(
        (PROMPT_TEMPLATES[t].split("{input}")[0] for t in SUM_TYPES),
        key=len,
    )
    overhead = len(enc.encode(longest_header))
    budget = target_tokens - overhead

    expressions: list[dict] = []
    used_tokens = 0
    while True:
        expr, answer = _generate_expression(max_operands)
        line_tokens = len(enc.encode(f"[{len(expressions) + 1}]{expr}\n"))
        if used_tokens + line_tokens > budget:
            break
        expressions.append({"expr": expr, "answer": answer})
        used_tokens += line_tokens
    return expressions, overhead + used_tokens


def _build_sum_sample(
    expressions: list[dict],
    max_operands: int,
    sample_type: str,
    approx_prompt_tokens: int,
    rng: random.Random,
) -> dict:
    """Render one arithmetic-chain row."""
    numbered = list(enumerate(expressions, start=1))

    if sample_type == "unnumbered_streaming_sum":
        input_text = "\n".join(e["expr"] for _, e in numbered)
        expected_output = "\n".join(
            f"{e['expr']}={e['answer']}" for _, e in numbered
        )
    elif sample_type in {"streaming_sum", "shuffled_streaming_sum"}:
        if sample_type == "shuffled_streaming_sum":
            input_order = numbered.copy()
            rng.shuffle(input_order)
        else:
            input_order = numbered
        input_text = "\n".join(f"[{n}]{e['expr']}" for n, e in input_order)
        expected_output = "\n".join(
            f"[{n}]{e['expr']}={e['answer']}" for n, e in numbered
        )
    else:
        raise ValueError(f"unknown sum sample_type: {sample_type}")

    prompt = PROMPT_TEMPLATES[sample_type].replace("{input}", input_text)
    return {
        "type": sample_type,
        "question": prompt,
        "expected_output": expected_output,
        "expressions": expressions,
        "n_expressions": len(expressions),
        "max_operands": max_operands,
        "approx_prompt_tokens": approx_prompt_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# UUID-sort generation
# ─────────────────────────────────────────────────────────────────────────────

def _new_uuid(rng: random.Random) -> str:
    """Deterministic 8-char lowercase hex token drawn from the provided RNG.

    We use just the first segment of a uuid4 (32 bits, 8 hex chars) so the
    tokens are short enough to fit many lines per 100K-token prompt while
    still being unique enough across the dataset (~1 in 4B collision odds).
    """
    return f"{rng.getrandbits(32):08x}"


def _generate_uuid_lines(
    uuids_per_line: int,
    enc,
    seed_key: tuple,
    target_tokens: int,
) -> tuple[list[list[str]], int]:
    """Generate UUID lines that fit a target_tokens-budget prompt.

    Each inner list is the canonical (input-presentation) UUIDs for that
    line. Token budget is computed against the longer (shuffled) header.
    """
    longest_header = max(
        (PROMPT_TEMPLATES[t].split("{input}")[0] for t in UUID_SORT_TYPES),
        key=len,
    )
    overhead = len(enc.encode(longest_header))
    budget = target_tokens - overhead

    rng = random.Random(seed_key)
    lines: list[list[str]] = []
    used_tokens = 0
    while True:
        idx = len(lines) + 1
        uuids = [_new_uuid(rng) for _ in range(uuids_per_line)]
        line_str = f"[{idx}]" + ",".join(uuids) + "\n"
        line_tokens = len(enc.encode(line_str))
        if used_tokens + line_tokens > budget:
            break
        lines.append(uuids)
        used_tokens += line_tokens
    return lines, overhead + used_tokens


def _build_uuid_sample(
    uuid_lines: list[list[str]],
    uuids_per_line: int,
    sample_type: str,
    approx_prompt_tokens: int,
    rng: random.Random,
) -> dict:
    """Render one UUID-sort row.

    `uuid_lines` is the canonical (input-presentation) per-line UUIDs. The
    UUIDs are deliberately NOT pre-sorted — that's the model's task. For the
    shuffled variant, the LINE order in the input is shuffled but the model
    must still emit ascending [N].
    """
    numbered = list(enumerate(uuid_lines, start=1))

    if sample_type == "unnumbered_uuid_sort":
        input_text = "\n".join(",".join(uuids) for _, uuids in numbered)
        expected_output = "\n".join(",".join(sorted(uuids))
                                    for _, uuids in numbered)
        prompt = PROMPT_TEMPLATES[sample_type].replace("{input}", input_text)
        return {
            "type": sample_type,
            "question": prompt,
            "expected_output": expected_output,
            "uuid_lines": uuid_lines,
            "expressions": [{"expr": list(uuids), "answer": sorted(uuids)} for uuids in uuid_lines],
            "n_lines": len(uuid_lines),
            "uuids_per_line": uuids_per_line,
            "approx_prompt_tokens": approx_prompt_tokens,
        }

    if sample_type == "streaming_uuid_sort":
        input_order = numbered
    elif sample_type == "shuffled_streaming_uuid_sort":
        input_order = numbered.copy()
        rng.shuffle(input_order)
    else:
        raise ValueError(f"unknown uuid sample_type: {sample_type}")

    def _fmt_line(n: int, uuids: list[str]) -> str:
        return f"[{n}]" + ",".join(uuids)

    input_text = "\n".join(_fmt_line(n, uuids) for n, uuids in input_order)
    expected_output = "\n".join(
        _fmt_line(n, sorted(uuids)) for n, uuids in numbered
    )
    prompt = PROMPT_TEMPLATES[sample_type].replace("{input}", input_text)

    return {
        "type": sample_type,
        "question": prompt,
        "expected_output": expected_output,
        "uuid_lines": uuid_lines,
        "expressions": [{"expr": list(uuids), "answer": sorted(uuids)} for uuids in uuid_lines],
        "n_lines": len(uuid_lines),
        "uuids_per_line": uuids_per_line,
        "approx_prompt_tokens": approx_prompt_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV permutation generation
# ─────────────────────────────────────────────────────────────────────────────

def _new_csv_cell(rng: random.Random) -> str:
    """Random UUID string (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx) truncated to 1-36 chars."""
    hex_str = f"{rng.getrandbits(128):032x}"
    uuid_str = f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"
    return uuid_str[: rng.randint(1, 36)]


def _build_csv_input(grid: list[list[str]], row_order: list[int], col_order: list[int]) -> str:
    """Render the input CSV (original row/col order) with permutation spec and header."""
    N = len(grid)
    row_order_str = ",".join(f"[R{i}]" for i in row_order)
    col_order_str = ",".join(f"[C{j}]" for j in col_order)
    header = "," + ",".join(f"[C{j}]" for j in range(N))
    rows = [f"[R{i}]," + ",".join(grid[i][j]
                                  for j in range(N)) for i in range(N)]
    csv_str = "\n".join([header] + rows)
    return PROMPT_TEMPLATES["csv_permutation"].format(
        row_order=row_order_str,
        col_order=col_order_str,
        input=csv_str,
    )


def _build_csv_expected_output(grid: list[list[str]], row_order: list[int], col_order: list[int]) -> str:
    """Render the expected permuted CSV."""
    header = "," + ",".join(f"[C{j}]" for j in col_order)
    rows = [f"[R{i}]," + ",".join(grid[i][j]
                                  for j in col_order) for i in row_order]
    return "\n".join([header] + rows)


def _generate_csv_grid(enc, seed_key, target_tokens: int) -> tuple[list[list[str]], int]:
    """Find the largest square N×N grid fitting target_tokens and return it with its token count.

    Uses a binary search: each probe regenerates the grid deterministically from seed_key
    with an identity permutation (same spec length for any permutation of the same N).
    """
    def _probe(N: int) -> int:
        rng = random.Random(seed_key)
        grid = [[_new_csv_cell(rng) for _ in range(N)] for _ in range(N)]
        prompt = _build_csv_input(grid, list(range(N)), list(range(N)))
        return len(enc.encode(prompt))

    lo, hi = 2, 300
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _probe(mid) <= target_tokens:
            lo = mid
        else:
            hi = mid - 1

    N = lo
    rng = random.Random(seed_key)
    grid = [[_new_csv_cell(rng) for _ in range(N)] for _ in range(N)]
    return grid, _probe(N)


def _apply_perm_fraction(N: int, perm_fraction: float, rng: random.Random) -> list[int]:
    """Return a permutation of range(N) where ~perm_fraction of positions are shuffled."""
    n_permuted = max(2, int(round(N * perm_fraction)))
    positions = sorted(rng.sample(range(N), n_permuted))
    values = positions.copy()
    for _ in range(20):  # retry until non-identity shuffle
        rng.shuffle(values)
        if values != positions:
            break
    order = list(range(N))
    for pos, val in zip(positions, values):
        order[pos] = val
    return order


def _build_csv_sample(
    grid: list[list[str]],
    perm_fraction: float,
    rng: random.Random,
    approx_prompt_tokens: int,
) -> dict:
    N = len(grid)
    row_order = _apply_perm_fraction(N, perm_fraction, rng)
    col_order = _apply_perm_fraction(N, perm_fraction, rng)
    prompt = _build_csv_input(grid, row_order, col_order)
    expected_output = _build_csv_expected_output(grid, row_order, col_order)
    return {
        "type": "csv_permutation",
        "question": prompt,
        "expected_output": expected_output,
        "expressions": [
            {"expr": grid[i][j], "answer": grid[i][j]}
            for i in row_order
            for j in col_order
        ],
        "n_rows": N,
        "n_cols": N,
        "perm_fraction": perm_fraction,
        "approx_prompt_tokens": approx_prompt_tokens,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top-level orchestration
# ─────────────────────────────────────────────────────────────────────────────

def generate(force: bool = False) -> None:
    if OUTPUT_FPATH.exists() and not force:
        count = sum(1 for line in OUTPUT_FPATH.open() if line.strip())
        print(
            f"long_transduction benchmark already exists: {count} examples in {OUTPUT_FPATH}"
        )
        return

    enc = _get_encoder()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    with OUTPUT_FPATH.open("w") as out:
        for target_tokens in TARGET_TOKENS_LIST:
            print(f"\n── target_tokens={target_tokens:,} ──────────────────────────────────────────")

            # Arithmetic chain: 3 variants per (max_operands, sample_idx).
            for max_operands in MAX_OPERANDS_RANGE:
                print(
                    f"Generating sum max_operands={max_operands} ({N_SAMPLES} pairs)...")
                for i in range(N_SAMPLES):
                    expressions, n_tokens = _generate_expressions(
                        max_operands, enc, target_tokens)
                    for sample_type in SUM_TYPES:
                        rng = random.Random((target_tokens, max_operands, i, sample_type))
                        sample = _build_sum_sample(
                            expressions, max_operands, sample_type, n_tokens, rng
                        )
                        sample["target_tokens"] = target_tokens
                        out.write(json.dumps(sample) + "\n")
                        total += 1
                    print(
                        f"  sum[{i + 1:2d}/{N_SAMPLES}] "
                        f"{len(expressions)} expressions, ~{n_tokens:,} tokens, "
                        f"emitted {len(SUM_TYPES)} variants"
                    )

            # UUID sort: 3 variants per (uuids_per_line, sample_idx). The same
            # MAX_OPERANDS_RANGE knob is reused — for these types it is the exact
            # count of UUIDs per line.
            for uuids_per_line in MAX_OPERANDS_RANGE:
                print(
                    f"Generating uuid_sort uuids_per_line={uuids_per_line} "
                    f"({N_SAMPLES} samples)..."
                )
                for i in range(N_SAMPLES):
                    uuid_lines, n_tokens = _generate_uuid_lines(
                        uuids_per_line, enc,
                        seed_key=("uuid", target_tokens, uuids_per_line, i),
                        target_tokens=target_tokens,
                    )
                    for sample_type in UUID_SORT_TYPES:
                        rng = random.Random((target_tokens, uuids_per_line, i, sample_type))
                        sample = _build_uuid_sample(
                            uuid_lines, uuids_per_line, sample_type, n_tokens, rng
                        )
                        sample["target_tokens"] = target_tokens
                        out.write(json.dumps(sample) + "\n")
                        total += 1
                    print(
                        f"  uuid[{i + 1:2d}/{N_SAMPLES}] "
                        f"{len(uuid_lines)} lines × {uuids_per_line} uuids, "
                        f"~{n_tokens:,} tokens, emitted {len(UUID_SORT_TYPES)} variants"
                    )

            # CSV permutation: N_SAMPLES grids × PERM_FRACTIONS difficulties.
            # Each grid is shared across all perm_fractions for that sample index.
            print(
                f"Generating csv_permutation ({N_SAMPLES} grids × {len(PERM_FRACTIONS)} perm_fractions)...")
            for i in range(N_SAMPLES):
                grid, n_tokens = _generate_csv_grid(
                    enc, seed_key=("csv", target_tokens, i), target_tokens=target_tokens
                )
                N = len(grid)
                for perm_fraction in PERM_FRACTIONS:
                    rng = random.Random(("csv", target_tokens, i, perm_fraction))
                    sample = _build_csv_sample(grid, perm_fraction, rng, n_tokens)
                    sample["target_tokens"] = target_tokens
                    out.write(json.dumps(sample) + "\n")
                    total += 1
                print(
                    f"  csv[{i + 1:2d}/{N_SAMPLES}] "
                    f"{N}×{N} grid, ~{n_tokens:,} tokens, "
                    f"emitted {len(PERM_FRACTIONS)} perm_fraction variants"
                )

    print(f"Done. Wrote {total} examples to {OUTPUT_FPATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Regenerate even if output exists"
    )
    args = parser.parse_args()
    generate(force=args.force)


if __name__ == "__main__":
    main()
