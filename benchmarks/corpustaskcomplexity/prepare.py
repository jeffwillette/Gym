# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download published CTC parquet rungs and convert them to Gym tasks.

Run with ``python -m benchmarks.corpustaskcomplexity.prepare --help``.
Each parquet is downloaded independently; no training corpus or seed pool is needed.
"""

import argparse
import json
import logging
from collections.abc import Iterator
from itertools import batched, islice
from pathlib import Path

from pydantic import JsonValue

from resources_servers.corpustaskcomplexity.contract import (
    HF_DATASET,
    QUERY_POSITION,
    ROSTER,
    RUNG_TOKENS,
    generation_budget,
    get_spec,
)


LOG = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent / "data" / "shared"


def convert_example(*, task: str, rung: str, index: int, example: dict[str, JsonValue]) -> dict:
    """Preserve the whole example, including task-specific gold and metadata."""
    if task not in ROSTER or rung not in ROSTER[task].rungs:
        raise ValueError(f"Unknown published task/rung: {task}:{rung}")
    prompt = get_spec(task).build_prompt(example, query_position=QUERY_POSITION)
    return {
        "id": f"{task}:{rung}:{index}",
        "responses_create_params": {
            "input": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_output_tokens": generation_budget(task),
        },
        "verifier_metadata": {"task": task, "rung": rung, "example": example},
    }


def read_examples(path: Path) -> Iterator[dict[str, JsonValue]]:
    """Stream bounded parquet batches rather than loading a whole long rung."""
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
        yield from batch.to_pylist()


def prepare(
    *,
    tasks: list[str] | None = None,
    rungs: list[str] | None = None,
    output_dir: Path | str = DATA_DIR,
    revision: str = "main",
    cache_dir: Path | str | None = None,
    tokenizer_name: str | None = None,
    max_model_len: int | None = None,
    allow_overlength: bool = False,
    limit: int | None = None,
    local_data_root: Path | str | None = None,
    resume: bool = False,
) -> Path:
    """Materialize each selected rung and an auditable preparation manifest.

    Overlength rows are never truncated or scored as wrong answers. By default,
    any such row makes preparation fail; an explicit opt-in produces a filtered
    run whose omitted counts are recorded per task/rung in the manifest.
    """
    from huggingface_hub import HfApi, hf_hub_download

    # Gym forwards CLI config paths as strings; argparse supplies Path objects.
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else None
    local_data_root = Path(local_data_root) if local_data_root is not None else None
    selected_tasks = list(tasks) if tasks is not None else list(ROSTER)
    selected_rungs = list(rungs) if rungs is not None else list(RUNG_TOKENS)[:5]
    if not selected_tasks or not selected_rungs:
        raise ValueError("Select at least one task and rung")
    if set(selected_tasks) - ROSTER.keys() or set(selected_rungs) - RUNG_TOKENS.keys():
        raise ValueError("Unknown task or rung selection")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if max_model_len is not None and (max_model_len <= 0 or not tokenizer_name):
        raise ValueError("A positive max_model_len requires a tokenizer")
    tokenizer = None
    if tokenizer_name:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_revision = revision if local_data_root else HfApi().dataset_info(HF_DATASET, revision=revision).sha
    manifest = {
        "dataset": HF_DATASET,
        "revision": resolved_revision,
        "query_position": QUERY_POSITION,
        "prompt_mode": "raw_completion",
        "tokenizer": tokenizer_name,
        "max_model_len": max_model_len,
        "allow_overlength": allow_overlength,
        "limit_per_rung": limit,
        "tasks": selected_tasks,
        "rungs": selected_rungs,
        "files": [],
        "complete": False,
    }
    manifest_path = output_dir / "manifest.json"
    previous_files = {}
    if resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in (
            "dataset",
            "revision",
            "query_position",
            "prompt_mode",
            "tokenizer",
            "max_model_len",
            "allow_overlength",
            "limit_per_rung",
            "tasks",
            "rungs",
        ):
            if previous[key] != manifest[key]:
                raise ValueError(f"Cannot resume: preparation setting {key} changed; use another output directory")
        previous_files = {(item["task"], item["rung"]): item for item in previous["files"]}
    combined_path = output_dir / "corpustaskcomplexity_benchmark.jsonl"
    overlength_total = 0
    total = 0
    with combined_path.with_suffix(".jsonl.partial").open("w") as combined:
        for task in selected_tasks:
            roster_row = ROSTER[task]
            for rung in selected_rungs:
                if rung not in roster_row.rungs:
                    continue
                output = output_dir / task / f"{rung}.jsonl"
                previous_file = previous_files.get((task, rung))
                if previous_file is not None and output.is_file():
                    LOG.info("Reusing completed %s:%s (%d rows)", task, rung, previous_file["kept"])
                    copied = 0
                    with output.open() as stream:
                        for line in stream:
                            combined.write(line)
                            copied += 1
                    if copied != previous_file["kept"]:
                        raise ValueError(f"Cannot resume: row count changed in {output}")
                    manifest["files"].append(previous_file)
                    total += copied
                    overlength_total += previous_file["overlength"]
                    continue
                filename = f"data/{roster_row.subset}/{rung}.parquet"
                LOG.info("Loading %s:%s (%s)", task, rung, filename)
                if local_data_root:
                    source_path = local_data_root / filename
                else:
                    source_path = Path(
                        hf_hub_download(
                            HF_DATASET,
                            filename,
                            repo_type="dataset",
                            revision=resolved_revision,
                            cache_dir=str(cache_dir) if cache_dir else None,
                        )
                    )
                output.parent.mkdir(exist_ok=True)
                kept = seen = overlength = 0
                max_prompt_tokens = 0
                with output.with_suffix(".jsonl.partial").open("w") as stream:
                    examples = islice(read_examples(source_path), limit)
                    for batch in batched(enumerate(examples), 16):
                        rows = [
                            convert_example(task=task, rung=rung, index=index, example=example)
                            for index, example in batch
                        ]
                        token_counts = [None] * len(rows)
                        if tokenizer is not None:
                            prompts = [row["responses_create_params"]["input"][0]["content"] for row in rows]
                            encoded = tokenizer(
                                prompts,
                                add_special_tokens=False,
                                return_attention_mask=False,
                                return_token_type_ids=False,
                            )
                            token_counts = [len(ids) for ids in encoded["input_ids"]]
                        for row, tokens in zip(rows, token_counts, strict=True):
                            seen += 1
                            if tokens is not None:
                                row["prompt_tokens"] = tokens
                                max_prompt_tokens = max(max_prompt_tokens, tokens)
                                if max_model_len is not None and tokens + generation_budget(task) > max_model_len:
                                    overlength += 1
                                    continue
                            line = json.dumps(row, ensure_ascii=False) + "\n"
                            stream.write(line)
                            combined.write(line)
                            kept += 1
                            if seen % 25 == 0:
                                LOG.info("%s:%s seen=%d kept=%d overlength=%d", task, rung, seen, kept, overlength)
                if not seen:
                    raise ValueError(f"Empty published input: {source_path}")
                output.with_suffix(".jsonl.partial").replace(output)
                total += kept
                overlength_total += overlength
                manifest["files"].append(
                    {
                        "task": task,
                        "rung": rung,
                        "path": str(output.resolve()),
                        "seen": seen,
                        "kept": kept,
                        "overlength": overlength,
                        "max_prompt_tokens": max_prompt_tokens,
                    }
                )
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                LOG.info("Finished %s:%s seen=%d kept=%d overlength=%d", task, rung, seen, kept, overlength)
    if overlength_total and not allow_overlength:
        raise ValueError(
            f"{overlength_total} rows exceed the context budget; see {manifest_path}. "
            "Choose smaller rungs or explicitly pass --allow-overlength for a filtered run."
        )
    if not total:
        raise ValueError("No rows fit the selected context budget")
    combined_path.with_suffix(".jsonl.partial").replace(combined_path)
    manifest["complete"] = True
    manifest["total_rows"] = total
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    LOG.info("Prepared %d rows in %s", total, combined_path)
    return combined_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=list(ROSTER))
    parser.add_argument("--rungs", nargs="+", choices=list(RUNG_TOKENS))
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--tokenizer", dest="tokenizer_name")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--allow-overlength", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--local-data-root", type=Path)
    parser.add_argument("--resume", action="store_true", help="Reuse completed rungs when preparation settings match")
    args = vars(parser.parse_args())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prepare(**args)


if __name__ == "__main__":
    main()
