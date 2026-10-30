# CTC-Bench

Implements all 22 published corpus-task-complexity tasks and all 177 task/rung
combinations, with each task's original ladder ceiling. Public data comes from
`PrasannSinghal/ctc-suite-eval`. No training data generation is needed.

Run from the Gym root. Keep preparation dependencies in a dedicated environment;
the resources server's runtime requirements remain separate.

```bash
uv venv --python .venv/bin/python benchmarks/corpustaskcomplexity/.venv
uv pip install --python benchmarks/corpustaskcomplexity/.venv/bin/python \
  -e . -r benchmarks/corpustaskcomplexity/requirements.txt
benchmarks/corpustaskcomplexity/.venv/bin/python -m benchmarks.corpustaskcomplexity.prepare \
  --revision 04ab86006124b3f34c566f44bbc82d4fec98fcda
```

The default selection includes every task at every published rung through r32k
(the paper grid: 108 task/rung combinations).
Use `--tasks ctc_nq ctc_contradiction` and `--rungs r2k r4k` for a smaller run.
Choose rungs appropriate to the serving context. `--limit 1` produces a smoke sample per task/rung, recorded in the
manifest; it is not a full benchmark result.

The Gym benchmark config exposes the same rung and output selection:

```bash
# From the Gym worktree, build the default paper grid.
benchmarks/corpustaskcomplexity/.venv/bin/gym eval prepare --benchmark corpustaskcomplexity

# Build additional published rungs in a separate directory.
benchmarks/corpustaskcomplexity/.venv/bin/gym eval prepare --benchmark corpustaskcomplexity \
  '++ctc_rungs=[r2k,r4k,r8k,r16k,r32k,r64k,r128k,r256k,r512k,r1m]' \
  ++ctc_data_dir=benchmarks/corpustaskcomplexity/data/extended
```

Use `+prepare_script_args.tasks=[ctc_nq]` or `+prepare_script_args.limit=1`
for a bounded CLI check. A larger selection uses the existing published examples;
it does not synthesize additional examples. `--rungs` and `--output-dir` provide
the corresponding arguments when invoking the Python module directly.

Each selected parquet is downloaded separately from a single resolved dataset
revision. Preparation creates one JSONL per task/rung, a combined
`corpustaskcomplexity_benchmark.jsonl`, and `manifest.json`. The original example,
including special gold fields and metadata, travels to the verifier unchanged.
Use `--local-data-root` with the same `data/<subset>/<rung>.parquet` tree for
offline preparation. Run from the Gym root using `python -m`, as shown above.
Add `--resume` to reuse completed rungs with the same preparation settings.

**Prompt fidelity:** config.yaml uses `/v1/completions` with the verbatim reference
Alpaca-style prompt and `query_position="both"`. It does not apply the model's
chat template or add document-marker tokens used by historical training runs.
This matches the recommended olmo-eval harness. Temperature is 0, and each row
contains the reference task's output allowance. Do not globally override it.

**Context fidelity:** rung labels are build targets, not measured prompt lengths.
Token counting is optional: add `--tokenizer /path/to/model --max-model-len N`
to count the actual prompt and reserve that task's output allowance. Without
these options, the same prepared inputs can be shared by different models,
and the serving engine enforces its context limit. Overlength examples are never truncated.
When counting tokens, the default behavior fails on any overlength row. An explicit
`--allow-overlength` opts into a filtered run; its omitted counts remain in the
manifest and reported metrics. Use different output directories for different
tokenizers, rung selections or sample limits.

Scoring and parse rate must be reported by task/rung. Kendall tau can be negative,
and rerank's primary is CE-positive recall, not MRR. See
[the verifier README](../../resources_servers/corpustaskcomplexity/README.md).
