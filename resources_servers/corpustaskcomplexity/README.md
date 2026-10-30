# Corpus Task Complexity (CTC-Bench)

Stateless verification for the complete published 22-task CTC suite. Reuses
`simple_agent` and the vLLM model server. See
[benchmark preparation](../../benchmarks/corpustaskcomplexity/README.md).

The public dataset is [PrasannSinghal/ctc-suite-eval](https://huggingface.co/datasets/PrasannSinghal/ctc-suite-eval).
The benchmark author recommends the
[olmo-eval CTC harness](https://github.com/allenai/olmo-eval/tree/prasann/ctc-suite-grader-fixes/src/olmo_eval/evals/tasks/ctc_suite).
Its prompt builders, parsers, scoring functions and stopping rules are integrated
into the normal `evaluation` package and follow Gym lint/pre-commit rules. The
[evaluation module docstring](evaluation/__init__.py) records the exact source
revision, license, and local adaptations. Original licenses are retained as
`CTC_LICENSE` and `OLMO_EVAL_LICENSE`.

`verifier_metadata` carries `task` (the published `ctc_*` row name), `rung`,
and the complete upstream `example`. The verifier returns the task's primary
score as `reward`, all secondary scores in `task_metrics`, and `parse_ok`.
Retrieval/absence gold indices are zero based; pair gold is one based;
query/document matching pairs are ordered. Reranking uses CE-positive recall,
OOLONG uses partial credit, and reorder uses Kendall tau in **[-1, 1]**.
Rewards are neither binarized nor clipped. Metrics remain separate by task/rung;
there is no meaningful suite-wide average across these different metrics.

Reference stopping runs after generation, preserving pretty-printed JSON,
required answer preambles and completed `<think>` blocks. The adapter fixes
two empty-answer edge cases: blank completions are parse failures, and actual
empty gold sets remain empty. Invalid model-output types become parse failures;
invalid gold data remains an error.

```bash
.venv/bin/pytest resources_servers/corpustaskcomplexity/tests
```

The committed five published examples cover retrieval, contradiction, reranking,
OOLONG and passage ordering. `verified: false` remains until model baselines
have been reviewed.
