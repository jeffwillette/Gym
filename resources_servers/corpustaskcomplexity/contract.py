# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Published CTC roster and evaluation contract.

Roster adapted from https://github.com/allenai/olmo-eval/tree/69729fadd898e50e383e42803bb9a6250eb1dbbf/src/olmo_eval/evals/tasks/ctc_suite
Revision: 69729fadd898e50e383e42803bb9a6250eb1dbbf; Apache-2.0.
The evaluation package docstring records the reference snapshot and local adaptations.
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import JsonValue

from .evaluation.eval.stopping import STOP_PRESETS, apply
from .evaluation.format import registry
from .evaluation.format.prompts import GROUPING_INSTRUCTION
from .evaluation.format.registry import TaskSpec
from .evaluation.tasks import load_all
from .evaluation.tasks._grouping import make_grouping_spec


HF_DATASET = "PrasannSinghal/ctc-suite-eval"
QUERY_POSITION = "both"
LOG = logging.getLogger(__name__)
load_all()
if "grouping" not in registry.names():
    registry.register(
        make_grouping_spec(
            name="grouping",
            description="Partition abstracts into unnamed field clusters.",
            instruction=GROUPING_INSTRUCTION,
            rungs=("2k", "4k", "8k", "16k", "32k"),
            query_builder=lambda ex: f"{GROUPING_INSTRUCTION}\n\n{ex['queries'][0]}"
            if ex.get("queries")
            else GROUPING_INSTRUCTION,
            sources=("openalex",),
        )
    )

#: Rung label -> token budget in the local filenames. ``contradiction_iid`` aliases r2k to 2560:
#: its 2048-doc-count file sat below the training minimum, so the IID ladder starts one step up
#: (the remap is explicit in the source repo's ``build_suite_table.py`` as well).
RUNG_TOKENS: dict[str, int] = {
    "r2k": 2048,
    "r4k": 4096,
    "r8k": 8192,
    "r16k": 16384,
    "r32k": 32768,
    "r64k": 65536,
    "r128k": 131072,
    "r256k": 262144,
    "r512k": 524288,
    "r1m": 1048576,
}

_LADDER_2K_32K = ("r2k", "r4k", "r8k", "r16k", "r32k")
_LADDER_FULL = tuple(RUNG_TOKENS)  # 2k .. 1m
_LADDER_TO_512K = _LADDER_FULL[:-1]


class CTCClass(StrEnum):
    """How much of the corpus a row forces the model to track at once.

    This is the axis the suite is named for, so it is declared per row rather than inferred:
    :data:`LOW` rows are answerable by finding the right document(s) -- work that scales O(N) in
    corpus size and that a retriever could in principle do -- while :data:`HIGH` rows require
    relating documents to each other (O(N^2) pair-finding, O(NM) clustering, O(N^3) triples) and
    have no single answer-bearing span. The split is 11/11 and is what ``ctc:low`` / ``ctc:high``
    select; it is orthogonal to context length, which ``ctc:figure`` / ``ctc:xlong`` select.
    """

    LOW = "low"
    HIGH = "high"


@dataclass(frozen=True)
class RosterRow:
    """One row of the 22-task suite.

    :param subset: HF config name == local ladder directory name.
    :param spec: The evaluation :class:`TaskSpec` that formats and grades this row.
    :param ctc_class: :class:`CTCClass` -- low (O(N), retrieval-shaped) or high (O(N^2)+,
        relational/structural). Declared with no default: a new row must classify itself, because
        a silently-defaulted row would quietly change what ``ctc:low``/``ctc:high`` mean.
    :param complexity: The class in the notation the README table uses (``"O(N)"``, ``"O(N^2)"``,
        ``"O(NM)"``, ``"O(N^3)"``), recorded so the coarse two-way split stays auditable.
    :param rungs: Rung labels with data, ascending.
    :param eval_size: ``{rung_label: rows}`` for any rung below the 500-example floor. A number
        quoted from one of these must carry the size inline -- a small eval inflates noise into
        apparent findings.
    :param rung_alias: Rung label -> token budget override for the local filename.
    :param note: Anything a reader of the numbers has to know.
    """

    subset: str
    spec: str
    ctc_class: CTCClass
    complexity: str
    rungs: tuple[str, ...] = _LADDER_2K_32K
    eval_size: dict[str, int] = field(default_factory=dict)
    rung_alias: dict[str, int] = field(default_factory=dict)
    note: str = ""


_SUB500_XLONG = {"r256k": 125, "r512k": 125, "r1m": 125}

#: The frozen 22-row roster (records/ctc-final-suite.md, 2026-08-12). Row order is figure order.
ROSTER: dict[str, RosterRow] = {
    "ctc_fiqa": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="fiqa",
        spec="retrieval",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_nq": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="nq",
        spec="retrieval",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_hpqa": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="hotpotqa",
        spec="retrieval",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_qdmatch_fiqa": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="qdmatch_fiqa",
        spec="qdmatch",
        rungs=_LADDER_TO_512K,
        eval_size=dict(_SUB500_XLONG),
        note="caps at 512k: the 6,148 recoverable BEIR-FiQA units are the whole labeled universe, "
        "and a 1M example needs ~8k distinct units",
    ),
    "ctc_qdmatch_nq": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="qdmatch_nq",
        spec="qdmatch",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_qdmatch_hpqa": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="qdmatch_hpqa",
        spec="qdmatch",
        rungs=_LADDER_FULL[:-2],
        eval_size=dict(_SUB500_XLONG),
        note="caps at 256k: 4,000 recoverable HotpotQA units, and a 512k example needs ~7k",
    ),
    "ctc_outlier_amzn": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="outlier_amzn",
        spec="outlier",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_outlier": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(NM)",
        subset="outlier",
        spec="outlier",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
        note="64k+ rungs are the true scale-K construction (K ~ n/9); the retired fixed-K xlong files are not used",
    ),
    "ctc_outlier_fixedm": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="outlier_fixedM",
        spec="outlier",
        rungs=_LADDER_TO_512K,
        eval_size=dict(_SUB500_XLONG),
        note="K pinned at 3 -- the control for the scale-K row. Caps at 512k: three majority "
        "topics need ~2,400 same-topic chunks each at 1M and the wiki pool cannot supply that",
    ),
    "ctc_oolong": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="oolong",
        spec="oolong",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_grouping": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(NM)",
        subset="grouping",
        spec="grouping",
    ),
    "ctc_absence": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="absence_gutenberg",
        spec="absence",
        rungs=("r2k", "r4k", "r8k", "r16k"),
        note="corpus is a contiguous Gutenberg passage; rung ceiling is bounded by book length",
    ),
    "ctc_xabsence": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="xabsence",
        spec="xabsence",
        note="the one-sided EXACT-COPY Gutenberg build (2026-08-14): B-side twins are "
        "byte-identical and orphans sit A-side only. Supersedes the PubMed paraphrase build, "
        "kept in the dataset as the xabsence_paraphrase config",
    ),
    "ctc_rerank": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="rerank",
        spec="rerank",
        rungs=_LADDER_TO_512K,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_msmarco": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="msmarco",
        spec="retrieval",
        rungs=_LADDER_TO_512K,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_reorder": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="reorder",
        spec="reorder",
        rungs=("r2k", "r4k", "r8k", "r16k"),
        note="chunks are contiguous in one book; rung ceiling is bounded by book length",
    ),
    "ctc_obliq": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="obliq_twitter",
        spec="retrieval",
        rungs=_LADDER_FULL,
        eval_size={
            "r2k": 123,
            **{r: 126 for r in _LADDER_FULL[1:7]},
            "r256k": 125,
            "r512k": 125,
            "r1m": 125,
        },
        note="123-126 examples per rung -- flag the size inline",
    ),
    "ctc_niah": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="niah",
        spec="retrieval",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
    ),
    "ctc_contradiction": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="contradiction_iid",
        spec="contradiction",
        rungs=_LADDER_FULL,
        eval_size=dict(_SUB500_XLONG),
        rung_alias={"r2k": 2560},
        note="the IID realistic-mode ladder; never mix with the retired both-mode ladder",
    ),
    "ctc_strmatch": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^2)",
        subset="strmatch",
        spec="strmatch",
    ),
    "ctc_textgroups": RosterRow(
        ctc_class=CTCClass.HIGH,
        complexity="O(N^3)",
        subset="textgroups",
        spec="textgroups",
    ),
    "ctc_scifact": RosterRow(
        ctc_class=CTCClass.LOW,
        complexity="O(N)",
        subset="scifact",
        spec="retrieval",
        eval_size={r: 300 for r in _LADDER_2K_32K},
        note="300 examples at every rung -- flag the size inline",
    ),
}


def get_spec(task: str) -> TaskSpec:
    """Resolve a published suite row to its frozen task specification."""
    return registry.get(ROSTER[task].spec)


def generation_budget(task: str) -> int:
    """Match the recommended harness's per-task output allowance."""
    spec = get_spec(task)
    return max(spec.max_new_tokens, STOP_PRESETS[spec.stop].max_new_tokens)


def score_answer(task: str, prediction: str, example: dict[str, JsonValue]) -> tuple[dict[str, float], bool]:
    """Apply reference stopping, parsing, gold resolution, and scoring without rounding."""
    spec = get_spec(task)
    cleaned = apply(prediction, STOP_PRESETS[spec.stop])
    # The frozen pair parser interprets blank output as an empty set. A missing
    # model answer must remain a parse failure, even when the gold set is empty.
    try:
        parsed = spec.parse(cleaned, len(example["documents"])) if cleaned.strip() else None
    except (ValueError, TypeError, OverflowError) as error:
        # Parsing is the untrusted model-output boundary. Gold/scoring errors
        # remain exceptions so malformed datasets cannot become model failures.
        LOG.warning("Malformed CTC answer for %s: %s", task, error)
        parsed = None
    if spec.extra.get("score_takes_example") or spec.name == "oolong":
        gold = example
    else:
        # Preserve a real empty gold set; truthiness fallback would replace it
        # with the entire example and corrupt empty-answer scoring.
        gold = example[spec.extra.get("gold_field", "gold_doc_indices")]
    scores = {key: float(value) for key, value in spec.score(parsed, gold).items()}
    return scores, parsed is not None
