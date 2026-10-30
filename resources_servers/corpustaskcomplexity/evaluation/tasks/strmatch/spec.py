# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CTC evaluation support, adapted from the recommended olmo-eval harness.

Source: https://github.com/allenai/olmo-eval/tree/69729fadd898e50e383e42803bb9a6250eb1dbbf/src/olmo_eval/evals/tasks/ctc_suite
Revision: 69729fadd898e50e383e42803bb9a6250eb1dbbf
Original file: ctc/tasks/strmatch/spec.py
License: Apache-2.0; see ../CTC_LICENSE and ../OLMO_EVAL_LICENSE.
Local adaptation: package relocation, provenance documentation, and normal Gym lint/formatting.

The ``strmatch`` eval contract.

A list of short strings; find every pair satisfying a string-similarity criterion -- a shared
contiguous run of words, or a matching word count. Synthetic, so the criterion varies per example
and is carried in ``queries[0]`` rather than being fixed by the instruction.

That per-example criterion is the one structural difference from :mod:`ctc.tasks.redundancy`: the
positioned ask is the instruction *followed by* the example's criterion. Everything else -- answer
format, parser, scorer, gold convention -- is the shared pair-family machinery.
"""

from __future__ import annotations

from typing import Dict

from ...format.prompts import STRMATCH_INSTRUCTION
from .._pairs import build_target, make_pair_spec  # noqa: F401 (build_target re-exported)


__all__ = ["SPEC", "build_query", "build_target"]


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example whose ``queries[0]`` states this example's criterion.

    :returns: The instruction followed by the criterion, in that order -- the order is part of the
        format the shards were built with.
    """
    return f"{STRMATCH_INSTRUCTION}\n\n{example['queries'][0]}"


SPEC = make_pair_spec(
    name="strmatch",
    description="Find every pair of strings matching a per-example similarity criterion.",
    instruction=STRMATCH_INSTRUCTION,
    serializer="strmatch",
    rungs=("2k", "4k", "8k", "16k", "32k"),
    query_builder=build_query,
    max_new_tokens=200,
)
