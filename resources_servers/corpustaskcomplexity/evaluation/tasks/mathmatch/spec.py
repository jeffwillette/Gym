# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CTC evaluation support, adapted from the recommended olmo-eval harness.

Source: https://github.com/allenai/olmo-eval/tree/69729fadd898e50e383e42803bb9a6250eb1dbbf/src/olmo_eval/evals/tasks/ctc_suite
Revision: 69729fadd898e50e383e42803bb9a6250eb1dbbf
Original file: ctc/tasks/mathmatch/spec.py
License: Apache-2.0; see ../CTC_LICENSE and ../OLMO_EVAL_LICENSE.
Local adaptation: package relocation, provenance documentation, and normal Gym lint/formatting.

The ``mathmatch`` eval contract.

A list of arithmetic expressions; find every pair whose values satisfy a closeness criterion (e.g.
``|val(a) - val(b)| <= x``). Synthetic, with the criterion carried per example in ``queries[0]``.

Structurally identical to :mod:`ctc.tasks.strmatch` -- instruction then criterion, shared pair
machinery -- and differs only in what the items are and which template renders them.
"""

from __future__ import annotations

from typing import Dict

from ...format.prompts import MATHMATCH_INSTRUCTION
from .._pairs import build_target, make_pair_spec  # noqa: F401 (build_target re-exported)


__all__ = ["SPEC", "build_query", "build_target"]


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example whose ``queries[0]`` states this example's criterion.

    :returns: The instruction followed by the criterion.
    """
    return f"{MATHMATCH_INSTRUCTION}\n\n{example['queries'][0]}"


SPEC = make_pair_spec(
    name="mathmatch",
    description="Find every pair of arithmetic expressions whose values meet a criterion.",
    instruction=MATHMATCH_INSTRUCTION,
    serializer="mathmatch",
    rungs=("2k", "4k", "8k", "16k", "32k"),
    query_builder=build_query,
    max_new_tokens=200,
)
