# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CTC evaluation support, adapted from the recommended olmo-eval harness.

Source: https://github.com/allenai/olmo-eval/tree/69729fadd898e50e383e42803bb9a6250eb1dbbf/src/olmo_eval/evals/tasks/ctc_suite
Revision: 69729fadd898e50e383e42803bb9a6250eb1dbbf
Original file: ctc/tasks/xabsence/spec.py
License: Apache-2.0; see ../CTC_LICENSE and ../OLMO_EVAL_LICENSE.
Local adaptation: package relocation, provenance documentation, and normal Gym lint/formatting.

The ``xabsence`` eval contract (cross-corpus absence).

Two numbered corpora, A and B, under one shared index. Almost every claim has a paraphrase in the
*other* corpus; a few are unmatched. Find the unmatched ones.

Harder than :mod:`ctc.tasks.absence` in a specific way: absence compares an item against its own
copy, so the match is near-exact, whereas here the match is a paraphrase and must be recognised
semantically across the corpus boundary. That is why it is grouped with the N-squared tasks despite
producing the same flat id-set answer.

The answer anchors on ``Unmatched:`` rather than ``Missing:``; the shared parser accepts both.

.. note::
   Pre-2026-07-26 xabsence shards may be affected by the oolong ``--item-regex`` leak (a bare
   ``'||'`` matched every line). Check the build date of any shard before trusting a number.
"""

from __future__ import annotations

from typing import Dict

from ...format.prompts import XABSENCE_INSTRUCTION
from .._absence import make_absence_spec


__all__ = ["SPEC", "build_query", "build_target"]


def build_query(example: Dict) -> str:
    """
    :param example: A unified-format example. ``queries`` is unused -- both corpora are already in
        the rendered context, so the instruction is the whole ask.

    :returns: The positioned ask.
    """
    return XABSENCE_INSTRUCTION


def build_target(example: Dict) -> str:
    """
    :param example: A unified-format example with 0-based ``gold_doc_indices``.

    :returns: The ``Unmatched: [i], [j]`` answer line, with ids rendered 1-based.
    """
    from .._absence import build_target as _bt

    return _bt(example, "Unmatched")


SPEC = make_absence_spec(
    name="xabsence",
    description="Find claims with no paraphrase in the other corpus (cross-corpus absence).",
    instruction=XABSENCE_INSTRUCTION,
    serializer="xabsence",
    rungs=("2k", "4k", "8k", "16k", "32k"),
    query_builder=build_query,
    max_new_tokens=200,
    sources=("pubmed", "abstracts"),
)
