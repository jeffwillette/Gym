# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CTC evaluation support, adapted from the recommended olmo-eval harness.

Source: https://github.com/allenai/olmo-eval/tree/69729fadd898e50e383e42803bb9a6250eb1dbbf/src/olmo_eval/evals/tasks/ctc_suite
Revision: 69729fadd898e50e383e42803bb9a6250eb1dbbf
Original file: ctc/__init__.py
License: Apache-2.0; see ../CTC_LICENSE and ../OLMO_EVAL_LICENSE.
Local adaptation: package relocation, provenance documentation, and normal Gym lint/formatting.

Prompt construction, answer parsing, scoring, and stopping for the published CTC suite.

The recommended harness embeds the CTC snapshot identified as c2b345fba (2026-09-09).
Its task definitions differ from the current canonical CTC repository; this integration
uses the published harness definitions and the 22-task roster in ../contract.py.

All evaluation modules follow Gym's ordinary lint and pre-commit rules. They are
maintained integration code, not byte-for-byte reference copies. Prompt strings,
index conventions, stopping behavior, and metric definitions are preserved.
The adapter in ../contract.py treats blank outputs as parse failures, preserves
empty gold sets, and catches malformed prediction conversions at the parsing boundary.

The reference loader imports its complete task registry; only the 22 published
roster entries are selectable by the benchmark. No generation or inference backend
is bundled. Source licenses are retained in the parent directory.

Golden regression fixtures and independent comparisons on 110 local published r2k
examples protect prompt and scoring behavior across formatting/layout changes.
"""

__version__ = "0.1.0"
