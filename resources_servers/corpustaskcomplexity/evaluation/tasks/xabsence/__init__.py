# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CTC evaluation support, adapted from the recommended olmo-eval harness.

Source: https://github.com/allenai/olmo-eval/tree/69729fadd898e50e383e42803bb9a6250eb1dbbf/src/olmo_eval/evals/tasks/ctc_suite
Revision: 69729fadd898e50e383e42803bb9a6250eb1dbbf
Original file: ctc/tasks/xabsence/__init__.py
License: Apache-2.0; see ../CTC_LICENSE and ../OLMO_EVAL_LICENSE.
Local adaptation: package relocation, provenance documentation, and normal Gym lint/formatting.

The ``xabsence`` task. Importing this package registers its spec.
"""

from ...format import registry
from .spec import SPEC


registry.register(SPEC)

__all__ = ["SPEC"]
