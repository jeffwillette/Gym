# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Behavior snapshots captured before integrating the reference package into normal Gym checks."""

import builtins
import hashlib
import json
import re
from pathlib import Path

import pytest

from resources_servers.corpustaskcomplexity.contract import QUERY_POSITION, generation_budget, get_spec, score_answer


REFERENCE = json.loads((Path(__file__).parent / "fixtures/evaluation-reference.json").read_text())


@pytest.mark.parametrize("case", REFERENCE["cases"], ids=lambda case: case["task"])
def test_prompts_budgets_and_scores_match_reference(case: dict) -> None:
    task, example, expected = case["task"], case["example"], case["expected"]
    prompt = get_spec(task).build_prompt(example, query_position=QUERY_POSITION)
    assert hashlib.sha256(prompt.encode()).hexdigest() == expected["prompt_sha256"]
    assert generation_budget(task) == expected["budget"]
    for prediction, recorded in zip(REFERENCE["predictions"], expected["scores"], strict=True):
        if "error" in recorded:
            exception = getattr(builtins, recorded["error"])
            with pytest.raises(exception, match=re.escape(recorded["message"])):
                score_answer(task, prediction, example)
        else:
            metrics, parsed = score_answer(task, prediction, example)
            assert [metrics, parsed] == recorded["value"], (task, prediction)
