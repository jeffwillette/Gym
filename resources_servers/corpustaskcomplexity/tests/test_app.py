# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.corpustaskcomplexity.app import (
    CTCResourcesServer,
    CTCResourcesServerConfig,
    CTCVerifyRequest,
)
from resources_servers.corpustaskcomplexity.contract import ROSTER, generation_budget, get_spec, score_answer


def example(*, gold: object = None) -> dict:
    return {
        "queries": ["find the answer"],
        "answers": ["Paris"],
        "documents": [{"text": f"Document {i}"} for i in range(4)],
        "gold_doc_indices": [0] if gold is None else gold,
    }


def test_published_roster() -> None:
    assert len(ROSTER) == 22
    assert sum(len(row.rungs) for row in ROSTER.values()) == 177
    assert sum(row.ctc_class == "high" for row in ROSTER.values()) == 11
    assert ROSTER["ctc_contradiction"].subset == "contradiction_iid"
    assert ROSTER["ctc_xabsence"].subset == "xabsence"
    assert ROSTER["ctc_outlier_fixedm"].subset == "outlier_fixedM"
    for task in ROSTER:
        assert generation_budget(task) >= get_spec(task).max_new_tokens


@pytest.mark.parametrize("task", [task for task, row in ROSTER.items() if row.spec == "retrieval"])
def test_retrieval_index_base(task: str) -> None:
    assert score_answer(task, "[1]", example())[0]["f1"] == 1
    assert score_answer(task, "[0]", example())[0]["f1"] == 0
    assert score_answer(task, "[1], [2]", example())[0]["f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "task", ["ctc_contradiction", "ctc_strmatch", "ctc_qdmatch_nq", "ctc_qdmatch_hpqa", "ctc_qdmatch_fiqa"]
)
def test_pair_scoring_and_stopping(task: str) -> None:
    data = example(gold=[[1, 3]])
    data["gold_pairs"] = [[1, 3]]
    scores, parsed = score_answer(task, "<think>\nreasoning\n</think>\n```json\n[[1, 3]]\n```", data)
    assert parsed and scores["f1"] == 1
    assert score_answer(task, "[[1, 2]]", data)[0]["f1"] == 0
    assert score_answer(task, "", data)[1] is False


@pytest.mark.parametrize("task", ["ctc_outlier", "ctc_outlier_amzn", "ctc_outlier_fixedm"])
def test_outlier_required_preamble(task: str) -> None:
    scores, parsed = score_answer(task, "Most documents are science.\nOutliers: [1]\nMore explanation.", example())
    assert parsed and scores["f1"] == 1


@pytest.mark.parametrize("task, prefix", [("ctc_absence", "Missing"), ("ctc_xabsence", "Unmatched")])
def test_absence_index_base(task: str, prefix: str) -> None:
    assert score_answer(task, f"{prefix}: [1]", example())[0]["f1"] == 1


def test_grouping_labels_are_arbitrary() -> None:
    scores, parsed = score_answer("ctc_grouping", "[[3, 4], [1, 2]]", example(gold=[[0, 1], [2, 3]]))
    assert parsed and scores["pairwise_f1"] == 1


def test_textgroups_set_of_groups() -> None:
    assert score_answer("ctc_textgroups", "[[3, 1, 2]]", example(gold=[[1, 2, 3]]))[0]["f1"] == 1


def test_reorder_preserves_negative_tau() -> None:
    data = example()
    data["gold_order"] = [1, 2, 3, 4]
    assert score_answer("ctc_reorder", "[1, 2, 3, 4]", data)[0]["kendall_tau"] == 1
    assert score_answer("ctc_reorder", "[4, 3, 2, 1]", data)[0]["kendall_tau"] == -1
    assert score_answer("ctc_reorder", "[1, 2]", data)[1] is False


def test_rerank_uses_ce_positive_recall() -> None:
    data = example()
    data["ce_scores"] = [4, -10, 2, -8]
    assert score_answer("ctc_rerank", "Ranking: [1], [3]", data)[0]["ce_pos_recall"] == 1
    assert score_answer("ctc_rerank", "Ranking: [1]", data)[0]["ce_pos_recall"] == 0.5


def test_oolong_numeric_partial_credit() -> None:
    data = example(gold=[])
    data["answers"] = ["10"]
    data["_meta"] = {"answer_type": "NUMERIC", "gold_list": ["10"]}
    assert score_answer("ctc_oolong", "Let me count.\nAnswer: 11\nignored", data)[0]["score"] == 0.75


def test_invalid_gold_is_not_suppressed() -> None:
    with pytest.raises(ValueError, match="sorted"):
        score_answer("ctc_contradiction", "[[1, 3]]", example(gold=[[3, 1]]))


@pytest.mark.parametrize("task", ["ctc_contradiction", "ctc_strmatch", "ctc_textgroups", "ctc_qdmatch_nq"])
def test_empty_gold_and_missing_answer(task: str) -> None:
    data = example(gold=[])
    data["gold_pairs"] = []
    assert score_answer(task, "[]", data)[0]["f1"] == 1
    assert score_answer(task, "", data)[0]["f1"] == 0
    assert score_answer(task, "<think>reasoning</think>\n", data)[1] is False


def test_malformed_model_output_is_a_parse_failure() -> None:
    scores, parsed = score_answer("ctc_contradiction", '[["bad", null]]', example(gold=[[1, 3]]))
    assert not parsed and scores["f1"] == 0


@pytest.mark.parametrize("prediction, expected", [("[1]", 1.0), ("[2]", 0.0), ("", 0.0)])
async def test_verify(prediction: str, expected: float) -> None:
    server = CTCResourcesServer(
        config=CTCResourcesServerConfig(host="127.0.0.1", port=8001, entrypoint="app.py", name="ctc"),
        server_client=MagicMock(spec=ServerClient),
    )
    response = NeMoGymResponse.model_validate(
        {
            "id": "response",
            "created_at": 0,
            "model": "test",
            "object": "response",
            "status": "completed",
            "parallel_tool_calls": False,
            "tool_choice": "none",
            "tools": [],
            "output": [
                {
                    "id": "msg",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": prediction, "annotations": []}],
                }
            ],
        }
    )
    body = CTCVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "prompt"}]},
            "response": response,
            "verifier_metadata": {"task": "ctc_nq", "rung": "r2k", "example": example()},
        }
    )
    result = await server.verify(body)
    assert result.reward == expected
    assert result.primary_metric == "f1"
    assert result.task_metrics["f1"] == expected
    assert result.verifier_metadata.example == example()


def test_metrics_keep_tasks_and_rungs_separate() -> None:
    server = CTCResourcesServer(
        config=CTCResourcesServerConfig(host="127.0.0.1", port=8001, entrypoint="app.py", name="ctc"),
        server_client=MagicMock(spec=ServerClient),
    )
    rows = [
        [{"verifier_metadata": {"task": "ctc_nq", "rung": "r2k"}, "task_metrics": {"f1": 1.0}, "parse_ok": True}],
        [
            {
                "verifier_metadata": {"task": "ctc_reorder", "rung": "r2k"},
                "task_metrics": {"kendall_tau": -1.0},
                "parse_ok": True,
            }
        ],
        [{"mask_sample": True}],
    ]
    metrics = server.compute_metrics(rows)
    assert metrics["ctc_nq/r2k/f1"] == 1
    assert metrics["ctc_reorder/r2k/kendall_tau"] == -1
    assert metrics["ctc_nq/r2k/eval_size"] == 1
    assert "reward" not in metrics
    key = server.get_key_metrics(metrics)
    assert key["ctc_reorder/r2k/kendall_tau"] == -1
    assert key["ctc_nq/r2k/parse_rate"] == 1
    assert "ctc_nq/r2k/eval_size" not in key
