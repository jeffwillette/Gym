# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stateless verifier for all 22 published CTC-Bench tasks.

Prompt and scoring reference: https://github.com/allenai/olmo-eval/tree/69729fadd898e50e383e42803bb9a6250eb1dbbf/src/olmo_eval/evals/tasks/ctc_suite
Revision: 69729fadd898e50e383e42803bb9a6250eb1dbbf; Apache-2.0.
The evaluation package documents the source snapshot and local adaptations.
Responses are graded with the task-specific parser, stopping rule, and primary metric;
metrics are reported separately by task and context rung.
"""

from collections import defaultdict
from statistics import mean

from pydantic import BaseModel, JsonValue

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from resources_servers.corpustaskcomplexity.contract import ROSTER, get_spec, score_answer


class CTCResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE = ReverifyMode.STATELESS


class CTCVerifierMetadata(BaseModel):
    task: str
    rung: str
    example: dict[str, JsonValue]


class CTCVerifyRequest(BaseVerifyRequest):
    verifier_metadata: CTCVerifierMetadata


class CTCVerifyResponse(CTCVerifyRequest, BaseVerifyResponse):
    primary_metric: str
    task_metrics: dict[str, float]
    parse_ok: bool


class CTCResourcesServer(SimpleResourcesServer):
    config: CTCResourcesServerConfig
    ray_enabled = False

    async def verify(self, body: CTCVerifyRequest) -> CTCVerifyResponse:
        metadata = body.verifier_metadata
        scores, parse_ok = score_answer(metadata.task, body.response.output_text or "", metadata.example)
        primary = get_spec(metadata.task).primary_metric
        return CTCVerifyResponse(
            **body.model_dump(),
            reward=scores[primary],
            primary_metric=primary,
            task_metrics=scores,
            parse_ok=parse_ok,
        )

    def compute_metrics(self, tasks: list[list[dict]]) -> dict[str, float]:
        """Keep heterogeneous primary metrics separate for each task and rung."""
        subsets = defaultdict(list)
        for repeats in tasks:
            for row in repeats:
                if row.get("mask_sample", False):
                    continue
                metadata = row["verifier_metadata"]
                subsets[(metadata["task"], metadata["rung"])].append(row)
        result = {}
        for (task, rung), rows in subsets.items():
            prefix = f"{task}/{rung}"
            result[f"{prefix}/eval_size"] = float(len(rows))
            result[f"{prefix}/parse_rate"] = mean(float(row["parse_ok"]) for row in rows)
            for metric in sorted({key for row in rows for key in row["task_metrics"]}):
                result[f"{prefix}/{metric}"] = mean(
                    row["task_metrics"][metric] for row in rows if metric in row["task_metrics"]
                )
        return result

    def get_key_metrics(self, agent_metrics: dict) -> dict:
        """Show the primary score and parse rate for each task/rung."""
        return {
            key: value
            for key, value in agent_metrics.items()
            if key.endswith("/parse_rate")
            or any(key.startswith(f"{task}/") and key.endswith(f"/{get_spec(task).primary_metric}") for task in ROSTER)
        }


if __name__ == "__main__":
    CTCResourcesServer.run_webserver()
