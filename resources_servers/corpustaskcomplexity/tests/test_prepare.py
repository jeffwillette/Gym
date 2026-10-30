# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from benchmarks.corpustaskcomplexity.prepare import convert_example, prepare


def test_conversion_preserves_prompt_and_metadata() -> None:
    data = {"queries": ["capital?"], "documents": [{"text": "Paris"}], "gold_doc_indices": [0]}
    row = convert_example(task="ctc_nq", rung="r2k", index=4, example=data)
    assert row["id"] == "ctc_nq:r2k:4"
    assert row["verifier_metadata"]["example"] == data
    assert "Paris" in row["responses_create_params"]["input"][0]["content"]
    assert row["responses_create_params"]["temperature"] == 0
    assert row["responses_create_params"]["max_output_tokens"] == 64


def test_invalid_task_rung() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        convert_example(task="ctc_absence", rung="r1m", index=0, example={})


def test_local_parquet_preparation(tmp_path: Path) -> None:
    from omegaconf import OmegaConf

    source = tmp_path / "source/data/nq"
    source.mkdir(parents=True)
    data = {"queries": ["capital?"], "documents": [{"text": "Paris"}], "gold_doc_indices": [0]}
    pq.write_table(pa.Table.from_pylist([data, data]), source / "r2k.parquet")
    output = tmp_path / "output"
    # Native Gym CLI passes OmegaConf lists and string paths.
    result = prepare(
        tasks=OmegaConf.create(["ctc_nq"]),
        rungs=OmegaConf.create(["r2k"]),
        output_dir=str(output),
        local_data_root=str(tmp_path / "source"),
    )
    assert len(result.read_text().splitlines()) == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["complete"] and manifest["total_rows"] == 2
    assert manifest["files"][0]["overlength"] == 0


def test_resume_reuses_completed_rungs_and_rejects_changed_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source/data/nq"
    source.mkdir(parents=True)
    data = {"queries": ["capital?"], "documents": [{"text": "Paris"}], "gold_doc_indices": [0]}
    pq.write_table(pa.Table.from_pylist([data, data]), source / "r2k.parquet")
    options = {
        "tasks": ["ctc_nq"],
        "rungs": ["r2k"],
        "output_dir": tmp_path / "output",
        "local_data_root": tmp_path / "source",
    }
    result = prepare(**options)
    original = result.read_text()

    def unexpected_read(path: Path) -> None:
        raise AssertionError("Completed input should not be read again")

    monkeypatch.setattr("benchmarks.corpustaskcomplexity.prepare.read_examples", unexpected_read)
    assert prepare(**options, resume=True).read_text() == original
    with pytest.raises(ValueError, match="setting limit_per_rung changed"):
        prepare(**options, resume=True, limit=1)


def test_batch_tokenization_and_overlength_are_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from transformers import AutoTokenizer

    source = tmp_path / "source/data/nq"
    source.mkdir(parents=True)
    small = {"queries": ["capital?"], "documents": [{"text": "Paris"}], "gold_doc_indices": [0]}
    large = {"queries": ["capital?"], "documents": [{"text": "LARGE"}], "gold_doc_indices": [0]}
    pq.write_table(pa.Table.from_pylist([small] * 17 + [large]), source / "r2k.parquet")
    batch_sizes = []

    class Tokenizer:
        def __call__(self, prompts: list[str], **kwargs: object) -> dict:
            batch_sizes.append(len(prompts))
            assert kwargs["add_special_tokens"] is False
            return {"input_ids": [[1] * (500 if "LARGE" in prompt else 100) for prompt in prompts]}

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: Tokenizer())
    options = {
        "tasks": ["ctc_nq"],
        "rungs": ["r2k"],
        "output_dir": tmp_path / "output",
        "local_data_root": tmp_path / "source",
        "tokenizer_name": "test",
        "max_model_len": 400,
    }
    with pytest.raises(ValueError, match="1 rows exceed"):
        prepare(**options)
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    assert not manifest["complete"]
    assert batch_sizes == [16, 2]
    result = prepare(**options, allow_overlength=True)
    assert len(result.read_text().splitlines()) == 17
    manifest = json.loads((tmp_path / "output/manifest.json").read_text())
    assert manifest["complete"] and manifest["files"][0]["overlength"] == 1


@pytest.mark.parametrize("kwargs", [{"tasks": []}, {"rungs": ["bad"]}, {"limit": 0}, {"max_model_len": 2048}])
def test_invalid_preparation_options(tmp_path: Path, kwargs: dict) -> None:
    with pytest.raises(ValueError):
        prepare(output_dir=tmp_path, local_data_root=tmp_path, **kwargs)
