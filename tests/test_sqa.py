import json
from pathlib import Path

import pytest

from src.data import sqa


class FakeDataset:
    def __init__(self, rows):
        self.rows = rows

    def map(self, fn, with_indices=False):
        mapped = []
        for idx, row in enumerate(self.rows):
            example = dict(row)
            if with_indices:
                mapped.append(fn(example, idx))
            else:
                mapped.append(fn(example))
        return FakeDataset(mapped)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _make_flac(tmp_path: Path, speech_path: str):
    flac_path = tmp_path / Path(speech_path).with_suffix(".flac")
    flac_path.parent.mkdir(parents=True, exist_ok=True)
    flac_path.write_bytes(b"fake flac")
    return flac_path


def test_english_sqa_uses_flac_fallback(monkeypatch, tmp_path):
    speech_path = "LibriSpeech/train-clean-360/1/2/1-2-0000.wav"
    expected_audio = _make_flac(tmp_path, speech_path)

    monkeypatch.setattr(sqa, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        sqa,
        "load_librisqa_source_dataset",
        lambda split="train": FakeDataset(
            [{"question": "What is said?", "answer": "Hello", "speech_path": speech_path}]
        ),
    )

    ds = sqa.load_sqa_dataset("train", "en")
    item = ds[0]

    assert item["audio_path"] == str(expected_audio)
    assert item["messages"][1]["content"] == "Hello"


def test_translated_sqa_uses_train_split_and_flac_fallback(monkeypatch, tmp_path):
    speech_path = "LibriSpeech/train-clean-360/3/4/3-4-0001.wav"
    expected_audio = _make_flac(tmp_path, speech_path)

    translated_dir = tmp_path / "sqa_translated"
    translated_dir.mkdir(parents=True)
    with open(translated_dir / "librisqa_de.jsonl", "w", encoding="utf-8") as f:
        f.write(
            json.dumps({"id": 0, "question": "Was wird gesagt?", "answer": "Hallo"})
            + "\n"
        )

    monkeypatch.setattr(sqa, "get_data_dir", lambda: tmp_path)
    monkeypatch.setattr(
        sqa,
        "load_librisqa_source_dataset",
        lambda split="train": FakeDataset(
            [{"question": "What is said?", "answer": "Hello", "speech_path": speech_path}]
        ),
    )

    ds = sqa.load_translated_sqa_dataset("train", "de")
    item = ds[0]

    assert item["audio_path"] == str(expected_audio)
    assert item["messages"][1]["content"] == "Hallo"
    assert "Was wird gesagt?" in item["messages"][0]["content"]


def test_translated_sqa_rejects_non_train_split(tmp_path):
    translated_dir = tmp_path / "sqa_translated"
    translated_dir.mkdir(parents=True)
    (translated_dir / "librisqa_de.jsonl").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="only the train split"):
        sqa.load_translated_sqa_dataset("test", "de")
