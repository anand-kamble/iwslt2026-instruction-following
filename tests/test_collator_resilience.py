import pytest
import torch

from src.data.collator import SpeechTextCollator


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        _ = (tokenize, add_generation_prompt)
        return " ".join(m.get("content", "") for m in messages)

    def __call__(self, texts, return_tensors="pt", padding=True, truncation=True, max_length=4096):
        _ = (return_tensors, padding, truncation, max_length)
        # Include assistant header token id (99) so label masking path is exercised.
        rows = [[10, 99, 20] for _ in texts]
        input_ids = torch.tensor(rows, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, text, add_special_tokens=False):
        _ = (text, add_special_tokens)
        return [99]


class _FakeProcessor:
    def __call__(
        self,
        audios,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    ):
        _ = (sampling_rate, return_tensors, padding, return_attention_mask)
        bsz = len(audios)
        features = torch.zeros((bsz, 5, 160), dtype=torch.float32)
        mask = torch.ones((bsz, 5), dtype=torch.long)
        return {"input_features": features, "attention_mask": mask}


def _messages():
    return [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": "Answer"},
    ]


def test_collator_drops_bad_audio_samples(monkeypatch):
    import torchaudio

    def fake_load(path):
        if "bad" in path:
            raise RuntimeError("corrupt file")
        return torch.zeros((1, 32000), dtype=torch.float32), 16000

    monkeypatch.setattr(torchaudio, "load", fake_load)

    collator = SpeechTextCollator(
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
        text_only=False,
    )
    features = [
        {"audio_path": "/tmp/bad-1.wav", "messages": _messages(), "_task": "asr"},
        {"audio_path": "/tmp/good.wav", "messages": _messages(), "_task": "asr"},
        {"audio_path": "/tmp/bad-2.wav", "messages": _messages(), "_task": "asr"},
    ]

    batch = collator(features)

    assert batch["input_features"].shape[0] == 1
    assert batch["input_ids"].shape[0] == 1
    assert batch["_tasks"] == ["asr"]


def test_collator_raises_when_all_audio_samples_fail(monkeypatch):
    import torchaudio

    def fake_load(_path):
        raise RuntimeError("corrupt file")

    monkeypatch.setattr(torchaudio, "load", fake_load)

    collator = SpeechTextCollator(
        processor=_FakeProcessor(),
        tokenizer=_FakeTokenizer(),
        text_only=False,
    )
    features = [
        {"audio_path": "/tmp/bad-1.wav", "messages": _messages(), "_task": "asr"},
        {"audio_path": "/tmp/bad-2.wav", "messages": _messages(), "_task": "asr"},
    ]

    with pytest.raises(RuntimeError, match="all audio samples in batch failed to load"):
        collator(features)
