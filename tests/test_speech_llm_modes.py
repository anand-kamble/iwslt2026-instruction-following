import types

import torch.nn as nn

from src.model.speech_llm import SpeechLLM


class _DummySpeechEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.config = types.SimpleNamespace(
            hidden_size=1024,
            adaptor_kernel_size=3,
            adaptor_stride=2,
            num_adapter_layers=1,
        )


class _DummyLLM(nn.Module):
    def __init__(self, trainable: bool):
        super().__init__()
        # Match SpeechLLM._get_embed_tokens() expectation.
        self.model = types.SimpleNamespace(embed_tokens=nn.Embedding(16, 8))
        self.adapter = nn.Linear(8, 8)
        if not trainable:
            for p in self.parameters():
                p.requires_grad = False


class _DummyTokenizer:
    pass


def _build_model(*, llm_trainable: bool) -> SpeechLLM:
    speech = _DummySpeechEncoder()
    for p in speech.parameters():
        p.requires_grad = False
    speech.eval()

    llm = _DummyLLM(trainable=llm_trainable)
    projector = nn.Linear(8, 8)
    return SpeechLLM(speech, projector, llm, _DummyTokenizer())


def test_train_keeps_frozen_speech_encoder_in_eval_mode():
    model = _build_model(llm_trainable=True)
    model.train()

    assert model.speech_encoder.training is False
    assert model.llm.training is True


def test_train_keeps_fully_frozen_llm_in_eval_mode():
    model = _build_model(llm_trainable=False)
    model.train()

    assert model.speech_encoder.training is False
    assert model.llm.training is False
