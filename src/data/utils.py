"""Shared data loading utilities."""

from __future__ import annotations

from pathlib import Path


def get_data_dir() -> Path:
    """Return the project data directory."""
    return Path(__file__).parent.parent.parent / "data"


def load_audio(path: str | Path, target_sr: int = 16000):
    """Load an audio file and resample to target sample rate.

    Returns:
        (waveform, sample_rate) — waveform is (1, num_samples)
    """
    import torch
    import torchaudio

    waveform, sr = torchaudio.load(str(path))
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    # Mix to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform, target_sr


# MCIF fixed prompts per task and language
TASK_PROMPTS = {
    "asr": {
        "en": "Transcribe the English content.",
    },
    "s2tt": {
        "de": "Übersetze den englischem Inhalt nach Deutsch.",
        "it": "Traduci il contenuto inglese in italiano.",
        "zh": "将英文内容翻译成中文。",
    },
    "sqa": {
        "en": "Answer the following question given the English content: {QUESTION}",
        "de": "Beantworte die folgende Frage basierend auf dem englischen Inhalt: {QUESTION}",
        "it": "Rispondi alla seguente domanda dato il contenuto inglese: {QUESTION}",
        "zh": "根据所给的英文内容，回答以下问题： {QUESTION}",
    },
    "s2tsum": {
        "en": "Summarize the English content in an abstract of approximately 200 words.",
        "de": "Fasse den englischen Inhalt in einem Abstract mit maximal 200 Wörtern zusammen.",
        "it": "Riassumi il contenuto inglese in un abstract di circa 200 parole.",
        "zh": "用400个字左右概括所给的英语内容。",
    },
    "achap": {
        "en": "Identify the chapters and their titles in the English content.",
        "de": "Identifiziere die Kapitel und ihre Titel im englischen Inhalt.",
        "it": "Identifica i capitoli e i loro titoli nel contenuto inglese.",
        "zh": "识别英文内容中的章节及其标题。",
    },
}

# Unanswerable responses for SQA
UNANSWERABLE = {
    "en": "Not answerable.",
    "de": "Nicht zu beantworten.",
    "it": "Non è possibile rispondere.",
    "zh": "无法回答。",
}

# Speech placeholder token
SPEECH_PLACEHOLDER = "<|speech|>"


def format_text_only_messages(instruction: str, response: str | None = None) -> list[dict]:
    """Format instruction and response as ChatML messages WITHOUT speech placeholder.

    Used for Stage 2 text-only LoRA training.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that follows instructions."},
        {"role": "user", "content": instruction},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages


def format_chat_messages(instruction: str, response: str | None = None) -> list[dict]:
    """Format instruction and response as ChatML messages.

    The instruction should already contain SPEECH_PLACEHOLDER where audio goes.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that processes speech and follows instructions."},
        {"role": "user", "content": f"{SPEECH_PLACEHOLDER}\n{instruction}"},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages
