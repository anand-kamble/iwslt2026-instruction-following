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


# Task prompts following NAVER IWSLT 2025 (arxiv 2506.01808, Table 4).
# Prompt text only — structural prefixes (Content:, Question:, Your answer:)
# are added by format_chat_messages / format_text_only_messages.
TASK_PROMPTS = {
    "asr": {
        "en": "Can you transcribe the Speech content into English text?",
    },
    "s2tt": {
        # Exact NAVER prompts (Table 4): translated via SeamlessM4T-v2-large
        "de": "Können Sie den Inhalt der Rede in den deutschen Text übersetzen?",
        "it": "Puoi tradurre il contenuto del discorso in testo italiano?",
        "zh": "你能把演讲内容翻译成中文吗?",
    },
    "sqa": {
        # NAVER SQA: question IS the instruction (Table 4)
        "en": "{QUESTION}",
        "de": "{QUESTION}",
        "it": "{QUESTION}",
        "zh": "{QUESTION}",
    },
    "s2tsum": {
        "en": "Can you summarize the Speech content in an abstract of approximately 200 words?",
        "de": "Können Sie den Inhalt der Rede in einem Abstract von ungefähr 200 Wörtern zusammenfassen?",
        "it": "Puoi riassumere il contenuto del discorso in un abstract di circa 200 parole?",
        "zh": "你能用大约400个字概括演讲内容吗?",
    },
    "achap": {
        "en": "Can you identify the chapters and their titles in the Speech content?",
        "de": "Können Sie die Kapitel und ihre Titel im Inhalt der Rede identifizieren?",
        "it": "Puoi identificare i capitoli e i loro titoli nel contenuto del discorso?",
        "zh": "你能识别演讲内容中的章节及其标题吗?",
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


def format_text_only_messages(
    instruction: str, response: str | None = None, source_text: str | None = None,
) -> list[dict]:
    """Format text-only instruction as ChatML messages (NAVER Table 4 format).

    Used for Stage 2 text-only LoRA training and Stage 3 text interleaving.
    Follows NAVER IWSLT 2025 (arxiv 2506.01808) template:
        Content: <text>[source text]</text>
        Question: [instruction]
        Your answer:

    Args:
        instruction: Task instruction (e.g., translation prompt, question).
        response: Expected output (assistant turn). None for inference.
        source_text: English transcript or context for <text> tags.
            If provided, wrapped as Content: <text>...</text>.
    """
    if source_text:
        content = f"Content: <text>{source_text}</text>\nQuestion: {instruction}\nYour answer:"
    else:
        content = f"Question: {instruction}\nYour answer:"
    messages = [{"role": "user", "content": content}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages


def format_chat_messages(instruction: str, response: str | None = None) -> list[dict]:
    """Format speech + instruction as ChatML messages (NAVER Table 4 format).

    Follows NAVER IWSLT 2025 (arxiv 2506.01808) prompt template:
        Content: <speech>[SPEECH EMBEDDINGS]</speech>
        Question: [instruction]
        Your answer:

    The <speech>/<text> boundary tags and Content:/Question:/Your answer:
    structural prefixes give the model consistent parsing cues across tasks.
    The <|speech|> placeholder is replaced with speech embeddings in
    _merge_speech_and_text; boundary tags stay as regular token embeddings.
    No system prompt — matches MCIF evaluation and Qwen3 defaults.
    """
    content = (
        f"Content: <speech>{SPEECH_PLACEHOLDER}</speech>\n"
        f"Question: {instruction}\n"
        f"Your answer:"
    )
    messages = [{"role": "user", "content": content}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages
