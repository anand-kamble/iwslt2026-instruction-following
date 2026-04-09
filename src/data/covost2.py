"""CoVoST2 dataset loading and preprocessing.

Uses fixie-ai/covost2 on HuggingFace (pre-packaged with embedded audio).
No manual download required — audio is streamed from HF and cached.

Language pairs: en-de, en-zh (Simplified Chinese).
"""

from datasets import load_dataset

from src.data.utils import TASK_PROMPTS, format_chat_messages, format_text_only_messages

COVOST2_LANG_CONFIGS = {
    "de": "en_de",
    "zh": "en_zh-CN",
}

_HF_REPO = "fixie-ai/covost2"


def load_covost2_asr(split: str = "train", max_samples: int = None):
    """Load CoVoST2 for ASR (English transcription).

    Loads from en_de config — English audio and transcription are
    identical across all target-language configs.
    """
    ds = load_dataset(_HF_REPO, "en_de", split=split)

    if max_samples is not None and len(ds) > max_samples:
        ds = ds.select(range(max_samples))

    def _format(example):
        instruction = TASK_PROMPTS["asr"]["en"]
        example["messages"] = format_chat_messages(instruction, example["sentence"])
        return example

    ds = ds.map(_format, num_proc=8)
    print(f"CoVoST2 ASR: {len(ds)} samples")
    return ds


def load_covost2_s2tt(target_lang: str, split: str = "train"):
    """Load CoVoST2 for speech translation (EN -> target_lang).

    Supported targets: de, zh.
    """
    if target_lang not in COVOST2_LANG_CONFIGS:
        raise ValueError(
            f"CoVoST2 doesn't support EN->{target_lang}. "
            f"Available: {list(COVOST2_LANG_CONFIGS.keys())}"
        )

    ds = load_dataset(_HF_REPO, COVOST2_LANG_CONFIGS[target_lang], split=split)

    def _format(example):
        instruction = TASK_PROMPTS["s2tt"][target_lang]
        example["messages"] = format_chat_messages(instruction, example["translation"])
        return example

    ds = ds.map(_format, num_proc=8)
    print(f"CoVoST2 S2TT en-{target_lang}: {len(ds)} samples")
    return ds


def load_mt_covost2(target_lang: str, split: str = "train"):
    """Load CoVoST2 text-only MT pairs (EN sentence -> target translation).

    No audio loaded. Supported targets: de, zh.
    """
    if target_lang not in COVOST2_LANG_CONFIGS:
        raise ValueError(f"CoVoST2 text doesn't support EN->{target_lang}")

    ds = load_dataset(_HF_REPO, COVOST2_LANG_CONFIGS[target_lang], split=split)

    if "audio" in ds.column_names:
        ds = ds.remove_columns(["audio"])

    def _format(example):
        instruction = TASK_PROMPTS["s2tt"][target_lang]
        example["messages"] = format_text_only_messages(
            instruction,
            example["translation"],
            source_text=example["sentence"],
        )
        return example

    ds = ds.map(_format, num_proc=8)
    print(f"CoVoST2 MT en-{target_lang}: {len(ds)} samples")
    return ds
