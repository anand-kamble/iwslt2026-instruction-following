"""Speech-to-Text Translation dataset loading and preprocessing.

Permitted datasets (constrained): EuroParlST, GigaST
Target languages: DE, IT, ZH
"""

from datasets import load_dataset

from src.data.utils import TASK_PROMPTS, format_chat_messages


def load_europarlst_s2tt(target_lang: str, split: str = "train"):
    """Load EuroParlST for speech translation (EN → target_lang).

    Uses CLAPv2/Europarl-st English audio samples and matches them with
    translations from tj-solergibert/Europarl-ST by English transcription text.
    """
    lang_map = {"de": "de", "it": "it"}
    if target_lang not in lang_map:
        raise ValueError(f"EuroParlST doesn't support EN→{target_lang}")

    from src.data.asr import _extract_transcription

    # Load audio dataset (English samples only)
    audio_ds = load_dataset("CLAPv2/Europarl-st", split=split)
    audio_ds = audio_ds.filter(
        lambda ex: ex["text"].startswith("A person is saying:"),
        num_proc=8,
    )

    # Load text dataset and build lookup: english_text -> target translation
    text_ds = load_dataset("tj-solergibert/Europarl-ST", split=split)
    translation_lookup = {}
    for ex in text_ds:
        transcriptions = ex.get("transcriptions", {})
        en_text = (transcriptions.get("en") or "").strip()
        tgt_text = (transcriptions.get(target_lang) or "").strip()
        if en_text and tgt_text:
            translation_lookup[en_text] = tgt_text

    # Match audio samples with translations by English transcription
    def _add_translation(example):
        en_text = _extract_transcription(example["text"]).strip()
        translation = translation_lookup.get(en_text)
        example["_translation"] = translation or ""
        return example

    audio_ds = audio_ds.map(_add_translation)
    audio_ds = audio_ds.filter(lambda ex: bool(ex["_translation"]))

    def _format(example):
        instruction = TASK_PROMPTS["s2tt"][target_lang]
        example["messages"] = format_chat_messages(instruction, example["_translation"])
        return example

    return audio_ds.map(_format)


def load_s2tt_dataset(dataset_name: str, target_lang: str, split: str = "train"):
    """Load a speech translation dataset by name and target language."""
    if dataset_name == "gigast":
        from src.data.gigast import load_gigast_s2tt

        return load_gigast_s2tt(target_lang, split)

    if dataset_name == "covost2":
        from src.data.covost2 import load_covost2_s2tt

        return load_covost2_s2tt(target_lang, split)

    loaders = {
        "europarlst": load_europarlst_s2tt,
    }
    if dataset_name not in loaders:
        raise ValueError(
            f"Unknown S2TT dataset: {dataset_name}. "
            f"Available: {list(loaders.keys()) + ['gigast', 'covost2']}"
        )
    return loaders[dataset_name](target_lang, split)
