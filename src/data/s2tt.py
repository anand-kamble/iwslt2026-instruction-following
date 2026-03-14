"""Speech-to-Text Translation dataset loading and preprocessing.

Permitted datasets: EuroParlST, CoVoST2, GigaST
Target languages: DE, IT, ZH
"""

from datasets import load_dataset

from src.data.utils import TASK_PROMPTS, format_chat_messages


def load_europarlst_s2tt(target_lang: str, split: str = "train"):
    """Load EuroParlST for speech translation (EN → target_lang)."""
    lang_map = {"de": "en-de", "it": "en-it"}
    if target_lang not in lang_map:
        raise ValueError(f"EuroParlST doesn't support EN→{target_lang}")

    ds = load_dataset(
        "facebook/europarl_st", lang_map[target_lang], split=split, trust_remote_code=True
    )

    def _format(example):
        instruction = TASK_PROMPTS["s2tt"][target_lang]
        example["messages"] = format_chat_messages(instruction, example["text"])
        return example

    return ds.map(_format)


def load_covost2_s2tt(target_lang: str, split: str = "train"):
    """Load CoVoST2 for speech translation (EN → target_lang)."""
    ds = load_dataset(
        "facebook/covost2", f"en_{target_lang}", split=split, trust_remote_code=True
    )

    def _format(example):
        instruction = TASK_PROMPTS["s2tt"][target_lang]
        example["messages"] = format_chat_messages(instruction, example["translation"])
        return example

    return ds.map(_format)


def load_s2tt_dataset(dataset_name: str, target_lang: str, split: str = "train"):
    """Load a speech translation dataset by name and target language."""
    if dataset_name == "gigast":
        from src.data.gigast import load_gigast_s2tt
        return load_gigast_s2tt(target_lang, split)

    loaders = {
        "europarlst": load_europarlst_s2tt,
        "covost2": load_covost2_s2tt,
    }
    if dataset_name not in loaders:
        raise ValueError(
            f"Unknown S2TT dataset: {dataset_name}. Available: {list(loaders.keys()) + ['gigast']}"
        )
    return loaders[dataset_name](target_lang, split)
