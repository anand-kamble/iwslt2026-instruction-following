"""Speech-to-Text Summarization dataset loading and preprocessing.

Permitted dataset: NUTSHELL

Uses lazy audio loading: the HF dataset is memory-mapped from Arrow files,
audio stays on disk and is decoded per-sample in __getitem__. No .map() or
.filter() is called on the audio column, avoiding Arrow int32 overflow.
"""

from datasets import load_dataset

from src.data.lazy_audio import LazyAudioDataset
from src.data.utils import TASK_PROMPTS, format_chat_messages


def load_s2tsum_dataset(split: str = "train", language: str = "en",
                        max_samples: int = None):
    """Load the NUTSHELL summarization dataset with lazy audio loading.

    Args:
        split: dataset split
        language: target language for the prompt
        max_samples: limit number of samples (applied before audio loading)
    """
    ds = load_dataset("maikezu/nutshell", split=split, trust_remote_code=True)

    instruction = TASK_PROMPTS["s2tsum"][language]

    # Build index list without calling .filter() (avoids Arrow overflow)
    indices = list(range(len(ds)))
    if max_samples and len(indices) > max_samples:
        indices = indices[:max_samples]

    def format_fn(sample):
        summary = sample.get("abstract") or sample.get("summary") or ""
        return {"messages": format_chat_messages(instruction, summary)}

    print(f"  S2TSUM/{language}: {len(indices)} samples (lazy audio)")
    return LazyAudioDataset(ds, indices, format_fn, audio_key="audio")
