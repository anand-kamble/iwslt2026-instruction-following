"""Audio Chaptering dataset loading and preprocessing.

Permitted dataset: YTSeg (retkowski/ytseg)
Output format: [CSTART] Title [CEND] section text...

YTSeg configs:
  - "text": transcripts only (no audio blob), ~1.86 GB
  - "audio": embedded 16kHz mono audio, ~68 GB
  - "titles": per-chapter title generation, ~726 MB

Key fields:
  - target_text_ref: structured "[CSTART] Title [CEND] text..." output
  - chapter_titles: list of chapter title strings
  - chapter_timestamps: list of start times (seconds)

Uses lazy audio loading: the HF dataset is memory-mapped from Arrow files,
audio stays on disk and is decoded per-sample in __getitem__. No .map() or
.filter() is called on the audio column, avoiding Arrow int32 overflow.
"""

from datasets import load_dataset

from src.data.lazy_audio import LazyAudioDataset
from src.data.utils import TASK_PROMPTS, format_chat_messages


def load_achap_dataset(split: str = "train", language: str = "en",
                       use_audio: bool = False, max_samples: int = None):
    """Load the YTSeg audio chaptering dataset with lazy audio loading.

    Args:
        split: dataset split ("train", "validation", "test")
        language: target language for the prompt
        use_audio: if True, load "audio" config (68GB); otherwise "text" config
        max_samples: limit number of samples (applied before audio loading)
    """
    config_name = "audio" if use_audio else "text"
    ds = load_dataset("retkowski/ytseg", config_name, split=split)

    instruction = TASK_PROMPTS["achap"][language]

    # Build valid indices without calling .filter() (avoids Arrow overflow).
    # Access column-first (ds["col"][i]) to read only text, not audio.
    # ds["column"] reads one Arrow column without decoding audio.
    target_refs = ds["target_text_ref"]
    indices = [i for i, ref in enumerate(target_refs) if ref]
    if max_samples and len(indices) > max_samples:
        indices = indices[:max_samples]

    def format_fn(sample):
        return {"messages": format_chat_messages(instruction, sample["target_text_ref"])}

    audio_key = "audio" if use_audio else None
    print(f"  ACHAP/{language}: {len(indices)} samples (lazy audio={use_audio})")
    return LazyAudioDataset(ds, indices, format_fn, audio_key=audio_key or "_no_audio_")
