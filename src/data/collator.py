"""Custom data collator for speech + text multimodal training.

Supports three modes:
- Standard: audio + text (Stage 1 and 3)
- Text-only: no audio processing (Stage 2)
- Precomputed: load pre-extracted speech features from .pt files
"""

import torch
from dataclasses import dataclass

from src.model.speech_llm import SPEECH_PLACEHOLDER


@dataclass
class SpeechTextCollator:
    """Collator that handles both audio features and tokenized text.

    Each sample in the batch should have:
        - "audio": raw audio waveform (or "precomputed_features": tensor)
        - "messages": list of chat messages

    The collator:
        1. Processes audio through the Seamless M4T v2 processor (or loads precomputed)
        2. Tokenizes messages via the LLM tokenizer
        3. Creates labels (masking everything except the assistant response)
    """

    processor: object  # SeamlessM4T processor (None for text-only)
    tokenizer: object  # Qwen tokenizer
    max_seq_length: int = 4096
    text_only: bool = False
    max_audio_seconds: int = 15  # truncate audio to avoid OOM in speech encoder

    def __post_init__(self):
        self._resamplers = {}

    def __call__(self, features: list[dict]) -> dict:
        batch = {}

        # Pass task labels through for per-task loss logging (multitask training)
        tasks = [f.get("_task") for f in features]
        if any(tasks):
            batch["_tasks"] = tasks

        # Auto-detect text-only batches: skip audio processing if no samples have audio.
        # This supports interleaved text batches in Stage 3 (NAVER IWSLT 2025).
        # Require TRUTHY values, not just key presence — MT/QA text-only datasets
        # sometimes carry stale `audio_path=None` fields from their source schemas.
        def _sample_has_audio(f: dict) -> bool:
            if "precomputed_features" in f and f["precomputed_features"] is not None:
                return True
            if f.get("audio_path"):
                return True
            if f.get("audio_array") is not None:
                return True
            audio = f.get("audio")
            if isinstance(audio, dict):
                return audio.get("array") is not None
            return audio is not None

        has_audio = not self.text_only and any(_sample_has_audio(f) for f in features)
        if has_audio:
            if "precomputed_features" in features[0]:
                # Precomputed mode: pad pre-extracted features
                feats = [f["precomputed_features"] for f in features]
                max_len = max(f.shape[0] for f in feats)
                dim = feats[0].shape[1]
                padded = torch.zeros(len(feats), max_len, dim, dtype=feats[0].dtype)
                mask = torch.zeros(len(feats), max_len, dtype=torch.long)
                for i, f in enumerate(feats):
                    padded[i, : f.shape[0]] = f
                    mask[i, : f.shape[0]] = 1
                batch["input_features"] = padded
                batch["input_features_attention_mask"] = mask
            else:
                # Raw audio mode — supports both dict format and pre-extracted arrays
                # Resample to 16kHz if needed (SeamlessM4T expects 16kHz)
                import sys
                import torchaudio
                audios = []
                kept_features = []
                for f in features:
                    try:
                        if "audio_path" in f and f["audio_path"]:
                            # Load audio from file path (e.g., SQA → LibriSpeech)
                            waveform, sr = torchaudio.load(f["audio_path"])
                            if waveform.shape[0] > 1:
                                waveform = waveform.mean(dim=0, keepdim=True)
                            audio = waveform.squeeze(0).numpy()
                        elif "audio_array" in f:
                            audio = f["audio_array"]
                            sr = f.get("sampling_rate", 16000)
                        elif isinstance(f.get("audio"), dict):
                            audio = f["audio"]["array"]
                            sr = f["audio"].get("sampling_rate", 16000)
                        else:
                            audio = f["audio"]
                            sr = 16000

                        if sr != 16000:
                            if sr not in self._resamplers:
                                self._resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
                            waveform = torch.as_tensor(audio, dtype=torch.float32).unsqueeze(0)
                            audio = self._resamplers[sr](waveform).squeeze(0).numpy()

                        # Truncate long audio to avoid OOM in speech encoder
                        max_samples = self.max_audio_seconds * 16000
                        if len(audio) > max_samples:
                            audio = audio[:max_samples]

                        audios.append(audio)
                        kept_features.append(f)
                    except Exception as e:
                        try:
                            err = repr(e)
                        except Exception:
                            err = f"{type(e).__name__}(<repr failed>)"
                        path = f.get("audio_path", "<embedded>")
                        print(
                            f"[collator] dropping sample: task={f.get('_task')!r} "
                            f"path={path!r} err={err}",
                            file=sys.stderr,
                            flush=True,
                        )

                if not audios:
                    # Every audio sample failed to load. This happens when a
                    # text-only task (mt_*, qa_*) carries stale `audio_path`
                    # strings inherited from its source schema. Fall back to
                    # treating the batch as text-only so training — and DDP
                    # collectives — stay in sync across ranks.
                    print(
                        "[collator] all audio samples failed to load; "
                        f"falling back to text-only (tasks={set(tasks) if tasks else set()})",
                        file=sys.stderr,
                        flush=True,
                    )
                    has_audio = False
                else:
                    if len(kept_features) != len(features):
                        features = kept_features
                        tasks = [f.get("_task") for f in features]
                        if any(tasks):
                            batch["_tasks"] = tasks

                    audio_inputs = self.processor(
                        audios=audios,
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True,
                        return_attention_mask=True,
                    )
                    batch["input_features"] = audio_inputs["input_features"]
                    batch["input_features_attention_mask"] = audio_inputs["attention_mask"]

        # Tokenize text (full conversation including response)
        texts = []
        for f in features:
            text = self.tokenizer.apply_chat_template(
                f["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)

        text_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
        )

        batch["input_ids"] = text_inputs["input_ids"]
        batch["attention_mask"] = text_inputs["attention_mask"]

        # Create labels: mask everything except assistant response
        # Find the assistant response boundary by searching for the assistant
        # header token sequence in the tokenized IDs, avoiding re-tokenization
        # boundary mismatches.
        assistant_header = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        header_len = len(assistant_header)

        labels = text_inputs["input_ids"].clone()
        for i in range(len(features)):
            ids = text_inputs["input_ids"][i].tolist()
            # Find the LAST assistant header (in case of multi-turn)
            prompt_end = 0
            for j in range(len(ids) - header_len + 1):
                if ids[j : j + header_len] == assistant_header:
                    prompt_end = j + header_len
            # Mask everything up to and including the assistant header
            labels[i, :prompt_end] = -100

        # Mask padding in labels
        labels[text_inputs["attention_mask"] == 0] = -100
        batch["labels"] = labels

        return batch
