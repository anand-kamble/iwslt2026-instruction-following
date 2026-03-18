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

    def __post_init__(self):
        self._resamplers = {}

    def __call__(self, features: list[dict]) -> dict:
        batch = {}

        if not self.text_only:
            if "precomputed_features" in features[0]:
                # Precomputed mode: pad pre-extracted features
                feats = [f["precomputed_features"] for f in features]
                max_len = max(f.shape[0] for f in feats)
                dim = feats[0].shape[1]
                padded = torch.zeros(len(feats), max_len, dim)
                mask = torch.zeros(len(feats), max_len, dtype=torch.long)
                for i, f in enumerate(feats):
                    padded[i, : f.shape[0]] = f
                    mask[i, : f.shape[0]] = 1
                batch["input_features"] = padded
                batch["input_features_attention_mask"] = mask
            else:
                # Raw audio mode — supports both dict format and pre-extracted arrays
                # Resample to 16kHz if needed (SeamlessM4T expects 16kHz)
                import torchaudio
                audios = []
                for f in features:
                    if "audio_array" in f:
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
                        waveform = torch.tensor(audio).unsqueeze(0)
                        audio = self._resamplers[sr](waveform).squeeze(0).numpy()

                    audios.append(audio)

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
        labels = text_inputs["input_ids"].clone()
        for i, f in enumerate(features):
            prompt_messages = [m for m in f["messages"] if m["role"] != "assistant"]
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = self.tokenizer(
                prompt_text, truncation=True, max_length=self.max_seq_length
            )["input_ids"]
            prompt_len = len(prompt_ids)

            # Mask prompt tokens in labels
            labels[i, :prompt_len] = -100

        # Mask padding in labels
        labels[text_inputs["attention_mask"] == 0] = -100
        batch["labels"] = labels

        return batch
