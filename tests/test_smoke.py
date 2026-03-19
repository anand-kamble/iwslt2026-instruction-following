"""End-to-end smoke tests using real models on CPU.

Tests the full pipeline (model construction, data collation, forward pass)
without requiring real datasets or GPU. Uses CPU with synthetic audio inputs
but real speech encoder, projector, and LLM.
"""

import numpy as np
import torch
import pytest


@pytest.fixture(scope="module")
def real_processor():
    """Load the real SeamlessM4T processor (cached across all tests in module)."""
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")


@pytest.fixture(scope="module")
def real_speech_encoder():
    """Load the real SeamlessM4T speech encoder on CPU (cached across all tests in module)."""
    from transformers import SeamlessM4Tv2Model
    full_model = SeamlessM4Tv2Model.from_pretrained(
        "facebook/seamless-m4t-v2-large", torch_dtype=torch.float32,
    )
    speech_encoder = full_model.speech_encoder
    del full_model
    speech_encoder.eval()
    for param in speech_encoder.parameters():
        param.requires_grad = False
    return speech_encoder


def _build_test_model(speech_encoder):
    """Build a minimal SpeechLLM for testing on CPU with the real speech encoder."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    from src.model.adapters import SpeechProjector
    from src.model.speech_llm import SpeechLLM, SPEECH_PLACEHOLDER

    # Use a tiny LLM for testing
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", trust_remote_code=True, torch_dtype=torch.float32,
    )

    # Add speech placeholder token
    if SPEECH_PLACEHOLDER not in tokenizer.get_vocab():
        tokenizer.add_tokens([SPEECH_PLACEHOLDER], special_tokens=True)
        llm.resize_token_embeddings(len(tokenizer))

    llm_dim = llm.config.hidden_size

    # Projector: real encoder outputs 1024-dim, downsample to keep sequence manageable
    projector = SpeechProjector(speech_dim=1024, llm_dim=llm_dim, downsample_factor=5)

    model = SpeechLLM(speech_encoder, projector, llm, tokenizer)
    return model, tokenizer


def _make_audio_inputs(processor, batch_size=1):
    """Create real processor outputs from synthetic audio waveforms."""
    # 1-second audio at 16kHz
    audios = [np.random.randn(16000).astype(np.float32) for _ in range(batch_size)]
    inputs = processor(
        audios=audios,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    return inputs["input_features"], inputs["attention_mask"]


@pytest.mark.slow
class TestForwardPass:
    """Test forward pass through SpeechLLM with real encoder."""

    def test_training_forward(self, real_speech_encoder, real_processor):
        """Forward pass with labels should return a loss."""
        model, tokenizer = _build_test_model(real_speech_encoder)
        model.eval()

        from src.data.utils import format_chat_messages

        messages = format_chat_messages("Transcribe the English content.", "Hello world.")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        text_inputs = tokenizer(text, return_tensors="pt", padding=True)

        input_features, input_features_mask = _make_audio_inputs(real_processor)

        # Build labels (mask prompt, keep response)
        labels = text_inputs["input_ids"].clone()
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_len = len(tokenizer(prompt_text)["input_ids"])
        labels[0, :prompt_len] = -100

        with torch.no_grad():
            outputs = model(
                input_features=input_features,
                input_features_attention_mask=input_features_mask,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                labels=labels,
            )

        assert outputs.loss is not None
        assert outputs.loss.item() > 0
        assert not torch.isnan(outputs.loss)

    def test_generation(self, real_speech_encoder, real_processor):
        """Generate method should produce token IDs."""
        model, tokenizer = _build_test_model(real_speech_encoder)
        model.eval()

        from src.data.utils import format_chat_messages

        messages = format_chat_messages("Transcribe the English content.")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text_inputs = tokenizer(text, return_tensors="pt")

        input_features, input_features_mask = _make_audio_inputs(real_processor)

        output_ids = model.generate(
            input_features=input_features,
            input_features_attention_mask=input_features_mask,
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            max_new_tokens=10,
        )

        assert output_ids.shape[0] == 1
        assert output_ids.shape[1] > 0

    def test_forward_with_downsampling(self, real_speech_encoder, real_processor):
        """Forward pass with downsample_factor=5 should work correctly."""
        model, tokenizer = _build_test_model(real_speech_encoder)
        model.eval()

        from src.data.utils import format_chat_messages

        messages = format_chat_messages("Transcribe the English content.", "Test output.")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        text_inputs = tokenizer(text, return_tensors="pt", padding=True)

        input_features, input_features_mask = _make_audio_inputs(real_processor)

        labels = text_inputs["input_ids"].clone()
        labels[0, :5] = -100

        with torch.no_grad():
            outputs = model(
                input_features=input_features,
                input_features_attention_mask=input_features_mask,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                labels=labels,
            )

        assert outputs.loss is not None
        assert not torch.isnan(outputs.loss)

    def test_batch_forward(self, real_speech_encoder, real_processor):
        """Forward pass with batch_size > 1."""
        model, tokenizer = _build_test_model(real_speech_encoder)
        model.eval()

        from src.data.utils import format_chat_messages

        batch_messages = [
            format_chat_messages("Transcribe the English content.", "Hello."),
            format_chat_messages("Translate to German.", "Hallo."),
        ]
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
            for m in batch_messages
        ]
        text_inputs = tokenizer(texts, return_tensors="pt", padding=True)

        input_features, input_features_mask = _make_audio_inputs(real_processor, batch_size=2)

        labels = text_inputs["input_ids"].clone()
        labels[:, :5] = -100

        with torch.no_grad():
            outputs = model(
                input_features=input_features,
                input_features_attention_mask=input_features_mask,
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                labels=labels,
            )

        assert outputs.loss is not None


class TestMultiTaskDataset:
    """Test multi-task dataset and sampler."""

    def test_multitask_dataset_indexing(self):
        """MultiTaskDataset should correctly index across tasks."""
        from torch.utils.data import Dataset as TorchDataset
        from src.data.multitask import MultiTaskDataset

        class FakeDataset(TorchDataset):
            def __init__(self, size):
                self.size = size
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                return {"text": f"sample_{idx}"}

        ds = MultiTaskDataset({
            "asr": FakeDataset(10),
            "s2tt_de": FakeDataset(5),
            "sqa_en": FakeDataset(3),
        })

        assert len(ds) == 18
        item = ds[0]
        assert item["_task"] == "asr"
        item = ds[12]
        assert item["_task"] == "s2tt_de"
        item = ds[16]
        assert item["_task"] == "sqa_en"

    def test_temperature_sampler_distribution(self):
        """TemperatureSampler should bias toward smaller datasets at low temperature."""
        from torch.utils.data import Dataset as TorchDataset
        from src.data.multitask import MultiTaskDataset, TemperatureSampler
        from collections import Counter

        class FakeDataset(TorchDataset):
            def __init__(self, size):
                self.size = size
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                return {"text": f"sample_{idx}"}

        ds = MultiTaskDataset({
            "large": FakeDataset(10000),
            "small": FakeDataset(100),
        })

        # With T=0.5, small task should be sampled more than its natural proportion
        sampler = TemperatureSampler(ds, temperature=0.5, total_samples=1000, seed=42)
        indices = list(sampler)
        assert len(indices) == 1000

        # Count task distribution
        task_counts = Counter()
        for idx in indices:
            task_name, _ = ds._index_map[idx]
            task_counts[task_name] += 1

        # Natural proportion of "small" is 100/10100 ≈ 1%
        # With T=0.5, it should be boosted significantly (expect ~10%+)
        small_ratio = task_counts["small"] / 1000
        assert small_ratio > 0.05, f"Small task ratio {small_ratio} should be boosted above natural 1%"

    def test_temperature_1_preserves_proportions(self):
        """Temperature=1.0 should approximately preserve natural proportions."""
        from torch.utils.data import Dataset as TorchDataset
        from src.data.multitask import MultiTaskDataset, TemperatureSampler
        from collections import Counter

        class FakeDataset(TorchDataset):
            def __init__(self, size):
                self.size = size
            def __len__(self):
                return self.size
            def __getitem__(self, idx):
                return {"text": f"sample_{idx}"}

        ds = MultiTaskDataset({
            "a": FakeDataset(500),
            "b": FakeDataset(500),
        })

        sampler = TemperatureSampler(ds, temperature=1.0, total_samples=2000, seed=42)
        indices = list(sampler)

        task_counts = Counter()
        for idx in indices:
            task_name, _ = ds._index_map[idx]
            task_counts[task_name] += 1

        # Should be roughly 50/50
        ratio = task_counts["a"] / 2000
        assert 0.4 < ratio < 0.6


class TestProjectorDownsampling:
    """Test speech projector with various downsample factors."""

    def test_downsample_factor_5(self):
        from src.model.adapters import SpeechProjector

        proj = SpeechProjector(speech_dim=1024, llm_dim=2560, downsample_factor=5)
        x = torch.randn(2, 100, 1024)
        out = proj(x)
        assert out.shape == (2, 20, 2560)

    def test_downsample_factor_8(self):
        from src.model.adapters import SpeechProjector

        proj = SpeechProjector(speech_dim=1024, llm_dim=2560, downsample_factor=8)
        x = torch.randn(2, 160, 1024)
        out = proj(x)
        assert out.shape == (2, 20, 2560)

    def test_long_audio_sequence(self):
        """Simulate 6-min audio: 360s × 62.5 ≈ 22500 tokens, downsample by 5 → 4500."""
        from src.model.adapters import SpeechProjector

        proj = SpeechProjector(speech_dim=1024, llm_dim=2560, downsample_factor=5)
        # Use smaller sequence to avoid OOM in tests, but verify math
        x = torch.randn(1, 500, 1024)  # 500 tokens → 100 after downsample
        out = proj(x)
        assert out.shape == (1, 100, 2560)


class TestCollator:
    """Test SpeechTextCollator modes."""

    def test_text_only_collation(self):
        """Collator should handle text-only samples (no audio)."""
        from src.data.collator import SpeechTextCollator
        from src.data.utils import format_text_only_messages
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)

        collator = SpeechTextCollator(
            processor=None,
            tokenizer=tokenizer,
            max_seq_length=512,
            text_only=True,
        )

        features = [
            {"messages": format_text_only_messages("Translate to German.", "Hallo Welt.")},
            {"messages": format_text_only_messages("Answer the question.", "Yes.")},
        ]

        batch = collator(features)
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        assert "input_features" not in batch
        assert batch["input_ids"].shape[0] == 2


class TestTransformerProjector:
    """Test the new Transformer-based projector."""

    def test_output_shape(self):
        from src.model.adapters import TransformerProjector
        proj = TransformerProjector(speech_dim=1024, llm_dim=2560, downsample_factor=3)
        x = torch.randn(2, 150, 1024)
        out = proj(x)
        assert out.shape == (2, 50, 2560)

    def test_downsample_factor_1(self):
        from src.model.adapters import TransformerProjector
        proj = TransformerProjector(speech_dim=1024, llm_dim=2560, downsample_factor=1)
        x = torch.randn(2, 100, 1024)
        out = proj(x)
        assert out.shape == (2, 100, 2560)

    def test_param_count(self):
        from src.model.adapters import TransformerProjector
        proj = TransformerProjector(speech_dim=1024, llm_dim=2560, downsample_factor=3)
        total = sum(p.numel() for p in proj.parameters())
        assert 35_000_000 < total < 45_000_000  # ~39M expected


class TestGradientFlow:
    """Test that gradients flow correctly through the merge + padding."""

    @pytest.mark.slow
    def test_projector_receives_gradients(self, real_speech_encoder, real_processor):
        """Projector must receive non-zero gradients through the LLM."""
        model, tokenizer = _build_test_model(real_speech_encoder)
        model.train()

        # Freeze everything except projector (like Stage 1)
        for param in model.speech_encoder.parameters():
            param.requires_grad = False
        for param in model.llm.parameters():
            param.requires_grad = False
        for param in model.projector.parameters():
            param.requires_grad = True

        from src.data.utils import format_chat_messages

        messages = format_chat_messages("Transcribe the English content.", "Hello world.")
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        text_inputs = tokenizer(text, return_tensors="pt", padding=True)

        input_features, input_features_mask = _make_audio_inputs(real_processor)

        labels = text_inputs["input_ids"].clone()
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_len = len(tokenizer(prompt_text)["input_ids"])
        labels[0, :prompt_len] = -100

        outputs = model(
            input_features=input_features,
            input_features_attention_mask=input_features_mask,
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            labels=labels,
        )

        outputs.loss.backward()

        # Check that projector parameters received gradients
        has_grad = False
        for name, param in model.projector.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "Projector received zero gradients — gradient flow is broken"


class TestBuildModel:
    """Test build_model factory functions."""

    def test_build_transformer_projector(self):
        from src.model.build_model import build_projector
        proj = build_projector(projector_type="transformer", downsample_factor=3, device="cpu")
        assert hasattr(proj, "transformer")
        x = torch.randn(1, 90, 1024)
        out = proj(x)
        assert out.shape == (1, 30, 2560)

    def test_build_legacy_projector(self):
        from src.model.build_model import build_projector
        proj = build_projector(projector_type="linear", downsample_factor=1, device="cpu")
        x = torch.randn(1, 100, 1024)
        out = proj(x)
        assert out.shape == (1, 100, 2560)
