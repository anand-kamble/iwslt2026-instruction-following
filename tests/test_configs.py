"""Smoke tests for config loading and basic module imports."""

from pathlib import Path

import yaml

CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def test_all_configs_parse():
    """Every YAML config should parse without errors."""
    for config_path in CONFIGS_DIR.glob("*.yaml"):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config is not None, f"{config_path.name} parsed as None"


def test_base_config_has_model():
    """Base config must specify both SFM and LLM."""
    with open(CONFIGS_DIR / "base.yaml") as f:
        config = yaml.safe_load(f)
    assert "model" in config
    assert "sfm" in config["model"]
    assert "llm" in config["model"]
    assert "seamless" in config["model"]["sfm"]
    assert "Qwen3" in config["model"]["llm"]


def test_task_configs_have_required_fields():
    """Each task config must have task, data, and training sections."""
    task_files = ["asr.yaml", "s2tt.yaml", "sqa.yaml", "s2tsum.yaml", "achap.yaml"]
    for filename in task_files:
        config_path = CONFIGS_DIR / filename
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert "task" in config, f"{filename} missing 'task'"
        assert "data" in config, f"{filename} missing 'data'"
        assert "training" in config, f"{filename} missing 'training'"
        assert "train" in config["data"], f"{filename} missing 'data.train'"


def test_task_prompts_available():
    """All task prompts should be defined."""
    from src.data.utils import TASK_PROMPTS

    assert "asr" in TASK_PROMPTS
    assert "s2tt" in TASK_PROMPTS
    assert "sqa" in TASK_PROMPTS
    assert "s2tsum" in TASK_PROMPTS
    assert "achap" in TASK_PROMPTS

    # ASR has English
    assert "en" in TASK_PROMPTS["asr"]
    # S2TT has all target languages
    assert "de" in TASK_PROMPTS["s2tt"]
    assert "it" in TASK_PROMPTS["s2tt"]
    assert "zh" in TASK_PROMPTS["s2tt"]


def test_unanswerable_responses():
    """All unanswerable responses should be defined."""
    from src.data.utils import UNANSWERABLE

    assert "en" in UNANSWERABLE
    assert "de" in UNANSWERABLE
    assert "it" in UNANSWERABLE
    assert "zh" in UNANSWERABLE


def test_format_chat_messages():
    """Chat message formatting should produce valid structure."""
    from src.data.utils import format_chat_messages, SPEECH_PLACEHOLDER

    messages = format_chat_messages("Transcribe this.", "Hello world")
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert SPEECH_PLACEHOLDER in messages[1]["content"]
    assert messages[2]["content"] == "Hello world"

    # Without response
    messages = format_chat_messages("Transcribe this.")
    assert len(messages) == 2


def test_speech_projector_shapes():
    """SpeechProjector should produce correct output shapes."""
    import torch
    from src.model.adapters import SpeechProjector

    proj = SpeechProjector(speech_dim=1024, llm_dim=2560, downsample_factor=1)
    x = torch.randn(2, 100, 1024)
    out = proj(x)
    assert out.shape == (2, 100, 2560)

    # With downsampling
    proj_ds = SpeechProjector(speech_dim=1024, llm_dim=2560, downsample_factor=4)
    out_ds = proj_ds(x)
    assert out_ds.shape == (2, 25, 2560)


def test_config_loader():
    """Config loader should merge base and task configs."""
    from src.config import load_config

    config = load_config(str(CONFIGS_DIR / "asr.yaml"))
    # Should have merged base config values
    assert "model" in config
    assert "training" in config
    assert config["task"] == "asr"
    # Task-specific training overrides base
    assert "num_epochs" in config["training"]
