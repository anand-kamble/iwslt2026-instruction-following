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


def test_multitask_configs_have_required_fields():
    """Multitask configs must have tasks section with per-task data sources."""
    for filename in ["multitask_short.yaml", "multitask_long.yaml"]:
        config_path = CONFIGS_DIR / filename
        with open(config_path) as f:
            config = yaml.safe_load(f)
        assert config["task"] == "multitask", f"{filename} task should be 'multitask'"
        assert "tasks" in config, f"{filename} missing 'tasks'"
        assert "track" in config, f"{filename} missing 'track'"
        assert "sampling" in config, f"{filename} missing 'sampling'"
        assert "temperature" in config["sampling"]

        # Short track must have asr, s2tt, sqa
        tasks = config["tasks"]
        assert "asr" in tasks
        assert "s2tt" in tasks
        assert "sqa" in tasks


def test_multitask_long_has_all_tasks():
    """Long track config must include s2tsum and achap."""
    with open(CONFIGS_DIR / "multitask_long.yaml") as f:
        config = yaml.safe_load(f)
    tasks = config["tasks"]
    assert "s2tsum" in tasks
    assert "achap" in tasks


def test_gigast_in_asr_and_s2tt_configs():
    """GigaST should be listed in ASR and S2TT train data; CoVoST2 removed."""
    for filename in ["asr.yaml", "s2tt.yaml"]:
        with open(CONFIGS_DIR / filename) as f:
            config = yaml.safe_load(f)
        assert "gigast" in config["data"]["train"], f"{filename} missing gigast"
        assert "covost2" not in config["data"]["train"], f"{filename} should not have covost2 (removed from constrained setting)"


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
    # ACHAP has all target languages
    assert "en" in TASK_PROMPTS["achap"]
    assert "de" in TASK_PROMPTS["achap"]
    assert "it" in TASK_PROMPTS["achap"]
    assert "zh" in TASK_PROMPTS["achap"]


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
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert SPEECH_PLACEHOLDER in messages[0]["content"]
    assert messages[1]["content"] == "Hello world"

    # Without response
    messages = format_chat_messages("Transcribe this.")
    assert len(messages) == 1


def test_speech_projector_shapes():
    """SpeechProjector should produce correct output shapes."""
    import torch
    from src.model.adapters import SpeechProjector

    proj = SpeechProjector(speech_dim=1024, llm_dim=2560, downsample_factor=1)
    x = torch.randn(2, 100, 1024)
    out, lengths = proj(x)
    assert out.shape == (2, 100, 2560)

    # With downsampling
    proj_ds = SpeechProjector(speech_dim=1024, llm_dim=2560, downsample_factor=4)
    out_ds, lengths_ds = proj_ds(x)
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


def test_stage_configs_parse():
    """Stage configs should parse and have required fields."""
    for filename, expected_stage in [
        ("stage1_alignment.yaml", 1),
        ("stage2_text_lora.yaml", 2),
        ("stage3_multimodal.yaml", 3),
    ]:
        with open(CONFIGS_DIR / filename) as f:
            config = yaml.safe_load(f)
        assert config["stage"] == expected_stage


def test_base_config_defaults():
    """Base config should have correct quantization and LoRA defaults."""
    with open(CONFIGS_DIR / "base.yaml") as f:
        config = yaml.safe_load(f)
    # Quantization off by default (CUDA-only, enabled in CUDA-specific configs)
    assert config["training"]["load_in_4bit"] is False
    assert config["training"]["lora_r"] == 8
    assert config["training"]["lora_alpha"] == 16
    assert config["training"]["optim"] == "adamw_torch"


def test_stage1_no_lora():
    """Stage 1 must have no_lora=true."""
    with open(CONFIGS_DIR / "stage1_alignment.yaml") as f:
        config = yaml.safe_load(f)
    assert config["training"]["no_lora"] is True


def test_stage2_text_only():
    """Stage 2 must be text-only."""
    with open(CONFIGS_DIR / "stage2_text_lora.yaml") as f:
        config = yaml.safe_load(f)
    assert config["training"]["text_only"] is True


def test_format_text_only_messages():
    """Text-only messages should NOT contain speech placeholder."""
    from src.data.utils import format_text_only_messages, SPEECH_PLACEHOLDER
    messages = format_text_only_messages("Translate to German.", "Hallo Welt.")
    assert len(messages) == 3
    for m in messages:
        assert SPEECH_PLACEHOLDER not in m["content"]
    assert messages[1]["content"] == "Translate to German."
    assert messages[2]["content"] == "Hallo Welt."
