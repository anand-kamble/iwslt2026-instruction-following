#!/usr/bin/env python
"""Smoke check: does vLLM in this env support Gemma-4?

Reports:
  1. vLLM version
  2. Whether Gemma4ForConditionalGeneration is registered
  3. If yes, attempts a minimal load of the text model (Gemma-4-31B)
     with a tiny batch to confirm it actually works on this GPU

Usage (on GPU node, with .venv_gen and vllm installed):
    cd /work/nvme/bggk/atathe/IWSLT/iwslt2026-instruction-following
    source .venv_gen/bin/activate
    module load cuda/12.8
    export HF_HOME=/work/nvme/bggk/atathe/hf_cache
    python scripts/smoke_vllm_check.py
"""
from __future__ import annotations
import os
import sys
import traceback

PROJECT_ROOT = "/work/nvme/bggk/atathe/IWSLT/iwslt2026-instruction-following"
TEXT_MODEL = os.environ.get("SMOKE_TEXT_MODEL", f"{PROJECT_ROOT}/models/gemma-4-31B")
AUDIO_MODEL = os.environ.get("SMOKE_AUDIO_MODEL", f"{PROJECT_ROOT}/models/gemma-4-E4B-it")
TARGET = os.environ.get("SMOKE_TARGET", "text").lower()  # "text" or "audio"


def _section(title: str):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}", flush=True)


def check_versions():
    _section("1. Environment")
    print("python:", sys.version.split()[0])
    try:
        import torch
        print("torch:", torch.__version__, "| cuda ok:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("gpu  :", torch.cuda.get_device_name(0),
                  f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
    except Exception as e:
        print("torch: FAILED ->", e)

    try:
        import vllm
        print("vllm :", vllm.__version__)
        return vllm.__version__
    except Exception as e:
        print("vllm : FAILED ->", e)
        return None


def check_arch_registered():
    _section("2. Gemma4 architecture support in vLLM")
    try:
        from vllm.model_executor.models.registry import ModelRegistry
        archs = ModelRegistry.get_supported_archs()
        gemma_archs = sorted(a for a in archs if "gemma" in a.lower())
        print("Gemma-related archs registered in vLLM:")
        for a in gemma_archs:
            print(" ", a)
        supported = "Gemma4ForConditionalGeneration" in archs
        print()
        print("Gemma4ForConditionalGeneration supported:", supported)
        return supported
    except Exception as e:
        print("registry check FAILED ->", e)
        traceback.print_exc()
        return False


def check_config_loadable(path: str, label: str):
    _section(f"3. HF config loadable for {label}")
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        print("architectures:", cfg.architectures)
        print("model_type   :", cfg.model_type)
        has_audio = getattr(cfg, "audio_config", None) is not None
        print("has audio cfg:", has_audio)
        return True
    except Exception as e:
        print("config load FAILED ->", e)
        traceback.print_exc()
        return False


def attempt_text_load(path: str):
    _section(f"4. Attempt vLLM load of TEXT model ({path})")
    try:
        from vllm import LLM, SamplingParams
        print("Loading LLM (this may take 1-2 min)...", flush=True)
        llm = LLM(
            model=path,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
            max_model_len=2048,
            max_num_seqs=4,
            enforce_eager=True,  # skip CUDA graph capture for smoke
            trust_remote_code=True,
        )
        print("Load OK. Running one generation...", flush=True)
        out = llm.generate(
            ["Extract 3 keywords: The IPCC released its sixth assessment report today."],
            SamplingParams(temperature=0.7, top_p=0.9, max_tokens=32),
        )
        print()
        print("--- generated ---")
        print(out[0].outputs[0].text)
        print("-----------------")
        return True
    except Exception as e:
        print("vLLM load FAILED ->", type(e).__name__, e)
        print()
        print("Last 40 lines of traceback:")
        tb = traceback.format_exc().splitlines()
        print("\n".join(tb[-40:]))
        return False


def main():
    os.environ.setdefault("HF_HOME", "/work/nvme/bggk/atathe/hf_cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # don't hit network during smoke

    vllm_version = check_versions()
    if vllm_version is None:
        print("\nRESULT: vLLM not importable. Activate the right env first.")
        sys.exit(2)

    arch_ok = check_arch_registered()
    check_config_loadable(TEXT_MODEL, "Gemma-4-31B (text)")
    check_config_loadable(AUDIO_MODEL, "Gemma-4-E4B-it (audio-aware)")

    if not arch_ok:
        print("\nRESULT: Gemma4ForConditionalGeneration NOT in vLLM registry.")
        sys.exit(3)

    path = AUDIO_MODEL if TARGET == "audio" else TEXT_MODEL
    label = "audio-aware (Gemma-4-E4B-it)" if TARGET == "audio" else "text (Gemma-4-31B)"
    load_ok = attempt_text_load(path)
    if load_ok:
        print(f"\nRESULT: vLLM is ready for {label}.")
        sys.exit(0)
    else:
        print(f"\nRESULT: Arch is registered but load failed for {label}. See traceback above.")
        sys.exit(4)


if __name__ == "__main__":
    main()
