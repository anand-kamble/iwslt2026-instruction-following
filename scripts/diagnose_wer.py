"""Diagnostic script to trace exactly why WER is 98%.

Run on your training machine with a checkpoint:
    python scripts/diagnose_wer.py --checkpoint ./outputs/stage3_multimodal \
        --config configs/stage3_multimodal.yaml --data-dir /path/to/mcif/audio

This prints every intermediate value at each pipeline step to identify
exactly where things go wrong.
"""

import argparse
import os
import sys
import numpy as np
import torch
import yaml

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir", default=".", help="Base dir for MCIF audio files")
    parser.add_argument("--sample-idx", type=int, default=0, help="Which MCIF sample to test")
    args = parser.parse_args()

    config = {}
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)

    # ================================================================
    # STEP 1: Load checkpoint
    # ================================================================
    print("=" * 70)
    print("STEP 1: Loading checkpoint")
    print("=" * 70)
    from src.evaluate import load_checkpoint
    model, processor, tokenizer = load_checkpoint(args.checkpoint, config)

    print(f"  Tokenizer len: {len(tokenizer)}")
    print(f"  Tokenizer pad_token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"  Tokenizer eos_token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

    from src.model.speech_llm import SPEECH_PLACEHOLDER
    speech_id = tokenizer.convert_tokens_to_ids(SPEECH_PLACEHOLDER)
    print(f"  Speech token: {SPEECH_PLACEHOLDER} (id={speech_id})")
    print(f"  Speech token == unk? {speech_id == tokenizer.unk_token_id}")

    print(f"  Projector type: {type(model.projector).__name__}")
    print(f"  Projector params: {sum(p.numel() for p in model.projector.parameters()):,}")
    print(f"  LLM type: {type(model.llm).__name__}")

    # Check LLM embed_tokens size
    embed = model._get_embed_tokens()
    print(f"  embed_tokens shape: {embed.weight.shape}")
    if hasattr(model.llm, 'config'):
        conf = model.llm.config if not hasattr(model.llm, 'base_model') else model.llm.base_model.config
        print(f"  Model config vocab_size: {conf.vocab_size}")

    # ================================================================
    # STEP 2: Load one MCIF sample
    # ================================================================
    print()
    print("=" * 70)
    print("STEP 2: Loading MCIF sample")
    print("=" * 70)
    from datasets import load_dataset
    ds = load_dataset("FBK-MT/MCIF", "short_fixedprompt", split="test")
    sample = ds[args.sample_idx]

    print(f"  Sample ID: {sample['id']}")
    print(f"  Available keys: {list(sample.keys())}")

    # Show prompt for each language
    for lang in ["en", "de", "it", "zh"]:
        key = f"prompt_{lang}"
        if key in sample:
            prompt = sample[key]
            print(f"  {key}: {prompt[:100]}...")

    # ================================================================
    # STEP 3: Process audio
    # ================================================================
    print()
    print("=" * 70)
    print("STEP 3: Processing audio")
    print("=" * 70)

    # Try loading audio - from dict (embedded) or from file path
    audio_data = sample.get("audio")
    if isinstance(audio_data, dict) and "array" in audio_data:
        audio_array = audio_data["array"]
        sr = audio_data.get("sampling_rate", 16000)
        print(f"  Audio source: embedded in dataset")
    elif isinstance(audio_data, str):
        from src.data.utils import load_audio
        audio_path = os.path.join(args.data_dir, audio_data)
        print(f"  Audio source: file path: {audio_path}")
        if not os.path.exists(audio_path):
            print(f"  ERROR: Audio file not found! This would cause evaluation to crash.")
            return
        waveform, sr = load_audio(audio_path)
        audio_array = waveform.squeeze(0).numpy()
    else:
        print(f"  ERROR: Unexpected audio format: {type(audio_data)}")
        print(f"  Audio value: {audio_data}")
        return

    print(f"  Audio shape: {np.array(audio_array).shape}")
    print(f"  Audio dtype: {np.array(audio_array).dtype}")
    print(f"  Audio sample rate: {sr}")
    print(f"  Audio duration: {len(audio_array) / sr:.2f}s")
    print(f"  Audio range: [{np.min(audio_array):.4f}, {np.max(audio_array):.4f}]")
    print(f"  Audio RMS: {np.sqrt(np.mean(np.array(audio_array)**2)):.6f}")

    if np.sqrt(np.mean(np.array(audio_array)**2)) < 1e-6:
        print("  WARNING: Audio is effectively silent! RMS is near zero.")

    # Process through SeamlessM4T processor
    audio_inputs = processor(
        audios=audio_array,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    )
    device = model.device
    audio_inputs = {k: v.to(device) for k, v in audio_inputs.items()}

    print(f"  Processor output input_features shape: {audio_inputs['input_features'].shape}")
    print(f"  Processor output attention_mask shape: {audio_inputs['attention_mask'].shape}")
    print(f"  input_features range: [{audio_inputs['input_features'].min():.4f}, {audio_inputs['input_features'].max():.4f}]")

    # ================================================================
    # STEP 4: Tokenize instruction
    # ================================================================
    print()
    print("=" * 70)
    print("STEP 4: Tokenizing instruction (English)")
    print("=" * 70)
    from src.data.utils import format_chat_messages
    instruction = sample["prompt_en"]
    messages = format_chat_messages(instruction)
    print(f"  Instruction: {instruction}")
    print(f"  Messages: {messages}")

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print(f"  Chat template output: {repr(text)}")

    text_inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = text_inputs["input_ids"]
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  Token IDs:")
    for i, tid in enumerate(input_ids[0].tolist()):
        marker = " *** SPEECH ***" if tid == speech_id else ""
        print(f"    {i}: {tid} -> {repr(tokenizer.decode([tid]))}{marker}")

    num_speech_tokens = (input_ids == speech_id).sum().item()
    print(f"  Number of speech placeholders: {num_speech_tokens}")
    if num_speech_tokens == 0:
        print("  CRITICAL ERROR: No speech placeholder found in tokenized input!")
        print("  The model will not receive any speech information.")

    # ================================================================
    # STEP 5: Trace through generate() step by step
    # ================================================================
    print()
    print("=" * 70)
    print("STEP 5: Tracing generate() internals")
    print("=" * 70)

    with torch.inference_mode():
        # 5a: Encode speech
        print("  5a: Encoding speech...")
        speech_features, speech_lengths = model.encode_speech(
            audio_inputs["input_features"],
            audio_inputs["attention_mask"],
        )
        print(f"      speech_features shape: {speech_features.shape}")
        print(f"      speech_lengths: {speech_lengths}")
        print(f"      speech_features range: [{speech_features.min():.4f}, {speech_features.max():.4f}]")
        print(f"      speech_features norm (mean): {speech_features.norm(dim=-1).mean():.4f}")

        # 5b: Project to LLM space
        print("  5b: Projecting to LLM space...")
        speech_embeds, proj_lengths = model.projector(speech_features, speech_lengths)
        speech_embeds = speech_embeds.to(model.dtype)
        print(f"      projected shape: {speech_embeds.shape}")
        print(f"      projected lengths: {proj_lengths}")
        print(f"      projected range: [{speech_embeds.min():.4f}, {speech_embeds.max():.4f}]")
        print(f"      projected norm (mean): {speech_embeds.norm(dim=-1).mean():.4f}")

        # 5c: Get text embeddings for comparison
        print("  5c: Getting text embeddings...")
        text_embeds = model._get_embed_tokens()(input_ids)
        print(f"      text_embeds shape: {text_embeds.shape}")
        print(f"      text_embeds range: [{text_embeds.min():.4f}, {text_embeds.max():.4f}]")
        print(f"      text_embeds norm (mean): {text_embeds.norm(dim=-1).mean():.4f}")

        # CRITICAL CHECK: Are speech and text embeddings in the same scale?
        speech_norm = speech_embeds[0, :proj_lengths[0]].norm(dim=-1).mean().item()
        text_norm = text_embeds.norm(dim=-1).mean().item()
        ratio = speech_norm / text_norm if text_norm > 0 else float('inf')
        print(f"      SCALE CHECK: speech_norm={speech_norm:.4f}, text_norm={text_norm:.4f}, ratio={ratio:.4f}")
        if ratio > 10 or ratio < 0.1:
            print("      CRITICAL WARNING: Speech and text embeddings are on very different scales!")
            print("      The LLM will likely ignore the speech embeddings or be overwhelmed by them.")

        # 5d: Merge
        print("  5d: Merging speech and text...")
        inputs_embeds, new_attn_mask, _ = model._merge_speech_and_text(
            speech_embeds, text_embeds, input_ids,
            text_inputs["attention_mask"], labels=None,
            speech_lengths=proj_lengths,
        )
        print(f"      merged inputs_embeds shape: {inputs_embeds.shape}")
        print(f"      merged attention_mask shape: {new_attn_mask.shape}")
        print(f"      attention_mask sum (active tokens): {new_attn_mask.sum().item()}")

        # 5e: Generate
        print("  5e: Generating...")
        output_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=new_attn_mask,
            max_new_tokens=512,
            do_sample=False,
        )
        print(f"      output_ids shape: {output_ids.shape}")
        print(f"      output_ids (first 50 tokens):")
        for i, tid in enumerate(output_ids[0].tolist()[:50]):
            print(f"        {i}: {tid} -> {repr(tokenizer.decode([tid]))}")
        if output_ids.shape[1] > 50:
            print(f"        ... ({output_ids.shape[1] - 50} more tokens)")

    # ================================================================
    # STEP 6: Decode and analyze output
    # ================================================================
    print()
    print("=" * 70)
    print("STEP 6: Decoding output")
    print("=" * 70)

    raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    clean_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    print(f"  Raw output (with special tokens):")
    print(f"    {repr(raw_text[:500])}")
    print()
    print(f"  Clean output (skip_special_tokens=True):")
    print(f"    {repr(clean_text[:500])}")
    print()
    print(f"  Output length: {len(output_ids[0])} tokens, {len(clean_text)} chars")

    # Check for common issues
    if len(clean_text.strip()) == 0:
        print("  CRITICAL: Output is EMPTY after decoding!")
        print("  The model immediately generated EOS.")
    elif len(output_ids[0]) >= 510:
        print("  WARNING: Output hit max_new_tokens limit (512).")
        print("  Model may not have generated EOS — could be repetitive.")

    # Check for repetition
    words = clean_text.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        print(f"  Unique word ratio: {unique_ratio:.2f} ({len(set(words))}/{len(words)} unique)")
        if unique_ratio < 0.3:
            print("  WARNING: Output is highly repetitive!")

    # Check language (rough heuristic)
    import unicodedata
    cjk_count = sum(1 for c in clean_text if unicodedata.category(c).startswith('Lo'))
    total_alpha = sum(1 for c in clean_text if c.isalpha())
    if total_alpha > 0:
        cjk_ratio = cjk_count / total_alpha
        print(f"  CJK character ratio: {cjk_ratio:.2f}")
        if cjk_ratio > 0.5:
            print("  WARNING: Output appears to be primarily CJK (Chinese/Japanese/Korean)!")
            print("  For English ASR, this indicates the model is not following instructions.")

    # ================================================================
    # STEP 7: Compare with training data format
    # ================================================================
    print()
    print("=" * 70)
    print("STEP 7: Training data format verification")
    print("=" * 70)

    # Simulate what the collator does during training
    train_messages = format_chat_messages(instruction, response="[REFERENCE TEXT]")
    train_text = tokenizer.apply_chat_template(train_messages, tokenize=False, add_generation_prompt=False)
    print(f"  Training template (with response):")
    print(f"    {repr(train_text)}")

    train_ids = tokenizer(train_text, return_tensors="pt")["input_ids"][0].tolist()

    # Find assistant header
    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    print(f"  Assistant header tokens: {header}")

    prompt_end = 0
    for j in range(len(train_ids) - len(header) + 1):
        if train_ids[j:j+len(header)] == header:
            prompt_end = j + len(header)

    print(f"  Label masking prompt_end: {prompt_end}")
    if prompt_end == 0:
        print("  CRITICAL ERROR: Could not find assistant header in tokenized training text!")
        print("  Labels would NOT be masked — model trains on entire sequence including prompt!")
    else:
        masked_tokens = train_ids[:prompt_end]
        unmasked_tokens = train_ids[prompt_end:]
        print(f"  Masked (prompt) tokens: {len(masked_tokens)}")
        print(f"  Unmasked (response) tokens: {len(unmasked_tokens)}")
        print(f"  Unmasked text: {repr(tokenizer.decode(unmasked_tokens))}")

    print()
    print("=" * 70)
    print("DIAGNOSIS COMPLETE — review the output above for CRITICAL/WARNING messages")
    print("=" * 70)


if __name__ == "__main__":
    main()
