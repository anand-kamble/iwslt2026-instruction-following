# SWIFT Migration Design

Migrate the IWSLT 2026 instruction-following training pipeline from custom HuggingFace Trainer code to ms-swift (ModelScope SWIFT). This replaces `src/train.py`, the custom collator, and data loaders with SWIFT's training infrastructure while keeping the core model architecture.

**Dependency:** `ms-swift>=4.0,<5.0` — pin in `pyproject.toml`.

## Architecture

3-stage pipeline:

| Stage | Model | Trainable | SWIFT flags |
|-------|-------|-----------|-------------|
| 1 — Projector alignment | Custom SpeechLLM | Projector only | `freeze_vit=True, freeze_aligner=False, freeze_llm=True` |
| 2 — Text LoRA | Native Qwen3-4B | LoRA on LLM | Standard `swift sft` with Qwen3 |
| 3 — Multimodal merge | Custom SpeechLLM + Stage 1 projector + Stage 2 LoRA | Projector + LoRA | `freeze_vit=True, freeze_aligner=False, freeze_llm=False, aligner_lr=1e-4` |

Stage 0 (MSE pre-alignment) is dropped — audit showed degenerate loss.

## Model: `SpeechLLMForConditionalGeneration`

Full `PreTrainedModel` implementation.

### `SpeechLLMConfig(PretrainedConfig)`

```python
class SpeechLLMConfig(PretrainedConfig):
    model_type = "speech_llm"

    def __init__(
        self,
        sfm_model_name="facebook/seamless-m4t-v2-large",
        llm_model_name="Qwen/Qwen3-4B-Instruct-2507",
        speech_dim=1024,
        llm_dim=2560,
        downsample_factor=3,
        projector_num_layers=4,
        projector_num_heads=8,
        projector_ffn_dim=2048,
        **kwargs,
    ):
        self.sfm_model_name = sfm_model_name
        self.llm_model_name = llm_model_name
        self.speech_dim = speech_dim
        self.llm_dim = llm_dim
        self.downsample_factor = downsample_factor
        self.projector_num_layers = projector_num_layers
        self.projector_num_heads = projector_num_heads
        self.projector_ffn_dim = projector_ffn_dim
        super().__init__(**kwargs)
```

### `SpeechLLMForConditionalGeneration(PreTrainedModel)`

```python
class SpeechLLMForConditionalGeneration(PreTrainedModel):
    config_class = SpeechLLMConfig
    _no_split_modules = ["TransformerProjector"]

    def __init__(self, config: SpeechLLMConfig):
        super().__init__(config)
        # Sub-models are loaded programmatically by SpeechLLMLoader.get_model(),
        # NOT by from_pretrained(). __init__ only creates the projector.
        self.projector = TransformerProjector(
            speech_dim=config.speech_dim,
            llm_dim=config.llm_dim,
            downsample_factor=config.downsample_factor,
            num_layers=config.projector_num_layers,
            num_heads=config.projector_num_heads,
            ffn_dim=config.projector_ffn_dim,
        )
        # speech_encoder and llm are set by the ModelLoader after construction
        self.speech_encoder = None
        self.llm = None

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def get_output_embeddings(self):
        return self.llm.get_output_embeddings()

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, **kwargs):
        """Forward pass. Receives inputs_embeds (speech+text already merged by Template._post_encode)."""
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
```

**Key design decision:** `__init__` only creates the projector. The speech encoder and LLM are set by `SpeechLLMLoader.get_model()` after construction, because they are large pretrained models that need custom loading logic (quantization, device placement, checkpoint loading). This avoids fighting HuggingFace's `from_pretrained()` serialization for composite models. The config stores the model names so reconstruction is deterministic.

### Model arch registration

```python
register_model_arch(MultiModelKeys(
    'speech_llm',
    language_model=['llm.model', 'llm.lm_head'],  # explicit LM head for correct freeze/LoRA
    vision_tower=['speech_encoder'],
    aligner=['projector'],
))
```

Note: `language_model` explicitly lists both `llm.model` and `llm.lm_head` (following the Qwen2.5-Omni pattern) so SWIFT's freeze/LoRA logic correctly captures the full LLM.

## SWIFT Plugin: `swift_plugin/register.py`

### SpeechLLMLoader(ModelLoader)

- `get_config(model_dir)` — loads `SpeechLLMConfig` from `model_dir/config.json`, or creates default config if no config file exists (first run).
- `get_processor(model_dir, config)` — returns a composite processor: SeamlessM4T `AutoProcessor` for audio + Qwen3 `AutoTokenizer` for text. The tokenizer is the primary processor (SWIFT uses it for tokenization); the audio processor is stored as an attribute.
- `get_model(model_dir, config, processor, model_kwargs)`:
  1. Create `SpeechLLMForConditionalGeneration(config)` — initializes projector
  2. Load SeamlessM4T encoder: `SeamlessM4Tv2Model.from_pretrained(config.sfm_model_name).speech_encoder`, freeze all params, assign to `model.speech_encoder`
  3. Load Qwen3: `AutoModelForCausalLM.from_pretrained(config.llm_model_name, **model_kwargs)`, assign to `model.llm`
  4. Add `<|speech|>` token, resize embeddings
  5. If `STAGE1_CHECKPOINT` env var set: load `projector.pt` from that path
  6. If `STAGE2_CHECKPOINT` env var set: load LoRA adapters onto `model.llm` sub-module via `PeftModel.from_pretrained(model.llm, stage2_path, is_trainable=True)` — this handles the prefix mismatch because LoRA is applied directly to the `llm` sub-module, whose internal parameter names match the native Qwen3 checkpoint exactly.

### Stage 2 → Stage 3 LoRA loading

Stage 2 trains native Qwen3 with LoRA. Parameter names: `model.layers.0.self_attn.q_proj.lora_A.weight`.
Stage 3 wraps Qwen3 as `self.llm`. Internal parameter names: `llm.model.layers.0...`.

Solution: load Stage 2 LoRA onto `model.llm` directly (not the full `SpeechLLMForConditionalGeneration`). `PeftModel.from_pretrained(model.llm, checkpoint_dir)` works because `model.llm` IS a Qwen3 model with matching parameter names. SWIFT then sees `llm.model.layers.0.self_attn.q_proj.lora_A.weight` as part of `language_model` and applies freeze/unfreeze correctly.

The `<|speech|>` token: Stage 2 does NOT add it (native Qwen3). Stage 3's `get_model()` adds it and resizes embeddings AFTER loading LoRA. The new embedding row is randomly initialized, which is fine — it's the placeholder token whose embedding is replaced by speech features in `_post_encode`.

### SpeechLLMTemplate(Template)

```python
class SpeechLLMTemplate(Template):
    use_model = True
    placeholder_tokens = ['<|SPEECH|>']

    def replace_tag(self, media_type, index, inputs):
        """Load audio file, return speech placeholder token string."""
        if media_type == 'audio':
            inputs.audios[index] = load_audio(inputs.audios[index], 16000)
            return ['<|speech_bos|><|SPEECH|><|speech_eos|>']

    def _encode(self, inputs):
        """Tokenize text + process audio through SeamlessM4T processor."""
        encoded = super()._encode(inputs)

        if not inputs.audios:
            return encoded

        # Process audio through SeamlessM4T processor
        audio_processor = self.processor.audio_processor  # stored by SpeechLLMLoader
        audio_inputs = audio_processor(
            audio=inputs.audios[0],  # single audio per sample
            sampling_rate=16000,
            return_tensors='pt',
        )
        encoded['input_features'] = audio_inputs['input_features'].squeeze(0)
        encoded['input_features_attention_mask'] = audio_inputs['attention_mask'].squeeze(0)

        # Expand <|SPEECH|> placeholder to speech_length tokens
        # (actual length determined after encoder runs in _post_encode)
        # Use a fixed estimate; _post_encode handles the actual merge
        input_ids = encoded['input_ids']
        labels = encoded['labels']
        speech_token_id = self.processor.convert_tokens_to_ids('<|SPEECH|>')
        # ... expand placeholder using findall() + _extend_tokens() ...

        return encoded

    def _post_encode(self, model, inputs):
        """Run speech encoder + projector, merge into inputs_embeds."""
        input_ids = inputs['input_ids']
        input_features = inputs.get('input_features')

        # Get text embeddings
        embed_tokens = model.get_input_embeddings()
        inputs_embeds = embed_tokens(input_ids)

        if input_features is None:
            return {'inputs_embeds': inputs_embeds}

        # Run frozen speech encoder
        with torch.no_grad():
            speech_features = model.speech_encoder(
                input_features=input_features,
                attention_mask=inputs.get('input_features_attention_mask'),
                return_dict=True,
            ).last_hidden_state

        # Project to LLM dim
        speech_embeds = model.projector(speech_features)  # (1, T', llm_dim)

        # Find speech placeholder token positions, replace with speech embeddings
        speech_token_id = self.processor.convert_tokens_to_ids('<|SPEECH|>')
        placeholder_mask = (input_ids == speech_token_id)
        placeholder_indices = placeholder_mask.nonzero(as_tuple=True)[0]

        if len(placeholder_indices) == 0:
            return {'inputs_embeds': inputs_embeds}

        # Build merged sequence: [text_before] [speech_embeds] [text_after]
        first_pos = placeholder_indices[0].item()
        before = inputs_embeds[:first_pos]
        after = inputs_embeds[first_pos + len(placeholder_indices):]
        merged = torch.cat([before, speech_embeds.squeeze(0), after], dim=0)

        # Rebuild labels: speech positions get -100
        if 'labels' in inputs:
            labels = inputs['labels']
            labels_before = labels[:first_pos]
            labels_speech = torch.full((speech_embeds.shape[1],), -100, dtype=labels.dtype, device=labels.device)
            labels_after = labels[first_pos + len(placeholder_indices):]
            inputs['labels'] = torch.cat([labels_before, labels_speech, labels_after])

        return {'inputs_embeds': merged}

    def _data_collator_mm_data(self, batch):
        """Batch multimodal tensors."""
        res = super()._data_collator_mm_data(batch)
        input_features = [b['input_features'] for b in batch if b.get('input_features') is not None]
        masks = [b['input_features_attention_mask'] for b in batch if b.get('input_features_attention_mask') is not None]
        if input_features:
            res['input_features'] = torch.stack(input_features)
            res['input_features_attention_mask'] = torch.stack(masks)
        return res
```

### Template registration

```python
register_template(TemplateMeta(
    'speech_llm',
    prefix=[],
    prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
    chat_sep=['<|im_end|>\n'],
    suffix=['<|im_end|>'],
    default_system=None,  # no system prompt — matches MCIF eval and Qwen3 defaults
    stop_words=['<|endoftext|>'],
    template_cls=SpeechLLMTemplate,
))
```

## Data Preparation: `swift_plugin/prepare_data.py`

Offline script, two modes:

1. **Speech datasets** (Stage 1 & 3): Stream from CLAPv2/Europarl-st, resample to 16kHz, save as `.wav` files to `data/europarlst/audio/`, generate JSONL manifests.

2. **Text-only datasets** (Stage 2): Load from tj-solergibert/Europarl-ST, extract EN->DE/IT pairs with English source in instruction, generate JSONL.

JSONL format (SWIFT standard):

Speech (ASR):
```json
{"messages": [{"role": "user", "content": "<audio>\nTranscribe the English content."}, {"role": "assistant", "content": "Good morning."}], "audios": ["data/europarlst/audio/00001.wav"]}
```

Speech (S2TT):
```json
{"messages": [{"role": "user", "content": "<audio>\nÜbersetze den englischem Inhalt nach Deutsch."}, {"role": "assistant", "content": "Guten Morgen."}], "audios": ["data/europarlst/audio/00001.wav"]}
```

Text-only (MT):
```json
{"messages": [{"role": "user", "content": "Übersetze den englischem Inhalt nach Deutsch.\n\nGood morning everyone."}, {"role": "assistant", "content": "Guten Morgen allerseits."}]}
```

No system prompt. Prompts match MCIF evaluation exactly (including "concisely" in SQA, "ungefahr" in S2TSUM DE).

Script is idempotent — skips existing files.

**Future datasets** (GigaST, LibriSQA, NUTSHELL, YTSeg): The existing data loaders in `src/data/` are kept. When ready, `prepare_data.py` will be extended with additional subcommands to generate JSONL for each dataset. The JSONL format is the same — only the source data and prompt templates differ.

## File Structure

```
swift_plugin/
  __init__.py
  register.py          # SpeechLLMConfig, SpeechLLMForConditionalGeneration,
                       # SpeechLLMLoader, SpeechLLMTemplate, all registrations
  prepare_data.py      # Download, resample, generate JSONL

configs/swift/
  stage1_projector.yaml
  stage2_text_lora.yaml
  stage3_multimodal.yaml

data/
  europarlst/
    audio/              # 16kHz .wav files (gitignored)
    asr_train.jsonl
    s2tt_de_train.jsonl
    s2tt_it_train.jsonl
  mt_text/
    europarlst_de.jsonl
    europarlst_it.jsonl

scripts/
  train_stage1.sh
  train_stage2.sh
  train_stage3.sh
```

## SWIFT Training Configs

### Stage 1 — Projector alignment

```yaml
model: ./swift_plugin
model_type: speech_llm
template: speech_llm
external_plugins: swift_plugin/register.py
dataset: data/europarlst/asr_train.jsonl
freeze_vit: true
freeze_aligner: false
freeze_llm: true
tuner_type: full
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 8
learning_rate: 1e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
logging_steps: 10
save_steps: 200
save_total_limit: 3
max_length: 4096
output_dir: output/stage1
torch_dtype: float32  # MPS compatibility; bf16 on CUDA
```

### Stage 2 — Text LoRA (native Qwen3)

```yaml
model: Qwen/Qwen3-4B-Instruct-2507
dataset: data/mt_text/europarlst_de.jsonl data/mt_text/europarlst_it.jsonl
tuner_type: lora
lora_rank: 16
lora_alpha: 16
target_modules: all-linear
num_train_epochs: 1
max_steps: 2000
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 3e-4
lr_scheduler_type: cosine
warmup_ratio: 0.05
logging_steps: 10
save_steps: 500
save_total_limit: 3
max_length: 4096
output_dir: output/stage2
torch_dtype: float32
```

### Stage 3 — Multimodal merge

```yaml
model: ./swift_plugin
model_type: speech_llm
template: speech_llm
external_plugins: swift_plugin/register.py
dataset: data/europarlst/asr_train.jsonl
freeze_vit: true
freeze_aligner: false
freeze_llm: false
tuner_type: lora
lora_rank: 16
lora_alpha: 16
target_modules: all-linear
aligner_lr: 1e-4
learning_rate: 1e-5
max_steps: 1000
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
gradient_checkpointing: true
lr_scheduler_type: cosine
warmup_ratio: 0.05
logging_steps: 5
save_steps: 200
save_total_limit: 3
max_length: 4096
output_dir: output/stage3
torch_dtype: float32
```

## Checkpoint Flow

- Stage 1: saves full model. Key artifact is `projector` weights within the checkpoint.
- Stage 2: saves LoRA adapters in standard PEFT format. Parameter names match native Qwen3 (no `llm.` prefix).
- Stage 3 loading: `SpeechLLMLoader.get_model()` loads Stage 1 projector weights, then applies Stage 2 LoRA to `model.llm` sub-module directly via `PeftModel.from_pretrained(model.llm, ...)`. This handles the prefix mismatch transparently.
- Stage 3: saves projector + LoRA adapters.

## MPS Compatibility

SWIFT uses `accelerate` for device placement, which supports MPS. However, SWIFT's multimodal training on MPS is not well-tested upstream. Mitigations:

- Test `swift sft` on MPS with native Qwen3 (Stage 2) as the first integration test
- Use `torch_dtype: float32` (not bf16) on MPS — same constraint as current pipeline
- Keep existing `src/train.py` as fallback until SWIFT on MPS is validated
- Actual Stage 3 training will happen on CUDA (A100 cluster)

## What Gets Removed

| File | Status |
|------|--------|
| `src/model/speech_llm.py` | Replaced by `SpeechLLMForConditionalGeneration` in register.py |
| `src/model/build_model.py` | Replaced by `SpeechLLMLoader` |
| `src/model/adapters.py` | Kept — `TransformerProjector` reused |
| `src/train.py` | Kept as fallback until SWIFT on MPS validated, then removed |
| `src/train_stage0.py` | Removed — Stage 0 dropped |
| `src/data/collator.py` | Removed — SWIFT Template handles collation |
| `src/data/asr.py` | Removed — replaced by prepare_data.py |
| `src/data/text_only.py` | Removed — replaced by prepare_data.py |
| `src/data/utils.py` | Kept — prompts used by prepare_data.py |
| `src/data/multitask.py` | Removed — SWIFT handles mixing |
| `src/data/s2tt.py`, `sqa.py`, `achap.py`, `gigast.py`, `s2tsum.py` | Kept — used by prepare_data.py for future datasets |
| `src/evaluate.py`, `src/inference.py` | Updated to load from SWIFT checkpoints |
| `configs/stage*.yaml` | Replaced by configs/swift/*.yaml |
| `scripts/precompute_alignments.py` | Removed — was for Stage 0 |

## Testing

- Smoke test: `SpeechLLMForConditionalGeneration` instantiation from config, forward pass with mock inputs
- Template test: `_encode` and `_post_encode` produce correct shapes
- Integration test: 1-step SWIFT training on synthetic data (both MPS and CUDA)
- Regression test: Compare loss trajectory of first 10 steps between old pipeline and SWIFT pipeline on same data
