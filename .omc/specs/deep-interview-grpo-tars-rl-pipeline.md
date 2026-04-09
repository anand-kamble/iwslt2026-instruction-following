# Deep Interview Spec: GRPO + TARS Alignment RL Pipeline (Stage 4)

## Metadata
- Interview ID: iwslt-surprise-task-autoresearch
- Rounds: 8
- Final Ambiguity Score: 16%
- Type: brownfield
- Generated: 2026-04-04
- Threshold: 20%
- Status: PASSED

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.95 | 0.35 | 0.333 |
| Constraint Clarity | 0.75 | 0.25 | 0.188 |
| Success Criteria | 0.80 | 0.25 | 0.200 |
| Context Clarity | 0.80 | 0.15 | 0.120 |
| **Total Clarity** | | | **0.840** |
| **Ambiguity** | | | **0.160** |

## Goal
Build a Stage 4 RL post-training pipeline using GRPO + TARS alignment within the existing HuggingFace/TRL ecosystem. The pipeline applies reinforcement learning after SFT (Stages 1-3) to improve instruction-following performance across all tasks, with special emphasis on cross-lingual zero-shot tasks (SQA/S2TSUM/ACHAP in DE/ZH) and the surprise task — scenarios where task-specific training data is absent.

**GRPO** provides task-specific reward optimization (WER, COMET, BERTScore) for well-covered tasks. **TARS-style alignment** provides reference-free modality alignment rewards that work even for tasks with no ground-truth outputs, closing the speech↔text performance gap that limits zero-shot generalization.

## Constraints
- **Framework**: HuggingFace ecosystem — TRL v1.0 GRPOTrainer + PEFT (LoRA)
- **Hardware**: H200 GPUs (80GB HBM3e, bf16/tf32), multiple days of compute available
- **Track**: IWSLT 2026 constrained track — only permitted datasets
- **Deadline**: System submission April 15, 2026; system paper April 24, 2026
- **Architecture**: Frozen Seamless M4T v2 encoder + trainable projector + Qwen3-4B-Instruct-2507 with LoRA
- **Prerequisite**: Requires a Stage 3 SFT checkpoint. Pipeline must accept any checkpoint (not hardcoded).
- **No SWIFT**: Dropped due to custom model registration overhead. Stay in existing HF Trainer ecosystem.
- **Stage 4 only**: RL is applied post-SFT, not integrated into or replacing Stages 1-3.

## Non-Goals
- Replacing the existing 3-stage SFT pipeline
- Implementing DPO, PPO, or other RL methods (GRPO + TARS only)
- Building a custom reward model (use metric-based rewards only)
- Migrating to MS-SWIFT or any other framework
- Running full-scale training experiments as part of the build task (separate step)
- Implementing the surprise task itself (unknown until test data drops)

## Acceptance Criteria
- [ ] GRPO training script (`src/train_grpo.py`) using TRL's GRPOTrainer with SpeechLLM + LoRA
- [ ] Task-specific reward functions implemented:
  - [ ] WER reward for ASR (using `jiwer`)
  - [ ] COMET reward for S2TT (using `comet` or `unbabel-comet`)
  - [ ] BERTScore reward for SQA/S2TSUM (using `bert_score`)
- [ ] TARS alignment reward implemented:
  - [ ] Representation alignment: cosine similarity of hidden states between speech and text paths
  - [ ] Behavior alignment: BERTScore between speech-path and text-path generated outputs
  - [ ] Composite reward: `task_reward + λ_rep * rep_alignment + λ_beh * beh_alignment`
- [ ] Data pipeline adapted for GRPO (prompts + audio, reusing existing dataset loaders)
- [ ] YAML config for Stage 4 GRPO hyperparameters (group_size, lr, KL coeff, λ weights)
- [ ] Smoke test passes: load checkpoint → generate N samples → compute all rewards → one gradient step → no crash + sensible reward values
- [ ] TARS alignment smoke test: verify text-path and speech-path hidden states are extractable and cosine similarity is computable
- [ ] Pipeline accepts any Stage 3 checkpoint path via config
- [ ] Works on H200 with 4-bit Qwen3-4B + LoRA (memory fits)

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| SWIFT is needed for GRPO | Simplifier: is SWIFT the simplest path? Custom model registration is ~200 lines. | Dropped SWIFT. TRL GRPOTrainer in HF ecosystem is simpler with existing infrastructure. |
| RL replaces or modifies Stage 3 | Where does RL fit in the pipeline? | Stage 4 post-SFT. All speech RL papers validate this placement. |
| Only GRPO is useful | What other RL methods? | Considered DPO, KTO, ORPO, PPO, RFT. GRPO is most validated for speech. Added TARS for zero-shot. |
| Task-specific rewards are sufficient | How do you optimize for tasks without ground truth? | TARS alignment is reference-free — the only reward signal for cross-lingual zero-shot + surprise task. |
| TARS requires audio generation | Does TARS generate audio after Stage 3? | No. TARS runs two forward passes (speech path + text path) through the same LLM, comparing hidden states. |
| Autoresearch should run experiments | Should autoresearch run full training? | No. Build the pipeline + smoke test. Manual experiment runs after. |

## Technical Context

### Existing Codebase
- `src/train.py` — HF Trainer + PEFT, supports single-task and multitask training (Stages 1-3)
- `src/model/speech_llm.py` — SpeechLLM wrapper: frozen encoder + trainable projector + LoRA LLM
- `src/model/build_model.py` — Model construction, LoRA setup, projector building
- `src/data/` — Per-task dataset loaders (asr.py, s2tt.py, sqa.py, s2tsum.py, achap.py, multitask.py)
- `src/inference.py` — Inference pipeline with custom/surprise task support
- `configs/` — YAML configs for all stages and tasks

### New Files to Create
- `src/train_grpo.py` — GRPO training script using TRL's GRPOTrainer
- `src/rewards/` — Reward function module:
  - `src/rewards/__init__.py`
  - `src/rewards/task_rewards.py` — WER, COMET, BERTScore reward functions
  - `src/rewards/tars_alignment.py` — TARS representation + behavior alignment rewards
- `configs/stage4_grpo.yaml` — Stage 4 GRPO config
- `tests/test_grpo_smoke.py` — Smoke test for the full GRPO + TARS pipeline

### Architecture: GRPO + TARS Integration

```
Stage 4 GRPO Training Loop (per batch):

1. Sample prompts + audio from training data (reuse existing loaders)

2. Generate N samples per prompt via SPEECH PATH:
   Audio → Seamless M4T v2 (frozen) → Projector → Qwen3-4B (LoRA) → N outputs
   Collect: generated texts + hidden states at each LLM layer

3. Generate reference via TEXT PATH (for TARS):
   Transcript → Qwen3-4B (same LoRA, no encoder/projector) → reference output
   Collect: generated text + hidden states at each LLM layer

4. Compute rewards for each of the N samples:
   a. Task-specific reward (when ground truth available):
      - ASR: 1.0 - WER(generated, reference_transcript)
      - S2TT: COMET(generated, reference_translation)
      - SQA/S2TSUM: BERTScore(generated, reference_answer)
   b. TARS representation alignment:
      - cosine_sim(speech_hidden_states, text_hidden_states) per layer, averaged
   c. TARS behavior alignment:
      - BERTScore(speech_output, text_output)
   d. Composite: task_reward + λ_rep * rep_align + λ_beh * beh_align

5. GRPO policy update:
   - Group-relative advantage: A_i = (r_i - mean(r)) / std(r)
   - Policy gradient with KL penalty against reference model
   - Update: projector weights + LoRA adapters

Trainable during Stage 4: Projector + LoRA (same as Stage 3)
Frozen during Stage 4: Seamless M4T v2 encoder (same as all stages)
```

### Key Research References
| Paper | arxiv | Relevance |
|-------|-------|-----------|
| GRPO for Speech Understanding | [2509.16990](https://arxiv.org/abs/2509.16990) | Validates GRPO for SQA + S2TT on speech LLMs |
| GRPO for Speech Recognition | [2509.01939](https://arxiv.org/abs/2509.01939) | 18.4% WER improvement, rule-based rewards |
| TARS: Closing Modality Reasoning Gap | [2601.05543](https://arxiv.org/abs/2601.05543) | Asymmetric alignment rewards for speech LLMs |
| CORD: Cross-modal Self-distillation | [2601.16547](https://arxiv.org/abs/2601.16547) | Alternative alignment approach (lighter weight) |
| RBA: Reinforced Behavior Alignment | [2509.03526](https://arxiv.org/abs/2509.03526) | Teacher LLM alignment for speech |
| NAVER LABS IWSLT 2025 Winner | — | Brief Stage 3 preserves zero-shot generalization |

### Reward Signal Strategy

| Task | Has Ground Truth? | Task Reward | TARS Role |
|------|-------------------|-------------|-----------|
| ASR (EN) | Yes | WER | Supplementary |
| S2TT EN→DE | Yes (EuroParlST, GigaST) | COMET | Supplementary |
| S2TT EN→ZH | Yes (GigaST) | COMET | Supplementary |
| S2TT EN→IT | Yes (EuroParlST) | COMET | Supplementary |
| SQA (EN) | Yes (LibriSQA) | BERTScore | Supplementary |
| SQA (DE/ZH) | **No** | **None** | **Primary** |
| S2TSUM (EN) | Yes (NUTSHELL) | BERTScore | Supplementary |
| S2TSUM (DE/ZH) | **No** | **None** | **Primary** |
| ACHAP (EN) | Yes (YTSeg) | Task-specific | Supplementary |
| ACHAP (DE/ZH) | **No** | **None** | **Primary** |
| Surprise | **No** | **None** | **Primary (only signal)** |

## Ontology (Key Entities)

| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| SpeechLLM | core domain | speech_encoder, projector, LLM_backbone, LoRA | Target of RL optimization; trained via Stages 1-3, refined by Stage 4 |
| GRPO + TARS alignment | technique | group_size, KL_coeff, λ_rep, λ_beh, task_rewards | Applied to SpeechLLM as Stage 4 post-SFT |
| Surprise task | core domain | EN→DE, EN→ZH, ICL-based, no_training_data | Evaluated on SpeechLLM; TARS is only RL signal |
| Stage 4 post-SFT | pipeline phase | checkpoint_path, learning_rate, max_steps | Follows Stage 3, precedes inference/submission |
| Task-specific rewards | supporting | WER, COMET, BERTScore | Primary signal for covered tasks |
| TARS alignment rewards | supporting | rep_alignment, beh_alignment, λ_weights | Primary signal for zero-shot + surprise tasks |
| Cross-lingual zero-shot | gap | SQA/S2TSUM/ACHAP in DE/ZH | No training data; relies on TARS + LLM generalization |
| Constrained track | constraint | permitted_datasets_only | Limits training data to EuroParlST, GigaST, LibriSQA, NUTSHELL, YTSeg |
| TRL GRPOTrainer | framework | PEFT/LoRA, vLLM_optional, HF_ecosystem | Implementation vehicle for GRPO |
| RL pipeline | deliverable | train_grpo.py, rewards/, configs, smoke_test | The artifact being built |
| Smoke test | validation | forward_pass, generate, rewards, gradient_step | Proves pipeline works end-to-end |
| System paper | deliverable | deadline_Apr_24, documents_RL_experiments | Documents findings from RL experiments |

## Ontology Convergence

| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|----------------|
| 1 | 5 | 5 | - | - | N/A |
| 2 | 7 | 2 | 0 | 5 | 71% |
| 3 | 9 | 2 | 0 | 7 | 78% |
| 4 | 11 | 2 | 0 | 8 | 80% (1 removed) |
| 5 | 11 | 1 | 0 | 10 | 91% |
| 6 | 12 | 2 | 1 | 9 | 91% (1 removed) |
| 7 | 12 | 0 | 2 | 10 | 100% |
| 8 | 12 | 0 | 0 | 12 | 100% |

Ontology fully converged at Round 7. All entities stable through Round 8.

## Interview Transcript
<details>
<summary>Full Q&A (8 rounds)</summary>

### Round 1
**Q:** What specifically should autoresearch improve or prove for the surprise task?
**A:** Consider more methods from RL like GRPO, or some other things that we can add to this model to improve the performance.
**Ambiguity:** 66% (Goal: 0.40, Constraints: 0.30, Criteria: 0.20, Context: 0.50)

### Round 2
**Q:** What does 'improved performance' mean for your autoresearch mission? Should it optimize for all tasks jointly, focus on one, or target zero-shot ICL?
**A:** Consider all the tasks that are not already covered in the given datasets of IWSLT. The model has to optimize for all known tasks, but more focused on tasks not in the constrained track dataset.
**Ambiguity:** 52% (Goal: 0.60, Constraints: 0.35, Criteria: 0.40, Context: 0.55)

### Round 3
**Q:** What's the current state of your SFT training pipeline? Timeline and compute constraints?
**A:** Both improve submission and paper. We have multiple days worth of H200 hours. Autoresearch should not run whole experiments.
**Ambiguity:** 46% (Goal: 0.65, Constraints: 0.55, Criteria: 0.35, Context: 0.55)

### Round 4 [CONTRARIAN MODE]
**Q:** Autoresearch can't run experiments — so what should it produce? Should we drop autoresearch and build the pipeline instead?
**A:** Build the RL pipeline and run smoke test to confirm the pipeline will work.
**Ambiguity:** 36% (Goal: 0.75, Constraints: 0.60, Criteria: 0.55, Context: 0.60)

### Round 5
**Q:** What exactly should the RL pipeline include and what does the smoke test verify?
**A:** Please consider MS-SWIFT's GRPO implementation (https://swift.readthedocs.io/en/latest/Instruction/GRPO/GetStarted/GRPO.html).
**Ambiguity:** 32% (Goal: 0.80, Constraints: 0.60, Criteria: 0.60, Context: 0.65)

### Round 6 [SIMPLIFIER MODE]
**Q:** Is SWIFT the simplest path? SWIFT requires ~200+ lines of custom model registration for your SpeechLLM.
**A:** Forget about SWIFT GRPO, stay in current HF ecosystem. Also consider other RLs apart from GRPO. Which stage does the RL apply to — stage 1, 2, 3, or post-training?
**Ambiguity:** 31% (Goal: 0.75, Constraints: 0.70, Criteria: 0.60, Context: 0.65)

### Round 7
**Q:** Which RL method(s) should the pipeline support? GRPO, DPO, TARS, or multiple?
**A:** GRPO + TARS alignment (Recommended).
**Ambiguity:** 24% (Goal: 0.90, Constraints: 0.70, Criteria: 0.65, Context: 0.70)

### Round 8
**Q:** How does TARS help when we have expected outputs from the dataset?
**A:** [Explanation provided: TARS is reference-free, works for cross-lingual zero-shot + surprise where task rewards fail]
**Follow-up Q:** Proceed to spec crystallization?
**A:** Yes, proceed to spec (GRPO + TARS).
**Ambiguity:** 16% (Goal: 0.95, Constraints: 0.75, Criteria: 0.80, Context: 0.80)

</details>
