# Open Questions

## grpo-tars-pipeline-plan (R2) - 2026-04-03

### Carried Forward (from R1)

- [ ] **TARS lambda values (lambda_rep, lambda_beh):** The plan defaults to 0.3 for both. The TARS paper may suggest different values. Should these be tuned, or are defaults acceptable for the constrained submission? -- Affects reward balance and training dynamics.

- [ ] **num_generations tradeoff:** 4 generations per prompt is conservative for memory but may not give enough variance for GRPO advantage estimation. Should we try 8 (still fits in H200 budget) or stick with 4 for faster iteration? -- Affects training quality vs speed.

- [ ] **TARS hidden state layer selection:** The plan uses the last hidden state for representation alignment. Some papers use an average of the last N layers or a specific intermediate layer. Which layer(s) should be used? -- Affects alignment quality.

- [ ] **COMET model variant:** Which COMET model to use? `Unbabel/wmt22-comet-da` (default) or a smaller/faster variant? COMET inference during reward computation adds latency per batch. -- Affects reward computation speed.

- [ ] **BERTScore model variant:** Default BERTScore uses `roberta-large`. For multilingual S2TT tasks, should we use `bert-base-multilingual-cased` instead? -- Affects cross-lingual reward accuracy.

- [ ] **Stage 3 checkpoint readiness:** Is a Stage 3 checkpoint available for Stage 4 to start from, or does Stage 3 training need to complete first? -- Blocks Step 6 integration validation.

- [ ] **Pin TRL version:** Plan recommends using TRL reward function signature convention but does not depend on TRL trainer. Confirm TRL is still needed as a dependency or can be dropped. -- Affects dependency footprint.

### New (from R2 revision)

- [ ] **s2tsum dataset loader:** Config references `nutshell` dataset for s2tsum training. Verify that `src/data/s2tsum.py` has a working `load_s2tsum_dataset("train")` implementation. -- Blocks s2tsum reward computation during training.

- [ ] **achap dataset loader:** Config references `ytseg` dataset for achap training. Verify that `src/data/achap.py` has a working `load_achap_dataset("train", "en")` implementation. -- Blocks achap reward computation during training.

- [ ] **Differential LR in GRPOSpeechTrainer:** Stage 3 uses differential LR (separate rates for projector vs LoRA). Should Stage 4 GRPO also use differential LR? The plan specifies `lr_projector: 1e-6` and `lr_lora: 5e-7` but the `GRPOSpeechTrainer` needs to implement the same per-group optimizer as `src/train.py`. -- Affects training dynamics.

- [ ] **GRPO epsilon value:** The plan defaults to `epsilon: 0.2` (standard PPO/GRPO clipping). DeepSeekMath uses 0.2 but some GRPO implementations use smaller values (0.1). Which value to use? -- Affects policy update magnitude.

- [ ] **Reward function batching vs per-sample:** COMET and BERTScore are much faster with batch computation. The current pseudocode shows per-sample loops. Should the executor implement batch collection + batch inference within each reward function? -- Affects training throughput (potentially 4-10x for reward computation).

### Resolved (from R1)

- [x] **TARS text-path caching:** Resolved by making behavior alignment OFF by default. When enabled, caching text-path outputs per prompt (not per generation) reduces cost by N_generations factor.

- [x] **Reward function batching:** Moved to open question about executor implementation detail (see above).
