# REML-LoRA: RL-Based Meta-Learning for LoRA Adapter Composition

## Research Specification

### 1. Overview

Extend REML from composing neural network layers for sinusoidal regression to composing **LoRA adapters for large language models**. An RL agent learns to select and compose subsets of pre-trained LoRA adapters for unseen NLP tasks, replacing hand-crafted merging heuristics with a learned compositional policy.

**Base model:** Llama-3-8B (or Llama-3.2-3B for faster iteration)
**Gap in literature:** No published work uses RL as a meta-learner for LoRA adapter composition (as of Feb 2026). Existing methods use Nelder-Mead (LoRAHub), attention fusion (AdapterFusion), static heuristics (TIES-Merging, DARE), MoE routing (MOELoRA), or supervised fusion gates (LoRA-Flow).

### 2. Architecture

```
                    ┌─────────────────────────────┐
                    │   RL Policy (PPO + LSTM)     │
                    │                               │
                    │  State: task_embedding,       │
                    │         selected_adapters,    │
                    │         eval_loss             │
                    │                               │
                    │  Action: select adapter_i     │
                    │          OR set weight_i      │
                    │                               │
                    │  Reward: -loss on task         │
                    └──────────────┬────────────────┘
                                   │ compose
                    ┌──────────────▼────────────────┐
                    │      LoRA Adapter Pool         │
                    │                                │
                    │  adapter_1 (SST-2 sentiment)   │
                    │  adapter_2 (NLI)               │
                    │  adapter_3 (QA)                │
                    │  adapter_4 (summarization)     │
                    │  adapter_5 (translation)       │
                    │  adapter_6 (code gen)          │
                    │  adapter_7 (math reasoning)    │
                    │  adapter_8 (NER)               │
                    │  adapter_9 (paraphrase)        │
                    │  adapter_10 (toxicity)         │
                    └──────────────┬────────────────┘
                                   │ merge into
                    ┌──────────────▼────────────────┐
                    │     Llama-3-8B (frozen)        │
                    │  + merged LoRA weights         │
                    │                                │
                    │  Evaluate on held-out task     │
                    └───────────────────────────────┘
```

### 3. LoRA Adapter Pool

#### 3.1 Training Adapters

Train 10 LoRA adapters, one per diverse NLP task:

| # | Adapter | Dataset | Task Type | LoRA Config |
|---|---------|---------|-----------|-------------|
| 1 | sentiment | SST-2 | classification | r=16, alpha=32 |
| 2 | nli | MultiNLI | NLI | r=16, alpha=32 |
| 3 | qa_extractive | SQuAD v2 | extractive QA | r=16, alpha=32 |
| 4 | summarization | CNN/DailyMail | seq2seq | r=16, alpha=32 |
| 5 | translation | WMT14 en-de | seq2seq | r=16, alpha=32 |
| 6 | code | CodeAlpaca-20k | generation | r=16, alpha=32 |
| 7 | math | GSM8K | reasoning | r=16, alpha=32 |
| 8 | ner | CoNLL-2003 | token classification | r=16, alpha=32 |
| 9 | paraphrase | QQP | classification | r=16, alpha=32 |
| 10 | toxicity | Jigsaw Toxicity | classification | r=16, alpha=32 |

**LoRA config (uniform across all adapters):**
- Rank: 16
- Alpha: 32
- Target modules: q_proj, k_proj, v_proj, o_proj (attention layers)
- Dropout: 0.05
- Training: 3 epochs, lr=2e-4, batch_size=8, AdamW

#### 3.2 Adapter Storage

Each adapter is stored as a dict of (A, B) low-rank matrices per target module per layer. Total parameter overhead per adapter: ~4M params (for r=16 on Llama-3-8B attention).

### 4. RL Agent Design

#### 4.1 MDP Formulation

**State** (at step t):
- `task_embedding` (dim 768): Mean-pooled Llama hidden states over a few-shot prompt describing the target task (frozen, computed once)
- `selected_mask` (dim 10): Binary vector of which adapters are selected so far
- `current_weights` (dim 10): Mixing weights assigned to each adapter
- `eval_loss` (dim 1): Current loss on a small validation batch
- Total state dim: ~789

**Action space** (two-phase per step):
- Phase 1 (selection): Discrete(10) — select/deselect adapter i
- Phase 2 (weighting): Box(10) — set mixing coefficient for each selected adapter (softmax-normalized)

Simplified alternative: Discrete(10) selection over K steps (K=3-5), then uniform/learned weighting.

**Episode structure:**
- T=5 steps (select 3-5 adapters from pool of 10)
- At each step: select one adapter, merge current selection into base model, evaluate on small batch
- Terminal reward: negative loss on validation set of the target task

**Reward:**
- Non-terminal steps: 0 (sparse) OR small intermediate signal from validation loss delta
- Terminal step: `-validation_loss` (min-max scaled across tasks for stability)

#### 4.2 Adapter Merging

At each step, merge selected adapters using weighted linear combination:

```
merged_A = sum(w_i * A_i for i in selected)
merged_B = sum(w_i * B_i for i in selected)
```

where `w_i` are the mixing weights (uniform initially, or learned by the policy).

This is applied per-module, per-layer. The merged LoRA is applied to the frozen base model.

#### 4.3 Policy Network

- PPO with LSTM policy (same as original REML, via sb3-contrib RecurrentPPO)
- Hidden size: 256
- Learning rate: 3e-4
- Clip range: 0.2
- N_steps: 128 (short episodes)

### 5. Training Procedure

#### Phase 1: Train LoRA Adapters (offline, ~10 GPU hours on A100)
1. For each of the 10 tasks, fine-tune a LoRA adapter on Llama-3-8B
2. Save adapter weights to disk
3. Evaluate each adapter on all 10 tasks (cross-evaluation matrix)

#### Phase 2: Train RL Composition Policy (~5 GPU hours on A100)
1. Define held-out evaluation tasks (3-5 tasks from different distributions):
   - BoolQ (yes/no QA — different from SQuAD)
   - MRPC (paraphrase detection — related to QQP but different distribution)
   - XSUM (abstractive summarization — different from CNN/DailyMail)
   - HumanEval (code generation — different from CodeAlpaca)
   - ARC-Challenge (reasoning — different from GSM8K)

2. Meta-training loop:
   ```
   for epoch in range(N_epochs):
       for task in training_tasks:  # rotate through held-out tasks
           env = LoRACompositionEnv(base_model, adapter_pool, task)
           policy.learn(env, timesteps=2048)
   ```

3. The RL agent learns which adapter combinations work well for which task types.

#### Phase 3: Evaluate (~2 GPU hours)
- On fully held-out tasks never seen during meta-training
- Compare REML-LoRA against baselines

### 6. Baselines

| Method | Description |
|--------|-------------|
| **Single best adapter** | For each eval task, pick the adapter with lowest loss (oracle upper bound for single adapter) |
| **Uniform merge** | Average all 10 adapters with equal weights |
| **Random selection (k=3)** | Randomly select 3 adapters, uniform merge |
| **LoRAHub** | Nelder-Mead optimization over mixing coefficients (Huang et al., 2023) |
| **TIES-Merging** | Sign-aware static merging heuristic (Yadav et al., 2023) |
| **REML-LoRA (ours)** | RL policy selects and weights adapters |

### 7. Evaluation Metrics

For each eval task, measure:
1. **Task performance** (primary):
   - Classification: accuracy, F1
   - Generation: ROUGE-L, BLEU
   - QA: exact match, F1
2. **Few-shot adaptation speed**: Performance after 0, 5, 10, 20 gradient steps of fine-tuning the merged adapter
3. **Composition efficiency**: Number of adapters selected (fewer is better, all else equal)
4. **Inference overhead**: Latency comparison vs baselines

### 8. Ablation Studies

1. **Policy vs random selection** (same as original REML ablation, but for LoRA):
   - REML-selected adapters vs random k adapters from the trained pool
2. **Number of selected adapters** (k=1,2,3,4,5): How many adapters does the policy learn to select?
3. **Adapter pool diversity**: Train adapters on more/fewer task types, measure composition quality
4. **LoRA rank sensitivity**: r=4, 8, 16, 32 — does higher rank change the optimal composition?
5. **State representation**: With/without task embedding, with/without eval loss feedback

### 9. Compute Budget

| Phase | Hardware | Estimated Time | Cost (cloud) |
|-------|----------|---------------|--------------|
| Adapter training (10x) | 1x A100 80GB | ~10 hours | ~$20 |
| RL policy training | 1x A100 80GB | ~5 hours | ~$10 |
| Evaluation + ablations | 1x A100 80GB | ~5 hours | ~$10 |
| **Total** | | **~20 hours** | **~$40** |

Google Colab Pro+ with A100: feasible in ~2-3 sessions.

For faster iteration, use Llama-3.2-3B (halves compute) or reduce adapter pool to 5.

### 10. Implementation Plan

| Step | Task | Files | Time Est. |
|------|------|-------|-----------|
| 1 | LoRA adapter training script | `src/train_lora_adapters.py` | 1 day |
| 2 | Cross-evaluation matrix | `src/evaluate_adapters.py` | 0.5 day |
| 3 | LoRA composition gym environment | `src/lora_composition_env.py` | 2 days |
| 4 | RL training loop | `src/train_reml_lora.py` | 1 day |
| 5 | Baselines implementation | `src/baselines.py` | 1 day |
| 6 | Evaluation + plots | `src/evaluate_reml_lora.py` | 1 day |
| 7 | Ablation experiments | `src/lora_ablations.py` | 1 day |
| 8 | Colab notebook | `notebooks/reml_lora.ipynb` | 0.5 day |
| **Total** | | | **~8 days** |

### 11. Key Risks and Mitigations

1. **RL instability with LLM evaluation in the loop**
   - Mitigation: Cache adapter merging results; use small eval batches (32 examples); normalize rewards across tasks
   
2. **Adapter merging may not be compositional**
   - Mitigation: Use task-arithmetic framework (Ilharco et al., 2023) which shows LoRA deltas ARE compositional in practice
   
3. **Compute cost of LLM forward pass in RL loop**
   - Mitigation: Pre-compute adapter merge matrices offline; only run LLM forward pass at terminal step; use 8-bit quantization

4. **Task embedding may leak information**
   - Mitigation: Use a generic task descriptor (natural language prompt) rather than labeled examples; ablate with/without embedding

### 12. Expected Contributions

1. **First RL-based meta-learner for LoRA adapter composition** — fills a clear gap in the literature
2. **Learned composition vs heuristic merging** — empirical comparison on diverse NLP tasks
3. **Generalizable framework** — the REML formulation naturally extends from layers to adapters; demonstrates the generality of the RL-as-meta-learner paradigm
4. **Practical tool** — if the policy learns meaningful composition, it provides an automatic adapter selection system for practitioners with adapter libraries
