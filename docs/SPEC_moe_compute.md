# SPEC — MoE compute, MoE ablations, transformer variants (Opus P5)

*Written 2026-10-21 by Opus. Status: ready for Sonnet. Depends on `SPEC_2khz_emg.md`: its stem keeps the expert input at T=64, which removes the memory pressure that motivated this task.*

## 1. Memory: do the cheap, exact fix and stop there
- **Do:** the running-sum rewrite at all six `torch.stack(expert_feats)` sites (`MOE_encoder.py:710, 949, 1145, 1337, 1618, 1643`).
  - Accumulate `h = h + w[:, e, None, None] * expert_e(x)` instead of materialising `(B, E, C, T)` and its weighted copy.
  - This is identical math. Test equivalence to atol 1e-6, in float64, on the tiny config, for every MoE class.
- **Also do:** an optional `config["moe_expert_checkpointing"]` that wraps each expert call in `torch.utils.checkpoint` (`use_reentrant=False`). Default `False`, set explicitly in `make_base_config`.
  - Test that gradients through `functional_call` match the non-checkpointed path. MAML here is first-order, so this is expected to hold.
- **Do not do:**
  - Sparse dispatch, grouped/chunked evaluation, or Soft-MoE *for memory*. With T=64 the dense bank is cheap.
  - Sparse dispatch also runs into B14: experts with no routed samples get no gradient under MAML, which is a training-dynamics change and not an efficiency tweak.
- **Debug script:** a peak-memory script (`system/MOE/debug_peak_memory.py`) prints `torch.cuda.max_memory_allocated` for M0 at T=64 and for the EMG-RAW stem at 6,144 samples, B = support + query of a 10-way 5-shot episode (100 samples), old vs new mixing. Kai runs it once on the debug partition.

## 2. MoE ablations worth cluster time (answer "does MoE help, and why?")
The archived evidence (LIMITATIONS A4/C1/C2):
- routing is about 98% of maximum entropy
- M0's gain over the parameter-matched A4 looks like ensembling, not specialisation

Do not spend weeks on gating variants before knowing whether MoE survives `DECISION_method.md`. One compact matrix on dev users, at the primary cell, with M0's tuned hyperparameters:

| ID | Experts E | Routing | Purpose |
|---|---|---|---|
| X0 | 1 (= A4, param-matched) | — | no-MoE reference |
| X1 | 22, top-9 soft (current M0) | softmax + top-k mask | current |
| X2 | 22, dense uniform weights (gate frozen at 1/E) | none | **the key control: is the gate doing anything?** If X2 ≈ X1, the benefit is ensembling. |
| X3 | 8, top-2 + load-balance + router z-loss (ST-MoE style) | sparse-ish | standard modern MoE |
| X4 | 8, top-2 + task-conditioned routing: **new code**, there is no such flag in the repo yet (the archived PLAN's `MOE_task_routing_mode` was never implemented). The router input is the per-sample context concatenated with the mean context of the episode's support set; the support set is available inside the inner loop. | conditional | the only routing idea with a plausible originality claim against MoEMeta (archived PLAN 1.5 C6) |

- **Promotion to 7 folds:** whichever of X1/X3/X4 is best on dev, plus X0 and X2.
- **Report:** routing entropy, per-gesture and per-user routing mutual information (existing `MOE_analysis.py`), and accuracy.
- **If X2 ≈ X1 within MDE:** the paper states plainly that the MoE benefit is an ensembling effect. That is a legitimate finding and better than a reviewer discovering it.

## 3. Transformer variants Kai asked for
All variants:
- Implement `forward(x_emg, x_imu=None, demographics=None, return_routing=False)`. MoE variants return `(logits, routing_dict)` with the same keys as `DeepCNNLSTM_EncoderMOE`.
- Pass the `tests/test_models_functional.py` suite. Add every class to `ALL_MODELS` there.

**Attention implementation:**
- Do **not** use `nn.MultiheadAttention` or `nn.TransformerEncoderLayer`. Their fast path bypasses `functional_call` substitution in eval mode. The existing TST test passes only because `mamlpp_adapt` forces `train()`.
- Write a small `FunctionalSelfAttention` with explicit `q, k, v, o` `nn.Linear` layers and `torch.nn.functional.scaled_dot_product_attention`.
- Add a test asserting that substituting `q.weight` via `functional_call` changes the output in **eval** mode.

**Common block:** pre-LN transformer block with learned positional embedding, GELU MLP (4x), dropout. Token sequence = the 64 time steps.

**Param budget:**
- **Matched** preset: M0's parameter count ±10%, enforced by `assert_param_budget` in model construction and in HPO (`optuna.TrialPruned`).
- **Naive** preset: the shipped small TST, about 0.2M.

| ID | Architecture | Notes |
|---|---|---|
| T0 | Single transformer over the 88-channel input (current `TST`, patch_len ∈ {1, 2, 4}, CLS token) | Exists. Needs the matched preset, the functional attention swap, and its own HPO. Run as both MAML (like A4) and supervised fine-tune (like A2). |
| T1 | EncoderMoE with the shared 3-layer BiLSTM replaced by a transformer encoder over the mixed feature map `(B, 256, T)` → tokens `(B, T, 256)` → 2–4 blocks → mean pool → head | Tests "why an LSTM?". Cheapest of the MoE variants. |
| T2 | EncoderMoE with each expert a small transformer: a shared linear patch/channel embedding to d=128, then each expert a 2-block transformer → `(B, T, 128)`, then weighted mix, then the existing LSTM | E reduced to 8 to stay within budget; top-2 routing as X3 |
| T3 | T2 experts + T1 head (all attention) | Run only if T1 or T2 is within MDE of, or better than, M0 on dev users |

**Inner-loop parameters for MAML** (`inner_param_filter`): head + last block for T0/T1/T3; head + LSTM for T2 (mirrors M0). The choice is an HPO dimension, `maml_inner_scope ∈ {head, head+last_block, all}`, selected on dev users.

**Order and promotion:**
- Order: T0 matched → T1 → T2 → T3 (conditional).
- Promote to 7 folds x 3 seeds: T0 matched and the best of T1/T2/T3 on dev.
- The naive T0 runs 7 folds x 1 seed (it is cheap), to show the effect of scale honestly.

## 4. Tests Sonnet adds
- Running-sum equivalence for all MoE classes, forward and gradients.
- Checkpointing gradient equivalence.
- `FunctionalSelfAttention` substitution in eval mode.
- Param-budget assertion: each matched preset is within ±10% of M0.
- Shape and routing-dict contract for T1/T2/T3.
- X2's frozen uniform gate: gradients to gate parameters are `None` or zero, and the outputs equal the mean of the experts.
