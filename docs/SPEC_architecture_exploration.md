# SPEC — architecture exploration: MoE revamp, fusion, transformers, alternatives, raw EMG

*Written 2026-10-30 (Opus) from Kai's questions of the same day; §3-4 revised the same afternoon after his review. Builds on `SPEC_moe_compute.md` (X/T matrices), `SOTA_COMPARISONS.md` (baseline list), `SPEC_2khz_emg.md` (representations), `SPEC_hpo.md` (HPO protocol) and `DECISION_method.md` (go/no-go rules). Status: design agreed with nobody yet. Kai to confirm the candidate list (§2) and the budgets (§3).*

## 0. Ground rules
1. **Dev users only.** All exploration uses the 4 dev users (P005, P109, P116, P131) and their cached dev episodes, via the HPO machinery (`ablation_hpo.py`, objective = per-user mean at 10-way 1-shot). The 28 fold users are touched only for the final runs of the finalists.
2. **Representation is fixed: `fs64`** (dev decision, notebook 01). Every candidate uses it, plus one raw-EMG ablation (§2E).
3. **The bar to beat.** A no-training classical baseline on the same fs64 input:
   - dev users (notebook 01): balanced-fusion 1-NN L1 = 78.0% at 10-way 1-shot;
   - test users: kNN-L1 = 72.9% at 10-way 1-shot.
   A learned model that cannot beat 78% on dev is not a method; `DECISION_method.md` R4 applies.
4. **One Optuna study per architecture, equal budgets.** Do not put the architecture in one study as a categorical choice:
   - TPE then spends most trials on whichever arm looks good early, so the losing arms are under-tuned and their numbers are not reportable;
   - pruning compares arms that train at different speeds;
   - reviewers want "each variant tuned with the same budget", which a mixed study cannot show.
5. **Every screened variant is listed** in an appendix table (dev numbers). This pre-empts the "you tried many things and reported the best" critique.

## 1. What EncoderMoE is today (from the code, 2026-10-30)
- **Fusion:** early. EMG and IMU channels are concatenated at the input (`torch.cat([x_emg, x_imu], dim=1)`, `DeepCNNLSTM_EncoderMOE.backbone`). On fs64 that is 112 + 72 = 184 channels. Every expert sees both modalities.
- **MoE:**
  - E = 22 expert CNNs (3-layer, 64 base filters), top-k = 9, softmax gate with temperature 1.53;
  - router = a context projector over the raw input (`MOE_routing_signal = context_proj`);
  - load-balancing aux loss 0.033 (outer loop); no shared expert.
  - The expert outputs are mixed by a running weighted sum, then fed through 3 LSTM layers (64 hidden) and a linear head.
- **Adaptation:** MAML++, and the inner loop updates **everything** (experts, gate, LSTM, head).
- **Known behaviour:**
  - routing is close to uniform: about 98% of maximum entropy, and every expert is used by 30/32 users;
  - Lpuc's point stands: E = 8 matched E = 22 in the old sweep.
  - The current entropy diagnostic is saturated at ln(top_k), so it cannot show collapse. Fix it before trusting new routing plots (debug results, issue 5).
- **The tuning search already covers the MoE size:** E ∈ {4, 8, 16, 22, 28, 36}, top-k ∈ {1, 2, 4, 6, 9, 12} (clipped to E). **Kai's "too many experts / k too high" is answered by tuning, not by new code.**

## 2. Candidates

### A. MoE revamp (structural variants; E and k are tuned inside each)
| ID | What | Why | Code | Priority |
|---|---|---|---|---|
| X0 | No MoE, param-matched (= A4) | reference | exists | required |
| X1 | Current M0 (soft top-k, context router) | the submitted method | exists | required |
| X2 | Same experts, **uniform fixed gate** (1/E, no router) | **the key control:** routing or just ensembling? | small (`_forced_gate_weights` hook exists) | **required** |
| X3 | Sparse top-1/2 + load-balance + router z-loss (ST-MoE style), small E | the standard modern MoE recipe; directly tests "fewer experts, lower k" | small–medium | high |
| X4 | Task-conditioned routing: router input = sample context ⊕ mean context of the episode's support set | the only routing idea with an originality case against MoEMeta | medium | high |
| X5 | Shared expert + routed experts (DeepSeek-MoE style; `MOE_use_shared_expert` exists) | common fix when routing is near-uniform; the shared expert carries common structure | trivial (flag) | medium |
| X6 | Cross-attention-gated routing: router input = EMG features cross-attended to IMU (or vice versa) | Kai's idea; it couples fusion and routing | medium | after the F screen |

### B. EMG–IMU fusion
| ID | What | Code | Priority |
|---|---|---|---|
| F0 | Early concat (current) | exists | reference |
| F1 | Separate per-modality CNN encoders, concat features before the temporal model (mid fusion) | small | high (the cheapest real alternative) |
| F2 | **FiLM:** the IMU branch produces per-channel γ, β that modulate the EMG features (and/or the reverse) | small (a `FiLMConditioner` exists in `AttentiveMatchingNetwork.py`; it must be ported functional-call-safe) | high |
| F3 | **Cross-attention fusion:** EMG tokens attend to IMU tokens over the 64 steps (EMG-CrossFormer style) | medium (needs the functional attention block, `SPEC_moe_compute.md` §3) | high, shares code with the transformers |
| F4 | Modality-specific expert pools (EMG experts and IMU experts, each routed) | medium | medium |

- **How to choose without a factorial explosion:** screen F0-F3 on the **cheaper no-MoE backbone (X0/A4)**, pick the best fusion, then carry it into the MoE variants.
- Fusion and MoE interact weakly compared with their cost. If F-best on X0 beats F0 clearly, re-check it on X1 once.
- X6 lives in the MoE family because it changes routing, not just the representation.

### C. Transformers (reviewer-required: Lpuc, fkxS)
| ID | Kai's label | What | Priority |
|---|---|---|---|
| T0 | (a) | Single transformer encoder (64 time tokens), MAML++, no MoE, param-matched to M0 | **required** |
| T2 | (d) | EncoderMoE with CNN experts, LSTM replaced by a transformer | **required** (the most direct answer to "why LSTM?") |
| T4 | (b) | Transformer whose FFN layers are sparse MoE layers (Switch / ST-MoE): the *standard* MoE-transformer, which reviewers will recognise | recommended |
| T1 | (c) | EncoderMoE with transformer experts (+ LSTM) | low: MAML through 4-36 transformer experts is the most expensive variant, and T4 is the standard way to "put MoE in a transformer" |
| T3 | (c)+(d) | Transformer experts and a transformer temporal model | only if T1 and T2 both win |

- Constraints from `SPEC_moe_compute.md` §3:
  - use functional attention (no `nn.MultiheadAttention`, whose fast path breaks `functional_call` in eval);
  - a param-matched preset (±10% of M0) plus the shipped small TST as the "naive" preset;
  - every model passes `tests/test_models_functional.py`.
- Each T gets its own HPO with the same budget. Transformers need their own lr/warmup/dropout ranges, so do not reuse the CNN-LSTM space blindly.

### D. Alternatives to EncoderMoE (different learning approach, not a different encoder)
| ID | What | Why | Priority |
|---|---|---|---|
| P1 | **ProtoNet trained episodically** on fs64 (same encoder family, metric-based, no inner loop) | the closest prior work in this population (Sarwat et al. 2024, stroke, one-shot ProtoNet). With kNN already at 73%, a learned metric on fs64 is the most likely strong simple method. | **high** |
| P2 | Pretrain + fine-tune with a dev-tuned budget (Chen et al. 2019 "baseline++"; = A1/A2 with tuned ft) | exists; the tuning now covers the fine-tune budget | required (exists) |
| P3 | **ADANN-style (multi-domain DANN core):** supervised pretrain with a gradient-reversal user-ID head (users as domains), then fine-tune on the support set (Côté-Allard 2020) | EMG-recognisable cross-user baseline; tests whether *removing* user identity helps when labels are user-defined (prediction: it hurts, which is an informative finding) | medium |
| P4 | Pretrained encoder + kNN / nearest centroid (frozen features, no adaptation) | separates "better features" from "better adaptation"; A14 already does this for the M0/A4 backbones | exists (A14) |
| P5 | CNP / attentive neural process (context = support set) | Kai asked; low expected value with 10 reps × 10 classes per user and 20 training users; amortised inference tends to underfit task-specific mappings this idiosyncratic | low (Tier 3) |
| P6 | Learned `$B` / covariance metric (`system/nonparametric/metric_models.py`) | the natural "method" if a classical baseline wins (R4) | conditional on R4 |

### E. Raw 2 kHz EMG: features vs raw, and fairness to Meta GNI
- **Main table: `fs64` for every learned model.** Classical methods get their natural input, all derived from the same raw segments:
  - Riemannian on native-rate covariance;
  - `$B`/kNN on fs64 or env20;
  - Meta GNI (Kaifosh & Reardon) on raw 2 kHz with its own preprocessing (40 Hz high-pass, noise-floor normalisation).

  Giving each method its standard input is normal practice and defensible. Every method sees the same trials, segments and episodes.
- **What reviewers *will* attack** is an unpaired comparison: "your model gets engineered features, theirs gets raw". ARaC asked for exactly this ("apples-to-apples EncoderMoE featurisation"). Two runs close the gap:
  1. **EMG-RAW ablation** (`SPEC_2khz_emg.md` §3): M0 (and A4) with a learned strided conv stem on the native-rate 3 s crop, experts still at T = 64. This answers "do hand-crafted features matter?". If raw ≈ fs64, the features are a convenience, not the source of the gain.
  2. **The matched EMG-only comparison ARaC asked for:** M0 trained EMG-only on raw 2 kHz (stem) vs GNI fine-tuned EMG-only, same episodes.
- The stem is not built yet (`emg_repr = "raw3s"`, a raw builder and `emg_stem` are all pending). Medium effort. It is not needed before the weekend.

## 3. Protocol and budgets (revised 2026-10-30 after Kai's review)
**Why staged.** Kai's point: if the MoE scheme changes, M0 changes, and the param-matched controls (A1, A4) must be re-matched to it. Tuning them fully before the architecture is settled wastes the budget.

The other problem is budget. The full M0 space has about 17 HPs. 16 TPE trials over 17 dims is random search, which is too few to rank architectures (Kai: "isn't 16 way too small?" Yes, for the full space).

The fix is to separate the shared training HPs from each architecture's own HPs.

**Stage 0 (now): shared training HPs.**
- Full HPO of **A4** (MAML, no MoE, fixed width): the cheapest MAML model, and its architecture does not depend on the MoE outcome.
- Also **A2** (supervised, fixed arch), which is needed in every version of the paper.
- Freeze A4's best training HPs for stage S with `hpo_tools.py freeze`: outer_lr, wd, inner steps, both inner step sizes, LSLR on/off, MSL, episodes per epoch.

**Stage S (screen), dev users, fs64.**
- Each variant is its own study (`--study-suffix screen --fixed-params ...`) and searches only its own HPs, 2-6 dims:
  - X: bank size, dropout, router HPs;
  - F: branch width, attention width;
  - T: d_model, heads, depth, FF width, dropout;
  - metric learners: lr, wd, episodes, plus kind-specific HPs.
- **24 trials each**, with MedianPruner. At 2-6 dims this is a reasonable budget; at 17 dims it would not be.
- **Caveat:** frozen HPs tuned for a CNN-LSTM can disadvantage the transformers. So the T family freezes only the inner-loop HPs and keeps outer_lr / wd searched (a separate freeze file without those two keys).
- **Order:** the fusion screen on A4 (F1-F3 vs A4) and the MoE screen (X2-X6 vs M0 at the frozen HPs) can run in parallel. Then the transformers, then the alternatives.
- **Promotion rule (pre-stated):** promote a variant if its best-trial dev mean at 10-way 1-shot beats its family reference by ≥ 2 points, with ≥ 3/4 dev users better. Otherwise keep the simpler model.
- Required controls (X0 = A4, X1 = M0, X2, T0, T2, P1, P2) go to stage F regardless.

**E and top-k: a grid, not TPE.**
- On the best MoE-family config: E ∈ {4, 8, 16, 22, 36} × k ∈ {1, 2, 4, 9}, k ≤ E (17 configs), 2 seeds.
- Built with `hpo_tools.py grid` and enqueued with `ablation_hpo.py --enqueue-json ... --enqueue-only`.
- This separates the MoE size from the other HPs (Kai's concern) and gives reviewers the E=8-vs-22 figure.
- Inside TPE, E and k are entangled with the router HPs, and 60 trials cannot map them.

**Stage F (full), finalists only.**
- The finalists plus their re-matched controls (M0 or its successor, A1, and A4 re-matched to the final bank) get the **full space with 100 trials** if the trial cost allows, 60 at minimum.
- Then the top-3 × 3-seed replay, export to `best_hpo_configs.json`, and 7 folds × 3 seeds.
- Replay and export restore the frozen HPs from each trial's `fixed_params` attribute.

**Is this "architecture search"?** Yes, in the plain sense: a small set of hand-designed architectures, each with an equal small tuning budget, plus a pre-stated promotion rule. Neural architecture search (a searched cell space) is not warranted here, and reviewers would not expect it.

**Compute.**
- Every trial now records its wall time, GPU, peak memory and parameter count. `hpo_tools.py report` totals them.
- The debug runs (CURRENT_TODOs §2B) give the per-epoch time; set the budgets from that.
- Rough count: stage S ≈ 24 trials × ~14 variants ≈ 340 trials, most of them pruned or cheap (the metric learners have no inner loop). Stage F ≈ 100 × ~6 ≈ 600 trials.
- **If compute is short,** cut in this order: T1/T3, P5, X5, F4, X6, the 100 → 60 trials of stage F. Never cut X2, T0, T2 or P1.

## 4. Status (2026-10-30, end of the Opus code session)
All rows below are built and pass the local tests (356 in total). **None has run on the cluster yet**: CURRENT_TODOs §2B has the debug commands.

| Item | HPO token | Where |
|---|---|---|
| X2 uniform gate (dense, no router) | `X2` | `MOE_encoder.DeepCNNLSTM_EncoderMOE` (`MOE_gate_mode`) |
| X3 sparse top-1/2 + Switch balance + router z-loss, skipping unselected experts | `X3` | same class; z-loss in `mamlpp._compute_aux_loss` |
| X4 routing on [sample context, support-set mean context] | `X4` | same class; `mamlpp.task_support` sets the support set per episode (train, adapt, eval, collapse check) |
| X5 shared expert | `X5` | existing flag |
| X6 cross-attention router (EMG tokens attend to IMU tokens) | `X6` | `MOE_encoder.CrossModalRouter` |
| T2 M0's experts + transformer instead of the LSTM | `T2` | same class (`temporal_model`) |
| Functional attention, transformer block, token MoE-FFN | (shared) | `arch_variants/attention.py` |
| F1 mid, F2 FiLM (identity at init), F3 cross-attention fusion | `F1 F2 F3` | `arch_variants/flexnet.py` (also supports a fusion encoder as expert, for carrying the winner into the MoE) |
| T0 transformer on the 64 steps; T4 Switch-style MoE-FFN transformer | `T0 T4` | `flexnet.py` |
| P1 ProtoNet (A4 encoder or M0's bank), Matching Net, Relation Net, R2-D2, FEAT | `P1 P1_moe P1_match P1_rel P1_ridge P1_feat` | `arch_variants/metric_learners.py` |
| P3 ADANN-style (multi-domain DANN core; no domain-specific BN) | `P3` | `arch_variants/adann.py` |
| ANIL (head-only inner loop) | `A4_ANIL M0_ANIL` | existing `maml_inner_param_include` |
| Routing diagnostics: pre-top-k entropy / ln E, usage entropy, dead experts, I(expert; user), I(expert; gesture \| user), with permutation nulls | stored per MoE trial (`routing`) | `MOE/routing_metrics.py` |
| Per-trial timing, `hpo_tools.py` (report / freeze / grid), frozen HPs, study suffix, enqueue | all tokens | `ablation_hpo.py`, launcher |

**How to read X1 vs X2:**
- If X2 ≈ X1 and X1's routing shows soft entropy / ln E near 1 and I(expert; user) near its null, the gate is ensembling, not routing. The paper must then not claim user-specific routing.
- If X1 > X2 and the routing is sharp and user-informative, routing is doing work.

On Kai's question "is uniform routing good?": *balanced usage across the batch* is good, because it prevents collapse. *Uniform weights per sample* means the router does nothing. A healthy router has low per-sample entropy and high usage entropy.

**Not built:** EMG-RAW stem; F4; T1/T3; CNP (P5); learned `$B` (P6, only if R4 triggers); 7-fold eval scripts for the new tokens (finalists only, after stage S).
