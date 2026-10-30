# ANALYSIS — EncoderMoE vs MoEMeta, and the plan for originality

*Written 2026-10-30 (Opus) from Kai's questions of the same day. Sources: both papers in `docs/`, the OpenReview thread (`docs/papers/OpenReview_downloaded.pdf`), and the code in `system/`. Status: MoEMeta-TS is implemented (`system/arch_variants/moemeta_ts.py`, tokens `Q1*`, 2026-10-30 evening, Fable); the substitutions forced by the domain change are listed in its module docstring. The body-region line (§6) ran its Gate 0 and was stopped.*

## 1. Verdict
- **EncoderMoE and MoEMeta differ mechanically.** A time-series MoEMeta would not be EncoderMoE (§3).
- **The idea-level overlap cannot be argued away.** "MoE + meta-learning" has a NeurIPS 2025 precedent. Do not claim novelty there.
- **Citing MoEMeta alone is not enough for an ML venue.** ARaC asked for a comparison (Q1). It is probably enough for IMWUT, ASSETS or TNSRE.
- **Separating from MoEMeta does not fix originality by itself.** Only ARaC named MoEMeta. fkxS gave originality 1 because the parts are well known. ARaC also said no further experiment resolves originality.

## 2. Comparison
From the MoEMeta paper only. The official repo (`github.com/alexhw15/MoEMeta`) held only a README when checked on 2026-10-30, so their code could not be compared.

| | MoEMeta | EncoderMoE |
|---|---|---|
| Task | One relation, K positive triplets, rank candidates | One user, N classes, classify |
| New at test time | The relation; entities come from a known vocabulary | The person's signals and their classes |
| Role of the MoE | Builds one task vector from the support set; queries never enter it | Feature extractor; every sample passes through it |
| Experts | 32 MLPs over embeddings, top 5 | 22 CNNs over raw windows, top 9, load-balancing loss |
| Adapted per task | Task vector plus 3 projection vectors | All 5.5M parameters |
| Starting point | Task vector generated from the support set | Meta-learned init; the head starts with no class information |
| Scoring | Translational distance, margin loss | LSTM + MLP head, cross-entropy |

## 3. A time-series MoEMeta is a generalised ProtoNet
Translate a triplet as (user centroid `ū`, gesture, window `z`). With an identity expert at 1-shot, the score becomes `ū + (z_i − ū) − z_q = z_i − z_q`, which is 1-nearest-neighbour. Given that kNN on fs64 reaches 72.9% at 10-way 1-shot, it could be a real competitor.

## 4. Weak spots in what we already posted
1. **"Not for time series" is weak.** Our gate averages over time (`system/MOE/MOE_encoder.py:198`), and MixER (arXiv 2502.05335) already combines MoE, meta-learning and time series.
2. **Port B is ANIL, not MoEMeta.** It has no support-generated starting point. The prototype readout on the frozen M0 backbone got 78.7% against Port B's 59.76% (unpaired, 3-way, old code). Most of the 28.6-point cost may come from head initialisation rather than freezing.
3. **"The mixture must be adapted, not frozen" is at risk.** The 78.7% and untrained kNN on fs64 both point against it. Do not repeat it until the new runs support it.
4. **"Their mapping is shared, ours is not" does not separate us from MoEMeta.** With labels permuted per episode, we are also doing novel-class few-shot learning. The stronger differences: our inputs are new too, and only about 10% of cross-user motif matches share a label (notebook 03).
5. **X4 moves toward MoEMeta.** Support-mean routing is their signal, and Port A was already a null (91.24 vs 91.25). Keep X4 as a control. Drop "originality case" from `PLAN_resubmission.md` §5 and `SPEC_architecture_exploration.md`.
6. **The port baselines do not match.** Port A's baseline is 91.25 and Port B's is 88.4, both reported as 1-shot 3-way. The cause was not traced.

## 5. Track A: know what EncoderMoE can do (required)
This is the critical path and it ships whatever Track B does.

**MoEMeta-TS, the faithful port.** Add it beside `system/arch_variants/metric_learners.py`.
- **Encoder:** the same backbone family, trained end to end, initialised from supervised pretraining.
- **Experts:** each takes `(ū, z_i)`, with top-N gating as in their paper. Average per class to get `R_c`.
- **Adaptation:** update `R_c` and the projection vectors on the support set. Everything else stays frozen at test time.
- **Scoring:** rank the N classes by distance, with a margin loss using the other classes as negatives. Run cross-entropy as a sensitivity check.
- **Comparison:** the same HPO budget, cached episodes and 7 folds as every other method.
- **Variants:** no local adaptation, MLP instead of MoE, uniform gate.

**Novel-user × novel-gesture interaction.** Hold out gestures, cross seen users vs novel users, and compare M0 with MoEMeta-TS. Prediction: the gap shrinks when only labels are new and grows when the person is new. Pre-register this.

**Update `DECISION_method.md` before matched numbers exist.** R1 should name MoEMeta-TS as a baseline. R3 covers only non-MoE methods, so a tie or loss to MoEMeta-TS has no rule yet.

| Outcome | What the paper says |
|---|---|
| M0 wins | Full adaptation is justified for new users |
| MoEMeta-TS ties or wins | Adopt it and credit them; application and benchmark paper |
| Classical wins | Benchmark paper (`DECISION_method.md` R4) |

**Items promised in the NeurIPS discussion, still open:** transformer comparison, upper bound, EncoderMoE on 2 kHz EMG-only, modality ablation on the full protocol, user sweep at the top end, EMGBench with gate statistics, and the Rahimian 2021 and Antoniou citation fixes. X2 (uniform gate) runs first because it decides whether any routing claim survives.

**Writing rules.**
- Name MoEMeta in the introduction and related work, with a short version of the §2 table.
- Keep the application and empirical contribution labels unless a new variant wins.
- Cite ANIL, MMAML, task-similarity MAML, MixER and sEMG MoE work (check whether MoEMba is the Shabanpour 2025 paper already cited).
- Say a literal MoEMeta run is impossible, so we ported it, and list every substitution. Call Port B "ANIL".
- Leave independent development out of the paper.

## 6. Track B: make the method more original (time-boxed)
All of this uses dev users only, so the 28 test users stay clean.

> **Update 2026-10-30: Gate 0 stopped this candidate.** Anatomical groups barely beat random sensor groups (81.3 vs 80.4, 94th percentile against a pre-stated 95th), so deep region experts will not be built. Per-user selection helps, but the support-only rule tried does not work. Details: `docs/specs/SPEC_body_regions.md` §4. The text below is kept as the original proposal.

**Lead candidate: body-region experts with per-user sensor selection.**
- Each expert reads a fixed anatomical sensor group. A new user calibrates only the gate and class prototypes, so personalisation becomes "which body regions does this person use?"
- **The new capability is the point, not accuracy alone.** One calibration could recommend a small per-user sensor subset, which matters because a 28-sensor rig is not wearable. It also gives each user an interpretable body-use profile, and a way to test robustness to missing regions.
- MoEMeta has no analogue.

**Gate 0, no GPU, on the laptop: is there any headroom?**
1. Per-region 1-NN on dev users, plus an oracle that picks each user's best region subset on query data. The oracle is an upper bound, never a reportable method.
2. If the oracle does not clearly beat all-channel 1-NN (78% on dev), stop. Region routing has nothing to gain.
3. If it does, test a non-oracle rule that picks regions from the support set alone.

**Gate 1: the deep version, through the normal screen.** Controls:
- **Random sensor groups of the same sizes.** This is the key control. Region experts specialise by construction, so "experts differ" proves nothing. The claim is that anatomy beats random grouping.
- Uniform gate over regions.
- Full adaptation vs gate-only vs prototypes.
- Promotion uses the existing rule (≥ 2 points, ≥ 3/4 dev users).

**Ability-level experts: do not hard-wire them.** They would need ability labels at calibration, the groups are small (6, 11, 15), and notebook 03 found ability barely predicts sharing (Mantel r = 0.05-0.09). Test instead whether emergent region use differs by ability. That would be the better result, and it was not fitted by construction.

**Other work.**
- **Motifs vs bindings:** one analysis on whichever gate survives. Does routing follow motif rather than label? It needs non-uniform routing, so it comes after X2.
- **Per-timestep gating: dropped as a priority (Kai, 2026-10-30).** The current gate is already per sample (one gate vector per window), which is Kai's preference. Test it only as an optional late screen.
- **Not worth pursuing for novelty:** X4, more gating tricks, or transformers. Transformers are controls.

**Time box (proposal).** Decide Track B by 2026-11-20, which is roughly when the screen should have results and early enough to protect the ICML timeline. If nothing passes, the Track A paper goes out unchanged. Track B never blocks Track A.

## 7. Open items
- [x] **Kai:** find the sensor map (`docs/emg-sensor-placements.png`).
- **Kai's decisions, 2026-10-30 (evening):** MoEMeta-TS built. Track B is exploratory, no date; candidates go through `SPEC_gate0.md`. The `DECISION_method.md` edit is dropped: Kai's view is that if MoEMeta-TS beats EncoderMoE and that cannot be fixed, EncoderMoE is not publishable, which needs no rule. Body-region stays closed.
- [ ] Do a proper literature search before claiming that nobody has done MoE + MAML on sEMG. The check so far was one search.
- [ ] Trace the Port A vs Port B baseline mismatch, or let the rerun replace both.
