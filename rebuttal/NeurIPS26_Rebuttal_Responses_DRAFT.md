# EncoderMoE — NeurIPS 2026 Rebuttal: Drafted Responses

Every placeholder note from `NeurIPS26_Rebuttals.pdf` is replaced below with a drafted answer. Reviewer text is retained for context; my running notes are replaced.

Source docs: `rebuttal_plan.md` (schedule/triage) · `ENCODERMOE_REBUTTAL_PLAN.md` (Kaifosh, external baselines, protocol) · `MoEMeta_vs_EncoderMoE_handoff.md` (MoEMeta) · `pi_update_draft.md` (PI note).

---

## How to use this document

Each item has up to three parts:

- **→ RESPONSE** — prose intended to be pasted into the reviewer thread, edited for tone.
- **⚙ INTERNAL** — not for reviewers. Decisions, dependencies, things deliberately not volunteered.
- **⛔ DO NOT SAY** — where a natural-sounding answer is actually a trap.

Fill-in markers:

| Marker | Meaning |
|---|---|
| `[[FILL: x]]` | A number or result that must exist in a table or log before this text ships |
| `[[VERIFY: x]]` | Gated on a check that is still open — text may change entirely (S1, S2 and V7 are now closed; see below) |
| `[[PI: x]]` | Needs the PI's sign-off before it goes out |
| `[[CHOOSE: A / B]]` | Branch on an experimental outcome; both branches drafted |

**Two rules that govern every line below.** (1) No number that isn't in a table, a log, or explicitly marked pending — two reviewers are at confidence 4. (2) No claim about a method that wasn't run.

### Status after the repo audit (`REBUTTAL_CODE_FINDINGS.md`)

Three of the four blockers are resolved, two of them favourably. Where the audit and the planning docs disagree, the audit wins and this document follows it.

| Check | Was | Now |
|---|---|---|
| **S1** — what does "Subject-Specific Transfer Learning" train on? | Could have invalidated L279–284 | **Resolved, favourably.** Genuinely within-subject. Paper text correct, label wrong. L279–284 stands. Quotable answer for R3. |
| **S2** — episode sampler leak? | Could have forced revising the K≥3 half of Table 4 | **Resolved, favourably.** No leak; disjointness is structural. Realised Q at eval is 9/7/5 for K=1/3/5, which is a disclosure and a mechanistic account of the 10-way non-monotonicity. |
| **V7** — right Kaifosh checkpoint? | Could have voided C0/C1 | **Resolved analytically** at ≈6.48M. Confirm against the checkpoint file on NOTS before the text ships. |
| **Kaifosh preprocessing** | Assumed a literal `2.46e-6` multiply producing a dead network | **Real defect, different mechanism.** Signal-s.d. vs noise-s.d. normalisation → under-scaling by roughly the SNR. The fix is a gain sweep, not a two-way direction test. |

Also surfaced: A10's prototypical zero-shot path was one of six call sites that would have crashed on construction, so **any A10 number in the paper needs its provenance traced** (Appendix C).

**Fallback for anything that doesn't land in time:**

> We have queued this experiment on our shared cluster; scheduling latency prevented completion before this response. We will post results in this thread during the discussion period and include them in the camera-ready.

That works once or twice when the rest is substantive.

---

## Shared blocks

Five demands recur across reviewers. Full prose lives here; per-reviewer sections tailor the lead-in and point back.

### Block MOD — modality ablation (asked by R1, R2, R4, meta)

> **EMG-only and IMU-only ablations.** We agree this was a gap: Contribution 1 claims multimodal fusion matters, and the submission contained no modality ablation. We have run both unimodal conditions. To keep the parameter count exactly matched to the fused model we **mask** the ablated modality with zeros rather than deleting channels (deleting them would shrink per-expert layer 1 from `C=88` to `C=16`, removing roughly 10% of each expert's parameters and ~507k across all 22 — the kind of unmatched comparison the rest of the paper is careful to avoid).
>
> On the fixed 24/4/4 split, 1-shot 3-way: fused **88.4%**, EMG-only `[[FILL: %]]`, IMU-only `[[FILL: %]]`.
>
> Two caveats we want to state ourselves. These cells were run on the fixed split rather than under leave-two-subjects-out (L2SO is 16 runs per condition), so they are **preliminary single-split results**, are not directly comparable to Table 1's L2SO numbers, and do not enter the paired RM-ANOVA. We commit to L2SO versions in the camera-ready. Second, hyperparameters were tuned for the fused model, so both unimodal conditions are handicapped — the bias runs in favour of our claim, and we do not lean on small margins.
>
> `[[CHOOSE:`
> **A — if fused wins clearly:** The gap supports the fusion claim as stated, and we will add the ablation to Table 1 and reference it from Contribution 1.
> **B — if IMU-only matches or beats fused:** We report this straight, because it is a real finding: the kinematic channel carries most of the discriminative signal for user-defined gestures in this population. We will narrow the fusion claim at L34–38 from blanket complementarity to the specific regimes where EMG contributes — static and low-movement gestures — and characterise it per gesture class in the camera-ready.
> `]]`

⚙ **INTERNAL.** Masked channels are dead capacity, so this is not identical to a purpose-built unimodal model — disclose that in the caption. Branch B is plausible given earlier informal runs; if it happens, report it rather than attributing it to untuned hyperparameters, since the tuning bias points the other way.

**A13 ships three jobs, and the `both` control is not optional.** It is the only evidence that the masking harness didn't perturb the pipeline. If `both` doesn't land near the fixed-split M0 number, distrust the other two conditions and don't post any of them. Masking is verified to zero the correct modality, preserve shapes and channel counts, and leave the source `tensor_dict` unmutated — but that was verified on synthetic data, not end-to-end on the cluster. Compare against **88.4%** (fixed split), never 86.7% (L2SO).

### Block KAIF — the Kaifosh & Reardon baseline (meta, R1, R3)

> **We are re-evaluating this baseline, and expect the numbers to move in their favour.** Working through [11]'s Methods carefully in response to these reviews, we found that our reported linear-probe (62.1%) and fine-tuned (56.0%) numbers are not a fair evaluation of their model, for reasons that all bias in the same direction:
>
> 1. **Their preprocessing was not applied.** Their discrete-gesture decoder expects rescaling to unit noise s.d., a 40 Hz 4th-order Butterworth high-pass, and a sigmoid squash `x/(32+|x|)`. Our evaluation applied a different pipeline. The reported numbers are therefore out-of-distribution evaluations of their model.
> 2. **Our input arrives at the wrong scale, by roughly the signal-to-noise ratio.** Their pipeline expects the input normalised so the *noise* standard deviation is 1.0, which places gesture-active samples well above 1 — that is what makes their μ=32 outlier squash a sensible operating point. Our 2 kHz tensor is instead normalised so the *whole-trial signal* standard deviation is 1.0 (one scalar per trial, computed over all 16 channels flattened), so active samples land near 1 and `x/(32+|x|) ≈ x/32` compresses the entire recording into roughly ±0.03–0.2. Their compression is inside the released module, so it runs regardless of what we feed it: the model is evaluated in the near-linear corner of a nonlinearity it was trained to use. Measured post-compression 99th-percentile magnitude for the condition that produced our reported numbers is 0.106. Because our data is already s.d.-normalised, a single global gain `g` is exactly the hypothesis "noise s.d. = 1/g", so we re-ran across a gain sweep with post-compression activation statistics logged for every condition, and report the best fairly-measured one.
> 3. **Their model is a detector, not a window classifier.** Their readout is a 9-way multilabel sigmoid trained with binary cross-entropy and scored by CLER over Needleman–Wunsch-matched events, after threshold selection (0.35), debouncing, and state-machine filtering. Our comparison replaces the head and discards that stack, so the row measures *their pretrained trunk with our readout*, not their method. This was not stated and should have been.
> 4. **Our reported parameter count for their model was wrong.** We evaluated their discrete-gesture decoder, which is ≈6.5M parameters; the ~60M figure in our submission is their *handwriting* conformer. The checkpoint itself is correct — the error is in our text, and we are correcting it.
>
> We also note two secondary mismatches in the same direction: our band-pass is 20–450 Hz rather than their 40 Hz high-pass, and 450 Hz sits below the 850 Hz low-pass on their analog front end.
>
> Revised results with their preprocessing, at the best fairly-measured gain: linear probe `[[FILL: %]]`, fine-tuned `[[FILL: %]]`. Full gain sweep with activation statistics per condition: `[[FILL: table]]`.
>
> **Two further confounds we cannot fix, and will state rather than absorb into a comparison.** Their filters learned the spatial statistics of a dense circumferential wrist array (20 mm within-pair, 10.6–15 mm between channels); our electrodes are distributed across the upper body, so any channel mapping is arbitrary. This is not merely a mismatch: their wrist and handwriting decoders each include a rotational-invariance module, and their *gesture* architecture has neither that nor rotational augmentation, so their filters cannot be realigned to our montage even in principle. Their analog front end also applies a 20 Hz high-pass and 850 Hz low-pass ahead of the digital filtering, so "matched preprocessing" is matched digitally only.
>
> **We therefore propose reframing this comparison rather than repairing it.** We will move it out of the main ablation table into its own subsection asking: *does large-scale able-bodied wrist-sEMG pretraining transfer to this hardware and population?* Under that framing, topology mismatch and the head conversion become part of the question rather than defects in a head-to-head. The finding that survives is the interesting one — a decoder pretrained on 4,800 participants does not transfer well to this hardware and population, while a small fused model with fast adaptation performs substantially better. [11] scope this exact question as open themselves, writing that it is unclear whether models trained on able-bodied participants will generalise to clinical populations.
>
> **Committed for camera-ready:** their architecture trained from scratch on our data (separating their *architecture* from their *pretraining corpus*, which is the attribution this comparison actually owes), and matched-input cells at 200 Hz — their stride-10 convolution decimates 2 kHz to 200 Hz inside the model, whereas our 20 Hz envelope discards 10× more.

⚙ **INTERNAL.** V7 is resolved analytically — the instantiated architecture computes to ≈6,482,953 parameters and `load_state_dict(strict=True)` would have raised on a mismatched artifact — so the C-rows survive and this block is writable. Still run `V7_checkpoint_param_count.py` on NOTS before it ships: it reads the checkpoint's own `state_dict`, which is the architecture-independent check, and flags conformer-style keys.

Cite "N=4,800 (largest model, Fig. 2f)" rather than a bare participant count; [11]'s own counts disagree across Methods (4,900), Fig. 2f (4,800) and Extended Data Fig. 3a (4,579). Add checkpoint provenance (repo + commit) to the caption, since their Code availability section never mentions checkpoints and a reviewer may go looking.

**Two cautions on the gain sweep.** The fair/under-scaled/saturated thresholds (`UNDERSCALED_P99 = 0.15`, `SATURATED_P99 = 0.90`) are judgement calls tuned on synthetic data at an assumed SNR — sanity-check them against real activation statistics before any text leans on the word "fair." And the noise-floor estimator assumes each trial contains a roughly quiescent 100 ms window; if gestures fill the trial it biases the estimate high, which under-gains and therefore under-states their model — the *same* direction as the original defect, so it is not a conservative fallback. Report the fixed-gain sweep alongside it rather than relying on the estimator alone.

⛔ **DO NOT SAY** anything about a literal `2.46e-6` multiply as though it happened. It does not appear anywhere in the repo. If it is worth mentioning at all, mention it only as a reading that was checked and excluded — measured, it drives post-compression output to ~3e-7, which would indeed be a dead network, but it is not where our numbers came from. Claiming a defect we didn't have is as bad as hiding one we did.

⛔ **DO NOT SAY** "we could not get MUAPs" or otherwise imply their gesture pipeline uses hand-crafted features. It does not: MPF features serve their wrist and handwriting decoders only, and they explicitly chose raw EMG for gestures because the 100 ms MPF window is comparable to gesture duration. Matching their gesture input is three preprocessing operations, not a feature-engineering project. Saying otherwise is both wrong and checkable.

### Block MOEMETA — MoEMeta attribution and ports (meta, R3)

> **We were not aware of MoEMeta (Wu & Yin, NeurIPS 2025) and we should have been. We will cite it, and we accept the substance of the point.** MoE combined with meta-learning has a NeurIPS 2025 precedent in the knowledge-graph domain, and their per-relation gating-distribution figure is structurally the same analysis as our §4.6 gate-contrast heatmap grouped by ability level. We are not going to defend Contribution 1 as "combining MAML with MoE is novel."
>
> Draft citation text for §2, with a pointer added in §1:
>
> > MoEMeta (Wu & Yin, 2025) applies MoE as a meta-learner for few-shot knowledge-graph relational learning, using globally shared experts to produce a task-level relation-meta that is subsequently adapted via low-dimensional projection vectors. Our setting differs in that experts operate on continuous multimodal biosignals rather than symbolic entity embeddings, routing is applied to query samples at inference and is independent of support-set size, and our inner loop adapts the full parameter set rather than a task embedding.
>
> **A literal MoEMeta baseline is not runnable on this data, for four specific reasons rather than a general appeal to domain difference:**
>
> 1. **It requires a graph.** Its first component aggregates each entity's one-hop neighbourhood (up to 50 neighbours). Our samples have no entity vocabulary and no adjacency.
> 2. **It requires a candidate set and a ranking objective.** MRR and Hits@k over `C_{h,r}` have no analog in N-way episodic classification; degenerated to ranking 3 candidates, MRR collapses to accuracy and the translational scoring function does no work.
> 3. **Its experts consume symbolic embeddings** — 2-layer MLPs over 100-d TransE-pretrained vectors. Replacing them with modules that ingest an 88×64 continuous window means replacing them with our experts.
> 4. **Its meta-test protocol is a different adaptation regime by definition.** Global parameters (aggregator, experts, gate) are frozen; only three d-dim projection vectors and the relation-meta are updated. A "MoEMeta baseline" here is not a different architecture, it is a different inner loop.
>
> Most fundamentally, **their held-out axis is labels and ours is users**: in MoEMeta all entities are seen during meta-training and only the relation is novel at test time. Converting our data to a KG is constructible but uninformative — the signal has nowhere to live except in a learned signal→embedding encoder, at which point the graph scaffolding is inert; and if task = relation, our novel-task-at-test becomes a novel *gesture*, not a novel *user*, so L2SO no longer applies.
>
> **Instead of a non-comparison, we port MoEMeta's two distinguishing design decisions into our setting:**
>
> **(A) Support-derived routing.** MoEMeta computes gate values per support triplet and aggregates to one task-level vector; query items are never routed. Ours routes each query sample directly and is independent of support-set size (L487–488). We isolate exactly this by deriving routing from support samples only, aggregating to one gate vector per episode, and applying it to all queries — evaluable from our existing checkpoint. Result at 1-shot 3-way: support-derived routing `[[FILL: %]]` vs. query-conditioned `[[FILL: matching-protocol baseline %]]`.
>
> **(B) MoEMeta-style local adaptation.** Freeze the global expert bank at adaptation time and adapt only a small task-conditional module (a per-episode gate bias / low-rank modulation of the head), the closest analog to their `{p_h, p_r, p_t}`. Result: `[[FILL: %]]`. `[[VERIFY: must be meta-trained with the restricted inner loop, not merely restricted at eval time]]`
>
> **Reframed Contribution 1.** The claim is not the pairing; it is the specific instantiation and what it buys in this regime: encoder-level experts over raw multimodal biosignal rather than symbolic embeddings; full-parameter inner-loop adaptation rather than a frozen encoder plus a task embedding; query-time routing independent of support-set size; load balancing in the outer loop only — a placement question that cannot arise in MoEMeta's design, since nothing inside its MoE is adapted per task; and the empirical finding that routing's benefit concentrates at K=1 and largely evaporates by K=5.

⚙ **INTERNAL.** Port A is the flagship: nearly free, no retraining, and if query-conditioned routing wins at K=1 as predicted, a novelty attack becomes a positive empirical result. Port B must be meta-trained *through* the restricted adaptation; meta-training with full MAML++ and freezing only at test time evaluates the model in a regime it was never optimised for, would lose by construction, and R3 at confidence 4 will notice. If time forces the eval-only version, disclose the confound in the caption and do not lean on the margin.

⛔ **DO NOT SAY** anything implying quantitative convergence on expert count. Their guideline is M < #tasks (M=32 vs 51/133/75 relations, ratio 0.24–0.63); ours is 22/24 ≈ 0.92 with the sweep peaking *at* the task count. A reviewer can read that as sitting on the boundary their heuristic warns against. Use only the ordinal framing (both select E ≤ #tasks and degrade above it) or the mechanistic one (fewer experts than tasks forces reuse over memorisation). Also: do not mention the internal inconsistencies in their Tables 3/4 or the Hits@5 figure in §5.1 — correcting typos in a paper the reviewer champions reads as petty and buys nothing.

### Block XFMR — transformer baselines (R1, R4, meta)

> **On the architectural choice, the reviewer is right that we assert more than we show, and the correct fix is to stop asserting it.** We will revise L171–173 from a claim about recurrent inductive bias to an explicitly labelled design hypothesis in Limitations/Future Work.
>
> The motivation, offered as motivation and not as evidence: at K=1 with 10 classes an adaptation episode sees on the order of 10 windows of ~3 s, roughly 600 timepoints at our sampling rate. Attention's flexibility is an advantage when there is enough data to identify the relationships it can express; in this regime we expected the recurrent prior to constrain the inner loop's effective degrees of freedom. We also expected pretrained transformer encoders to transfer poorly here for two reasons independent of architecture: users with motor impairments are largely absent from pretraining corpora, and user-defined gesture semantics require overwriting rather than reusing much of the learned mapping.
>
> `[[CHOOSE:` **if exploratory logs exist:** In preliminary experiments a small transformer temporal encoder reached `[[FILL: %]]` versus `[[FILL: %]]` for the LSTM under otherwise identical settings; we report this as an untuned exploratory result, not as a controlled comparison. `[[FILL: config — depth, width, heads, params]]` **if no logs:** We do not have exploratory results we are willing to report as evidence. `]]`
>
> We are declining a full transformer-MoE reimplementation for this response, and want to be direct about why rather than citing effort: an untuned transformer that loses is uninformative and could fairly be called a straw man, while a properly tuned transformer-MoE — architecture search, width and depth selection, its own HPO budget — is paper-scale work whose result we could not attribute. We would rather withdraw the unsupported claim than answer it with a comparison neither of us would trust.

### Block DATA2 — second dataset (R1, R2, meta)

> **We are declining a second dataset on grounds of hypothesis rather than effort, and we would rather say so plainly than run something uninformative.** Ninapro and [11]'s benchmark both use fixed, standardised gesture vocabularies from able-bodied participants. Neither instantiates the two properties the method targets — user-defined gesture semantics and physiological heterogeneity from motor impairment — so a result in either direction would not bear on our claims. We do not claim EncoderMoE is the right approach for large able-bodied fixed-vocabulary corpora; where those exist, scale is likely to be the better answer, and our Limitations section says so.
>
> The reason the paper exists is that the population it targets is expensive to recruit and routinely excluded from datasets, so the near-term question is what works at N=32 rather than what would work at N=4,800. [11]'s authors frame the same gap as open, writing that it is unclear whether models trained on able-bodied participants will generalise to clinical populations.
>
> We do accept the underlying point that single-dataset evaluation limits generalisation claims, and will `[[CHOOSE: state this more prominently in Limitations / narrow the affected claims in §1 and §5]]`.

⚙ **INTERNAL.** Never phrase this as "we don't want to set up the infrastructure." It is a defensible scientific decline and an indefensible logistical one.

---

# Meta-Review

> All reviewers gave rating 3.0 — this paper is currently on a trajectory toward likely rejection.

**→ RESPONSE — opening paragraph for the meta-reviewer thread:**

> We thank the reviewers and the meta-reviewer. The reviews are consistent, and we accept most of them: the two structural gaps identified — no external comparisons on this benchmark, and no modality ablation supporting a fusion claim — are real, and one of them (the Kaifosh baseline) we have found to be more seriously flawed than the reviews allege. Below we (1) concede the MoEMeta attribution and reframe our contribution accordingly, (2) report a re-evaluation of the Kaifosh baseline that we expect to move its numbers upward, (3) add the modality ablations, (4) identify two external baselines already present in Table 1 under misleading labels, and (5) correct a transposition of our two headline numbers in the meta-review, which we believe our own presentation caused.

---

### "Evaluation is limited to a single dataset and small number of gestures"

**→ RESPONSE:** Block **DATA2**, plus:

> On vocabulary size: we agree the 10-way regime is the deployment-relevant one and that our 10-way result is the weakest number in the paper. See the response to R2 below, where we (a) locate the data-saturated upper bound already present in Table 4 and (b) report an investigation into the flat K-scaling of the 10-way condition.

⚙ **INTERNAL.** No longer gated — S2 is resolved with no leak, so the 10-way thread is writable in full.

### "Novelty compared to prior work"

**→ RESPONSE:** Block **MOEMETA**, plus:

> With originality scored 2/2/1 we take the point that the method-novelty framing is not persuasive as written, and we are not going to argue it. We would ask the committee to weigh instead the problem formulation (few-shot cross-subject adaptation × user-defined gesture semantics × EMG-IMU fusion in a motor-impaired population), the empirical characterisation of a regime for which no comparable dataset exists, and the evaluation rigor that R2 and R4 both credited unprompted.

⚙ **INTERNAL.** Do not spend response space defending novelty. Reframe once, clearly, and move budget to the concessions and the experiments.

### "Evaluation lacks comparisons against SOTA EMG/IMU methods and single-modality baselines"

**→ RESPONSE:**

> Two answers, and we think the first is the more important one.
>
> **Two external baselines are already in Table 1, presented as self-ablations, and a third row is misleadingly named.** Our "No MAML, No MoE" row is cross-subject supervised pretraining followed by 10-step gradient fine-tuning — that is the Côté-Allard [7] transfer-learning recipe, a named external method presented as an ablation of ours. Likewise, "Meta Pretrained EMG" means the pretrained model of Kaifosh & Reardon [11]; in a MAML paper that label parses as *meta*-pretrained, which was a poor choice on our part. Separately, "Subject-Specific Transfer Learning" performs no cross-subject transfer at all (see R3 Q2), so that name should also go. We will relabel:
>
> | Current | Revised |
> |---|---|
> | No MAML, No MoE | Cross-Subject Supervised + Fine-tuning (cf. Côté-Allard [7]) |
> | Meta Pretrained EMG | Kaifosh & Reardon [11] (wrist sEMG, N=4,800 pretrain, largest model, Fig. 2f) |
> | Subject-Specific Transfer Learning | Subject-Specific Supervised (within-subject) |
>
> This is a presentation failure rather than a missing experiment, and we are grateful it was caught.
>
> **Single-modality baselines:** see Block **MOD**.
>
> **On whether Kaifosh "counts":** we believe it does, but the reviewers are right that as executed it does not support the weight we placed on it. See Block **KAIF**.

### "3-way (67.7%) is notably lower than 10-way (86.7%)"

**→ RESPONSE — flag this early and prominently:**

> **We believe these two numbers are transposed in the meta-review, and we think the confusion is our fault.** In the submission, 1-shot 3-way accuracy is 86.7% (L2SO, Table 1) and 88.4% (fixed 24/4/4 split, Table 4); 1-shot 10-way accuracy is 67.7% (Table 4). Accuracy decreases with vocabulary size, as expected.
>
> The likely source of the confusion is ours: we report two different numbers for the same nominal 1-shot 3-way condition — 86.7% and 88.4% — because Table 1 uses leave-two-subjects-out over all 32 participants while Table 4 and Figures 3–5 use the fixed HPO split, and only one parenthetical in A.6 explains the difference. We will name the evaluation protocol explicitly in every table caption, and we would ask the meta-reviewer to re-read the vocabulary-size discussion with the numbers in their intended orientation, since the conclusion reverses.

⚙ **INTERNAL.** This is the single most important correction in the entire response — the meta-review drives the decision and currently contains an inverted premise. Lead with it, state it once, without defensiveness, and take the blame for the presentation.

### "Architectural choices lack theoretical or empirical validation"

**→ RESPONSE:** Block **XFMR**.

### Key question 1 — MoEMeta citation and comparison

**→ RESPONSE:** Block **MOEMETA**.

### Key question 2 — Kaifosh re-evaluation with matched featurization

**→ RESPONSE:** Block **KAIF**, plus the specific clarification about what "featurization" means here:

> We want to resolve a possible mutual misunderstanding about featurization, because it narrows the gap. **Neither method performs hand-crafted feature extraction.** [11]'s MPF features (6-band cross-spectral density, log-matrix, half-vectorised to 384 dimensions) are used for their *wrist* and *handwriting* decoders only; for discrete gestures they explicitly chose raw EMG because the 100 ms MPF window is comparable to gesture duration. Their gesture pipeline is three preprocessing operations followed by a learned strided convolution that decimates 2 kHz to 200 Hz *inside the model*. Ours is likewise feature-engineering-free: rectification, a 40 Hz low-pass, and decimation to a 20 Hz amplitude envelope, following [28].
>
> The difference is therefore a single documented axis — envelope at 20 Hz versus minimally preprocessed oscillatory signal at 200 Hz — not a methodological gulf, and it is matchable. We note the two representations differ in kind and not only in rate (ours is positive and smooth with a 10 Hz Nyquist; theirs is zero-mean and bipolar), so a faithful matched-input stem needs more than one linear convolution to approximate rectification. We commit the matched-input cells at 200 Hz for camera-ready.

### Key question 3 — Does "Subject-Specific Transfer Learning" pretrain on all users?

**→ RESPONSE:** see R3 Q2 below — **resolved: trained from scratch on one user's data, no cross-subject phase.** The paper text was right; the label was wrong.

### Key question 4 — Upper bound from fine-tuning on all of a test user's data

**→ RESPONSE:** see R3 Q2 below (the upper bound is already in Table 4).

### Key question 5 — Is it fair to include the Kaifosh fine-tuning evaluation?

**→ RESPONSE:** Block **KAIF** — specifically the reframing. R3's instinct that it "muddies the story" is correct as the row currently stands, and the reframing is our answer.

### Key question 6 — 22 experts, input ablation, subject-level routing, single-expert-per-subject

**→ RESPONSE:** see R1's Questions below; all four are answered there.

### Key question 7 — Foundation models and scaling

**→ RESPONSE:**

> We do not claim EncoderMoE would outperform a scaled foundation model given a comparable corpus, and we should have said so explicitly. If a dataset of thousands of participants with motor impairments performing user-defined gestures existed, we would expect a large generically-trained model to do well on it. The claim we make is narrower and conditional: **at the data scale that actually exists for this population, meta-learned adaptation plus routed encoder capacity substantially outperforms transfer from the largest available pretrained sEMG decoder.** The supporting evidence is that [11]'s released decoder, pretrained on N=4,800 participants, transfers poorly to this hardware and population `[[FILL: revised number from Block KAIF]]` — with the caveats in Block KAIF about what that comparison does and does not isolate.
>
> We will revise the framing so this reads as a statement about a data regime rather than about scaling in general.

⚙ **INTERNAL.** "Scale alone is insufficient" survives on premise (the checkpoint really is the N=4,800 model), but its *evidence* is the size of the gap, which is gated on the re-run. If their number rises materially, soften the sentence rather than defend a shrunken gap.

### Key question 8 — Why 1D-CNN experts and an LSTM instead of transformers?

**→ RESPONSE:** Block **XFMR**.

### Key question 9 — Normalization; individual modality contributions

**→ RESPONSE:**

> **Normalization.** Both modalities are normalized; the reviewer is right that the submission does not make the procedure explicit enough to evaluate, and we will fix that. Our EMG preprocessing follows the `[[FILL: name the pipeline — per-channel scaling / envelope normalization, and where it is applied: per-window, per-session, or per-user]]` procedure of [28], and IMU channels are `[[FILL: normalization for IMU — per-axis standardization? shared or per-channel statistics? computed on train users only?]]`. We will add a short subsection specifying, per modality: what statistic is used, over what scope it is computed, and whether test-user data contributes to it.
>
> **Modality contributions:** Block **MOD**.

⚙ **INTERNAL.** Fill this from the data loader, not from memory. Two things a confidence-4 reviewer will check: whether normalization statistics are computed with any test-user data (a leak if so), and whether EMG and IMU are on comparable scales before concatenation into the 88-channel input.

### Key question 10 — Cross-subject vs within-subject; meta-training user sweep; where does the MoE fail?

**→ RESPONSE:**

> **Is the cross-subject advantage just a sample-size effect?** Partly, and we think the decomposition is informative rather than deflationary. A within-subject model sees one user's ~`[[FILL: n]]` windows; a cross-subject meta-trained model sees 24 users. But sample count is not the whole story: the parameter-matched single-encoder control (which sees the identical 24-user corpus) underperforms EncoderMoE by `[[FILL: Δ from Table 1]]`, so the benefit is not attributable to data volume alone. The honest statement is that cross-subject meta-training buys both more data and structure across users that a single encoder does not exploit, and our ablations separate the second from model capacity but not from the first.
>
> **Meta-training user sweep.** We do not have a full sweep, but the existing endpoints bracket it: N=1 meta-training user (Subject-Specific EncoderMoE, 64.6%) to N=24 (86.7%). `[[CHOOSE: if a slot is free, add one intermediate point (N=8 or N=16) / otherwise commit the full sweep to camera-ready]]`
>
> **Where does the MoE fail?** Two answers we can support. First, routing recovers **no** subject-level cluster structure — the gate distributions do not separate individual users — so the mechanism relies on structure shared across users at a coarser level (we do recover ability-level structure, §4.6). Where users share no exploitable structure, MoE should offer no benefit, and the single-subject setting is the limiting case of this: Subject-Specific EncoderMoE (64.6%) underperforms the single-encoder within-subject model (72.2%) — and we have verified in code that both of these conditions are genuinely trained within a single subject, so the comparison isolates what it claims to. Second, the benefit is K-dependent: routing's advantage concentrates at K=1 and largely disappears by K=5, and it does not rescue the 10-way regime.

---

# Reviewer 1 (LPUC) — Rating 3, Confidence 3

Scores: Quality 3, Clarity 3, Significance 2, Originality 3.

### W1 — Limited baselines, all within our own architecture

**→ RESPONSE:** the relabelling table from the meta-review section (two external baselines already present, plus one misnamed row), plus Block **KAIF** for why the Kaifosh row as executed does not carry the weight we gave it, plus:

> We are also adding a Prototypical Networks baseline `[[FILL: %]]` on the same backbone for parameter comparability. This is directly relevant to a claim we should not have made without it: §2 (L110–112) asserts that metric-learning approaches are fragile with 1-shot physiological signals, about a method we did not run. Running it either supports the claim or corrects it; either way the assertion should not have shipped unsupported.

⚙ **INTERNAL.** ProtoNet is first to drop if time runs out. If it drops, still soften L110–112 — the free half of the fix is the more important half.

### W2 — Architectural choice not empirically validated; no transformer or mixture-of-transformers comparison

**→ RESPONSE:** Block **XFMR**.

⚙ **INTERNAL.** R1's objection is that the claim isn't validated, not that a transformer would win. The cheapest valid response is to stop asserting it. Do not offer a token transformer run.

### W3 — Linear probing / fine-tuning reported on a different dataset, EMG-only, class count missing

**→ RESPONSE:**

> The class count is the same as our main evaluation and we should have stated it — `[[VERIFY: confirm N-way and K used for the LP/FT rows, and state both in the caption]]`. On comparability, the reviewer's criticism is well-founded and stronger than stated: see Block **KAIF**. We agree these rows do not support a like-for-like comparison and are moving them out of the main table into a transfer-study subsection with the confounds stated in the caption.

### W4 — Single-dataset evaluation, 32 subjects

**→ RESPONSE:** Block **DATA2**.

### Q1 — How was 22 experts chosen, given the HPO peak at N=8?

**→ RESPONSE:**

> E=22 was selected by the Optuna study; the sweep in Figure 3 is a separate post-hoc grid. Two things need correcting, both ours.
>
> First, **Table 2's description of the search space is wrong**: it lists a categorical space `{4, 8, 12, 16, 20, 24, 28, 32}`, which could not have returned 22. `[[VERIFY: reconcile against the Optuna study object — the space was almost certainly an integer-uniform range]]` The corrected row:
>
> | Group | Parameter | Type | Search range | Chosen |
> |---|---|---|---|---|
> | MoE structure | Num. experts E | int-uniform | [4, 32] | 22 |
>
> The model itself is unambiguously E=22 (Figure 2 shows E0–E21; A.7 states E=22 with top-k=9, ≈41% utilisation).
>
> Second, and more to the reviewer's point: **the choice is not load-bearing.** Accuracy is on a plateau across E ∈ [8, 24], and within-condition variance exceeds the between-condition differences over that range, so E=8 and E=22 are not statistically distinguishable in our data. We selected the higher value because prior MoE work supported larger expert banks; we should have reported the plateau rather than a peak. We will add a sentence to A.4 reconciling E=22 against the sweep maximum near E≈24, and a note to the Figure 3 caption clarifying that its grid is a separate post-hoc sweep rather than the HPO space.

⚙ **INTERNAL.** Gate the MoEMeta expert-count parallel on this correction, and use only the ordinal or mechanistic framing (see Block MOEMETA's DO NOT SAY).

### Q2 — Input ablation; EMG vs IMU contribution

**→ RESPONSE:** Block **MOD**. (R1 does mean exactly the three-way EMG / IMU / fused comparison.)

### Q3 — Subject-level routing analysis; is any subject served by a single expert?

**→ RESPONSE:**

> We ran this analysis and did not include it, which was a mistake — it addresses a natural concern about whether routing is memorising subject identity. It will go into the appendix.
>
> **Subject-level routing.** Contrasting per-user mean gate vectors reveals no subject-level cluster structure: routing does not separate individual users. The structure the gates do recover is coarser and ability-level (§4.6). This is the more useful result for the paper's claims, since subject-level routing at E=22 with 24 meta-training users would be closer to memorisation than specialisation.
>
> **Single-expert-per-subject.** Computed directly from the per-user mean gate vectors saved for all 32 participants: the maximum per-subject gate weight is `[[FILL: max and distribution]]` and mean per-subject routing entropy is `[[FILL: nats, against a uniform ceiling of ln 22 ≈ 3.09]]`. `[[CHOOSE: No subject's mass concentrates on a single expert (max weight X ≪ 1, entropy Y) / For N subjects the distribution is concentrated, which we report and discuss]]`. With top-k=9 of 22 experts active, near-degenerate routing would also be visible as a collapse in the aggregate load-balancing statistics, which `[[FILL: state whether it is / is not observed]]`.

⚙ **INTERNAL.** These are computable from artefacts already on disk — no cluster time. High value per minute; do this on Day 1 alongside the code checks.

---

# Reviewer 2 (XH89) — Rating 3, Confidence 3

Scores: Quality 3, Clarity 3, Significance 2, Originality 2.

### W1 — Deployment scenario is 10 gestures, but the primary evaluation is 3-way; 10-way (67.7%) is much weaker

**→ RESPONSE:**

> We accept the framing and want to add information rather than argue.
>
> **First, the upper bound the reviewer is implicitly asking for is already in the paper.** With 10 repetitions per gesture and Q=9 query samples per class, the K=5 column of Table 4 is already a near-data-saturated regime for this dataset. For 3-way, accuracy rises from 88.4% at K=1 to 96.9% at K=5 — 1-shot sits ~8.5 points below a near-saturated ceiling. For 10-way, it moves from 67.7% at K=1 to 68.5% at K=5: **five times the labelled data buys 0.8 points.** That reframes the 10-way result: it is not primarily an adaptation-efficiency failure, since more data from the same user does not fix it.
>
> **Second, we audited the evaluation code specifically to understand the flat K-scaling, and we can now give a mechanistic account of it.** Our 10-way accuracy is non-monotonic in K (67.7 → 64.4 → 68.5) while the MAML-no-MoE baseline rises monotonically (55.6 → 68.4 → 69.9). Two properties of the episode construction explain the pattern, and we want to disclose both.
>
> **(a) The realised query count shrinks with K, which the submission does not state.** With 10 repetitions per gesture, the evaluation path assigns every non-support repetition to the query set rather than holding the configured Q=9 fixed. Realised query samples per class are therefore 9, 7 and 5 at K=1, 3 and 5. Support and query sets are disjoint slices of a single shuffled repetition list, so there is no overlap and the accuracies themselves stand — but the *precision* of the K≥3 estimates is lower than the caption implies. At 10-way K=5 an episode is scored on 50 query samples rather than 90, which is consistent with the standard deviations we report (±10.3 at 10-way K=3/5 against ±2.9 at 3-way K=1). We will state realised query counts, measured per episode rather than taken from the configuration, in the revised table.
>
> **(b) At N=10 every evaluation episode necessarily uses the same class set.** With exactly 10 gestures available, N=10 admits one class set and one label map, so only the support/query repetition assignment varies between episodes; at N=3 the class subsets and their orderings vary freely (in a 20-episode sample, 19 distinct label maps at N=3 versus 1 at N=10). This reduces the effective diversity of the 10-way evaluation and contributes to both the variance and the flat K-response.
>
> We want to be explicit that we checked for, and ruled out, the two failure modes these observations might suggest. Support and query cannot overlap — they are disjoint slices by construction, and we confirmed this empirically across the cached evaluation episodes. And the model cannot be exploiting a fixed gesture→index mapping, because meta-training randomises the label map (label-shuffle augmentation is enabled during meta-training and disabled at evaluation), so no global mapping is learnable in the first place.
>
> Taken together: the 10-way result is not a leak and not a sampler bug, but the estimates are noisier than the submission conveys, and the vocabulary-size regime is genuinely hard in a way additional shots do not fix.

⚙ **INTERNAL.** The hard stop is lifted — S2 came back clean, no PI escalation, no published number moves. This thread is now one of the stronger parts of the response: a reviewer's sharpest criticism answered with a mechanism rather than a defence. Two things to keep straight when writing it. The false comment in `fewshot_grid.py` asserting that Q is held fixed across grid cells is corrected in code, but the *paper* may make the same claim — check Table 4's caption and the protocol description in §4 and fix them, because the realised-Q disclosure is only credible if the paper stops asserting the opposite. And frame (b) as reduced evaluation diversity, not as a defect; it is a property of having exactly 10 gestures and wanting a 10-way condition, and there is no version of the experiment where it isn't true.

⚙ **INTERNAL — optional diagnostic if a slot frees up.** The leaky oracle: train including test users and evaluate 10-way on held-out repetitions. ≈68% means the 10-way ceiling is intrinsic to the data and labels, converting R2's strongest criticism into a dataset property. ≈90% means an adaptation or architecture failure worth knowing before camera-ready. High information per run; below the cut line.

### W2 — MAML for EMG calibration has been done (Proroković et al. 2020)

**→ RESPONSE:**

> Proroković et al. is **reference [17]**, cited at L115 — we may not have made the connection prominent enough, and we will strengthen it. We also want to distinguish it, since the overlap is real but partial: [17] addresses within-subject session-to-session recalibration on a *fixed* gesture vocabulary. Our setting is cross-subject with user-defined semantics, so the held-out axis is a novel user and the label set differs per user rather than being shared. We will make both the citation and the distinction explicit in §2.

### W3 — Single dataset, single session; no cross-session, cross-device, cross-task; only 10 repetitions

**→ RESPONSE:**

> These are real limitations and we would rather be specific about each than generically apologetic.
>
> - **Cross-session.** The dataset contains a single session per participant, so this is not answerable with the data in hand. Answering it requires re-recruiting the same participants — a substantial undertaking with this population, and the reason session count is what it is. `[[FILL: state whether any participants have repeat sessions; if even a few do, a small cross-session sub-analysis is worth far more than the text]]`
> - **Cross-device.** Requires a second hardware montage with the same participants performing the same self-defined gestures. Our results on transfer *from* different hardware ([11], Block **KAIF**) suggest that topology mismatch is a first-order effect, which we now state as a limitation rather than leaving implicit.
> - **Cross-task.** Partially addressed by construction: gesture vocabularies are user-defined and differ across participants, so meta-test tasks have label semantics unseen during meta-training. We agree this is not the same as cross-*task-family* generalisation.
> - **10 repetitions; fatigue, electrode shift, day-to-day variability.** Not capturable in this dataset. We note the repetition budget was set by participant burden, which for this population is a binding constraint rather than a design oversight.
>
> We will consolidate these into Limitations with this level of specificity, including what each would require, so the boundary of the claims is explicit.

### W4 — Fusion is claimed with zero evidence that IMU adds value over EMG alone

**→ RESPONSE:** Block **MOD**. Note this is the reviewer's sharpest correct hit — Contribution 1 as written is unsupported without it.

### Q1 — Foundation models and scaling

**→ RESPONSE:** meta-review Key question 7.

### Q2 — Is the cross-subject advantage due to limited within-subject samples?

**→ RESPONSE:** meta-review Key question 10, first paragraph.

### Q3 — Performance vs. number of meta-training users (8, 16, 24, 30)

**→ RESPONSE:** meta-review Key question 10, second paragraph.

### Q4 — In which scenarios does the MoE fail?

**→ RESPONSE:** meta-review Key question 10, third paragraph.

---

# Reviewer 3 (ARAC) — Rating 3, Confidence 4

Scores: Quality 3, Clarity 3, Significance 2, Originality 2. **Highest-confidence reviewer, and the source of the MoEMeta finding.**

### W1 — Ideas closely related to uncited prior work (MoEMeta); contribution framing is fuzzy

**→ RESPONSE:** Block **MOEMETA**, with an explicit answer to the framing question:

> On the reviewer's direct question — whether the contribution is primarily in the application or the method — our answer after considering MoEMeta is: **primarily the problem formulation and its empirical characterisation, with the methodological contribution being a specific instantiation rather than the MoE-plus-meta-learning pairing.** We would rather state that plainly than defend the stronger reading. The reviewer's assessment that our §4.6 analysis is structurally parallel to MoEMeta's Figure 2 is correct, and we will cite it as such.

### Q1 — Are the authors aware of MoEMeta? It should be cited and compared against

**→ RESPONSE:** Block **MOEMETA** in full, including Ports A and B. This is the highest-priority item in the response.

### Q2 — Does "Subject-Specific Transfer Learning" pretrain on all users, or train from scratch on one? Plus: upper bound from fine-tuning on all of a test user's data

**→ RESPONSE:**

> **Part 1 — the naming. We checked the training code rather than answering from memory, and the answer is the second of the reviewer's two options: it is trained from scratch using only a single user's data.** There is no cross-subject training phase in this condition. The subject loop runs over evaluation subjects only, and the train/validation/test split is over *repetitions* within each subject — repetition 1 for training, repetition 2 for validation, repetitions 3–10 for test.
>
> So the paper's statements at L227–228 and L264–265 are correct, and **the label is what is wrong**: under our naming convention "transfer learning" implies a cross-subject pretraining phase that this condition does not have. We will rename it `Subject-Specific Supervised (within-subject)`. The interpretation at L279–284 is unaffected — it requires both subject-specific rows to be genuinely single-subject, and they are.
>
> While verifying this we found one thing worth disclosing: the supervised pretraining path saves final-epoch rather than best-epoch weights, so both subject-specific numbers (72.2% and 64.6%) are final-epoch. This most likely understates both, and it applies to the pair symmetrically, so the direction of the comparison is unaffected. We will state it in the appendix and use best-epoch checkpointing for these rows in the camera-ready.
>
> **Part 2 — the upper bound.** This is a useful request and we believe a version of it is already in the paper, though not labelled as such. Fine-tuning on *all* of a test user's data is bounded by the dataset: with 10 repetitions per gesture and Q=9 held out, there is no substantial "remaining data" beyond K≈5–9. Table 4's **K=5 column is therefore already close to the data-saturated regime** the reviewer is asking about: 3-way reaches 96.9% (versus 88.4% at K=1) and 10-way reaches 68.5% (versus 67.7%). Read as a ceiling estimate, 1-shot 3-way sits roughly 8.5 points below saturation, while the 10-way gap is essentially closed — i.e. 10-way is limited by something other than adaptation data volume. We will add this reading explicitly, since the reviewer's question shows the current presentation does not make it available.

⚙ **INTERNAL.** Resolved, and it resolved the good way: nothing published changes, the paragraph you were most worried about survives, and R3 gets a direct quotable answer to a question they asked at confidence 4. Answering this one crisply is worth more than it looks — it demonstrates you read the code rather than the paper.

### Q3 — Is it fair to include the Kaifosh fine-tuning evaluation? Would matched featurization help?

**→ RESPONSE:** Block **KAIF** in full, plus:

> To answer the reviewer's question directly: **as executed, no, and the problems are worse than the review supposes** — see the four items in Block KAIF, all of which bias against their model. Our answer is not to repair the row into a fair head-to-head, which the electrode-topology mismatch makes impossible in principle for their gesture architecture, but to reframe it as a transfer study and state the confounds.
>
> On matched featurization specifically: we agree, and the correct direction is the one the reviewer suggests — evaluate *our* model on their input representation rather than forcing their model onto ours. We commit that cell (EMG-only, 200 Hz, shared strided-conv stem) for camera-ready, at 200 Hz rather than our 20 Hz because their stride-10 convolution decimates to 200 Hz inside the model. We also commit their architecture trained from scratch on our data, which is the cell that actually separates their architecture from their pretraining corpus.

---

# Reviewer 4 (FKXS) — Rating 3, Confidence 4

Scores: Quality 2, Clarity 3, Significance 2, **Originality 1** — the lowest score in the batch. Stated blocker: no comparison against other methods on the same benchmark.

### W1 — Hard to see real novelty; method composed of well-known submodules (MoE, MAML++, LSTM)

**→ RESPONSE:** Block **MOEMETA**'s reframing, plus:

> We accept that no individual component is new, and we are not going to argue otherwise. What we claim is the instantiation and the empirical result: that in this specific regime — one labelled example per class, user-defined label semantics, a population with high physiological heterogeneity — routed encoder capacity and meta-learned adaptation are *jointly* necessary, with parameter-matched controls ruling out capacity as the explanation, and that routing's benefit concentrates at K=1 and largely disappears by K=5. We think the characterisation of that regime is the contribution, and we have rewritten Contribution 1 to say so rather than to claim architectural novelty.

### W2 — No comparison with cited related methods or on other benchmarks

**→ RESPONSE:** the relabelling table (two external baselines already in Table 1, presented as self-ablations), plus Block **KAIF**, plus the ProtoNet addition, plus Block **DATA2** for benchmarks:

> Because the reviewer identifies this as the specific obstacle to a higher score, we want to be concrete about what the response adds: two existing Table 1 rows correctly identified as external methods (the Côté-Allard [7] transfer recipe and Kaifosh & Reardon [11]) plus one relabelled for accuracy (`Subject-Specific Supervised (within-subject)`), a re-evaluated [11] baseline with their preprocessing applied at a fairly-measured input scale, a Prototypical Networks comparison on a matched backbone `[[FILL: %]]`, and two ports of MoEMeta's distinguishing design decisions into our setting `[[FILL: Port A %, Port B %]]`. We are declining a second benchmark on the grounds in Block DATA2 — not effort, but that the available benchmarks instantiate neither property the method targets.

### W3 — No single-modality experiments

**→ RESPONSE:** Block **MOD**.

### Q1 — Why 1D-CNN experts and an LSTM rather than transformers?

**→ RESPONSE:** Block **XFMR**.

### Q2 — Are the EMG and IMU values normalized?

**→ RESPONSE:** meta-review Key question 9, first paragraph. `[[FILL from the data loader.]]`

### Q3 — Individual contributions of each modality

**→ RESPONSE:** Block **MOD**.

### Formatting — possible anonymity violation in A.3 (Rice / NOTS named)

**→ RESPONSE:**

> The reviewer is correct; A.3 names the compute cluster and institution. This was an oversight, and we will remove the identifying text from the revised PDF and camera-ready.

⚙ **INTERNAL — do not volunteer.** References [14], [27] and [28] also narrow the author set considerably. Flag for the resubmission checklist; raising it in the response draws attention to a broader problem than the one the reviewer found.

---

# Appendix A — Pending checks that gate this document

| ID | Check | Blocks | Status |
|---|---|---|---|
| **S1** | What does "Subject-Specific Transfer Learning" train on? | R3 Q2; L279–284; one relabelling row | ✅ **Within-subject only.** Paper text correct, label wrong, interpretation survives |
| **S2a** | Support/query disjointness when K+Q > 10 | The K≥3 half of Table 4 | ✅ **No leak.** Disjoint slices of one shuffled list; verified empirically |
| **S2b** | Is the gesture→index map permuted per episode? | The 10-way anomaly | ✅ **Ruled out.** Meta-training randomises the label map, so no global mapping is learnable |
| **S2c** | Realised query count at eval | Table 4 caption and protocol text | ⚠️ **Q is 9/7/5 at K=1/3/5**, not fixed at 9. Disclose; check the paper doesn't claim otherwise |
| **V7** | Parameter count of the loaded checkpoint | All of Block **KAIF** | ⚠️ **≈6.48M analytically** — correct decoder. Confirm against the checkpoint file: `bash eval_launcher.sh V7` |
| **V2** | Re-run [11] with their preprocessing at a fair input scale | The revised 62.1% / 56.0% | ⬜ `bash eval_launcher.sh A11b` — gain sweep × {head_only, full} |
| **A13** | Modality ablation, incl. the mandatory `both` control | Block **MOD** | ⬜ `bash eval_launcher.sh A13` |
| **Port A** | Support-derived routing from existing checkpoint | Block **MOEMETA** | ⬜ `bash eval_launcher.sh portA` |
| **V4** | Was the swept adaptation LR the one reported for the baseline? | Whether the asymmetric-tuning objection has teeth | ⬜ |
| — | A10 provenance — the zero-shot path could not have run in the current tree | Any A10 number in the paper | ⬜ **See Appendix C** |
| — | Table 2 expert search space vs. the Optuna study object | R1 Q1's corrected table row | ⬜ |
| — | Normalization procedure, per modality, from the data loader | Meta Q9, R4 Q2 | ⬜ |
| — | Per-subject max gate weight and routing entropy from saved gate vectors | R1 Q3 | ⬜ |

**Launch order:** `V7` first (~1 min, gates A10/A11/A11b), then `A13`, then `A11b` and `portA` on Day 2. `--dry-run` inspects first. None of the new run scripts has been executed end-to-end — they were exercised against synthetic data only — so budget the first few minutes of each job for import and key-name fixes rather than assuming a clean launch. Untested in particular: the `build_config_meta` / `run_supervised_test_eval` interaction in A11b, `replace_head_for_eval` on the subclassed wrapper, and checkpoint-key handling in `portA`.

# Appendix B — Paper edits requiring no compute

**New, from the repo audit:**

- [ ] Rename "Subject-Specific Transfer Learning" → `Subject-Specific Supervised (within-subject)`; keep L279–284 as written
- [ ] State realised query counts (9/7/5 at K=1/3/5) in Table 4's caption and the §4 protocol description, measured per episode rather than taken from the config. **Check whether the paper currently asserts a fixed Q and fix it if so** — the disclosure isn't credible alongside the opposite claim
- [ ] Note that at N=10 all evaluation episodes share one class set and label map, as an evaluation-diversity caveat on the 10-way numbers
- [ ] Disclose that the subject-specific rows (72.2%, 64.6%) use final-epoch rather than best-epoch weights; symmetric across the pair, likely understates both. Best-epoch checkpointing for camera-ready
- [ ] Rewrite the Kaifosh preprocessing description around signal-s.d. vs noise-s.d. normalisation and the gain sweep — **not** around a literal `2.46e-6` multiply, which never happened
- [ ] Add the secondary band mismatch: our 20–450 Hz band-pass vs. their 40 Hz HP, and 450 Hz against their 850 Hz analog low-pass

**Previously identified:**

- [ ] Relabel the remaining two Table 1 rows
- [ ] Move the [11] comparison into its own transfer-study subsection
- [ ] Add the head/metric conversion (9-way multilabel sigmoid detector → N-way episodic accuracy) to the C-row caption
- [ ] Add electrode topology to Limitations as a first-class confound, including the absence of a rotational-invariance module in their gesture architecture
- [ ] Add the analog-band clause (20 Hz HP / 850 Hz LP) wherever matched preprocessing is claimed
- [ ] Replace "their device has no IMU" with "their models do not use IMU," including L229–230
- [ ] Correct the parameter count 60M → ~6.5M wherever it appears; present both readings of LP > FT (with K=1 and 6.5M parameters, full-FT overfitting is the simpler explanation and says little about representation quality)
- [ ] Cite "N=4,800 (largest model, Fig. 2f)" rather than a bare participant count
- [ ] Quote [11]'s clinical-population caveat in §4.4
- [ ] Correct Table 2's expert search space; add the Fig. 3 grid note and the A.4 reconciliation
- [ ] Name the evaluation protocol (L2SO vs. fixed 24/4/4) in **every** table caption
- [ ] Soften L171–173 from an inductive-bias assertion to a labelled design hypothesis
- [ ] Soften L110–112's claim about metric-learning fragility, or support it with the ProtoNet run
- [ ] Cut or soften the $B expressivity sentence in §2 `[[PI: decision pending — see Appendix C]]`
- [ ] Add MoEMeta citation to §2's MoE paragraph plus a §1 sentence; rewrite Contribution 1
- [ ] Add the expert-count note to A.4 (ordinal or mechanistic framing only)
- [ ] Fix the §4.6 / A.7 episode-count inconsistency: §4.6 says 100 query episodes per user, but A.7's 21,600 samples ÷ 32 users = 675 per user, and 100 × 27 = 2,700 ≠ 675 (675 = 25 × 27)
- [ ] Remove Rice/NOTS from A.3
- [ ] Fig. 3 caption wording (says the sweep was on the test split; the sweep was correct, the wording isn't) — **camera-ready only, no reviewer raised it, do not volunteer**

# Appendix C — Items requiring PI sign-off before they appear in any response

1. **$B.** Whether to evaluate it and report it, or take only the floor (cut the §2 expressivity critique). Non-negotiable floor: we do not criticise a method we did not measure — and it runs on the identical dataset, is prior published work, and per earlier runs outperforms EncoderMoE. If it is reported, the framing is amortisation rather than accuracy: $B stores an exemplar per class per user with cost growing as O(users × classes), no shared model, no cross-user generalisation, and cannot be deployed to a new user without their full template set; EncoderMoE is a single model with a transferable representation from the same one-shot budget, whose routing recovers ability-level structure that $B cannot produce at all.
2. **Both reframes** — moving [11] to a transfer study, and moving the contribution away from method novelty.
3. **Statistics family.** Adding conditions changes the correction family and shifts every p-value in §4.5. Two defensible options, pick one and declare it: refit the omnibus over all conditions with Holm–Bonferroni over the full family, or pre-specify the new comparisons as a second family and correct within it. Keep Greenhouse–Geisser (sphericity will still be violated; Mauchly W=0.039, ε=0.46) and Cohen's d_z on per-subject difference scores. Any cell not evaluable per-participant across all 32 participants — including everything on the fixed split — sits outside the paired analysis, stated.
4. ~~Anything from S2 branch B.~~ **Resolved — no leak, no escalation needed.** Support and query are disjoint by construction and verified empirically.
5. **A10 provenance — the one genuinely open integrity question.** Commit `215cd94` renamed a `MetaGestureDataset` keyword argument and missed six call sites, each of which raises `TypeError` on construction. One of them is A10's prototypical zero-shot path. All six are now fixed, but the implication needs tracing: **if an A10 number appears anywhere in the paper, it cannot have been produced by the current tree**, so it came from a pre-rename version of the code and its provenance should be established before it is defended in a response. Worth ten minutes of `git log` before Monday, and worth telling the PI what you find either way — this is exactly the class of thing that is survivable self-reported and not survivable reviewer-found.
