# SOTA comparisons and baseline plan (Opus P6)

*Written 2026-10-21 by Opus, from web searches run the same day. Entries marked **[verified]** were checked against the paper or abstract page linked. Entries marked **[canonical]** are standard ML references cited from memory: check the bibliographic details before they go into the paper. Accuracies quoted from other papers are on their own datasets and protocols and are **never** comparable to ours. Comparisons happen only by running methods on our episode cache.*

## 0. Bottom line
1. **There is no single SOTA for sEMG+IMU gesture recognition.**
   - The field reports on NinaPro DB2/DB3/DB5/DB7 or on small self-collected sets, with fixed, experimenter-defined gestures and mostly within-subject windows.
   - A 2025 review of sEMG-IMU fusion confirms the field is fragmented ([review](https://www.sciencedirect.com/science/article/pii/S1566253525004956)).
   - None of these works uses user-defined gestures, disabled participants and a few-shot protocol together. That gap is our application contribution.
   - It also means "beating SOTA" has to be shown by **reimplementing representative SOTA methods on our data**, not by quoting numbers.
2. **The most important new prior work for framing:**
   - Cross-user zero-shot myoelectric control **works** for fixed gesture sets at scale: 612 users in EMG-EPN612 ([Eddy et al. 2024, Frontiers Bioeng.](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1463377/pdf)) **[verified, abstract]**.
   - Reviewers will ask why we need few-shot at all. The answer is user-defined gestures, and `SPEC_zeroshot.md` tests exactly that. Cite Eddy et al. as the contrast case.
3. **Personalising a cross-user encoder:**
   - On NinaPro DB1/DB2/DB5 (77 subjects), full fine-tuning beat prototypical adaptation at every calibration budget. Prototypes recovered 52–78% of fine-tuning's gain ([Odeyemi & Zhang, arXiv 2609.20296, Jul 2026](https://arxiv.org/abs/2609.20296)) **[verified, abstract; preprint]**.
   - Implication: our fine-tune baselines (A1/A2 full-ft) must get a **properly dev-tuned** adaptation budget. The current A1/A2 deliberately mirror MAML's 10-step budget (archived LIMITATIONS, A16 notes). Report both the mirrored and the dev-tuned budget, or a reviewer will say the fine-tune baseline was crippled.
4. **Closest prior work to our population and setting:**
   - [Sarwat et al., JNER 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11167772/) **[verified, abstract]**: 20 stroke survivors, forearm EMG + wrist FMG + IMU, 7 gestures, **one-shot prototypical networks**. 82.2% vs 63.2% for one-shot NN transfer.
   - Cite it, and include ProtoNet on a multimodal encoder as a core baseline. It is "the method that already worked for this population".
5. **Population and HCI motivation:**
   - ASSETS '23 (Wobbrock lab): people with upper-body motor disabilities design highly ability-specific gestures, even within one disability type ([PDF](https://faculty.uw.edu/wobbrock/pubs/assets-23.01.pdf)) **[verified, abstract]**. This is the strongest citation for "user-defined gestures" and for an IMWUT/ASSETS framing.
   - Xu et al. CHI 2022 (wrist accelerometer gesture customisation): 55.3/83.1/87.2% at 1/3/5 shots for new user-defined gestures ([arXiv 2203.15239](https://arxiv.org/pdf/2203.15239); [ACM](https://dl.acm.org/doi/10.1145/3491102.3501904)) **[verified, abstract]**. Good HCI reference point. Different data, so not comparable.

## 1. Candidate baselines

### A. Classical and non-parametric (cheap; run on the episode cache first)
| Method | Ref | Code | Fit to our data | Effort | Reviewer recognition |
|---|---|---|---|---|---|
| `$B` (per-template PCA, L1 1-NN) | prior work, notebook `run_$B.ipynb` | ours | exact | low (port) | must-have: suspected to beat EncoderMoE |
| kNN / ProtoNet on raw & PCA features | `eval_knn_proto.py` | ours | exact | low | expected |
| Riemannian: per-modality channel covariance → MDM (min-distance-to-Riemannian-mean, a natural few-shot rule) and tangent space + LR | Barachant et al. 2012 **[canonical]**; EMG: [Gowda & Miller, J. Neural Eng. 2024](https://arxiv.org/abs/2311.08548) **[verified]** | `pyriemann` | good; covariance of 16 EMG and 72 IMU channels over 64 steps (IMU block may need shrinkage, OAS) | low | high in the EMG/BCI community |
| MiniRocket + ridge, fit per episode on support | [Dempster et al., KDD 2021, arXiv 2012.08791](https://arxiv.org/abs/2012.08791) **[verified]** | `aeon` | multivariate (88 ch x 64); with K=1 the ridge has 10 samples, so expect it to be weak at K=1 and better at K=5 | low | high in time-series ML |

### B. Few-shot / meta-learning (generic ML; ICML reviewers expect these)
| Method | Ref | Status in repo | Effort |
|---|---|---|---|
| Fine-tune transfer (supervised pretrain, then adapt on support) | Chen et al. ICLR 2019 "A Closer Look at Few-shot Classification" **[canonical]** | A1/A2 (head-only and full) | low: add a dev-tuned budget |
| ProtoNet (episodic training, Euclidean prototypes) | Snell et al. NeurIPS 2017 **[canonical]** | A14 uses the M0 encoder; a ProtoNet *trained* episodically is missing | medium |
| MAML / MAML++ without MoE | Finn et al. ICML 2017; Antoniou et al. ICLR 2019 **[canonical]** | A3/A4 | done |
| ANIL (adapt head only) | Raghu et al. ICLR 2020 **[canonical]** | config flag over `inner_param_filter` | low |
| Proto-MAML (prototype-initialised head) | Triantafillou et al. ICLR 2020 **[canonical]** | archived PLAN Phase 1 C1 | medium |
| Matching networks | Vinyals et al. NeurIPS 2016 **[canonical]** | `AttentiveMatchingNetwork.py` (dead code) | medium. **Tier 3:** ProtoNet covers the metric family. |
| CNP / ANP | Garnelo et al. ICML 2018 **[canonical]** | none | medium. **Tier 3:** Kai mentioned it; low expected value at 10 reps per class. |
| ADANN (domain-adversarial across users) | [Côté-Allard et al., Front. Bioeng. 2020](https://www.frontiersin.org/articles/10.3389/fbioe.2020.00158/full); TADANN, TNSRE 2021 **[verified]** | none | medium (gradient-reversal head over user id, then fine-tune on support) |

### C. EMG-specific deep baselines (reviewer-recognisable)
| Method | Ref | Code | Notes | Effort |
|---|---|---|---|---|
| Côté-Allard transfer ConvNet | [TNSRE 2019, 27(4):760–771](https://arxiv.org/abs/1801.07756) **[verified]** | public (authors') | the classic "pretrain on many users, fine-tune" for EMG; our A2 is a close analogue, so cite A2 as this protocol rather than port their net, unless reviewers insist | low |
| FS-HGR (few-shot EMG, temporal conv + attention, meta-learning) | [Rahimian et al., TNSRE 2021](https://arxiv.org/abs/2011.06104) **[verified]** | check | the canonical few-shot sEMG baseline. **Include.** | medium |
| TEMGNet (ViT for sEMG) | [Rahimian et al., RA-L, arXiv 2109.12379](https://arxiv.org/abs/2109.12379) **[verified]** | check | an EMG transformer reviewers know; doubles as a transformer baseline | medium |
| EMG-CrossFormer (per-modality encoders + cascaded cross-attention fusion + learnable gesture queries; sEMG + inertial) | [Del Pup, Tentori, Atzori, arXiv 2607.22779, Jul 2026](https://arxiv.org/abs/2607.22779) **[verified]** | [github.com/deepPNClab/emg-crossformer](https://github.com/deepPNClab/emg-crossformer), MIT **[verified repo exists]** | **the best current SOTA EMG+IMU fusion candidate.** Recent, from the NinaPro group (Atzori), explicitly multimodal, code available. Inertial adds +13 to +28 points over sEMG-only on NinaPro DB2/3/7/10 (their numbers). Expects raw sEMG windows, so run it on the 2 kHz representation once `SPEC_2khz_emg.md` lands, and on the 20 Hz envelope as a fallback. Preprint, not peer-reviewed. | medium–high |
| BSCAL (CNN-LSTM + attention, dual-stream EMG/IMU) | [Wang et al., Int. J. Med. Robot. 2025](https://onlinelibrary.wiley.com/doi/10.1002/rcs.70115) **[abstract only; full text 403]** | unknown | architecturally close to ours (CNN-LSTM). Cite; do not port. | — |

### D. Transformers and foundation models
| Method | Ref | Use |
|---|---|---|
| TST (time-series transformer), param-matched + naive presets | Zerveas et al. KDD 2021 **[canonical]**; in repo | Kai's transformer ablations, `SPEC_moe_compute.md` |
| PatchTST | Nie et al. ICLR 2023 **[canonical]** | optional second generic TS transformer; one config change from TST with patching |
| TinyMyo (3.6M-param transformer, masked-reconstruction pretrained on public sEMG) | [arXiv 2512.15729](https://arxiv.org/abs/2512.15729) **[verified listing]** | frozen pretrained EMG encoder + ProtoNet head: answers "why not use a pretrained EMG foundation model?" Needs raw EMG at its expected rate, so it depends on `SPEC_2khz_emg.md`. |
| EMG foundation model (1,667 subjects) | [bioRxiv 2025.12.17.694831](https://www.biorxiv.org/content/10.64898/2025.12.17.694831v1.full) **[verified listing]** | cite as related work; port only if weights are public |
| Meta generic neuromotor interface | [Kaifosh & Reardon et al., Nature 645 (2025)](https://ideas.repec.org/a/nat/nature/v645y2025i8081d10.1038_s41586-025-09255-w.html) **[verified]** | existing A10/A11 (frozen and fine-tuned); redo under the 2 kHz spec; disclose the electrode-topology mismatch |

## 2. Prioritised build list for Sonnet (all on the episode cache, all through `SPEC_stats.md`)
**Tier 1 — before the full runs (cheap, decisive for `DECISION_method.md`):**
1. `$B` faithful port (A6).
2. kNN and ProtoNet, raw and PCA.
3. Riemannian MDM and tangent space + LR, per modality.
4. MiniRocket + ridge.
5. Fine-tune transfer (A2) with both the MAML-mirrored and the dev-tuned budget.
6. MAML without MoE (A4).
7. ANIL (flag).
8. Episodically trained ProtoNet on the A2/A4 backbone.
9. TST param-matched and naive.
10. Zero-shot Z1–Z4 (`SPEC_zeroshot.md`).

**Tier 2 — the "state of the art" answer (after Tier 1 and the 2 kHz spec):**

11. EMG-CrossFormer as (a) fine-tune transfer and (b) ProtoNet encoder.
12. FS-HGR.
13. ADANN + fine-tune.
14. Proto-MAML head on the main method.
15. TEMGNet.

**Tier 3 — only if time, or if a reviewer asks:** TinyMyo frozen-encoder ProtoNet, PatchTST, CNP, matching networks, Kaifosh A10/A11 redo.

## 3. Related-work positioning (one paragraph to adapt)
> Prior sEMG(+IMU) gesture recognition evaluates fixed, experimenter-defined gesture vocabularies, typically within-subject (NinaPro; recent multimodal transformers such as EMG-CrossFormer). Cross-user models can generalise zero-shot when every user performs the same gestures (Eddy et al., 2024), and pretrained encoders can be personalised with a few calibration repetitions (Odeyemi & Zhang, 2026; Rahimian et al., 2021). People with motor disabilities, however, choose ability-specific movements for the same intent (ASSETS '23), so a gesture label carries no shared movement across users. We study this harder regime: N-way K-shot recognition of user-defined gestures from fused sEMG and IMU, in 32 participants including people with upper-limb disabilities, and compare meta-learned, fine-tuned, metric, Riemannian and transformer approaches on identical episodes.

The sentence about labels carrying no shared movement is only valid if `SPEC_zeroshot.md` lands in its first row. Otherwise use that spec's wording.

## Sources
- [sEMG-IMU fusion review, Information Fusion 2025](https://www.sciencedirect.com/science/article/pii/S1566253525004956)
- [EMG armbands & multimodal fusion review (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12818089/)
- [BSCAL, Int. J. Med. Robot. 2025](https://onlinelibrary.wiley.com/doi/10.1002/rcs.70115)
- [EMG-CrossFormer, arXiv 2607.22779](https://arxiv.org/abs/2607.22779) · [code](https://github.com/deepPNClab/emg-crossformer)
- [FS-HGR, arXiv 2011.06104 / TNSRE 2021](https://arxiv.org/abs/2011.06104)
- [TEMGNet, arXiv 2109.12379](https://arxiv.org/abs/2109.12379)
- [Hybrid CViT vs CNN for low-channel sEMG (PMC12730683)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12730683/)
- [Lightweight test-time adaptation for EMG, arXiv 2601.04181](https://arxiv.org/pdf/2601.04181)
- [REACT user-adaptive sEMG, arXiv 2605.30127](https://arxiv.org/pdf/2605.30127)
- [Personalising a cross-user sEMG encoder, arXiv 2609.20296](https://arxiv.org/abs/2609.20296)
- [Post-stroke one-shot prototypical networks, JNER 2024 (PMC11167772)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11167772/)
- [Zero-shot cross-user EMG, Eddy et al. 2024](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1463377/pdf)
- [Côté-Allard et al. 2019 transfer learning, arXiv 1801.07756](https://arxiv.org/abs/1801.07756)
- [ADANN, Frontiers 2020](https://www.frontiersin.org/articles/10.3389/fbioe.2020.00158/full) · [TADANN, arXiv 1912.09380](https://arxiv.org/abs/1912.09380)
- [Gowda & Miller, Riemannian sEMG, arXiv 2311.08548](https://arxiv.org/abs/2311.08548)
- [MiniRocket, arXiv 2012.08791](https://arxiv.org/abs/2012.08791)
- [TinyMyo, arXiv 2512.15729](https://arxiv.org/abs/2512.15729)
- [EMG foundation model, bioRxiv](https://www.biorxiv.org/content/10.64898/2025.12.17.694831v1.full)
- [Kaifosh & Reardon, Nature 2025](https://ideas.repec.org/a/nat/nature/v645y2025i8081d10.1038_s41586-025-09255-w.html) · [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2024.02.23.581779v2)
- [ASSETS '23 personalised upper-body gestures](https://faculty.uw.edu/wobbrock/pubs/assets-23.01.pdf)
- [Wrist-worn gesture customisation, CHI 2022 (arXiv 2203.15239)](https://arxiv.org/pdf/2203.15239)
- [Few-shot continual learning for motor-impaired users, arXiv 2405.08969](https://arxiv.org/pdf/2405.08969)
