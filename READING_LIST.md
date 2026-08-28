# READING_LIST.md — what to read, and which problem each paper solves

Organised by **the problem it addresses in our paper**, not by topic, so the
reading maps onto `LIMITATIONS.md` and the resubmission plan. Each entry says
why it matters *for us* — several change what we should claim, not just what we
should cite.

**Priority key:** ★★★ read before writing another line · ★★ read before the
relevant experiment · ★ background / for the related-work section.

**Verification note.** Titles and venues below are given from memory and are
right in substance, but check every one against the actual PDF before it enters
the `.bib` — a mis-cited year in a rebuttal is a cheap own goal. Entries marked
**[FILL]** are ones only you have.

---

## 0. The four papers that would have prevented this review round

If there is time for four papers, these are the four.

| ★ | Paper | Why it is load-bearing for us |
|---|---|---|
| ★★★ | **Wang, Tran & Feiszli, "What Makes Training Multi-modal Classification Networks Hard?", CVPR 2020** | Multimodal networks *routinely underperform* their best unimodal branch, and they explain why (overfitting/generalisation mismatch between modalities) and fix it (Gradient-Blending). This is *exactly* our symptom in `LIMITATIONS.md` A1. It also means "our fused model does not beat unimodal" is a **known phenomenon with a named cause**, not an embarrassment — which is a much better position to write from. |
| ★★★ | **Raghu, Raghu, Bengio & Vinyals, "Rapid Learning or Feature Reuse? Towards Understanding the Effectiveness of MAML", ICLR 2020 (ANIL)** | Shows MAML's inner loop barely changes the body — almost all of the adaptation is the **head**. That is the direct theoretical backing for our 10-way diagnosis (A5) and for the prototype-head fix. Also gives us a nearly-free ablation (ANIL: adapt the head only) that is a *named baseline* rather than something we invented. |
| ★★★ | **Chen, Liu, Kira, Wang & Huang, "A Closer Look at Few-shot Classification", ICLR 2019** | A plain **cosine-classifier baseline** ("Baseline++") matches or beats sophisticated meta-learners, and the gap shrinks as the backbone gets better. Read together with **Tian et al., "Rethinking Few-Shot Image Classification: A Good Embedding Is All You Need?", ECCV 2020** — a simple embedding plus a nearest-centroid/linear readout beats meta-learning on standard benchmarks. **This is the literature that makes "PCA+KNN beats us" a well-documented result rather than a fatal flaw** (D2). It also tells us how to respond: the field's answer is a cosine/prototype readout on a good embedding, which is precisely the plan's prototype head. |
| ★★★ | **Bouthillier, Delaunay, Bronzi, et al., "Accounting for Variance in Machine Learning Benchmarks", MLSys 2021** | Decomposes benchmark variance into its sources (seed, data order, splits) and shows how much of the literature's reported improvements sit inside it. This is the citation for our 88.46 / 87.58 / 90.68 problem (B1) and the template for the variance decomposition the plan proposes. Pair with **Agarwal, Schwarzer, Castro, Courville & Bellemare, "Deep RL at the Edge of the Statistical Precipice", NeurIPS 2021** for small-N reporting practice (stratified bootstrap CIs, interquartile mean) — our n=32 is exactly the regime it was written for. |

---

## 1. Fusion — why EMG contributes nothing, and what to do (A1, A2, A3)

| ★ | Paper | Use |
|---|---|---|
| ★★★ | Wang, Tran & Feiszli, CVPR 2020 (above) | The diagnosis and Gradient-Blending. |
| ★★★ | **Neverova, Wolf, Taylor & Nebout, "ModDrop: Adaptive Multi-Modal Gesture Recognition", TPAMI 2016** | **Modality dropout, in gesture recognition, a decade ago.** Almost exactly our plan 0.6. Cite it as prior art *and* as validation — and note that it means modality dropout is not a contribution we can claim, only a control we should have run. |
| ★★★ | **Huang, Lin, Du, Yang, Huang & Wang, "Modality Competition: What Makes Joint Training of Multi-modal Network Fail in Deep Learning?", ICML 2022** | Theory for our exact failure: with joint training, one modality wins the optimisation and the other is never learned, even when it carries signal. Gives us the vocabulary ("modality competition", "modality laziness") to describe A1 precisely. |
| ★★ | **Peng, Wei, Deng, Wang & Hu, "Balanced Multimodal Learning via On-the-fly Gradient Modulation", CVPR 2022 (OGM-GE)** | A cheap, per-modality gradient rebalancing scheme. Directly implementable alongside the per-modality aux heads, and a stronger fix than modality dropout alone. |
| ★★ | **Du, Teng, Wang, Wang & Wang, "On Uni-modal Feature Learning in Supervised Multi-modal Learning", ICML 2023** | Argues the right recipe is often: learn unimodal features well *first*, then fuse. Relevant to whether our per-modality stems should be pretrained separately. |
| ★★ | **Nagrani, Yang, Arnab, Jansen, Schmid & Sun, "Attention Bottlenecks for Multimodal Fusion", NeurIPS 2021 (MBT)** | The cleanest modern mid-fusion design: a small set of shared bottleneck tokens instead of full cross-attention. A better `fusion_mode="cross_attention"` than naive all-pairs, and cheap at T=64. |
| ★ | **Tsai, Bai, Liang, Kolter, Morency & Salakhutdinov, "Multimodal Transformer for Unaligned Multimodal Language Sequences", ACL 2019 (MulT)** | The canonical cross-modal attention formulation to cite for the design. |
| ★ | **Perez, Strub, de Vries, Dumoulin & Courville, "FiLM: Visual Reasoning with a General Conditioning Layer", AAAI 2018** | We already have FiLM in the contrastive branch (dead in the production path). Cite properly if it gets revived; note ours conditions on *demographics*, not on the other modality, which is a different and weaker thing. |
| ★ | **Baltrušaitis, Ahuja & Morency, "Multimodal Machine Learning: A Survey and Taxonomy", TPAMI 2019** | For getting the early/mid/late fusion terminology right in §2. Our `torch.cat` at the input is unambiguously *early* fusion, and the paper should say so. |

**What this literature changes about our claims.** Contribution 1 currently asserts
that multimodal fusion matters. The honest, better-supported version — and one
this literature actively supports — is: *naive early fusion of channel-imbalanced
modalities is dominated by the high-dimensional one; making EMG contribute
requires per-modality capacity, scale equalisation, and an explicit anti-competition
mechanism.* That is a finding, not a defect.

---

## 2. The 10-way collapse and the prototype head (A5)

| ★ | Paper | Use |
|---|---|---|
| ★★★ | Raghu et al., ICLR 2020 (ANIL) (above) | The head *is* the adaptation. |
| ★★★ | **Snell, Swersky & Zemel, "Prototypical Networks for Few-shot Learning", NeurIPS 2017** | The prototype readout itself. **Read §3.3 specifically**: they report that training with a *higher* way than testing improves performance — directly relevant to whether we should meta-train at 10-way (plan C3). |
| ★★★ | **Triantafillou, Zhu, Dumoulin, et al., "Meta-Dataset", ICLR 2020** | Where **Proto-MAML** is defined: initialise the final layer from class prototypes, then run the MAML inner loop. This is the plan's `CosineProtoHead`, so it gives us the right name and citation instead of presenting it as novel. |
| ★★ | **Dhillon, Chaudhari, Ravichandran & Soatto, "A Baseline for Few-Shot Image Classification", ICLR 2020** | Transductive fine-tuning as a strong, simple baseline, plus a serious critique of few-shot evaluation practice (confidence intervals, number of episodes) — relevant to B1/B3/B7. |
| ★★ | **Vinyals, Blundell, Lillicrap, Kavukcuoglu & Wierstra, "Matching Networks for One Shot Learning", NeurIPS 2016** | The origin of episodic N-way K-shot evaluation, and the source of the "match train and test conditions" principle we currently violate by meta-training at 3-way and reporting 10-way. |
| ★★ | **Laenen & Bertinetto, "On Episodes, Prototypical Networks, and Few-shot Learning", NeurIPS 2021** | Argues episodic training is often unnecessary or harmful. An uncomfortable but important read given A4/D2 — if it holds here, a non-episodic embedding plus a prototype readout is the model to beat, and that is roughly what PCA+KNN is. |
| ★ | **Oreshkin, Rodríguez & Lacoste, "TADAM: Task dependent adaptive metric for improved few-shot learning", NeurIPS 2018** | Task-conditioned scaling and metric learning; the closest prior art to our planned task-conditional routing. |
| ★ | **Requeima, Gordon, Bronskill, Nowozin & Turner, "Fast and Flexible Multi-Task Classification Using Conditional Neural Adaptive Processes", NeurIPS 2019 (CNAPs)**, and **Bateni et al., "Improved Few-Shot Visual Classification", CVPR 2020 (Simple CNAPS)** | FiLM-based task conditioning, and then the finding that a *Mahalanobis* distance readout beats the learned adaptation. Another data point that the metric matters more than the adaptation machinery. |
| ★ | **Ye, Hu, Zhan & Sha, "Few-Shot Learning via Embedding Adaptation with Set-to-Set Functions", CVPR 2020 (FEAT)** | A transformer over the support set to adapt embeddings — a transformer-based few-shot design that is *not* just "swap the encoder", useful for the D1 transformer story. |

---

## 3. Making the MoE actually specialise (A4)

| ★ | Paper | Use |
|---|---|---|
| ★★★ | **Zoph, Bello, Kumar, et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models", 2022** | The practitioner's guide to MoE instability, including the **router z-loss** and a frank treatment of what the load-balancing loss does and does not buy. Most relevant single paper to our 98.2%-flat routing. |
| ★★★ | **Puigcerver, Riquelme, Mustafa & Houlsby, "From Sparse to Soft Mixtures of Experts", ICLR 2024 (Soft MoE)** | Replaces hard top-k dispatch with soft slot assignment. Important for us for two reasons: (a) it is the principled version of what our soft-masked top-k already approximates, and (b) it sidesteps the unselected-expert / `None`-gradient hazard (B14) that rules out true sparse dispatch under MAML. |
| ★★ | **Shazeer, Mirhoseini, Maziarz, et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", ICLR 2017** | The source of our `importance_loss` (CV²) and load loss. Read it to state correctly in the paper that **both are batch-marginal constraints**, which is why re-enabling `MOE_importance_coeff` cannot fix per-sample flatness. |
| ★★ | **Fedus, Zoph & Shazeer, "Switch Transformers", JMLR 2022** | The source of `topk_MOE_aux_loss`, and the reference for token-level MoE if the `TST_TokenMOE` row ships. |
| ★★ | **Chen, Deng, Wu, Gu & Li, "Towards Understanding the Mixture-of-Experts Layer in Deep Learning", NeurIPS 2022** | Theory on *when* MoE specialises: it requires cluster structure in the input that the router can find. Our routing recovers ability-level but **not** subject-level structure — this paper is how we frame that as an expected result rather than a disappointment, and it supports the "ensembling, not specialisation" conclusion if that is what the data says. |
| ★★ | **Dai, Deng, Zhao, et al., "DeepSeekMoE", ACL 2024** | Shared-expert isolation and fine-grained expert segmentation. We already have `SharedExpert` (currently off); this is the citation and the argument for turning it on, and fine-grained experts are directly relevant to the E=8-vs-E=22 plateau (C1). |
| ★ | **Jacobs, Jordan, Nowlan & Hinton, "Adaptive Mixtures of Local Experts", Neural Computation 1991** | The origin. One sentence in §2; it also usefully frames MoE as an old idea, which softens the originality attack in the other direction. |
| ★ | **Muqeeth, Liu & Raffel, "Soft Merging of Experts with Adaptive Routing", 2024 (SMEAR)** | Merges expert *parameters* by gate weight instead of averaging outputs. Cheaper than our stack-and-sum and worth a comparison. |
| ★★★ | **[FILL] Wu & Yin, MoEMeta, NeurIPS 2025** | The paper R3 flagged, and the highest-priority citation in the response. Read it properly rather than from our own summary — the ports (A: support-derived routing, B: frozen bank + local adaptation) must be faithful, and R3 is at confidence 4. |

---

## 4. Transformers, and strong cheap baselines we are missing (D1, D2)

| ★ | Paper | Use |
|---|---|---|
| ★★★ | **Barachant, Bonnet, Congedo & Jutten, "Multiclass Brain–Computer Interface Classification by Riemannian Geometry", IEEE TBME 2012** (and the `pyRiemann` toolbox) | **Read this one even though it is not on anyone's list.** Riemannian tangent-space mapping of channel covariance matrices plus a simple classifier is the strongest classical baseline for small-sample EMG/EEG, and it frequently beats deep networks in exactly our regime. We already have a `CovarianceEmbeddingNet` gesturing at this idea. Two implications: (a) if PCA+KNN beats us, **Riemannian tangent space + logistic regression will probably beat PCA+KNN**, so we should find that out ourselves rather than in review; (b) including it is a much stronger claim of baseline diligence than PCA+KNN alone. |
| ★★★ | **Dempster, Petitjean & Webb, "ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels", DMKD 2020** (and **MiniRocket**, KDD 2021) | Random convolutional kernels plus a linear classifier — near-SOTA time-series classification, trains in seconds, no tuning. The other strong cheap baseline we should include, and a fair one because it has essentially no hyperparameters to tune in our favour or against. |
| ★★ | **Zerveas, Jayaraman, Patel, Bhamidipaty & Eickhoff, "A Transformer-based Framework for Multivariate Time Series Representation Learning", KDD 2021** | Our existing `TST`. Read the config section: our shipped 197k-parameter / 9-token instance is far from theirs, which is what makes it a straw man (D1). |
| ★★ | **Nie, Nguyen, Sinthong & Kalagnanam, "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers", ICLR 2023 (PatchTST)** | Patching done properly, and the source of the intuition that patch length is a first-order hyperparameter — which is why our TST search space must include `patch_len ∈ {1,2}`. |
| ★★ | **Zeng, Chen, Zhang & Xu, "Are Transformers Effective for Time Series Forecasting?", AAAI 2023 (DLinear)** | A linear model beating transformers. Supports our design hypothesis honestly, *and* gives us the right way to phrase it: as an empirical claim in a low-data regime, not an assertion about inductive bias. |
| ★ | **Ismail Fawaz, Forestier, Weber, Idoumghar & Muller, "Deep learning for time series classification: a review", DMKD 2019**; **InceptionTime**, DMKD 2020 | The standard architecture set for time-series classification. Reviewers in the domain venues will expect at least one of these. |
| ★ | **Middlehurst, Schäfer & Bagnall, "Bake off redux: a review and experimental evaluation of recent time series classification algorithms", DMKD 2024** | The current benchmarking picture, and useful for choosing which classical baselines are worth the compute. |
| ★ | **Hudgins, Parker & Scott, "A new strategy for multifunction myoelectric control", IEEE TBME 1993** | The origin of the TD feature set our `feat_td5` implements. Cite it where we describe the engineered-feature baseline. |

---

## 5. Augmentation (plan 0.7)

| ★ | Paper | Use |
|---|---|---|
| ★★★ | **Um, Pfister, Pichler, et al., "Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring using Convolutional Neural Networks", ICMI 2017** | **The** source for IMU augmentation: jitter, scaling, **rotation**, permutation, magnitude-warp, time-warp — with the rotation argument made on physical grounds (sensor placement/orientation varies). This is the citation for plan 0.7's `IMURandomRotation`, and it validates rotation as the highest-value IMU augmentation. |
| ★★ | **Iwana & Uchida, "An empirical survey of data augmentation for time series classification with neural networks", PLOS ONE 2021** | Systematic comparison across many datasets and methods. Use it to *pick* our augmentation set defensibly rather than by intuition, and to set expectations about effect sizes. |
| ★★ | **Wen, Gao, Song, et al., "Time Series Data Augmentation for Deep Learning: A Survey", IJCAI 2021** | Broader taxonomy; good for the paper's augmentation subsection. |
| ★ | **Zhang, Cissé, Dauphin & Lopez-Paz, "mixup", ICLR 2018**; **Verma et al., "Manifold Mixup", ICML 2019** | For the record on why we are *deferring* mixup: it needs soft labels, which our `CrossEntropyLoss(logits, int_labels)` inner loop cannot consume without changing the published training path. |
| ★ | **Park, Chan, Zhang, et al., "SpecAugment", Interspeech 2019** | The masking idea, if we add time/channel masking. |
| ★ | **Eldele, Ragab, Chen, et al., "Time-Series Representation Learning via Temporal and Contextual Contrasting", IJCAI 2021 (TS-TCC)**; **Zhang, Zhao, Tsiligkaridis & Zitnik, "Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency", NeurIPS 2022 (TF-C)** | If the contrastive branch gets revived (plan Phase 2). Note our contrastive module currently has **no augmentations at all**, which is unusual for a contrastive method and is the first thing to fix there. |
| ★ | **Tang, Perez-Pozuelo, Spathis & Mascolo, "Exploring Contrastive Learning in Human Activity Recognition for Healthcare", 2021** | Cross-modal EMG↔IMU as augmented views — the Phase 2 reach item. |

---

## 6. Evaluation methodology and statistics (B1, B4, B7, C4)

| ★ | Paper | Use |
|---|---|---|
| ★★★ | Bouthillier et al., MLSys 2021 (above) | Variance decomposition. |
| ★★★ | Agarwal et al., NeurIPS 2021 (above) | Small-N reporting: stratified bootstrap CIs, performance profiles. Our n=32 subjects is precisely this setting, and adopting their reporting style would be a visible rigor upgrade. |
| ★★ | **Demšar, "Statistical Comparisons of Classifiers over Multiple Data Sets", JMLR 2006** | The standard reference for the paired non-parametric tests we should lead with (Wilcoxon over 32 paired subjects), and for multiple-comparison correction. Directly relevant to the Appendix C.3 statistics-family decision. |
| ★★ | **Melis, Dyer & Blunsom, "On the State of the Art of Evaluation in Neural Language Models", ICLR 2018** | Shows apparent architectural gains dissolve under equal HPO budgets. This is the argument for giving the transformer its own equal budget (D1), and the citation that makes our fairness protocol credible rather than performative. |
| ★★ | **Dror, Baumer, Shlomov & Reichart, "The Hitchhiker's Guide to Testing Statistical Significance in NLP", ACL 2018** | Practical test selection; readable and directly applicable. |
| ★ | **Reimers & Gurevych, "Reporting Score Distributions Makes a Difference", EMNLP 2017** | Single-seed reporting considered harmful. One-line citation for B1. |
| ★ | **Gorman & Bedrick, "We Need to Talk about Standard Splits", ACL 2019** | Fixed-split vs resampled evaluation — relevant to our dual-protocol confusion (E5) and to why L2SO is the better headline. |
| ★ | **Lucic, Kurach, Michalski, Gelly & Bousquet, "Are GANs Created Equal?", NeurIPS 2018**; **Sculley, Snoek, Wiltschko & Rahimi, "Winner's Curse? On Pace, Progress, and Empirical Rigor", ICLR 2018 Workshop**; **Lipton & Steinhardt, "Troubling Trends in Machine Learning Scholarship", 2018** | Background on the failure mode we are trying not to instantiate. Genuinely worth reading once for the framing of the resubmission. |
| ★★ | **Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization", JMLR 2012**; **Akiba et al., "Optuna", KDD 2019**; **Li et al., "Hyperband", JMLR 2018** | HPO methodology and the pruner. Hyperband/median-pruning is what plan C4 adds. |

---

## 7. Domain: EMG/IMU gesture recognition, and motor-impaired populations (D4, D5, D7)

| ★ | Paper | Use |
|---|---|---|
| ★★★ | **[FILL] Our dataset's source paper (Yamagami Lab, UIST 2024)** | Must be cited precisely and its protocol described consistently with ours. Also the source for the participant/session facts in D7 — including whether *any* participant has a repeat session, which is the one cheap partial answer to the cross-session limitation. |
| ★★★ | **[FILL] Kaifosh & Reardon et al. (our [11]), generic non-invasive neuromotor interface, Nature 2025** | Re-read the Methods properly for D4/D5: their preprocessing (noise-s.d. normalisation, 40 Hz high-pass, the `x/(32+\|x\|)` squash), the analog front-end band, the detector head and CLER scoring, and the absence of a rotational-invariance module in the *gesture* architecture. Every clause of Block KAIF depends on getting this right. |
| ★★★ | **Côté-Allard, Fall, Drouin, et al., "Deep Learning for Electromyographic Hand Gesture Signal Classification Using Transfer Learning", IEEE TNSRE 2019** (our [7]) | Already our A2 baseline, currently mislabelled as a self-ablation (D3). Also the model for how a TNSRE paper is framed if that becomes the target venue. |
| ★★ | **[FILL] Proroković, Wand & Schultz (our [17])**, MAML for sEMG session recalibration | Cited but not distinguished (D8). Read it to get the distinction right: within-subject session-to-session on a fixed vocabulary, vs our cross-subject with user-defined semantics. |
| ★★ | **Atzori, Gijsberts, Castellini, et al., "Electromyography data for non-invasive naturally-controlled robotic hand prostheses", Scientific Data 2014 (Ninapro)** | The benchmark we are declining, so we must characterise it accurately: fixed standardised vocabulary, able-bodied participants. The decline is defensible on hypothesis grounds and indefensible on effort grounds. |
| ★★ | **Campbell, Phinyomark & Scheme, "Current Trends and Confounding Factors in Myoelectric Control: Limb Position and Contraction Intensity", Sensors 2020** | The confound literature for our limitations section — electrode shift, limb position, contraction intensity, day-to-day variability. Makes D7 specific instead of generically apologetic. |
| ★★ | **Scheme & Englehart, "Electromyogram pattern recognition for control of powered upper-limb prostheses: State of the art and challenges for clinical use", JRRD 2011** | The canonical statement of the clinical-translation gap. Useful for motivating why N=32 with motor impairment matters more than N=4,800 able-bodied. |
| ★ | **Sivakumar, Seely, et al., "emg2qwerty", NeurIPS D&B 2024**; **Salter, Warren, et al., "emg2pose", NeurIPS D&B 2024** | Large-scale sEMG datasets. Relevant both as related work and as evidence that the D&B track takes sEMG seriously — which matters for the venue decision. |
| ★ | **[FILL] Ketykó, Kovács & Varga**, domain adaptation for sEMG gesture recognition with RNNs, IJCNN 2019 | Cross-user adaptation framing from the domain side. |

---

## 8. Accessibility and personalisation framing (relevant if we target CHI/ASSETS/IMWUT)

| ★ | Paper | Use |
|---|---|---|
| ★★★ | **Wobbrock, Kane, Gajos, Harada & Froehlich, "Ability-Based Design: Concept, Principles and Examples", ACM TACCESS 2011** | The framing that makes our contribution legible to an HCI audience: systems adapt to abilities rather than requiring users to adapt. Our user-defined gesture vocabulary *is* ability-based design, and saying so reframes the whole paper. |
| ★★ | **Lee & Kacorri, "Hands Holding Clues for Object Recognition in Teachable Machines", CHI 2019** and related teachable-interface work | Few-shot personalisation evaluated with real disabled users, in a top HCI venue, without needing architectural novelty. The closest existing model for how our paper could be framed and received. |
| ★★ | **Anthony, Kim & Findlater, "Analyzing user-generated YouTube videos to understand touchscreen use by people with motor impairments", CHI 2013**; **Findlater & Wobbrock, "Personalized input", CHI 2012** | Precedent for personalisation-for-motor-impairment as a first-class contribution. |
| ★ | **Gadiraju, Kane, et al. / Bragg, Kacorri, et al.** on inclusive dataset practice | Relevant if we release the dataset (the NeurIPS D&B path), and for the ethics/limitations sections those venues expect. |
| ★ | **Fallah, Mokhtari & Ozdaglar, "Personalized Federated Learning: A Meta-Learning Approach", NeurIPS 2020** | If a personalisation-at-scale / on-device framing is wanted; connects MAML to deployment economics. |

---

## 9. Reading order

**Week 1 — before writing anything.** §0's four papers. They will change what we
claim, in four places: the fusion contribution, the 10-way story, the response to
PCA+KNN, and how we report variance.

**Week 2 — before the first experiment.** Barachant 2012 and ROCKET (because they
may change what "the baseline to beat" even *is*), Snell §3.3 and Meta-Dataset
(they determine the prototype-head and train-way design), Um 2017 (determines the
augmentation set), and MoEMeta (highest-priority citation, and R3 is at
confidence 4).

**Week 3 — before the HPO and transformer runs.** ST-MoE and Soft MoE, PatchTST,
Melis 2018, Zoph's z-loss.

**Ongoing.** §7 and §8 as the venue decision firms up — the framing sections
differ far more by venue than the experiments do.
