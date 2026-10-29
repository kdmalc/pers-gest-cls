# LOG (context and design decisions that are not immediately critical)
Full history: `docs/FULL_RUNNING_LOG.md`. Status table: `docs/HIGH_LEVEL_STATUS.md`. Immediate blockers (long form): `docs/KAI_TODO.md`.

## 2026-10-25
- Created `CURRENT_TODOs.md` (short handoff) and this file. Existing `docs/KAI_TODO.md` remains the detailed version; keep the two consistent.
- Why the TD baselines: peer feedback says journal reviewers will ask for LDA / time-domain / PCA+KNN baselines. `$B` is PCA + L1 1-NN, so label it that way in the paper. `td_lda_pool` is the only variant using the fold's train users (pooled within-class covariance, Ledoit-Wolf).
- Building the td64 dict (laptop, raw CSVs from Box): `python system/universal_preprocessing/EMG_preprocessing/build_td64_tensor_dict.py --root <Box segmented_raw_data dir> --fused-dict dataset/meta-learning-sup-que-ds/segfilt_rts_tensor_dict.pkl --out dataset/meta-learning-sup-que-ds/segfilt_td64_tensor_dict.pkl`, then scp like the fs64 file. `.pkl` files stay git-ignored.
- Project rules that still apply: no silent fallbacks (raise); never commit reviewer PDFs or `.pkl`; only Kai launches cluster jobs; submit from `/scratch/my13/kai/runs/paper/sub2/`.
