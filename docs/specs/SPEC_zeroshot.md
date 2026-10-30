# SPEC — zero-shot cross-user experiment (Opus P3)

*Written 2026-10-21 by Opus. Status: ready for Sonnet. **The framing consequences in section 4 are fixed here, before any number exists, and must not be edited after results arrive.** If the outcome falls in a range we did not anticipate, add a dated note below the table; do not move the thresholds.*

## 1. Question
How much of a gesture's identity transfers across users when the label name is the only shared thing? The paper's framing is that each user defines their own movements, so there are effectively about 320 user-gesture pairs. If that is right, a model trained on other users' labels and given **no** support from a new user should be near chance (10%). If it is well above chance, the gestures are less idiosyncratic than claimed and the framing must be softened.

## 2. Design (per kfold7 fold k, for all 7 folds, so all 28 users are tested exactly once)
- **Data.** All 10 gestures x 10 reps of the fold's 4 test users, i.e. 100 samples per user. No support set and no adaptation of any kind.
- **Training.** The fold's 20 train users, with global labels (gesture id 0..9). Early stopping uses zero-shot accuracy on the fold's 4 val users.
- **Hyperparameters.** From dev users only. Train on the 20 fold-0 train users, select on the dev users, then reuse those settings on all folds.

### Models (report all)
| ID | Model | What it measures |
|---|---|---|
| Z1 | Supervised CNN-LSTM, the A2 architecture and parameter count, 10-way softmax head | Best-effort learned cross-user classifier (head included) |
| Z2 | Nearest class centroid in Z1's penultimate embedding. Centroids are the mean embedding of each gesture over the 20 train users; assign by cosine distance. | Representation-level transfer, not head-confounded |
| Z3 | Nearest class centroid in the M0 encoder embedding (`backbone` features of the fold-k M0 checkpoint), same centroid rule | Whether the meta-learned representation encodes cross-user gesture identity |
| Z4 | Riemannian tangent space of per-modality channel covariance (EMG and IMU blocks separately, concatenated) + logistic regression, trained on the 20 train users | Classical cross-user transfer; strong on EMG covariance in the literature |

The pre-adaptation accuracy of MAML models is **not** used. Labels are permuted per episode, so it is chance by construction.

### Outputs (via the standard results contract, with `per_episode` replaced by `per_sample`)
- Per user: accuracy over 100 samples.
- Per gesture id: accuracy pooled over the 28 users, plus the per-user spread.
- A 10x10 confusion matrix summed over users.
- Statistics follow `SPEC_stats.md`: 28-user mean, user-bootstrap CI, and a per-user binomial test against 0.10 reported as the count of users significantly above chance after Holm.

### Reference rows in the same figure
- Chance: 10%.
- The main method and `$B` at 10-way 1-shot on the same users. These show how much one support example per gesture buys over zero.

## 3. Primary statistic
**Z\* = the highest 28-user mean accuracy among Z1–Z4.** The best model is used so the conclusion cannot rest on a weak zero-shot model.

## 4. Framing consequences — fixed now (2026-10-21)
| Z\* (28-user mean) | Interpretation | What the paper does |
|---|---|---|
| **<= 20%** | Gesture identity barely transfers across users | Keep the "user-defined, effectively ~320 distinct gestures" framing. The zero-shot figure becomes supporting evidence in the intro and method motivation. The per-gesture table is shown, and any gesture at or above 40% is named as an exception. |
| **20–40%** | Partial shared structure | Soften: "largely user-specific, with partial cross-user regularities". Drop "effectively 320 distinct gestures". The per-gesture table identifies which gestures share structure. The zero-shot model enters the main results table as a calibration-free baseline. |
| **> 40%** | Substantial shared structure | Drop the idiosyncratic-gesture framing entirely. Reframe as "personalising a gesture interface from a few examples beats a strong cross-user model". The zero-shot model becomes a **primary baseline** in `SPEC_stats.md`'s family, and the main method must beat it with Holm-corrected significance at 10-way 1-shot, or the few-shot contribution claim is withdrawn. |

These rules hold in every row:
- The zero-shot result is reported in the paper whatever its value.
- If the CI of Z\* straddles a boundary, the **more conservative** row applies, meaning the higher-transfer interpretation.
- If one gesture is at or above 60% while Z\* is in the first row, that gesture is reported, and the claim is worded "for 9 of 10 gestures".

## 5. Implementation notes for Sonnet
- **New script.** `system/NOTS/paper/ablations/test_eval_files/Z_zeroshot_crossuser.py`:
  - `--fold-idx` required.
  - `--models` chooses from Z1 Z2 Z3 Z4.
  - Z3 needs the fold's M0 checkpoint, located with the same glob as the A14 token in `kfold_launcher.sh`.
- **Z1 training.** Reuse `pretrain` / `get_pretrain_dataloaders` with the fold's train and val PIDs. The val metric must be zero-shot accuracy on whole val users. Reuse `replace_head_for_eval` nowhere: the 10-way head **is** the zero-shot classifier.
- **Z4.** Needs `pyriemann` (already in `fl_torch_NOTS.yaml`). Covariance uses the shrinkage estimator `pyriemann.estimation.Covariances("oas")`. Tangent space uses the reference mean over training users.
- **Launcher.** Add tokens `Z_all` (Z1+Z2+Z4) and `Z3` (needs M0 fold checkpoints) to `kfold_launcher.sh`.
- **Tests.** A synthetic-data test where gestures are made identical across users must give accuracy far above chance. A test with random per-user gesture templates must be near chance. These guard against a pipeline that cannot detect transfer.
- **Order of work** (co-advisor's step 4). Run this **before** the heavy 3-seed runs. It needs only the zero-shot models plus the M0 fold checkpoints, and its outcome can change which experiments matter.
