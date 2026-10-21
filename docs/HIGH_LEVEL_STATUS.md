# HIGH_LEVEL_STATUS — EncoderMoE resubmission

*Last updated 2026-10-21 (evening). Details, history and decisions: `FULL_RUNNING_LOG.md`. Old docs: `archive/` (read with a grain of salt; dated).*

## Where we are (one paragraph)
EncoderMoE was rejected at NeurIPS'26 (low originality/significance). Goal: resubmit (ICML Jan if framing/baselines land; else IMWUT/ASSETS, CHI, NeurIPS, IEEE Sensors). Known problems: numbers in old docs mix evaluation protocols; `$B` (PCA + L1 nearest-neighbour) may beat EncoderMoE; transformer baseline was a 28x under-sized straw man; EMG is natively 2 kHz but everything uses a 20 Hz envelope, and extending to 2 kHz OOMs because every expert runs on every sample. PI meeting 2026-10-22.

## Current results
**None on the new protocol yet** — everything must be rerun on the cluster (Kai launches). Old numbers (mixed protocols, do not cite together): M0 3-way 1-shot 86.7 (32-fold L2SO) vs 88.4 (old fixed 24/4/4 split); 10-way 1-shot 67.7; `$B` 74.1 (10-way 1-shot, 32 users, notebook) — not comparable.

## Evaluation protocol (decided 2026-10-21)
28/4 outer split: 4 HPO-val users (P005, P131, P116, P109) never tested; the other 28 in 7 folds of 20 train / 4 val / 4 test, group-stratified. File: `system/fixed_user_splits/kfold28_hpo4_split.json`, loader `kfold_split.py`.

## Status board
| # | Item | Owner | Status |
|---|------|-------|--------|
| 1 | New 28/4 + 7-fold split JSON, loader, tests | Sonnet | **done** (tests pass locally) |
| 2 | Docs reorganised (archive + this file + log) | Sonnet | **done** |
| 3 | S1: wire new protocol into eval scripts, fix sweep leaks, new `kfold_launcher.sh` | Sonnet | **mostly done, unrun on cluster** (A10-12/A11b/A16/A5/portA-launcher deferred; see log) |
| 3a | Eval-pipeline audit for co-advisor (`EVAL_PIPELINE_AUDIT.md`) | Sonnet | **done, read-only**; fixes await approval |
| 3c | Co-advisor decisions folded in (permute labels, shared episode cache, 3 seeds, per-user stats) | Kai/Opus/Sonnet | audit fixes implemented 2026-10-21 (permute labels, rep ids, asserts); ordered split dropped by Kai; shared episode cache waits on Opus P1 spec |
| 3d | `q_query` cleanup: training-only cap, `q_query_eval_mode` flag removed, comments fixed, tests (110 pass locally in `fl_torch` env) | Sonnet | **done** |
| 3b | Redo all HPO on new split (+ transformers, + ablations tiers) | Kai runs; Sonnet prepares | HPO split wired; HPO-result plumbing + transformer HPO space pending |
| 4 | S4: running-sum expert mixing + peak-memory script | Sonnet | **done** (only ~14% saving; checkpointing unsafe under MAML; see log) |
| 4a | Shared episode cache + cache-backed eval + stats/aggregator | Sonnet | **done** (15 committed episode files, manifest hashes, 193 tests) |
| 5 | S2: `$B`, kNN, ProtoNet-raw, Riemannian, MiniRocket on the cache (`A6_*`) | Sonnet | **done**, unrun on cluster; A14/A7/A8/portA/A16 still to port; ANIL/ProtoNet-trained/ADANN/FS-HGR/EMG-CrossFormer not started |
| 6 | O3/P6: SOTA lit review (`SOTA_COMPARISONS.md`) | **Opus** | **done** |
| 7 | O1/P4: 2 kHz EMG design (`SPEC_2khz_emg.md`) | **Opus** | **done**; Sonnet implements; E1 diagnostics need Kai |
| 8 | O2/P5: MoE compute + transformer variants (`SPEC_moe_compute.md`) | **Opus** | **done**; Sonnet implements |
| 9 | P9: go/no-go (`DECISION_method.md`) | **Opus** | criteria pre-registered; verdict after Tier 1 numbers |
| 9a | P1 episode cache, P2 protocol, P3 zero-shot, P7 HPO, P8 stats specs | **Opus** | **done**; Sonnet implements (start with P1 episode cache) |
| 10 | S3: transformer baselines (param-matched) | Sonnet (after O2) | queued |
| 11 | Cluster rerun on new protocol | **Kai** | waiting on #3 |

## Next for Kai
See `KAI_TODO.md` (immediate blockers only). Opus work: `OPUS_GUIDE.md`.

## Open decisions for Kai
- HPO tiers for ablations: HPO every headline comparator (M0, A1, A2, A4, transformers) with equal budgets, frozen-M0-hyperparameters for the rest, both for A1-A4? (decided direction 2026-10-21; budget TBD). HPO trains on 20 users and validates on the 4 dev users only.
- Whether PI sign-off is needed to report `$B` if it wins.
