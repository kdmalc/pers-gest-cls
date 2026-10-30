# SPEC — shared episode cache (Opus P1)

*Written 2026-10-21 by Opus. Status: ready for Sonnet to implement. Everything that reports a few-shot number consumes episodes from this cache; nothing samples its own test episodes again.*

## Why
Audit finding 6.3: the neural models, `$B`/kNN/ProtoNet (`eval_knn_proto.py`, support = first k reps), A14 and the `$B` notebook each used a different sampler and different users. That makes every baseline comparison invalid. Fix: episodes are generated once, from a seed, with no access to data, written to disk, hashed, and read by every method. Every results file records the hash, so a mismatch is caught mechanically.

## Decisions
1. **What an episode is.** One user, N distinct gestures in a random order (the order is the label permutation: local label i = `classes[i]`), K support reps per class drawn at random from that class's 10 reps, query = all remaining 10-K reps of each of the N classes. Only ids are stored, never tensors. Q is not stored; it is derived and asserted.
2. **One sampler.** Random reps only. There is no ordered or temporal variant (Kai's decision, 2026-10-21).
3. **Grid.** Cells are N in {3, 5, 10} and K in {1, 3, 5}, 9 cells. The same grid is used for every split.
4. **Episodes per cell.**

   | Split | Users | Episodes per user per cell | Used for |
   |---|---|---|---|
   | test | the fold's 4 test users | 200 | reported numbers |
   | val | the fold's 4 val users | 50 | checkpoint selection |
   | dev | the 4 dev users | 100 | HPO and every eval-time hyperparameter |

   With 10 reps and correlated episodes, more than 200 adds almost nothing. User-to-user spread dominates, and the unit of analysis is the user (`SPEC_stats.md`).
5. **Training episodes are not cached.** Meta-training keeps sampling on the fly from train users. Caching applies to evaluation and selection only.
6. **Seeding.** Each (user, cell) gets its own `random.Random` stream, seeded by `int(sha256(f"{BASE_SEED}|{split}|{fold}|{user}|n{N}k{K}").hexdigest()[:16], 16)` with `BASE_SEED = 20261021`. Adding a cell or a user never changes existing episodes. Within a stream, for each episode: `classes = rng.sample(range(10), N)`, then for each class in label order `support = sorted(rng.sample(range(10), K))`. Only `random.Random.sample` is used; its algorithm is stable across CPython 3.10–3.12. The files are committed anyway, so the RNG is never trusted after generation.
7. **Files and hashing.**
   - `system/episodes/kfold7/test_fold{k}.json.gz` and `val_fold{k}.json.gz` for k = 0..6, plus `system/episodes/kfold7/dev.json.gz`. That is 15 files, about 1 MB total gzipped. They are committed.
   - `system/episodes/kfold7/MANIFEST.json` maps file name to sha256.
   - The hash is sha256 over the canonical JSON (`json.dumps(obj, sort_keys=True, separators=(",", ":"))`) of the uncompressed object. gzip bytes are not hashed, because they are not deterministic across zlib versions.
   - The loader recomputes the hash and raises on any mismatch with the manifest.
8. **No data dependency at generation.** The generator assumes 10 classes x 10 reps for every user. Every consumer asserts `emg.shape[0] == 10` for each (user, class) it touches and raises otherwise. The archived verification output shows (10, 64, 16) for all 32 users.

## File schema (one JSON object per file)
```json
{
  "schema_version": 1,
  "protocol": "kfold7",
  "split": "test",
  "fold": 3,
  "base_seed": 20261021,
  "n_reps": 10,
  "n_classes": 10,
  "cells": [[3,1],[3,3],[3,5],[5,1],[5,3],[5,5],[10,1],[10,3],[10,5]],
  "users": ["P010","P105","P112","P127"],
  "episodes": [
    {"id": "test-f3-P010-n10k1-0000", "user": "P010", "n_way": 10, "k_shot": 1,
     "classes": [7,2,9,0,4,1,8,3,6,5],
     "support": [[4],[0],[9],[2],[7],[1],[3],[8],[5],[6]]}
  ]
}
```
- `support[i]` holds the rep indices (0-based tensor rows, i.e. rep number minus 1) for `classes[i]`.
- `fold` is `null` and the split is `"dev"` for the dev file.
- Ids are unique and stable. Results are joined on them.

## Module: `system/episodes/episode_cache.py`
- The core is stdlib only (generation, hashing, loading). The torch and numpy adapters live in a second module, so that the generator and its tests run without torch.
- Functions:
  - `generate(split, fold, users, cells, n_per_cell, base_seed) -> dict`
  - `write(obj, path)`
  - `canonical_sha256(obj) -> str`
  - `load(path) -> EpisodeSet`, which verifies the manifest hash
  - `EpisodeSet.select(n_way, k_shot, users=None) -> list[Episode]`
  - `Episode.query(i) -> sorted reps not in support[i]`
- `make_episodes.py` is the CLI that writes all 15 files and the manifest. It refuses to overwrite a file whose hash differs unless `--force` is passed, and it prints the old and new hash.

**Torch adapter, `system/episodes/episode_torch.py`:**
- `CachedEpisodeDataset(tensor_dict, episodes, modality_mask)` returns, per item, exactly the dict `_build_episode` returns today: support and query sample lists with `emg`, `imu`, `demo`, `label`, `global_class`, `rep_idx`, plus `user_id`, `label_map`, and new `episode_id`, `n_way`, `k_shot`.
- `maml_mm_collate` passes `episode_id` through.
- Modality masking reuses `MetaGestureDataset._apply_modality_mask`. Factor it out into a function so both classes share it.

**Numpy adapter, `episode_numpy.py`:**
- `materialize(ep, tensor_dict) -> (Xs_emg, Xs_imu, ys, Xq_emg, Xq_imu, yq, meta)`.
- Arrays are channel-first `(n, C, T)` and labels are local labels.
- This is what `$B`, kNN, ProtoNet, Riemannian, ROCKET and the zero-shot nearest-centroid baseline consume.

## Required asserts (raise, no warnings)
1. **At generation:**
   - users are disjoint from the other splits of the same fold
   - users of `dev` equal the 4 dev users and appear in no fold
   - `len(set(classes)) == n_way`
   - support reps are distinct and in `[0, 10)`
   - ids are unique
2. **At load:**
   - the hash matches the manifest
   - `split` and `fold` in the file match the requested ones
   - the users match `kfold_split.get_fold(fold)[split]`, or the dev users
3. **At materialize:**
   - every (user, class) tensor has 10 reps
   - support and query reps are disjoint per class, already enforced in the eval path by `assert_support_query_rep_disjoint`
   - realised Q equals 10-K
4. **At aggregation** (`system/analysis/aggregate.py`, see `SPEC_stats.md`):
   - for each (fold, cell), every method's results reference the same sha256
   - every method has the identical set of episode ids
   - otherwise raise, naming the method and the missing or extra ids

## Results contract (every method, every fold)
Each method writes, per fold, a results JSON that contains, in addition to what it has today:
```json
"protocol": "kfold7", "fold": 3, "method": "M0", "train_seed": 0, "git_rev": "c41e34d",
"episode_source": {"file": "system/episodes/kfold7/test_fold3.json.gz", "sha256": "..."},
"per_episode": [{"id": "test-f3-P010-n10k1-0000", "user": "P010", "n_way": 10, "k_shot": 1,
                 "n_query": 90, "n_correct": 71}]
```
- `per_user` and the means are recomputed from `per_episode` by the aggregator.
- The scripts' own summaries remain for logging only and are never used for tables.
- `n_correct` is stored instead of `acc`, so pooled and per-user numbers are both exact.
- A small helper, `ablation_config.episode_results_block(episodes, outcomes)`, builds this block so that no script hand-rolls it.

## Who consumes what (migration, in this order)
1. **`ablation_config.run_episodic_test_eval` and `run_supervised_test_eval`:** replace `MetaGestureDataset(is_train=False, ...)` with `CachedEpisodeDataset` over `test_fold{k}`. Add a `cells` argument. Default: the cell the model was trained for, plus every cell when `--eval-grid` is passed. This covers M0, A1–A4, A13, A15, portB and the grids.
2. **Checkpoint selection** (`get_maml_dataloaders` val loader, and the supervised `pretrain` val path): use `val_fold{k}` at the training cell. `shared_maml.meta_evaluate` must return the per-user mean (mean over users of per-user mean accuracy) and select on that. Today it pools over queries (audit 6.1).
3. **Nonparametric and classical methods:**
   - `eval_knn_proto.run_one_shot_condition` gets an `episodes=` path that bypasses its own `build_episode`.
   - A6 (`$B`, faithful port from `run_$B.ipynb`), A14, Riemannian and MiniRocket go through `episode_numpy.materialize`.
   - The old within-user fixed-rep split (support = rep 1) is removed from reported paths.
4. **A7/A8 (subject-specific):**
   - Train from scratch on each episode's support, evaluate on its query.
   - Training length is fixed and chosen on dev users. There is no per-user early stopping, because with K=1 there is no spare rep for a val set.
   - This replaces the old rep 1 / rep 2 / reps 3–10 split. It is costly: one model per episode. So A7/A8 use 20 episodes per user per cell, as an explicit subset of ids (the first 20 ids per user and cell). The aggregator compares A7/A8 to others only on that shared subset.
5. **Sweeps** (inner steps, eval learning rates): `dev.json.gz` only.
6. **Exempt, labelled as such:**
   - A16 (leave-one-rep-out ceiling, not a few-shot protocol).
   - The zero-shot experiment (`SPEC_zeroshot.md`, which uses all reps of the test users and no support).

## Tests Sonnet adds (`tests/test_episode_cache.py`, stdlib plus torch-free where possible)
- **Determinism:** regenerating each committed file gives the manifest hash. This catches RNG drift across Python versions.
- **Independence:** changing `n_per_cell` for one cell leaves the other cells' episodes unchanged.
- **Structural:** all generation asserts above, over every committed file.
- **Label permutation:** at N=10, the fraction of identity label maps is below 1%; across episodes each class appears in every label slot.
- **Aggregator:** two fake method results with the same hash and ids pass; a differing hash, a missing id or an extra id each raise.
- **Torch adapter:** a `CachedEpisodeDataset` item, run through `maml_mm_collate` and `mamlpp_adapt_and_eval`, works on the synthetic fixture re-keyed to real PIDs, the same trick as `tests/test_eval_protocol.py`.

## Out of scope
Training-time sampling, the zero-shot experiment, A16, and A10–A12/A11b (2 kHz, redesigned under `SPEC_2khz_emg.md`). Those will use the same episode files, because rep ids are format-independent. Their consumer must assert that its data dict has the same 10x10 rep layout for every user.
