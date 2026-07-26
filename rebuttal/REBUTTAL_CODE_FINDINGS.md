# Rebuttal code findings — corrections to the three planning docs

Everything below is read out of this repo or verified against it, not recalled.
Where a planning doc and the code disagree, the code wins and the doc is
corrected here. Companions: `rebuttal_plan.md`, `ENCODERMOE_REBUTTAL_PLAN.md`,
`MoEMeta_vs_EncoderMoE_handoff.md`.

Headline: **three of the four items blocking the Monday response are resolved,
and two of them resolve in your favour.** The fourth — the Kaifosh
preprocessing defect — is real but has a different mechanism than the plan
assumed, which changes the experiment.

---

## 1. S1 — RESOLVED. The paper text is right; only the name is wrong.

`system/NOTS/paper/ablations/test_eval_files/A7_A8_subject_specific.py:14`

> `NOTE: Subject-specific models have no cross-subject training phase.`
> `The subject-level loop here is over evaluation subjects only.`
> `Val/test splits are over REPS within each subject, not over subjects.`

A7 is flat supervised pretraining on **one subject's rep 1**, val rep 2, test
reps 3–10 (`seed_idx` fixed to 0, line 19). There is no cross-subject phase.

Consequences:

- The paper's two statements (L227–228, L264–265) are **correct**.
- Under your naming convention the **label** is wrong. Rename to
  `Subject-Specific Supervised (within-subject)`.
- **L279–284 survives unchanged.** The interpretation "in the single-subject
  setting, MoE offers no benefit since there is only one signal morphology"
  requires both rows to be genuinely single-subject. They are. No rewrite.
- R3's question has a direct, quotable answer.

**New disclosure item found while checking this.**
`A7_A8_subject_specific.py:291` warns that `pretrain_trainer` returns the
**final epoch, not the best epoch**:

> `WARNING: saving FINAL epoch weights, not best-epoch weights.`

A7 (72.2%) and A8 (64.6%) are both final-epoch numbers. This most likely
understates both, and it applies to the pair symmetrically, so the comparison
direction is probably safe. One sentence in the appendix; not worth a re-run
inside 14 hours.

---

## 2. S2 — RESOLVED. No leak. You can write about 10-way.

Verified in code and then empirically (20/20 assertions pass; see §7).

### (a) Support and query cannot overlap

`system/MAML/maml_data_pipeline.py`, `_build_episode`:

```python
rng_instance.shuffle(trial_indices)
sup_idx = trial_indices[: self.k_shot]
qry_idx = trial_indices[self.k_shot :]          # eval branch
```

Disjoint slices of one shuffled list. Overlap is structurally impossible, not
merely unlikely. There is also already a leakage assertion at
`system/MAML/mamlpp.py:994–997` (`debug=True`) which compares support and query
tensors directly — switch it on if you want an independent confirmation in a
log you can point at.

**So: the K≥3 numbers do not need revising, and nothing needs to go to the PI.**

### (b) But the realised Q is not the configured Q

The eval branch **ignores `q_query` entirely**. The condition is
`if is_train and self.q_query is not None`, so at eval every non-support trial
becomes a query:

| K | configured Q | realised Q at eval | measured |
|---|---|---|---|
| 1 | 9 | 9 | 9 ✅ |
| 3 | 9 | 7 | 7 |
| 5 | 9 | 5 | 5 |

This is the benign branch the plan hoped for — **disclose, don't revise** — and
it explains the variance pattern directly: at 10-way K=5 an episode carries 50
query samples rather than 90, which is consistent with ±10.3 at 10-way K=3/5
against ±2.9 at 3-way K=1.

Note `fewshot_grid.py:62–63` asserts the opposite in a comment:

> `q_query is held fixed across all grid cells (standard practice).`
> `Changing k_shot does NOT change the number of query samples.`

That comment is false for the eval path. It has been corrected in place, and
the dataset now records realised counts in `episode_shape_log` so the paper can
state Q from measurement rather than from the config.

### (c) The fixed-label-map mechanism is ruled out

Training uses `use_label_shuf_meta_aug=True` (`ablation_config.py:300`); eval
uses `False` (`ablation_config.py:756, 817`, and every eval script). Because
**meta-training randomises the label map**, the model cannot learn a global
gesture→index mapping — so the hypothesised mechanism for the 10-way collapse
is not available to it.

What *is* true: at N=10 all ten classes are always present, so
`classes = sorted(...)` yields **one identical class set and label map for every
eval episode**. Measured: 6/6 episodes share one map at N=10, while N=3 gives 19
distinct maps in 20 episodes. Only the support/query trial assignment varies at
N=10. That is an eval-diversity and variance story, not a leak — and combined
with (b) it is a decent mechanistic account of the non-monotonicity
(67.7 → 64.4 → 68.5) without invoking a bug.

---

## 3. V7 — RESOLVED analytically. You loaded the right checkpoint.

`A10_A11_A12_meta_pretrained.py:157–165` instantiates:

```python
DiscreteGesturesArchitecture(
    input_channels=16, conv_output_channels=512, kernel_width=21,
    stride=10, lstm_hidden_size=512, lstm_num_layers=3, output_channels=9)
```

| component | count |
|---|---|
| `Conv1d(16, 512, k=21)` | 172,544 |
| 3 × `LSTM(512→512)` | 6,303,744 |
| 2 × `LayerNorm(512)` | 2,048 |
| `Linear(512, 9)` | 4,617 |
| **total** | **≈ 6,482,953** |

**≈6.48M — the discrete-gesture decoder, not the 60M handwriting conformer.**
The 60M figure in the paper never came from this code path. Also note
`load_state_dict(..., strict=True)` at line 196, so a mismatched checkpoint
would already have raised.

- C0/C1 are **not** void; the Kaifosh plan survives intact.
- "Scale alone is insufficient" keeps its premise (corpus N=4,800, largest
  model, Fig. 2f).
- Correct 60M → ~6.5M wherever it appears.

Still run `V7_checkpoint_param_count.py` on NOTS to confirm against the actual
file — it reads the checkpoint's own `state_dict`, which is architecture-
independent and therefore the decisive check. It also flags
conformer/attention-style keys if the wrong artifact ever gets swapped in.

---

## 4. The Kaifosh preprocessing defect — real, but not the mechanism in the plan

### Trap 1 as written did not happen

There is **no multiply by `2.46e-6` anywhere in the repo**, and no 40 Hz
Butterworth high-pass either. V2 is confirmed: their pipeline was never applied.
The docstring at `A10_A11_A12_meta_pretrained.py:47` even *instructs* the
normalisation that the file never performs.

So "dead network from a literal multiply" is a hypothetical, not a diagnosis.
Worth knowing that the hypothetical is correct — measured, the literal reading
drives post-squash output to `~6e-7`, i.e. genuinely dead — so it is a fair
thing to mention as a reading you checked and excluded. It is not what your
numbers came from.

### What actually happened: under-scaling by roughly the SNR

From `system/universal_preprocessing/EMG_preprocessing/build_2khz_tensor_dict.ipynb`
and `shared_processing.py`:

1. band-pass **20–450 Hz** + mean subtraction — not their 40 Hz HP, and 450 Hz
   against the 850 Hz low-pass on their analog front end;
2. `normalize_gestures_by_std_any_channels`, which divides each trial by **one
   scalar computed over all 16 channels flattened** → whole-trial **signal**
   s.d. = 1.0 (the notebook verifies "should be ~1.0");
3. resample to 4300 samples.

Their model expects **noise** s.d. = 1, which puts gesture-active samples well
above 1 — precisely why `μ = 32` is a sensible outlier knob. You supply
**signal** s.d. = 1, so active samples land near 1 and `x/(32+|x|) ≈ x/32`
compresses the whole recording into roughly ±0.03–0.2.

Their squash (`net.compression`) is **inside** the released module, so it runs
regardless. The network is therefore evaluated in the near-linear corner of a
nonlinearity it was trained to use.

Measured on a synthetic trial matched to our normalisation (SNR ≈ 12):

| gain condition | post-squash \|x\| p99 | verdict |
|---|---|---|
| `none` (**what produced 62.1% / 56.0%**) | 0.106 | **under-scaled — not a fair measurement** |
| `literal` (×2.46e-6) | 2.9e-07 | dead |
| `noise_floor` | 0.578 | plausible |
| gain 32 | 0.791 | plausible |
| gain 100 | 0.922 | saturated |

The bias still runs **against their model**, as the plan says. But the defect is
a **units mismatch fixed by a gain**, and because the data is already
s.d.-normalised, a single global gain `g` is exactly the hypothesis "noise
s.d. = 1/g". That makes the honest experiment a **gain sweep**, not a two-way
direction test. Implemented in `A11b_kaifosh_matched_preproc.py`.

### Rewrite the plan's §2 / §3 item 2 as

> Their preprocessing was never applied. Our 2 kHz tensor is normalised to
> whole-trial *signal* s.d. 1.0, while their pipeline expects *noise* s.d. 1.0,
> so the input arrives roughly an SNR factor too small and their internal
> μ=32 compression operates in its linear corner. We re-ran across a gain sweep
> with post-compression activation statistics logged, and report the best
> fairly-measured condition.

Keep everything the plan says about what a matched re-run does **not** fix:
electrode topology (their gesture architecture has no rotational-invariance
module), the analog band, and the head/metric conversion. Those are unchanged
and remain the reason to reframe the row as a transfer study.

---

## 5. A separate breakage: six call sites would have crashed on any re-run

Commit `215cd94` ("finally fixed the target_trials_indices renaming") renamed
the `MetaGestureDataset` kwarg `target_trial_indices` → `target_trial_reps` but
**missed six call sites**, each of which raises `TypeError` on construction:

| file | line |
|---|---|
| `ablations/ablation_hpo.py` | 813, 922 |
| `ablations/sweeps/M0_inner_steps_eval_sweep.py` | 200 |
| `ablations/sweeps/maml_eval_hp_sweep.py` | 202 |
| `ablations/sweeps/A11_eval_hpo_extended.py` | 404 |
| `test_eval_files/A10_A11_A12_meta_pretrained.py` | 252 (**A10 zero-shot path**) |

All six fixed, with a lookup tolerant of either config key name so the older
configs that still set `target_trial_indices` keep working. Worth knowing that
**A10's prototypical zero-shot path was among them** — if an A10 number appears
anywhere in the paper, check where it came from, because this path cannot have
produced it in the current tree.

---

## 6. What was added, and how to run it

All runs go to NOTS via the existing launcher. Nothing here runs locally — the
data and the Kaifosh checkpoint are both on the cluster.

```bash
# Day 1, in this order
bash eval_launcher.sh V7                    # ~1 min. Gates A10/A11/A11b.
bash eval_launcher.sh A13                   # 3 jobs: both / emg_only / imu_only

# Day 2
bash eval_launcher.sh A11b                  # gain sweep x {head_only, full}
bash eval_launcher.sh portA                 # no retraining

bash eval_launcher.sh V7 A13 A11b portA --dry-run    # inspect first
```

| file | purpose |
|---|---|
| `V7_checkpoint_param_count.py` | Reads the checkpoint's own `state_dict`; PASS ≈6.5M, FAIL ≈60M. Flags conformer keys. |
| `kaifosh_preprocessing.py` | Their steps 1–2, noise-floor estimator, activation logger, shared regime classifier. |
| `A11b_kaifosh_matched_preproc.py` | The mandatory re-run. Sweeps 8 gains × 2 ft modes, labels each condition fair / dead / under-scaled / saturated. |
| `A13_modality_ablation.py` | EMG-only / IMU-only by **masking**, plus the `both` control. Verifies the mask before spending the job. |
| `portA_support_routing.py` | MoEMeta Port A, paired against ours on the same episodes. |

Pipeline changes (`maml_data_pipeline.py`), all defaulting to existing
behaviour so **no published number moves**:

- `modality_mask` — `both` / `emg_only` / `imu_only`, zero-masked via
  `zeros_like` so the shared `tensor_dict` is never mutated and parameter count
  is unchanged.
- `q_query_eval_mode` — `all_remaining` (current) or `fixed`.
- `strict_n_way` + a `RuntimeWarning` replacing the silent class drop.
- `episode_shape_log` — realised support/query counts per episode.

`MOE_encoder.py`: a `_forced_gate_weights` hook in
`DeepCNNLSTM_EncoderMOE.backbone`, inert unless set.

### Two things to watch when these land

- **A13's `both` control is not optional.** It is the only evidence that the
  masking harness did not perturb the pipeline. If it does not land near the
  fixed-split M0 number, distrust the other two conditions.
- **Compare A13 against 88.4%, not 86.7%.** These are fixed-split cells; they
  cannot enter the paired RM-ANOVA. Label them preliminary single-split.

---

## 7. Verification actually performed

`torch` and `scipy` were installed in a scratch environment and the changed
code paths exercised against synthetic data. What passed:

- support/query disjointness across all cached eval episodes;
- realised `q_per_class` = 9 / 7 / 5 at K = 1 / 3 / 5, and `fixed` mode capping
  correctly;
- all three modality masks: correct modality zeroed, shapes and channel counts
  preserved, **source `tensor_dict` unmutated** in every case;
- class-drop warning fires, and `strict_n_way=True` raises;
- N=10 yields one label map across episodes while N=3 yields 19 of 20 distinct;
- the gain sweep and regime classifier reproduce the table in §4;
- 40 Hz high-pass attenuates a 5 Hz tone to 6e-4 and preserves shape/dtype;
- the Port A override is honoured through `torch.func.functional_call`,
  **survives substituted fast weights** (so it is live inside the MAML++ inner
  loop), clears correctly, and does not block gradients.

**Not verified, and the honest limits of the above:** none of the new run
scripts has been executed end-to-end, because that needs the cluster data and
the Kaifosh checkpoint. Untested in particular are the
`build_config_meta` / `run_supervised_test_eval` interaction in A11b, the
`replace_head_for_eval` path on the subclassed wrapper, and the checkpoint-key
handling in `portA`. Expect to spend the first few minutes of each job fixing
import and key-name issues rather than assuming a clean first launch. The
`UNDERSCALED_P99 = 0.15` and `SATURATED_P99 = 0.90` thresholds are judgement
calls tuned on synthetic data with an assumed SNR — sanity-check them against
the real activation statistics before leaning on the fair/unfair labels.

The noise-floor estimator assumes every trial contains at least one roughly
quiescent 100 ms window. If gestures fill the trial the estimate biases high,
which under-gains and therefore under-states their model — i.e. it fails in the
*same* direction as the original bug, so it is not a conservative choice. Report
the fixed-gain sweep alongside it rather than relying on `noise_floor` alone.
