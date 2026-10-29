# SPEC — using the 2 kHz EMG (Opus P4)

*Written 2026-10-21 by Opus. Status: ready for Sonnet. E1 (diagnostics) needs one cluster run by Kai before E2 is finalised.*

## 0. Key insight: the OOM is a representation problem, not an expert problem
EncoderMoE ran out of memory at 2 kHz because every expert (22 of them) was applied to a very long sequence. That was 4,300 steps, or 216 after the shared `front_end_stride=20` conv.

If the EMG is compressed **before** the expert bank, to the same ~64-step axis the IMU already uses, the experts never see long sequences. Memory then stays where it is today. So **the 2 kHz question does not require restructuring the experts.** `SPEC_moe_compute.md` treats expert efficiency separately, as an optional research question.

> **Implementation notes (Sonnet, 2026-10-21).** Built as `EMG_preprocessing/emg_fs64.py` + `build_fs64_tensor_dict.py` (local, from the raw CSVs on Box). Deviations from the text below: (1) NO 60 Hz notch by default, because the diagnostics found no mains line in the data; (2) feature standardisation uses the 4 DEV users (no fold information) instead of each fold's train users; (3) measured segment lengths (median 3.2 s, tightly cut, no rest margins) make the "trim to the active gesture" ablation unnecessary; (4) `emg_repr` config key (`env20` | `fs64`) and launcher `--emg-repr` select the representation.

## 1. Facts this spec builds on
- **EMG:** 16 channels, recorded at **2048 Hz** (Kai). The code assumes 2000 Hz (`FS=2000`), which mis-scales the filter cut-offs and spectral features by 2.4%. All new code uses `FS_EMG = 2048`, defined once in `system/universal_preprocessing/EMG_preprocessing/shared_processing.py`.
- **IMU:** 72 channels (12 sensors x 6 axes), 148 Hz.
- **Current fused input:** each modality's whole trial is linearly interpolated to 64 steps, i.e. *normalised time*, with no absolute duration. EMG is a 20 Hz MAV envelope (100 ms window, 50 ms step), so content above about 10 Hz of envelope modulation and all spectral shape is gone.
- **Current 2 kHz dict:** whole-trial FFT resample to 4,300 samples. This time-warps: a 3 s trial compressed to 2.1 s has all its frequencies scaled by 1.4x. **Do not use it** for anything spectral.
- **Segment lengths:** native trials span 0.8–12.2 s, while gestures last 0.5–3 s. Segments therefore include rest margins of variable length.

## 2. Decision: primary representation = windowed spectral feature sequence ("EMG-FS64")
For each trial and each EMG channel, compute features over **64 windows whose centres sit at the same normalised times as the 64 IMU steps**. The result is `emg_fs` with shape `(n_reps, 64, 16 x F)`, aligned one-to-one with the existing `imu` `(n_reps, 64, 72)`.

**Window length:**
- `W = max(150 ms, 2 x trial_len / 64)`, with reflect-padding at the trial edges.
- 150 ms is the usual lower bound for stable EMG spectral estimates.
- Windows overlap. That is fine, because they are features of one sample, not separate samples.

**Features per channel (F = 7):**
- The Khushaba TD-PSD spectral moments, as already implemented safely in `shared_processing.py:176-265`: `log m0`, `log m2`, `log m4`, sparsity, irregularity factor.
- Plus `log WL` (waveform length) and `log MAV`.
- Total: 16 x 7 = **112 EMG feature channels**.

**Preprocessing, in order, per trial, offline:**
1. Remove mean.
2. 4th-order Butterworth band-pass, 20–450 Hz, **zero-phase** (`sosfiltfilt`). Segments are classified offline, and zero-phase avoids the onset delay of causal filtering.
3. Notch at 60 Hz and its harmonics up to 420 Hz (Q=30).
4. `$B` block normalisation (one scalar std over all 16 channels of the trial). This keeps cross-channel amplitude ratios, which carry the class signal.
5. Features.
6. Per-feature-type z-score using statistics from **training users of the fold only**. These are stored with the model checkpoint, and the training-only rule is checked by an assert.

**Why this one first:**
- It keeps the CNN/LSTM meaningful: it is still a sequence of 64 vectors, now with spectral content.
- It aligns exactly with the IMU and plugs into the existing 88-channel concat by changing `emg_in_ch` from 16 to 112.
- It has no memory cost.
- It answers Kai's worry about "a CNN on 1D features": the features form a time series, not one vector per trial.

## 3. Secondary representation: learned stem on native-rate EMG ("EMG-RAW")
- **Input.** Native 2048 Hz EMG, cropped or padded to a fixed **3.0 s (6,144 samples)** window centred on the trial's activity centroid (energy-weighted centre of the 20–450 Hz envelope). Pad with zeros. No time-warping.
- **Matching IMU.** The IMU is cropped to the same absolute window at its native 148 Hz (444 samples), then linearly resampled to 64 steps. For this representation **both modalities use absolute time.** That changes the IMU input relative to EMG-FS64, so the comparison is EMG-RAW (abs time) vs EMG-FS64 (normalised time) plus the 20 Hz baseline. Report this confound.
- **Stem.**
  - A modality-specific EMG stem placed *before* the EMG/IMU concat: `Conv1d(16→32, k=11, s=4) → GN → GELU → Conv1d(32→64, k=9, s=4) → GN → GELU → Conv1d(64→64, k=7, s=4) → GN → GELU → AdaptiveAvgPool1d(64)`.
  - It outputs `(B, 64, 64)` and concatenates with IMU `(B, 72, 64)`.
  - It replaces the shared `front_end_stride` path. That path is applied after the concat and cannot handle mismatched T.
  - Add it as `config["emg_stem"] = "none" | "raw_strided"` in `DeepCNNLSTM_EncoderMOE` and the non-MoE `DeepCNNLSTM`, so A4/A2 get the same stem.
- **Memory.** The stem runs once per sample on 6,144 steps with at most 64 channels, which is negligible next to the expert bank. The experts see T=64 as today.

## 4. Classical baselines get their natural representation of the same raw data
The aim is to compare methods, not inputs, so each classical method gets the representation it is known to work with:
- **Riemannian** (MDM, tangent space): channel covariance of the **band-passed native-rate EMG** over the whole trial (16x16, OAS shrinkage) plus the IMU covariance (72x72, OAS). This is the standard Riemannian EMG input (Gowda & Miller 2024).
- **`$B`:** its published form, the current 20 Hz envelope plus IMU. Also run it on EMG-FS64 as a sensitivity check.
- **MiniRocket:** EMG-FS64 + IMU (88→184 channels x 64).
- **ProtoNet, kNN and the neural methods:** both 20 Hz and EMG-FS64. EMG-RAW only for the main method and the best neural baseline.

The paper states: "each method is evaluated on the input representation it was designed for, and all neural methods also on the shared representation."

## 5. Experiment order
- **E1 — diagnostics (cluster, CPU, Kai).** Script `system/universal_preprocessing/EMG_preprocessing/diagnose_2khz.py` reports:
  - the per-trial native length histogram, per user and gesture
  - an activity-fraction estimate (share of samples above rest-level envelope)
  - the median PSD per channel, with a check that power peaks at 50–150 Hz, as it should
  - the 60 Hz line amplitude
  - a check that the recorded sampling rate stored in the raw CSVs (if any) is 2048

  Outcome: confirm FS and notch, and decide whether trimming rest margins is needed. Trimming is **off** by default, for continuity with all previous results. If median activity fraction is below 40%, add trimming as an E2 ablation.
- **E2 — EMG-FS64.** Build `segfilt_fs64_tensor_dict.pkl`: same keys and rep layout as the current fused dict, `emg` replaced by `(10, 64, 112)`, `imu` unchanged.
  - Dev users first: run M0 and A4 with the 20 Hz config and HPO-tuned settings. A small re-tune is allowed for `outer_lr`, `dropout` and the first conv width only.
  - Then all 7 folds for M0, A4, ProtoNet, `$B` and MiniRocket at the primary cell.
  - **Decision rule:** adopt EMG-FS64 as the paper's default input if the per-user mean at 10-way 1-shot improves by at least the MDE (`SPEC_stats.md`) on the 28 users. Otherwise keep 20 Hz and report FS64 as an ablation.
- **E3 — EMG-RAW.** Only for the main method and A4. Run it in both cases: if E2 helped, to show whether learned features beat hand-crafted ones; if E2 did not help, to rule out the feature choice before concluding that higher-rate EMG does not help.
- **E4 — modality ablation (A13) repeated on the adopted representation.** This is the real test of whether EMG now contributes. Archived LIMITATIONS A1: IMU dominated and EMG collapsed at 10-way under the 20 Hz envelope.
- **Dropped:** the envelope-rate sweep {50, 100, 200} Hz. EMG-FS64 strictly contains the envelope (log MAV) plus spectral shape, so the sweep would answer a weaker question at the same cost.

## 6. A10–A12 / A11b (Meta / Kaifosh comparison)
- **Keep as a Tier 3 related-work comparison, not a headline baseline.** The electrode topology differs (archived LIMITATIONS D5), so a weak result is uninformative.
- **When rewired:**
  - Resample our band-passed 2048 Hz EMG to 2000 Hz with a polyphase resampler (`scipy.signal.resample_poly(x, 125, 128)`). This is a true rate conversion, not a time-warp.
  - Feed the 3.0 s crop from section 3.
  - Use the kfold7 test episodes and the results contract.
- **A12** (our model on EMG-only 2 kHz) is superseded by E4's EMG-only condition on the adopted representation. Retire the A12 script.
- **A11b** (gain sweep) is only needed if A10/A11 are reported at all.

## 7. Implementation notes for Sonnet
- **New builder:** `system/universal_preprocessing/EMG_preprocessing/build_fs64_tensor_dict.py`, a script rather than a notebook.
  - It reads the same segmented raw source the 2 kHz notebook reads (`load_segraw_data(...)`) and joins with the IMU dict on `(pid, gesture, rep)` via `rep_indices`.
  - It asserts that every (pid, class) has 10 reps in both modalities and that the gesture label encodings match. The notebook's sorted `GESTURE_TO_CLASS` was never diffed against the fused dict's LabelEncoder; assert equality explicitly.
  - It raises on any mismatch.
- **Feature functions:** reuse the safe `shared_processing.py` functions. Unit-test them against hand-computed values on synthetic sinusoids: for a pure sine of frequency f, m2/m0 = (2πf/fs)^2.
- **Config keys:** `emg_repr = "env20" | "fs64" | "raw3s"`, and `emg_in_ch` follows from it (16 / 112 / 16). `ablation_config` resolves the dict path from `emg_repr`. No silent default: `emg_repr` is required in `make_base_config` and set to `"env20"` explicitly.
- The episode cache is unchanged: rep ids are format-independent.

## 8. Sampling-rate policy (Kai + Sonnet, 2026-10-23; pending PI sign-off)
The PI's concern: real devices do not stream 2 kHz, but the Meta comparison needs it. Decision: **no sampling-rate sweep and no tunable knob** (that would be tuning a preprocessing choice on data). Two tables, rates fixed from the literature before any result:
- **Meta comparison table:** native rate (2048 Hz resampled to 2000 Hz by `resample_poly(125, 128)`), raw-stem input (§3), A10/A11.
- **Main table:** one realistic rate declared in advance. Proposed 1 kHz: a common research/embedded rate that keeps the whole 20-450 Hz band the features use. Implementation is a fixed anti-alias decimation before filtering in the builder, with the band edge kept below Nyquist. An optional 200 Hz (Myo-class) run may be added but is not part of the main claims. Rates to cite (verify when writing): lab/research sEMG 1-2 kHz; Meta band 2 kHz; Myo armband 200 Hz.
- **One dev-user sanity check:** FS64 at 1 kHz vs 2048 Hz. The features only use 20-450 Hz, so near-identical results are expected; if so, report a single table with a footnote.
- Neural methods run on the shared representation; classical methods on their natural inputs (§4), including the Hudgins time-domain features (`emg_td.py`, `emg_repr="td64"`).
