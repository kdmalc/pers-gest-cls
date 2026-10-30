# SPEC — body-region experts with per-user sensor selection (Track B)

*Written 2026-10-30 (Opus). Rules in §3 were committed to this file **before** the first Gate 0 run. Background: `docs/analysis/ANALYSIS_moemeta.md` §6. Dev users only (P005, P109, P116, P131); the 28 fold users are not touched.*

## 1. Sensor map
Source: `docs/emg-sensor-placements.png` (Kai, 2026-10-30), every user has the same placement. Code: `system/arch_variants/body_regions.py`.

| EMG ch (1-based) | Lead | Side | Muscle | Region |
|---|---|---|---|---|
| 1 / 2 | pod 1 / 2 | R | biceps / triceps | R_upper_arm |
| 3 / 4 | pod 3 / 4 | R | deltoid / trapezius | R_shoulder |
| 5 / 6 | pod 5 / 6 | L | biceps / triceps | L_upper_arm |
| 7 / 8 | pod 7 / 8 | L | deltoid / trapezius | L_shoulder |
| 9 / 10 | 9a / 9b | R | extensor / flexor bundle | R_forearm |
| 11 / 12 | 10a / 10b | R | first dorsal interosseous / abductor pollicis brevis | R_hand |
| 13 / 14 | 11a / 11b | L | extensor / flexor bundle | L_forearm |
| 15 / 16 | 12a / 12b | L | first dorsal interosseous / abductor pollicis brevis | L_hand |

**IMU.** The tensor's 72 channels come from recording slots `[1..8, 9, 11, 13, 15]` × `(ax, ay, az, vx, vy, vz)` (`segfilt_aggregating_unified_df.ipynb`). Pods 9-12 each take two EMG slots, so slots 9/11/13/15 are pods 9/10/11/12. Tensor IMU index 0-7 = pods 1-8, index 8-11 = pods 9-12. Pods 10 and 12 sit on the distal forearm, so their IMUs count as forearm and the hand regions are EMG-only.

**Verification (fs64 + IMU, 3200 trials, 32 users, within-user, label-free; no accuracy computed):** side assignment is confirmed. IMU activity splits into right (index 0-3, 8-9) and left (4-7, 10-11) clusters, and EMG splits the same way, with the EMG-IMU sign flipping across sides. Index 3/7 (the two trapezius pods) correlate 0.69, as expected for shared trunk motion. Position *within* a side is not confirmed by this test (index 8-9 correlate 0.99), so it rests on the recording headers and the image. EMG ch 12 (R abductor pollicis brevis) correlates unusually weakly with everything; check it for a poor contact.

## 2. Gate 0: is there any headroom? (laptop, no training)
**Baseline.** Balanced-fusion 1-NN L1 on fs64 + IMU, notebook 01's definition, 10-way 1-shot, all 100 dev episodes per user. It must reproduce 78.0%; if not, stop and find out why. The distance is additive across channels, so a region subset's distance is an exact sum of per-channel distances and all 255 subsets of the 8 regions can be evaluated.

**Quantities.**
- Single-region accuracy per user.
- **In-sample oracle:** best subset per user, picked on the same episodes it is scored on. Optimistic, never reportable as a method.
- **CV oracle:** subset picked on odd episodes, scored on even, and the reverse. This is the headroom number.
- **Global subset:** one subset for everyone, leave-one-dev-user-out. CV oracle minus this is the value of *per-user* selection.
- **Support-only selector:** an actual method. Rank regions by support-set spread (mean pairwise distance among support samples in that region) relative to the same region's spread in the other dev users, and keep the top m. Report every m in 1-8.
- **Budget curve:** CV-oracle accuracy restricted to subsets of at most m regions.
- **Random-partition control:** 200 random partitions of the 16 EMG and 12 IMU channels into groups with the same shape as the anatomical regions (six groups of 2+2, two of 2+0). CV oracle on each.

## 3. Pre-stated rules (written before the first run)
- **H1, accuracy headroom:** CV oracle ≥ all-channel + 2 points on average, and better on ≥ 3/4 dev users. (Same bar as the architecture-screen promotion rule.)
- **H2, anatomy matters:** anatomical CV oracle is above the 95th percentile of the random-partition control.
- **H3, sensor reduction:** a per-user CV-oracle subset of ≤ 4 of the 8 regions is within 1 point of all-channel on average.
- **H4, a real method exists:** the support-only selector, at some m ≤ 4, is within 2 points of all-channel on average.

| Outcome | Decision |
|---|---|
| H1 + H2 | Build the deep body-region model; the claim is accuracy through anatomy |
| H3 + H2 and H4, not H1 | Build it with a deployment claim only: fewer sensors at the same accuracy |
| H3 without H2 | Any channel subset works, so anatomy is not the reason. Report the redundancy as a finding (wearability) and do not build region experts |
| Neither H1 nor H3 | Stop Track B's region line |

With 4 users these are go/no-go signals, not results. m is reported in full rather than selected.
