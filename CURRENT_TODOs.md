# CURRENT TODOs (short; details in `LOG.md` and `docs/`; top-level plan: `docs/PLAN_resubmission.md`)
*Updated 2026-10-25.*

## 1. Main goal
Resubmit EncoderMoE (rejected NeurIPS'26): rerun everything on one clean protocol (28 users, 7 folds, shared cached episodes, 10-way 1-shot primary), add baselines reviewers expect, fix the EMG representation. Venue order: ICML (Jan) → IMWUT/ASSETS → CHI/NeurIPS → IEEE Sensors.

## 2. Latest technical work (uncommitted, all on `main` working tree)
- Per-ability-group analysis of the M0-vs-A4 gain (`system/analysis/group_analysis.py`, spec `docs/SPEC_group_analysis.md`); unit-tested, not run on real data.
- Reviewer-requested classical baselines on Hudgins time-domain features: `td_knn`, `td_lda`, `td_lda_pool` (`emg_td.py`, `build_td64_tensor_dict.py`, `emg_repr="td64"`, launcher tokens `A6_td_*`). PCA+KNN is already covered by `$B`/kNN.
- `docs/SPEC_deployment.md` (offline vs real-time discussion); EMG rate policy drafted in `SPEC_2khz_emg.md` §8.

## 3. Blocker
Waiting on Kai: cluster smoke-test output, cluster dict hash, and "hpo ok" (no cluster run can start without them). The EMG rate policy also needs PI sign-off.

## 4. Limitations / caveats
- Nothing new has been run on real data; no result is claimable yet.
- The old 20 Hz EMG time axis does not reproduce from the raw CSVs (unresolved anomaly, so old EMG results may be unreliable).
- `td64` dict is not built yet; the new TD baselines can't run until it is.
- `td_lda` at 1-shot is just nearest class mean (no within-class scatter).
- Dev-user FS64 numbers (77.8% vs 48.9% env20) are 4 users, trivial classifier: not a result.
- The working tree is uncommitted (11 modified, 9 new files): commit/push before wiping the laptop.

## 5. Kai TODOs
**A. Cluster smoke test of the new code (~20 min).**
```bash
cd /projects/my13/kai/meta-pers-gest/pers-gest-cls && git pull   # after the commit is pushed
L=/projects/my13/kai/meta-pers-gest/pers-gest-cls/system/NOTS/paper/ablations/launchers
cp $L/kfold_launcher.sh $L/hpo_ablation_launcher.sh /scratch/my13/kai/runs/paper/sub2/
cd /scratch/my13/kai/runs/paper/sub2
bash kfold_launcher.sh M0 A1 A2 A4 --folds 0 --debug --hpo-config legacy --dry-run
bash kfold_launcher.sh M0 A1 A2 A4 --folds 0 --debug --hpo-config legacy
```
When done, paste (or tell me where M0/A4 got to if they hit the 15 min cap):
```bash
LOG=/scratch/my13/kai/runs/paper/sub2/eval/logs
for f in $LOG/kfold7_*_f0_*.out; do echo "=== $f"; grep -E "GIT_REV|smoke|JOB_|PYTHON_EXIT|Traceback|Error|\[Test\]|Saved to|FINAL" "$f" | tail -25; done
```
**B. Cluster: paste the dict hash** (mine: `eb60c8af8294550f0cf407d472daa1df6679340de564ee299436e93df1fac1c4`):
```bash
sha256sum /projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/meta-learning-sup-que-ds/segfilt_rts_tensor_dict.pkl
```
**C. Cluster: install packages** (needed for classical baselines):
```bash
module purge && module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh && source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba install -p /projects/my13/kai/meta-pers-gest/envs/fl-torch -c conda-forge pyriemann=0.9 aeon pytest
```
**D. Laptop → cluster file copy** (before any `--emg-repr fs64` run):
```powershell
scp "C:\Users\kdmen\Repos\pers-gest-cls\dataset\meta-learning-sup-que-ds\segfilt_fs64_tensor_dict.pkl" km82@<NOTS login host>:/projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/meta-learning-sup-que-ds/
```
**E. Replies:** "hpo ok"; "start hpo" (only after A passes); OK to build the td64 dict (needs raw CSVs from Box; I'll give the command); do old L2SO per-user M0/A4 JSONs still exist on the cluster (path)?; PI sign-off on the EMG rate policy.
