# KAI_TODO — only things blocked on you right now

*Updated 2026-10-21 (night). Done items get deleted. Everything else lives in `FULL_RUNNING_LOG.md`.*

## 1. Cluster: refresh the scratch copies, pull, smoke-test the NEW code (≈20 min, debug partition)
The launcher now reads the git revision from the repo under projects (`git -C $CODE_DIR`) and **refuses to run if your scratch copy differs from the repo copy**, so refresh both copies after every `git pull`:
```bash
cd /projects/my13/kai/meta-pers-gest/pers-gest-cls && git pull
L=/projects/my13/kai/meta-pers-gest/pers-gest-cls/system/NOTS/paper/ablations/launchers
cp $L/kfold_launcher.sh $L/hpo_ablation_launcher.sh /scratch/my13/kai/runs/paper/sub2/
cd /scratch/my13/kai/runs/paper/sub2
bash kfold_launcher.sh M0 A1 A2 A4 --folds 0 --debug --hpo-config legacy --dry-run   # check the printed job
bash kfold_launcher.sh M0 A1 A2 A4 --folds 0 --debug --hpo-config legacy             # submit (they queue one at a time)
squeue -u $USER
```
(`--hpo-config legacy` = the old Trial-89 values, smoke only. Results and logs go to `/scratch/my13/kai/runs/paper/sub2/eval/`.) When all four have finished, **paste**:
```bash
LOG=/scratch/my13/kai/runs/paper/sub2/eval/logs
for f in $LOG/kfold7_*_f0_*.out; do echo "=== $f"; grep -E "GIT_REV|smoke|JOB_|PYTHON_EXIT|Traceback|Error|val episodes from cache|Epoch 0 val|Val loss/acc|\[Test\]|Test head|Test full|Saved to|FINAL" "$f" | tail -25; done
```
If M0/A4 still hit the 15-minute cap, tell me how far they got; do NOT raise the limit yet.

## 2. Cluster: install the new packages (your dry run was clean)
```bash
module purge && module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba install -p /projects/my13/kai/meta-pers-gest/envs/fl-torch -c conda-forge pyriemann=0.9 aeon pytest 2>&1 | tail -20
```
Your dry run was clean (only `pooch`, `lazy-loader`, `mne-base` added, `pyriemann` 0.12 → 0.9; no torch/cuda): **run the same command without `--dry-run`**. (Needed only for the `A6_*` classical baselines, not for the smoke test or HPO.)

## 3. Copy the new EMG file to the cluster (it is built; 151 MB; do this before any `--emg-repr fs64` run, not before the smoke test)
`dataset/meta-learning-sup-que-ds/segfilt_fs64_tensor_dict.pkl` was built on this laptop from the raw CSVs (the raw data is in Box). Copy it next to the existing tensor dict on the cluster (the repo's `dataset/` folder under projects, where the runs already read `segfilt_rts_tensor_dict.pkl`), e.g.
```powershell
scp "C:\Users\kdmen\Repos\pers-gest-cls\dataset\meta-learning-sup-que-ds\segfilt_fs64_tensor_dict.pkl" km82@<NOTS login host>:/projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/meta-learning-sup-que-ds/
```

## 4. One quick check: is the cluster's tensor dict the same file as mine? (30 s, on the projects partition)
P005 is fine in my local `segfilt_rts_tensor_dict.pkl` (thanks - I also verified it by matching the dict to the raw folders). The corrected parquet you mention is a *derived* dataset, so I need to know the cluster uses the same tensor dict:
```bash
sha256sum /projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/meta-learning-sup-que-ds/segfilt_rts_tensor_dict.pkl
```
**Paste the hash.** Mine is `eb60c8af8294550f0cf407d472daa1df6679340de564ee299436e93df1fac1c4` (72,316,340 bytes). If they differ, tell me where the cluster's copy came from.

## 5. Decisions (one word each)
- **"hpo ok"** — HPO every headline comparator (M0, A1, A2, A4, later the transformers) with equal 60-trial budgets, and run A1–A4 both with frozen M0 hyperparameters and re-tuned. Rough cost at 10-way: M0 ≈ 60 trials × ~6 h, A4 ≈ 60 × ~3 h, A1/A2 ≈ 60 × <1 h each (≈ 600 GPU-hours for all four). Say if you want fewer trials for M0.
- **"start hpo"** — only after item 1 passes: I then give you the exact `hpo_ablation_launcher.sh` commands (M0 first, then A4, A2, A1; journals in `sub2/hpo/optuna_dbs`).
