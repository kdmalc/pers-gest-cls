# KAI_TODO — only things blocked on you right now

*Updated 2026-10-21 (night). Done items get deleted. Everything else lives in `FULL_RUNNING_LOG.md`.*

## 1. Cluster: pull and smoke-test the NEW code (≈20 min, debug partition)
The first smoke run exercised the old code and never reached test evaluation for M0/A4. This one runs the whole chain (train → cached val → cached test → save) in about 5 minutes per job (`--smoke` is added automatically by `--debug`).
```bash
cd /projects/my13/kai/meta-pers-gest/pers-gest-cls && git pull
cd system/NOTS/paper/ablations/launchers
bash kfold_launcher.sh M0 A1 A2 A4 --folds 0 --debug --hpo-config legacy --dry-run   # check the printed job
bash kfold_launcher.sh M0 A1 A2 A4 --folds 0 --debug --hpo-config legacy             # submit (they queue one at a time)
squeue -u $USER
```
(`--hpo-config legacy` = the old Trial-89 values, smoke only. Logs and results now go to `/scratch/my13/kai/runs/paper/sub2/eval/`.) When all four have finished, **paste**:
```bash
LOG=/scratch/my13/kai/runs/paper/sub2/eval/logs
for f in $LOG/kfold7_*_f0_*.out; do echo "=== $f"; grep -E "GIT_REV|smoke|JOB_|PYTHON_EXIT|Traceback|Error|val episodes from cache|Epoch 0 val|Val loss/acc|\[Test\]|Test head|Test full|Saved to|FINAL" "$f" | tail -25; done
```
If M0/A4 still hit the 15-minute cap, tell me how far they got; do NOT raise the limit yet.

## 2. Cluster: the new packages (dry run first, paste the output)
```bash
module purge && module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba install --dry-run -p /projects/my13/kai/meta-pers-gest/envs/fl-torch -c conda-forge pyriemann=0.9 aeon pytest 2>&1 | tail -40
```
If it does **not** list `pytorch`/`torch`/`cuda` among the packages it would change, run it again without `--dry-run`. Otherwise paste it to me. (Needed only for the `A6_*` classical baselines, not for the smoke test or HPO.)

## 3. Copy the new EMG file to the cluster (after I tell you it is built; ≈160 MB)
I am building `dataset/meta-learning-sup-que-ds/segfilt_fs64_tensor_dict.pkl` on this laptop from the raw CSVs (the raw data IS on this laptop, in Box). When I say it is done, copy it next to the existing tensor dict on the cluster, e.g.
```powershell
scp "C:\Users\kdmen\Repos\pers-gest-cls\dataset\meta-learning-sup-que-ds\segfilt_fs64_tensor_dict.pkl" km82@<NOTS login host>:/projects/my13/kai/meta-pers-gest/pers-gest-cls/dataset/meta-learning-sup-que-ds/
```

## 4. One data question (answer in chat)
In the raw folder `…/segmented_raw_data/P005/` all 560 files are named `…_P105_…` (none says P005), with different timestamps and bytes from the real `P105/` folder (P005's files are from 2306/2307, P105's from 2309). The original preprocessing used the *folder*, so P005's tensors come from those files. Is the P005 folder really P005's data with a mistyped ID in the file names? (I assume yes and made the new code accept exactly this one mismatch; anything else raises.)

## 5. Decisions (one word each)
- **"hpo ok"** — HPO every headline comparator (M0, A1, A2, A4, later the transformers) with equal 60-trial budgets, and run A1–A4 both with frozen M0 hyperparameters and re-tuned. Rough cost at 10-way: M0 ≈ 60 trials × ~6 h, A4 ≈ 60 × ~3 h, A1/A2 ≈ 60 × <1 h each (≈ 600 GPU-hours for all four). Say if you want fewer trials for M0.
- **"start hpo"** — only after item 1 passes: I then give you the exact `hpo_ablation_launcher.sh` commands (M0 first, then A4, A2, A1; journals in `sub2/hpo/optuna_dbs`).
