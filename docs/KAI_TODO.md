# KAI_TODO — only things blocked on you right now

*Updated 2026-10-21 (night). Done items get deleted. Anything not an immediate blocker lives in `FULL_RUNNING_LOG.md`.*

**Suggested order:** do 1 → 2 → 3 (everything is on `main`) (≈30 min of your time, mostly waiting) *while* Opus works on `OPUS_GUIDE.md` tasks P1, P2, P3 and P6 in parallel. The cluster steps do not depend on Opus and Opus does not depend on them.

## 1. Cluster: pull, check the environment, check the data file (NOTS login node)
```bash
cd /projects/my13/kai/meta-pers-gest/pers-gest-cls
git fetch origin && git checkout main && git pull   # all work is on main now

# (a) New packages for the baselines (pyriemann, aeon). DRY RUN FIRST and paste me the output:
module purge && module load Mamba/23.11.0-0
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh
source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh
mamba install --dry-run -p /projects/my13/kai/meta-pers-gest/envs/fl-torch -c conda-forge pyriemann aeon pytest 2>&1 | tail -40
```
If the dry run does **not** list `pytorch`, `torch` or `cuda` among the packages it would change (numpy/numba/scikit-learn bumps are OK), run the same command without `--dry-run`. If it does touch torch, do not run it, and paste the output to me instead (then we use `pip install --no-deps` into the env).
```bash
# (b) What can the data file tell us about repetition order? (about 2 minutes, debug partition)
sbatch --partition=debug --time=00:10:00 --mem=32G --cpus-per-task=2 --job-name=inspect_reps \
  --output=/scratch/my13/kai/runs/inspect_reps_%j.out --wrap='source /etc/profile.d/modules.sh; module purge; module load Mamba/23.11.0-0; source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/conda.sh; source /opt/apps/software/Mamba/23.11.0-0/etc/profile.d/mamba.sh; mamba activate /projects/my13/kai/meta-pers-gest/envs/fl-torch; cd /projects/my13/kai/meta-pers-gest/pers-gest-cls; python system/universal_preprocessing/inspect_rep_structure.py /scratch/my13/kai/meta-pers-gest/data/filtered_datasets/metadata_IMU_EMG_allgestures_allusers.pkl'
```
When it finishes: `tail -60 /scratch/my13/kai/runs/inspect_reps_*.out` and **paste it to me**.

## 2. Cluster: smoke test of the new protocol (debug partition, 15-minute cap)
```bash
cd /projects/my13/kai/meta-pers-gest/pers-gest-cls/system/NOTS/paper/ablations/launchers
bash kfold_launcher.sh M0 A1 A2 A4 --folds 0 --debug --dry-run     # sanity-check the printed commands
bash kfold_launcher.sh M0 A1 A2 A4 --folds 0 --debug               # submit
squeue -u $USER
```
M0/A4 are expected to hit the 15-minute wall time; I only need proof that each job starts, loads fold 0 (test users `P102 P108 P126 P132`) and trains at least one epoch. When the jobs end, **paste the output of**:
```bash
LOG=/scratch/my13/kai/runs/paper/ablations/eval/logs
for f in $LOG/kfold7_*_f0_*.out; do echo "=== $f"; grep -E "kfold7\] fold|JOB_|PYTHON_EXIT|Traceback|Error|Parameters|Epoch [0-9]+ of|Val loss/acc|test_PIDs" "$f" | tail -25; done
```

## 3. Decisions (one word each)
- **"hpo ok"** — HPO every headline comparator (M0, A1, A2, A4, transformers) with equal budgets, and run A1–A4 both with frozen M0 hyperparameters and re-tuned. Say what to change if not.
- **"10-way ok"** — make 10-way the primary task (10-way 1-shot = the single primary comparison cell; 3/5-way become the secondary grid). Reasoning in `SPEC_protocol.md` §1. The HPO objective depends on this, so it blocks HPO.
