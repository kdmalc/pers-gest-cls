"""
M0_inner_steps_eval_sweep.py
============================
Post-hoc sweep over maml_inner_steps_eval on the VAL set.
No training. Load a pretrained M0 checkpoint, sweep step counts, save plots + JSON.

Run via sbatch:
    sbatch inner_steps_sweep_launcher.sh
Or directly (with env vars set):
    python M0_inner_steps_eval_sweep.py
"""

import matplotlib
matplotlib.use("Agg")   # headless — must come before any other matplotlib import

import os
import sys
import copy
import json
import pickle
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

# =============================================================================
# PATHS — read from env vars (set by the SLURM launcher), with cluster defaults
# =============================================================================
CHECKPOINT_PATH = os.environ.get(
    "CHECKPOINT_PATH",
    "/scratch/my13/kai/runs/paper/ablations/hpo/M0/trial_25/trial_64_fold0_best.pt",
)
CODE_DIR = os.environ.get(
    "CODE_DIR",
    "/projects/my13/kai/meta-pers-gest/pers-gest-cls",
)
DATA_DIR = os.environ.get(
    "DATA_DIR",
    "/scratch/my13/kai/meta-pers-gest/data",
)
OUT_DIR = os.environ.get(
    "OUT_DIR",
    "/scratch/my13/kai/runs/paper/ablations/inner_steps_sweep",
)

os.environ["CODE_DIR"] = CODE_DIR
os.environ["DATA_DIR"] = DATA_DIR
os.environ["RUN_DIR"]  = OUT_DIR

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# =============================================================================
# sys.path
# =============================================================================
_code = Path(CODE_DIR)
for _p in [
    _code,
    _code / "system",
    _code / "system" / "MAML",
    _code / "system" / "MOE",
    _code / "system" / "pretraining",
]:
    sys.path.insert(0, str(_p))

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
print(f"OUT_DIR : {OUT_DIR}")

# =============================================================================
# Config — exactly as trial 64 ran, verified against the SLURM log
# =============================================================================
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'test_eval_files'))
from ablation_config import make_base_config, VAL_PIDS, FIXED_SEED, build_maml_moe_model, count_parameters

def build_config() -> dict:
    config = make_base_config(ablation_id="M0")

    # ── Fixed architecture constants (from ablation_hpo.py FIXED_* values) ───
    config["cnn_base_filters"]     = 64
    config["lstm_hidden"]          = 64
    config["groupnorm_num_groups"] = 8
    config["use_GlobalAvgPooling"] = True

    # ── HPO-tuned values (trial 64, confirmed from SLURM log) ─────────────────
    config["learning_rate"]        = 0.00010093304999603776   # outer_lr
    config["weight_decay"]         = 0.0008325426470137959
    config["maml_inner_steps"]     = 10
    config["maml_alpha_init"]      = 0.0009888781900907544
    config["maml_alpha_init_eval"] = 0.006717556958813548
    config["maml_use_lslr"]        = True
    config["use_lslr_at_eval"]     = False
    config["use_maml_msl"]         = "hybrid"
    config["maml_msl_num_epochs"]  = 39
    config["episodes_per_epoch_train"] = 250
    config["num_experts"]          = 24
    config["MOE_top_k"]            = 7
    config["top_k"]                = 7
    config["MOE_gate_temperature"] = 1.1879664247660187
    config["MOE_aux_coeff"]        = 0.08672942143224953
    config["MOE_ctx_out_dim"]      = 64
    config["MOE_ctx_hidden_dim"]   = 32
    config["MOE_dropout"]          = 0.022501513050004283
    # 'MOE_aux_loss_plcmt' in Optuna maps to 'apply_MOE_aux_loss_inner_outer' in config
    config["apply_MOE_aux_loss_inner_outer"] = "outer"
    config["label_smooth"]         = 0.05   # fixed for all MAML trials

    # ── Fixed MAML settings (from _suggest_maml_hps) ─────────────────────────
    config["meta_learning"]          = True
    config["meta_batchsize"]         = 24
    config["maml_opt_order"]         = "first"
    config["maml_first_order_to_second_order_epoch"] = 1_000_000
    config["enable_inner_loop_optimizable_bn_params"] = False

    # ── Fixed MoE settings (from _suggest_moe_hps) ───────────────────────────
    config["use_MOE"]             = True
    config["MOE_placement"]       = "encoder"
    config["MOE_expert_expand"]   = 1.0
    config["MOE_mlp_hidden_mult"] = 1.0
    config["MOE_log_every"]       = 5
    config["MOE_plot_dir"]        = None
    config["gate_type"]           = "context_feature_demo"
    config["expert_architecture"] = "MLP"

    # ── Eval / misc ───────────────────────────────────────────────────────────
    config["num_eval_episodes"] = 200   # full NUM_VAL_EPISODES (HPO used 100)
    config["padding"]           = 0
    config["use_batch_norm"]    = False

    return config


config = build_config()

print("\nKey config values:")
for k in ["cnn_base_filters", "lstm_hidden", "groupnorm_num_groups",
          "maml_inner_steps", "maml_alpha_init_eval", "use_lslr_at_eval",
          "num_experts", "MOE_top_k", "use_MOE"]:
    print(f"  {k}: {config[k]}")

# =============================================================================
# Build model & load checkpoint
# =============================================================================
from MAML.mamlpp import PerParamPerStepLSLR, named_param_dict

model = build_maml_moe_model(config)
print(f"\nParameters: {count_parameters(model):,}")

# _lslr is monkey-patched onto the model during training (not part of the
# architecture), so we must recreate it here before load_state_dict will accept
# the checkpoint's keys.
if config["maml_use_lslr"]:
    temp_params = named_param_dict(model, require_grad_only=True)
    model._lslr = PerParamPerStepLSLR(
        named_params = temp_params.items(),
        inner_steps  = config["maml_inner_steps"],   # must match train-time value (10)
        init_lr      = config["maml_alpha_init"],
        learnable    = True,
        device       = config["device"],
    ).to(config["device"])

print(f"\nLoading checkpoint: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location=config["device"])
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

print("Checkpoint loaded.")
if "best_val_acc" in ckpt:
    print(f"  best_val_acc : {ckpt['best_val_acc']:.4f}")
if "seed" in ckpt:
    print(f"  seed         : {ckpt['seed']}")

# =============================================================================
# Build val dataloader (built once, reused across all step counts)
# =============================================================================
from MAML.maml_data_pipeline import MetaGestureDataset, maml_mm_collate, reorient_tensor_dict
from torch.utils.data import DataLoader

tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")

print(f"\nLoading tensor dict: {tensor_dict_path}")
with open(tensor_dict_path, "rb") as f:
    full_dict = pickle.load(f)
tensor_dict = reorient_tensor_dict(full_dict, config)

NUM_VAL_EPISODES = config["num_eval_episodes"]

val_ds = MetaGestureDataset(
    tensor_dict,
    target_pids             = VAL_PIDS,
    target_gesture_classes  = config["maml_gesture_classes"],
    target_trial_indices    = config["target_trial_indices"],
    n_way                   = config["n_way"],
    k_shot                  = config["k_shot"],
    q_query                 = config["q_query"],
    num_eval_episodes       = NUM_VAL_EPISODES,
    is_train                = False,
    seed                    = FIXED_SEED,
    use_label_shuf_meta_aug = False,
)
val_dl = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=maml_mm_collate
)

print(f"Val PIDs     : {VAL_PIDS}")
print(f"Val episodes : {len(val_ds)}")

# =============================================================================
# Sweep
# =============================================================================
from MAML.mamlpp import mamlpp_adapt_and_eval

INNER_STEPS_SWEEP = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 75, 100]

print(f"\nSweeping maml_inner_steps_eval over: {INNER_STEPS_SWEEP}")
print(f"Train-time steps (for reference)   : {config['maml_inner_steps']}")
print(f"maml_alpha_init_eval               : {config['maml_alpha_init_eval']}")
print(f"use_lslr_at_eval                   : {config['use_lslr_at_eval']}")
print()

sweep_results = []

for n_steps in INNER_STEPS_SWEEP:
    sweep_config = copy.deepcopy(config)
    sweep_config["maml_inner_steps_eval"] = n_steps

    model.eval()
    user_accs  = defaultdict(list)
    user_losses = defaultdict(list)

    for batch in val_dl:
        uid     = batch["user_id"]
        # mamlpp_adapt_and_eval returns {"loss": ..., "acc": ..., "pre_adapt_acc": ...}
        # Note: the key is "loss", not "query_loss"
        metrics = mamlpp_adapt_and_eval(
            model, sweep_config, batch["support"], batch["query"]
        )
        user_accs[uid].append(metrics["acc"])
        user_losses[uid].append(metrics["loss"])

    per_user_acc  = {uid: float(np.mean(accs))   for uid, accs   in user_accs.items()}
    per_user_loss = {uid: float(np.mean(losses)) for uid, losses in user_losses.items()}

    all_accs   = list(per_user_acc.values())
    all_losses = list(per_user_loss.values())
    mean_acc   = float(np.mean(all_accs))
    std_acc    = float(np.std(all_accs))
    mean_loss  = float(np.mean(all_losses))
    std_loss   = float(np.std(all_losses))

    sweep_results.append({
        "inner_steps_eval": n_steps,
        "mean_acc":         mean_acc,
        "std_acc":          std_acc,
        "mean_loss":        mean_loss,
        "std_loss":         std_loss,
        "per_user_acc":     per_user_acc,
        "per_user_loss":    per_user_loss,
    })

    print(f"steps={n_steps:3d}  val_acc={mean_acc*100:.2f}% ± {std_acc*100:.2f}%"
          f"  val_loss={mean_loss:.4f} ± {std_loss:.4f}")

# =============================================================================
# Results table
# =============================================================================
print(f"\n{'inner_steps_eval':>18}  {'mean_acc (%)':>13}  {'std_acc (%)':>12}  {'mean_loss':>10}")
print("-" * 62)
for r in sweep_results:
    print(f"{r['inner_steps_eval']:>18}  {r['mean_acc']*100:>13.2f}  "
          f"{r['std_acc']*100:>12.2f}  {r['mean_loss']:>10.4f}")

# =============================================================================
# Plots
# =============================================================================
steps  = [r["inner_steps_eval"] for r in sweep_results]
accs   = [r["mean_acc"] * 100   for r in sweep_results]
stds   = [r["std_acc"]  * 100   for r in sweep_results]
losses = [r["mean_loss"]        for r in sweep_results]
loss_stds = [r["std_loss"]      for r in sweep_results]

best_idx    = int(np.argmax(accs))
train_steps = config["maml_inner_steps"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Accuracy ──────────────────────────────────────────────────────────────────
ax = axes[0]
ax.plot(steps, accs, marker="o", linewidth=2, color="steelblue", label="mean val acc")
ax.fill_between(
    steps,
    [a - s for a, s in zip(accs, stds)],
    [a + s for a, s in zip(accs, stds)],
    alpha=0.2, color="steelblue", label="±1 std (over val users)",
)
ax.axvline(
    steps[best_idx], color="red", linestyle="--", alpha=0.7,
    label=f"best: {steps[best_idx]} steps  ({accs[best_idx]:.2f}%)",
)
ax.axvline(
    train_steps, color="orange", linestyle=":", alpha=0.9,
    label=f"train steps = {train_steps}",
)
ax.set_xlabel("maml_inner_steps_eval", fontsize=12)
ax.set_ylabel("Val Accuracy (%)", fontsize=12)
ax.set_title("M0 — Inner Steps Eval Sweep (Val Set)", fontsize=13)
ax.xaxis.set_major_locator(mticker.FixedLocator(steps))
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# ── Loss ──────────────────────────────────────────────────────────────────────
ax2 = axes[1]
ax2.plot(steps, losses, marker="s", linewidth=2, color="coral", label="mean val loss")
ax2.fill_between(
    steps,
    [l - s for l, s in zip(losses, loss_stds)],
    [l + s for l, s in zip(losses, loss_stds)],
    alpha=0.2, color="coral",
)
ax2.axvline(
    train_steps, color="orange", linestyle=":", alpha=0.9,
    label=f"train steps = {train_steps}",
)
ax2.set_xlabel("maml_inner_steps_eval", fontsize=12)
ax2.set_ylabel("Val Loss", fontsize=12)
ax2.set_title("M0 — Query Loss vs Eval Steps", fontsize=13)
ax2.xaxis.set_major_locator(mticker.FixedLocator(steps))
ax2.tick_params(axis="x", rotation=45)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = str(Path(OUT_DIR) / "M0_inner_steps_eval_sweep.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nPlot saved: {plot_path}")
print(f"Best: {steps[best_idx]} steps → {accs[best_idx]:.2f}% ± {stds[best_idx]:.2f}%")

# ── Per-user breakdown at best step count ─────────────────────────────────────
best_result = sweep_results[best_idx]
per_user    = best_result["per_user_acc"]
uids        = sorted(per_user.keys())
u_accs      = [per_user[uid] * 100 for uid in uids]

fig2, ax3 = plt.subplots(figsize=(max(6, len(uids) * 1.5), 4))
ax3.bar(range(len(uids)), u_accs, color="steelblue", alpha=0.8)
ax3.axhline(float(np.mean(u_accs)), color="red", linestyle="--",
            label=f"mean = {np.mean(u_accs):.2f}%")
ax3.axhline(100.0 / config["n_way"], color="gray", linestyle=":",
            label=f"chance = {100.0/config['n_way']:.1f}%")
ax3.set_xticks(range(len(uids)))
ax3.set_xticklabels(uids, rotation=45, fontsize=9)
ax3.set_ylabel("Val Accuracy (%)", fontsize=12)
ax3.set_title(f"M0 — Per-user accuracy @ {steps[best_idx]} inner steps", fontsize=13)
ax3.legend(fontsize=9)
ax3.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
per_user_plot_path = str(Path(OUT_DIR) / "M0_per_user_best_steps.png")
plt.savefig(per_user_plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Per-user plot saved: {per_user_plot_path}")

# =============================================================================
# Save results JSON
# =============================================================================
serialisable = []
for r in sweep_results:
    entry = {k: v for k, v in r.items() if k not in ("per_user_acc", "per_user_loss")}
    entry["per_user_acc"]  = {str(k): v for k, v in r["per_user_acc"].items()}
    entry["per_user_loss"] = {str(k): v for k, v in r["per_user_loss"].items()}
    serialisable.append(entry)

output = {
    "ablation_id":            "M0",
    "sweep_type":             "maml_inner_steps_eval",
    "checkpoint":             CHECKPOINT_PATH,
    "val_pids":               VAL_PIDS,
    "num_val_episodes":       NUM_VAL_EPISODES,
    "maml_inner_steps_train": config["maml_inner_steps"],
    "maml_alpha_init_eval":   config["maml_alpha_init_eval"],
    "use_lslr_at_eval":       config["use_lslr_at_eval"],
    "inner_steps_sweep":      INNER_STEPS_SWEEP,
    "best_steps":             steps[best_idx],
    "best_mean_acc":          accs[best_idx] / 100.0,
    "results":                serialisable,
}

json_path = str(Path(OUT_DIR) / "M0_inner_steps_eval_sweep.json")
with open(json_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Results JSON saved: {json_path}")
