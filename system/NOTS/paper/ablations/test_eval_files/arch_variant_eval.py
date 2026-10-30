"""
arch_variant_eval.py
====================
7-fold (kfold7) evaluation of ANY architecture-exploration token (arch_variants.registry.ARCH_VARIANTS:
X2-X6, F1-F3, T0, T2, T4, P1 family, P3, A4_ANIL, M0_ANIL). Run only for screen finalists
(docs/specs/SPEC_architecture_exploration.md §3); the screen itself uses the dev users via ablation_hpo.py.

The token means exactly what it meant in the HPO: the same registry entry, the same model builder, and
(with HPO_CONFIG=tuned) the token's own exported HPO result from best_hpo_configs.json.

Training / evaluation by kind (registry.variant_kind):
  maml        MAML++ on the fold's 20 train users, checkpoint on its 4 val users, test on its 4 test users
              (mamlpp_pretrain + run_episodic_test_eval)                       -- X*, F*, T*, *_ANIL
  metric      episodic metric training, same selection / test                     -- P1 family
  supervised  supervised pretraining (ADANN for P3), then head-only and full fine-tune per test
              episode, exactly as A2 (A2_no_maml_no_moe.build_config is the base)  -- P3

    python arch_variant_eval.py --token X2 --fold-idx 0 [--seed 43] [--cells all] [--smoke]
kfold_launcher.sh token: V_<token> (e.g. V_X2, V_P1_feat).
"""

import os, sys, copy, json, argparse, pickle
import numpy as np
import torch

from pathlib import Path
CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
for _p in ("", "system", "system/MAML", "system/MOE", "system/pretraining"):
    sys.path.insert(0, str(CODE_DIR / _p))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ablation_config import (  # noqa: E402
    tensor_dict_file, make_base_config, set_seeds, FIXED_SEED,
    run_episodic_test_eval, run_supervised_test_eval, save_results, save_model_checkpoint,
    count_parameters, make_periodic_checkpoint_fn, apply_kfold_fold, add_fold_arg, kfold_tag,
    resolve_fold_ids, add_smoke_arg, apply_smoke, add_seed_arg, add_cells_arg, parse_cells,
)
from arch_variants.registry import (  # noqa: E402
    ARCH_VARIANTS, FAMILY_EVAL_DEFAULTS, A4_CNN_BASE_FILTERS, A4_LSTM_HIDDEN,
    variant_kind, uses_a4_widths, uses_moe, build_model_for_config,
)


# the block the headline number comes from (supervised tokens also report head-only, like A2)
MAIN_RESULT = {"maml": "test_results", "metric": "test_results", "supervised": "test_full_ft"}


def build_config(token: str, smoke: bool) -> dict:
    """Base config + the token's fixed structure. Tuned HPs arrive later, in apply_kfold_fold."""
    v = ARCH_VARIANTS[token]
    kind = variant_kind(token)
    if kind == "supervised":
        from A2_no_maml_no_moe import build_config as a2_config
        config = a2_config()
        config["ablation_id"] = token
    else:
        config = make_base_config(ablation_id=token)
    config["arch_family"] = v["family"]
    config.update(copy.deepcopy(v["set"]))
    config["use_MOE"] = uses_moe(token)
    if uses_a4_widths(token):
        config["cnn_base_filters"], config["lstm_hidden"] = A4_CNN_BASE_FILTERS, A4_LSTM_HIDDEN
    if kind == "metric":
        config["meta_learning"] = True
        config.update(FAMILY_EVAL_DEFAULTS["metric"])
    if "adann" in v["spaces"]:
        config.update(FAMILY_EVAL_DEFAULTS["adann"])
    if config["MOE_gate_mode"] == "uniform":
        config["MOE_top_k"] = config["top_k"] = config["num_experts"]
        config["MOE_aux_coeff"] = 0.0
    return apply_smoke(config, smoke)


def _fold_consistency(config: dict, token: str):
    """A tuned override must not change the token's defining structure (it was tuned WITH it)."""
    for k, val in ARCH_VARIANTS[token]["set"].items():
        if config[k] != val:
            raise ValueError(f"{token}: config[{k!r}]={config[k]!r} after the HPO overrides, "
                             f"but the token defines {val!r}")
    if config["MOE_gate_mode"] == "uniform" and config["MOE_top_k"] != config["num_experts"]:
        raise ValueError(f"{token}: uniform gate needs MOE_top_k == num_experts")


def run_one_fold(token: str, fold_id: str, seed: int, config: dict, cells) -> dict:
    set_seeds(seed)
    config = copy.deepcopy(config)
    config["seed"] = seed
    _fold_consistency(config, token)
    kind = variant_kind(token)
    tensor_dict_path = tensor_dict_file(config)

    model = build_model_for_config(config).to(config["device"])
    n_params = count_parameters(model)
    print(f"\n[{token} | {fold_id} | seed={seed}] kind={kind}  parameters={n_params:,}")
    print(f"  train={config['train_PIDs']}\n  val={config['val_PIDs']}\n  test={config['test_PIDs']}")

    if kind in ("maml", "metric"):
        from MAML.maml_data_pipeline import get_maml_dataloaders
        train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)
        if kind == "maml":
            from MAML.mamlpp import mamlpp_pretrain
            model, hist = mamlpp_pretrain(model, config, train_dl, episodic_val_loader=val_dl,
                                          periodic_checkpoint_fn=make_periodic_checkpoint_fn(config),
                                          checkpoint_every=10)
            eval_fn = None
        else:
            from arch_variants.metric_learners import metric_pretrain, metric_adapt_and_eval
            model, hist = metric_pretrain(model, config, train_dl, episodic_val_loader=val_dl)
            eval_fn = metric_adapt_and_eval
        best_val, best_state = hist["best_val_acc"], hist["best_state"]
        model.load_state_dict(best_state)
        test = {"test_results": run_episodic_test_eval(model, config, tensor_dict_path, config["test_PIDs"],
                                                       cells=cells, adapt_and_eval_fn=eval_fn)}
        routing = None
        if config["use_MOE"] and kind == "maml":
            # routing on the fold's VAL episodes (never test): the X1-vs-X2 diagnostics per fold
            from MOE.routing_metrics import collect_episode_routing, format_summary
            routing = collect_episode_routing(model, config, val_dl, adapted=True)
            if routing is not None:
                print(f"[{token} | {fold_id}] routing on val episodes:\n{format_summary(routing)}")
        test["routing_val"] = routing
    else:
        from MAML.maml_data_pipeline import reorient_tensor_dict
        from pretraining.pretrain_data_pipeline import get_pretrain_dataloaders
        with open(tensor_dict_path, "rb") as f:
            tensor_dict = reorient_tensor_dict(pickle.load(f), config)
        train_dl, val_dl, _ = get_pretrain_dataloaders(config, tensor_dict)
        if "adann_lambda" in config:
            from arch_variants.adann import adann_pretrain
            model, hist = adann_pretrain(model, train_dl, val_dl, config)
        else:
            from pretraining.pretrain_trainer import pretrain
            model, hist = pretrain(model, train_dl, val_dl, config)
        best_val = max(hist["val_acc"]) if hist["val_acc"] else float("nan")
        best_state = model.state_dict()
        test = {mode_key: run_supervised_test_eval(model, config, tensor_dict_path, config["test_PIDs"],
                                                   ft_mode=mode, cells=cells)
                for mode_key, mode in (("test_head_only", "head_only"), ("test_full_ft", "full"))}

    save_model_checkpoint({"fold_id": fold_id, "seed": seed, "model_state_dict": best_state,
                           "config": config, "best_val_acc": best_val}, config,
                          tag=f"{fold_id}_seed{seed}_best")
    headline = test[MAIN_RESULT[kind]]
    print(f"[{token} | {fold_id}] test {headline['mean_acc']*100:.2f}% "
          f"+/- {headline['std_acc']*100:.2f}% (over users; {MAIN_RESULT[kind]})")
    return {"fold_id": fold_id, "seed": seed, "test_PID": config["test_PIDs"], "val_PID": config["val_PIDs"],
            "best_val_acc": float(best_val), "n_params": n_params, **test}


def main():
    ap = argparse.ArgumentParser(description="kfold7 evaluation of an architecture-exploration token")
    ap.add_argument("--token", required=True, choices=sorted(ARCH_VARIANTS))
    add_fold_arg(ap)
    add_seed_arg(ap)
    add_cells_arg(ap)
    add_smoke_arg(ap)
    args = ap.parse_args()
    cells = parse_cells(args.cells)
    config = build_config(args.token, args.smoke)

    results = []
    for k in resolve_fold_ids(args.fold_idx):
        fold_config = apply_kfold_fold(config, k)
        results.append(run_one_fold(args.token, f"kfold7_fold{k}", args.seed, fold_config, cells))
    accs = [r[MAIN_RESULT[variant_kind(args.token)]]["mean_acc"] for r in results]
    summary = {
        "ablation_id": args.token, "description": ARCH_VARIANTS[args.token]["note"],
        "variant": {k: (list(v) if isinstance(v, tuple) else v) for k, v in ARCH_VARIANTS[args.token].items()},
        "kind": variant_kind(args.token), "test_procedure": "kfold7",
        "fold_idxs": resolve_fold_ids(args.fold_idx), "seed": args.seed,
        "n_params": results[0]["n_params"], "fold_results": results,
        "mean_test_acc": float(np.mean(accs)), "std_test_acc": float(np.std(accs)),
        "num_folds": len(results), "config_snapshot": {k: str(v) for k, v in config.items()},
    }
    save_results(summary, config, tag=kfold_tag(args.fold_idx, args.seed))
    print(f"[{args.token}] FINAL kfold7 {summary['fold_idxs']} seed {args.seed}: "
          f"{summary['mean_test_acc']*100:.2f}% +/- {summary['std_test_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
