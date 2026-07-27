"""
portB_local_adaptation.py
=========================
MoEMeta Port B: frozen global bank + small task-conditional local adaptation.

WHAT MoEMeta DOES
-----------------
At meta-test, MoEMeta freezes its global parameters Phi (neighbour aggregator,
expert bank, gate) and fine-tunes only three d-dim projection vectors
{p_h, p_r, p_t} plus the relation-meta R_T on the support set. That is
embedding-level adaptation over roughly 4d trainable scalars -- closer to MetaR
than to MAML++. Ours adapts the full parameter set with 10 first-order LSLR
steps.

This run tests that adaptation regime in our setting.

THE THING THAT MAKES OR BREAKS THIS RUN
---------------------------------------
MoEMeta's global parameters are optimised *through* its restricted local
adaptation in the outer loop. Meta-training with full MAML++ and then freezing
the encoder only at evaluation time would evaluate a model in a regime it was
never optimised for. It would lose by construction, the margin would be
meaningless, and R3 (confidence 4) would be right to notice.

So this script **meta-trains with the restricted inner loop**, via
`config["maml_inner_param_include"]`. The mechanism is deliberately not
`requires_grad=False` on the bank: the outer loop selects its parameters by
`requires_grad`, so freezing that way would stop the bank being meta-learned at
all. Instead the inner-loop parameter dict is filtered, and
`torch.func.functional_call` falls back to the module's live parameters for
every name absent from it. The bank therefore stays in the autograd graph and
outer-loop meta-gradients still reach it through the restricted inner loop.

Verified: with the inner loop restricted to the head, all of
`expert_cnns`, `gate` and the LSTM still receive non-zero outer-loop gradients.

WHICH LOCAL MODULE
------------------
  head       (default, faithful) Freeze experts, gate and LSTM; adapt only the
             classification head. This is the closest analog to MoEMeta that
             requires no architectural change -- a small terminal module
             adapted per task while a globally-shared bank stays fixed.
  head_gate  Also adapt the gate. Less faithful: in MoEMeta the gate is part of
             the frozen global bank, and nothing inside its MoE is adapted per
             task. Provided as a sensitivity check, not as the headline.

Report the `head` variant. If you report `head_gate`, say why.

WHAT TO EXPECT, AND HOW TO WRITE IT UP
--------------------------------------
This is a fair test, so the result is informative in both directions. If
full-parameter adaptation wins, that supports the reframed Contribution 1
(full inner-loop adaptation rather than a frozen encoder plus a task
embedding). If restricted adaptation matches it, that is a real finding worth
stating: most of the benefit is in the meta-learned representation rather than
the adaptation budget, and it also answers the open question at L362-365.

Costs one training configuration.

Usage (NOTS):
    python portB_local_adaptation.py
    python portB_local_adaptation.py --local-module head_gate
    python portB_local_adaptation.py --eval-only-freeze   # the INVALID variant
"""

import os
import sys
import copy
import argparse
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))
sys.path.insert(0, str(CODE_DIR / "system" / "MOE"))
sys.path.insert(0, str(CODE_DIR / "system" / "pretraining"))

from ablation_config import (
    make_base_config, build_maml_moe_model, set_seeds, FIXED_SEED,
    run_episodic_test_eval, save_results, save_model_checkpoint, count_parameters,
    make_periodic_checkpoint_fn, make_periodic_test_eval_fn,
)
from MAML.maml_data_pipeline import get_maml_dataloaders
from MAML.mamlpp import named_param_dict, inner_param_filter

# Substrings selecting the "local" module. Matched against parameter names.
LOCAL_MODULES = {
    "head":      ["head"],
    "head_gate": ["head", "gate"],
}

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def build_config(local_module: str, eval_only_freeze: bool) -> dict:
    config = make_base_config(ablation_id=f"portB_{local_module}")

    # REQUIRED: get_maml_dataloaders reads config["seed"] directly and
    # make_base_config does not set it (M0_full_model.py sets it explicitly at
    # line 89). Omitting this raises KeyError: 'seed' at the dataloader build,
    # i.e. AFTER model construction and any pre-flight checks have printed --
    # which is why the mask verification passed and the job still died.
    config["seed"] = FIXED_SEED
    config["test_procedure"] = "hpo_test_split"

    if not eval_only_freeze:
        # The faithful regime: restrict the inner loop during META-TRAINING.
        config["maml_inner_param_include"] = LOCAL_MODULES[local_module]
    else:
        # The invalid regime, retained only so the confound can be demonstrated
        # rather than argued about. Meta-train unrestricted, restrict at eval.
        config["maml_inner_param_include"] = None

    print(f"[portB] local_module          : {local_module} "
          f"-> {LOCAL_MODULES[local_module]}")
    print(f"[portB] eval_only_freeze      : {eval_only_freeze}")
    print(f"[portB] maml_inner_param_include (train): "
          f"{config.get('maml_inner_param_include')}")
    return config


def report_partition(model, config, local_module: str) -> dict:
    """
    Print and return exactly which parameters the inner loop will adapt. This is
    the claim the caption rests on, so it is measured rather than asserted.
    """
    inc = LOCAL_MODULES[local_module]
    adapted = named_param_dict(model, require_grad_only=True, include_substrings=inc)
    everything = named_param_dict(model, require_grad_only=True)

    n_adapt = sum(p.numel() for p in adapted.values())
    n_total = sum(p.numel() for p in everything.values())

    print(f"\n[portB] inner-loop parameter partition")
    print(f"        adapted per task : {n_adapt:,} ({100.0*n_adapt/max(n_total,1):.2f}% "
          f"of {n_total:,} trainable)")
    print(f"        adapted tensors  : {list(adapted.keys())}")
    frozen_families = sorted({k.split('.')[0] for k in everything if k not in adapted})
    print(f"        frozen in inner loop (still meta-learned): {frozen_families}")

    assert n_adapt > 0, (
        f"Inner loop would adapt nothing: no parameter name contains any of {inc}. "
        f"Available top-level families: "
        f"{sorted({k.split('.')[0] for k in everything})}"
    )
    assert n_adapt < n_total, (
        "Inner loop would adapt everything, so this is not a restricted regime. "
        f"Check LOCAL_MODULES[{local_module!r}]."
    )
    return {
        "n_adapted_params": n_adapt,
        "n_trainable_params": n_total,
        "frac_adapted": n_adapt / max(n_total, 1),
        "adapted_tensors": list(adapted.keys()),
        "frozen_families": frozen_families,
    }


def run(args) -> dict:
    local_module = args.local_module
    config = build_config(local_module, args.eval_only_freeze)
    set_seeds(FIXED_SEED)

    tensor_dict_path = os.path.join(config["dfs_load_path"], "segfilt_rts_tensor_dict.pkl")

    model = build_maml_moe_model(config)
    n_params = count_parameters(model)
    print(f"[portB] Parameters: {n_params:,} (identical to M0 -- only the inner-loop "
          f"parameter SET differs, not the architecture)")

    partition = report_partition(model, config, local_module)

    train_dl, val_dl = get_maml_dataloaders(config, tensor_dict_path=tensor_dict_path)

    from MAML.mamlpp import mamlpp_pretrain
    trained_model, train_history = mamlpp_pretrain(
        model, config, train_dl, episodic_val_loader=val_dl,
        periodic_checkpoint_fn=make_periodic_checkpoint_fn(config),
        periodic_test_eval_fn=make_periodic_test_eval_fn(
            tensor_dict_path, config["test_PIDs"]),
        checkpoint_every=10,
    )
    best_val_acc = train_history["best_val_acc"]
    print(f"[portB] Training complete. Best val acc = {best_val_acc:.4f}")

    save_model_checkpoint(
        {
            "local_module":     local_module,
            "eval_only_freeze": args.eval_only_freeze,
            "seed":             FIXED_SEED,
            "model_state_dict": train_history["best_state"],
            "config":           config,
            "best_val_acc":     best_val_acc,
        },
        config,
        tag=f"portB_{local_module}_seed{FIXED_SEED}_best",
    )
    trained_model.load_state_dict(train_history["best_state"])

    # Evaluation must use the SAME restriction the model was trained under.
    eval_config = copy.deepcopy(config)
    eval_config["maml_inner_param_include"] = LOCAL_MODULES[local_module]
    print(f"[portB] eval inner-loop restriction: "
          f"{eval_config['maml_inner_param_include']}")

    test_results = run_episodic_test_eval(
        trained_model, eval_config, tensor_dict_path, eval_config["test_PIDs"]
    )

    caveats = [
        "Named 'MoEMeta-style Local Adaptation'. It is a transfer of MoEMeta's "
        "adaptation mechanism to a non-graph modality, because the original "
        "cannot be run on this data (no graph, no candidate set or ranking "
        "objective, symbolic-embedding experts, and its held-out axis is "
        "relations rather than users).",
        "Architecture and parameter count are identical to M0; only the set of "
        "parameters adapted per task differs.",
        "Fixed 24/4/4 split: outside the paired RM-ANOVA; compare against the "
        "fixed-split baseline (88.4%), not L2SO (86.7%).",
    ]
    if args.eval_only_freeze:
        caveats.insert(0,
            "CONFOUNDED: meta-trained with the FULL inner loop and restricted "
            "only at evaluation, so the model is evaluated in a regime it was "
            "never optimised for. It is expected to lose by construction and "
            "the margin is not interpretable. Do not lean on it.")
    else:
        caveats.insert(0,
            "Meta-trained THROUGH the restricted inner loop: the global bank is "
            "frozen during per-task adaptation but still receives outer-loop "
            "meta-gradients, matching MoEMeta's regime.")

    result = {
        "ablation_id":       f"portB_{local_module}",
        "description":       ("MoEMeta Port B: global expert bank frozen at adaptation "
                              "time, only a small task-conditional module adapted."),
        "local_module":      local_module,
        "local_module_substrings": LOCAL_MODULES[local_module],
        "eval_only_freeze":  args.eval_only_freeze,
        "meta_trained_through_restriction": not args.eval_only_freeze,
        "inner_loop_partition": partition,
        "n_params":          n_params,
        "best_val_acc":      float(best_val_acc),
        "test_results":      test_results,
        "test_acc":          test_results["mean_acc"],
        "caveats":           caveats,
        "config_snapshot":   {k: str(v) for k, v in config.items()},
    }
    save_results(result, config, tag=f"portB_{local_module}_final")

    print(f"\n{'='*70}")
    print(f"[portB] FINAL {local_module}: {test_results['mean_acc']*100:.2f}% "
          f"± {test_results['std_acc']*100:.2f}%")
    print(f"        adapted {partition['n_adapted_params']:,} params per task "
          f"({partition['frac_adapted']*100:.2f}% of trainable)")
    print(f"        compare against fixed-split M0 (full-parameter adaptation)")
    if args.eval_only_freeze:
        print("        WARNING: eval-only freeze -- confounded, do not report as a margin")
    print(f"{'='*70}")
    return result


def main():
    ap = argparse.ArgumentParser(
        description="MoEMeta Port B: restricted local adaptation.")
    ap.add_argument("--local-module", choices=list(LOCAL_MODULES), default="head")
    ap.add_argument("--eval-only-freeze", action="store_true",
                    help="INVALID variant: meta-train unrestricted, restrict only at "
                         "eval. Only for demonstrating the confound.")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
