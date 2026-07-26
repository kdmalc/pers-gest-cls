"""
desk_checks.py
==============
The Appendix A "desk work, no cluster" items. Each one turns a [[FILL]] in the
rebuttal into a fact. Runs on a login node in seconds; no GPU, no training.

    python desk_checks.py                 # all checks
    python desk_checks.py --only norm     # one check
    python desk_checks.py --only gates --gate-vectors /path/to/per_user_gates.npz

Checks
------
  norm    Block NORM's six facts about normalization (meta Q9, R4 Q2)
  lpft    N-way and K for the Kaifosh LP/FT rows (R1 W3)
  qcount  Realised query counts per class at each K (R2 W1)
  gates   Per-subject max gate weight and routing entropy (R1 Q3)
  optuna  Locate the Optuna studies backing Table 2's search space (R1 Q1)
"""

import os
import re
import sys
import glob
import json
import argparse
from pathlib import Path

CODE_DIR = Path(os.environ.get("CODE_DIR", "./")).resolve()
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "system"))
sys.path.insert(0, str(CODE_DIR / "system" / "MAML"))
sys.path.insert(0, str(CODE_DIR / "system" / "MOE"))
sys.path.insert(0, str(CODE_DIR / "system" / "pretraining"))


def hdr(t):
    print(f"\n{'='*78}\n{t}\n{'='*78}")


# ---------------------------------------------------------------------------
# norm
# ---------------------------------------------------------------------------

def check_norm():
    hdr("NORM -- Block NORM's six facts (meta Q9, R4 Q2)")
    print("""
Read out of the code, not recalled. Two distinct paths exist and they differ;
the one that matters for Table 1 / Table 4 is the 20 Hz path.

MAIN MODEL, 20 Hz envelope path
  produced by : system/universal_preprocessing/tensor_saving_with_morphology.ipynb
  function    : _B_normalize_block -> preprocess_df_B_by_gesture
  consumed as : segfilt_rts_tensor_dict.pkl

  (1) statistic : standard deviation, after per-channel mean removal
  (2) axis      : per-channel DEMEAN, then divide by ONE shared std computed
                  over the whole block (block_np.ravel().std()). So the centring
                  is per-channel but the scaling is not.
  (3) scope     : PER TRIAL. preprocess_df_B_by_gesture treats each trial as a
                  contiguous trial_length=64 block and normalises it alone.
  (4) position  : after the upstream band-pass / envelope, applied to the
                  (T=64, D) trial block, before tensorisation.
  (5) IMU same? : same FUNCTION, applied SEPARATELY to the IMU block (72 ch) and
                  the EMG block (16 ch) -- biosignal_switch_ix=72 splits them.
  (6) comparable
      scales?   : YES, and this is the useful answer for the shared-expert
                  question. Because each modality block is independently scaled
                  to std 1.0, EMG and IMU both arrive at unit std before being
                  concatenated into the 88-channel input, so neither dominates
                  the shared experts by scale alone.

  LEAKAGE: none. The statistic is per trial, so no normalisation constant is
  estimated across subjects and no test-subject data informs the normalisation
  of any training sample. The draft response's claim is correct FOR THIS PATH.

2 kHz path (Kaifosh comparison ONLY -- A10/A11/A11b)
  produced by : EMG_preprocessing/build_2khz_tensor_dict.ipynb
  (1) statistic : standard deviation
  (2) axis      : ONE scalar over all 16 channels flattened
                  (normalize_gestures_by_std_any_channels)
  (3) scope     : per trial
  (4) position  : after band-pass 20-450 Hz + mean subtraction, before resampling
  -> whole-trial SIGNAL s.d. = 1.0, which is the wrong target for their model
     (it expects NOISE s.d. = 1.0). This is the defect A11b corrects.

CAVEAT WORTH CHECKING BEFORE YOU WRITE THE NO-LEAKAGE SENTENCE
  A third function exists: normalize_whole_dataset_features() in
  EMG_preprocessing/shared_processing.py. Its docstring is explicit -- it
  "divides every feature by the overall standard deviation across all features
  and all samples", i.e. ONE global constant estimated over the entire dataset
  INCLUDING test subjects.

  It is called from the spectral-moment / feature-matrix pipelines
  (*_segraw_moments_pipeline.ipynb), NOT from the 20 Hz tensor path the main
  model uses. So it does not affect Table 1 or Table 4.

  But: if ANY reported row is fed by the moments path (the kNN / $B-style
  feature baselines are the likely candidates), then for that row a single
  global scalar IS estimated using test data, and the blanket sentence "no
  test-subject data informs the normalization of any training sample" would be
  wrong as written. It is one shared scalar divisor carrying no label or
  per-subject information, so the practical leakage is minimal -- but say
  "per trial for every EncoderMoE condition and every ablation in Tables 1 and
  4" rather than an unqualified "both are per trial", and confirm which path
  feeds any baseline row you quote.
""".rstrip())


# ---------------------------------------------------------------------------
# lpft / qcount
# ---------------------------------------------------------------------------

def check_lpft():
    hdr("LPFT -- N-way and K for the Kaifosh LP/FT rows (R1 W3)")
    try:
        from ablation_config import make_base_config
        c = make_base_config("A11")
        print(f"  ablation_config base : n_way={c['n_way']}  k_shot={c['k_shot']}  "
              f"q_query={c['q_query']}")
        print(f"  ft_steps={c.get('ft_steps', c.get('maml_inner_steps_eval'))}  "
              f"ft_lr={c.get('ft_lr', c.get('maml_alpha_init_eval'))}")
    except Exception as e:
        print(f"  could not import ablation_config ({e}); falling back to source grep")
        src = (CODE_DIR / "system/NOTS/paper/ablations/test_eval_files/ablation_config.py")
        if src.exists():
            for ln in src.read_text().splitlines():
                if re.search(r'config\["(n_way|k_shot|q_query)"\]\s*=', ln):
                    print("   ", ln.strip())
    print("""
  A10/A11 inherit n_way and k_shot from make_base_config and override only the
  data-format keys (use_imu=False, emg_in_ch=16, sequence_length=4300), so the
  LP/FT rows are the SAME N-way K-shot protocol as the main evaluation.
  State both explicitly in the caption -- R1 is right that it was missing.""".rstrip())


def check_qcount():
    hdr("QCOUNT -- realised query samples per class (R2 W1)")
    print("""
  MetaGestureDataset._build_episode, is_train=False, q_query_eval_mode=
  "all_remaining" (the default and what every published number used):

      support = shuffled_reps[:K]        query = shuffled_reps[K:]

  Disjoint slices, so no overlap. With 10 reps per gesture:

      K=1 -> Q=9      K=3 -> Q=7      K=5 -> Q=5

  The configured q_query=9 is IGNORED on the eval path. Verified empirically.

  Before writing the disclosure, grep the paper for any claim that Q is fixed
  at 9 -- Table 4's caption and the §4 protocol text are the likely places. The
  disclosure only reads as a correction if the paper stops asserting the
  opposite. This script cannot check that; the .tex is not in this repo.

  Also report per-episode realised counts rather than this table: the dataset
  now records them in MetaGestureDataset.episode_shape_log.""".rstrip())


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------

def check_gates(gate_path=None):
    hdr("GATES -- per-subject max gate weight and routing entropy (R1 Q3)")
    import numpy as np

    candidates = []
    if gate_path:
        candidates.append(gate_path)
    for pat in ("**/per_user_gate*", "**/*gate_vectors*", "**/*gate_mean*",
                "**/*routing*.np[yz]", "**/*gate*.np[yz]", "**/*gate*.pt"):
        candidates += sorted(glob.glob(str(CODE_DIR / pat), recursive=True))

    seen, found = set(), None
    for c in candidates:
        if c in seen or not Path(c).exists():
            continue
        seen.add(c)
        found = c
        break

    if found is None:
        print("""
  No saved per-user gate vectors located automatically.

  The rebuttal says these are "already saved for all 32 users". They are
  produced by the routing-analysis notebooks:
      system/MOE/MOE_routing_analysis_ALL_USERS.ipynb
      system/MOE/MOE_routing_analysis_WITHHELD_USERS.ipynb
      system/MOE/MOE_analysis.py

  Point this check at the array with --gate-vectors <path>. Expected shape
  (n_users, E) with E=22. What the response needs:

      max per-subject gate weight   -> is any subject served by one expert?
      per-subject routing entropy   -> against the uniform ceiling ln(22)=3.091
      whether any subject's mass concentrates on a single expert

  With top-k=9 of 22, a fully uniform subject sits at ln(9)=2.197 nats, not
  ln(22): only 9 experts are ever active per sample. Quote ln(22) as the
  unrestricted ceiling but compare against ln(9) for what top-k actually
  permits, or the entropy will look artificially low.""".rstrip())
        return

    print(f"  loading: {found}")
    try:
        if found.endswith(".npz"):
            z = np.load(found, allow_pickle=True)
            key = [k for k in z.files][0]
            G = np.asarray(z[key], dtype=float)
            print(f"  npz keys: {list(z.files)}; using {key!r}")
        elif found.endswith(".npy"):
            G = np.asarray(np.load(found, allow_pickle=True), dtype=float)
        else:
            import torch
            obj = torch.load(found, map_location="cpu")
            G = np.asarray(obj if not isinstance(obj, dict)
                           else list(obj.values())[0], dtype=float)
    except Exception as e:
        print(f"  FAILED to load ({type(e).__name__}: {e}). Pass --gate-vectors explicitly.")
        return

    if G.ndim != 2:
        print(f"  unexpected shape {G.shape}; expected (n_users, E). Aborting.")
        return

    n_users, E = G.shape
    P = G / np.clip(G.sum(axis=1, keepdims=True), 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(P > 0, P * np.log(P), 0.0), axis=1)
    mx = P.max(axis=1)
    n_active = (G > 0).sum(axis=1)

    print(f"  users={n_users}  experts={E}")
    print(f"  max gate weight   : mean={mx.mean():.4f}  min={mx.min():.4f}  "
          f"max={mx.max():.4f}")
    print(f"  routing entropy   : mean={ent.mean():.4f}  min={ent.min():.4f}  "
          f"max={ent.max():.4f} nats")
    print(f"  active experts    : mean={n_active.mean():.2f}  min={n_active.min()}  "
          f"max={n_active.max()}")
    print(f"  ceilings          : ln({E})={np.log(E):.4f} (unrestricted), "
          f"ln(9)={np.log(9):.4f} (top-k=9)")
    for thr in (0.5, 0.8):
        n = int((mx > thr).sum())
        print(f"  subjects with >{thr:.0%} mass on one expert: {n}/{n_users}")
    verdict = ("no subject's routing concentrates on a single expert"
               if (mx > 0.5).sum() == 0 else
               f"{int((mx>0.5).sum())} subject(s) concentrate >50% on one expert -- "
               "report and discuss rather than claiming none")
    print(f"  -> {verdict}")


# ---------------------------------------------------------------------------
# optuna
# ---------------------------------------------------------------------------

def check_optuna():
    hdr("OPTUNA -- studies backing Table 2's expert search space (R1 Q1)")
    print("  (you said the expert-count item is already handled; this only "
          "locates the artifacts)")
    dbs = sorted(glob.glob(str(CODE_DIR / "dataset/optuna_dbs/**/*"), recursive=True))
    logs = [d for d in dbs if d.endswith(".log")]
    sqlite = [d for d in dbs if d.endswith((".db", ".sqlite", ".sqlite3"))]
    print(f"  logs   : {len(logs)}")
    for d in logs[:8]:
        print(f"    {Path(d).relative_to(CODE_DIR)}")
    if len(logs) > 8:
        print(f"    ... and {len(logs)-8} more")
    print(f"  sqlite study DBs: {len(sqlite)}")
    for d in sqlite:
        print(f"    {Path(d).relative_to(CODE_DIR)}")
    if not sqlite:
        print("""
    No .db/.sqlite study found in the repo, so the distribution type cannot be
    read back from a study object here -- only from the logs, which record
    sampled values rather than the declared search space. The authoritative
    source is the `suggest_*` call in the HPO script:
        system/NOTS/paper/ablations/M0_MOE_hpo.py
        system/NOTS/paper/ablations/ablation_hpo.py
    grep for num_experts / suggest_int / suggest_categorical there.""".rstrip())
    hits = []
    for f in ("system/NOTS/paper/ablations/M0_MOE_hpo.py",
              "system/NOTS/paper/ablations/ablation_hpo.py"):
        p = CODE_DIR / f
        if p.exists():
            lines = p.read_text().splitlines()
            for i, ln in enumerate(lines, 1):
                if "num_experts" in ln and "suggest" in ln:
                    # The value list often continues on following lines. Keep
                    # appending until a bracketed list is present -- testing for
                    # a bare "]" fails because config["num_experts"] has one.
                    blob = ln.strip()
                    j = i
                    while (not re.search(r"\[[0-9][0-9,.\s]*\]", blob)
                           and j < len(lines) and j - i < 4):
                        blob += " " + lines[j].strip()
                        j += 1
                    hits.append((f, i, blob))
    if hits:
        print("\n  declared search spaces for num_experts:")
        contains_22 = []
        for f, i, blob in hits:
            m = re.search(r"\[([0-9,.\s]+)\]", blob)
            vals = ([int(float(v)) for v in m.group(1).split(",") if v.strip()]
                    if m else [])
            has22 = 22 in vals
            if has22:
                contains_22.append((f, i, vals))
            kind = ("suggest_categorical" if "categorical" in blob
                    else "suggest_int" if "suggest_int" in blob else "?")
            print(f"    {f}:{i}")
            print(f"      type={kind}  contains 22: {'YES' if has22 else 'no'}")
            print(f"      space={vals if vals else blob}")

        print("""
  READ THIS BEFORE EDITING TABLE 2. The space is suggest_CATEGORICAL, not an
  integer-uniform range. A drafted correction that says "int-uniform [4, 32]"
  is therefore wrong, and R1 asked specifically how 22 was chosen -- this is a
  factual claim to a reviewer who will check it.""".rstrip())
        if contains_22:
            f, i, vals = contains_22[0]
            print(f"""
  E=22 is reachable only from {Path(f).name}:{i}, whose space is
      {vals}
  i.e. a SECOND-STAGE refined categorical search. Note it starts at 20, so the
  refined stage never reconsidered E=8. That is fine and it directly supports
  the drafted line "we selected the higher value because prior MoE work
  supported larger expert banks" -- but it does mean the plateau argument has to
  rest on the Figure 3 post-hoc sweep, not on the HPO search, since the HPO
  search that produced 22 could not have chosen 8.

  Suggested Table 2 row, phrased so it is true of the code:
      Num. experts E | categorical (two-stage) | stage 2: {vals} | 22""")
        else:
            print("""
  None of the declared spaces contains 22, which needs explaining before any
  Table 2 edit ships. Check for a third HPO script or a hand-set override.""")


CHECKS = {"norm": check_norm, "lpft": check_lpft, "qcount": check_qcount,
          "gates": None, "optuna": check_optuna}


def main():
    ap = argparse.ArgumentParser(description="Desk checks for the rebuttal [[FILL]]s.")
    ap.add_argument("--only", choices=list(CHECKS), default=None)
    ap.add_argument("--gate-vectors", default=None)
    args = ap.parse_args()

    order = [args.only] if args.only else list(CHECKS)
    for name in order:
        if name == "gates":
            check_gates(args.gate_vectors)
        else:
            CHECKS[name]()
    print()


if __name__ == "__main__":
    main()
