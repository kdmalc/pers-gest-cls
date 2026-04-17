"""
moe_analysis.py
===============
Tools for understanding and visualising expert routing in MoE models.

Core idea: collect (gate_weights, pid, gesture_label, demographics) tuples
across a full val/test pass, then ask:
  - Do different participants consistently land on different experts?
  - Do different gestures cluster into different experts?
  - Do demographic groups (e.g. disability status) route differently?

Public API
──────────
  RoutingCollector          : context manager / accumulator — wraps any eval loop
  RoutingAnalyzer           : runs all analyses on a collected RoutingRecord
  run_routing_analysis()    : one-shot convenience wrapper

  log_routing_to_wandb()    : optional W&B integration
  save_routing_record()     : save to .pt file for later analysis
  load_routing_record()     : load from .pt file

Usage
─────
    from moe_analysis import RoutingCollector, RoutingAnalyzer

    collector = RoutingCollector(num_experts=4, model_name="DeepCNNLSTM_Middle")

    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            emg    = batch["emg"].to(device)
            imu    = batch["imu"].to(device)
            labels = batch["labels"]
            pids   = batch["pid"]          # str list / int list
            demo   = batch.get("demographics")

            logits, routing_info = model(emg, imu, return_routing=True)
            gate_w = routing_info["gate_weights"]   # (B, E)

            collector.add(
                gate_weights = gate_w.cpu(),
                gesture_labels = labels,
                pids           = pids,
                demographics   = demo.cpu() if demo is not None else None,
            )

    record  = collector.finalize()
    analyzer = RoutingAnalyzer(record)
    report   = analyzer.full_report(print_report=True)
    figs     = analyzer.plot_all()          # dict of matplotlib figures
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoutingRecord:
    """
    All routing information collected over one eval pass.

    Attributes
    ----------
    gate_weights   : (N, E) float32 — soft routing weights per sample
    gesture_labels : (N,) int64 — true gesture class (0-indexed)
    pids           : list[str] length N — participant IDs ("P102", ...)
    demographics   : (N, D) float32 | None — raw demographics vector if available
    model_name     : str — tag for logging
    num_experts    : int
    """
    gate_weights   : torch.Tensor
    gesture_labels : torch.Tensor
    pids           : List[str]
    demographics   : Optional[torch.Tensor] = None
    model_name     : str = "model"
    num_experts    : int = 4


# ─────────────────────────────────────────────────────────────────────────────
# Collector
# ─────────────────────────────────────────────────────────────────────────────

class RoutingCollector:
    """
    Accumulates routing data across mini-batches.

    Usage — see module docstring.
    """

    def __init__(self, num_experts: int, model_name: str = "model"):
        self.num_experts = num_experts
        self.model_name  = model_name
        self._gate_weights:    List[torch.Tensor] = []
        self._gesture_labels:  List[torch.Tensor] = []
        self._pids:            List[str]          = []
        self._demographics:    List[torch.Tensor] = []

    def add(self,
            gate_weights:    torch.Tensor,
            gesture_labels:  torch.Tensor,
            pids:            List,
            demographics:    Optional[torch.Tensor] = None) -> None:
        """
        Add one batch.

        gate_weights   : (B, E) — routing weights (cpu tensor)
        gesture_labels : (B,)  — integer labels (cpu tensor or list)
        pids           : list of str or int of length B
        demographics   : (B, D) or None
        """
        self._gate_weights.append(gate_weights.detach().float())
        if not isinstance(gesture_labels, torch.Tensor):
            gesture_labels = torch.tensor(gesture_labels, dtype=torch.long)
        self._gesture_labels.append(gesture_labels.cpu().long())
        # Normalise pid to str
        self._pids.extend([str(p) for p in pids])
        if demographics is not None:
            self._demographics.append(demographics.detach().float().cpu())

    def finalize(self) -> RoutingRecord:
        """Concatenate all accumulated batches into a RoutingRecord."""
        gw = torch.cat(self._gate_weights, dim=0)
        gl = torch.cat(self._gesture_labels, dim=0)
        demo = torch.cat(self._demographics, dim=0) if self._demographics else None
        return RoutingRecord(
            gate_weights=gw,
            gesture_labels=gl,
            pids=self._pids,
            demographics=demo,
            model_name=self.model_name,
            num_experts=self.num_experts,
        )

    def reset(self) -> None:
        self._gate_weights.clear()
        self._gesture_labels.clear()
        self._pids.clear()
        self._demographics.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class RoutingAnalyzer:
    """
    Runs routing analysis on a RoutingRecord and returns dicts / matplotlib figs.

    All numeric outputs are plain Python floats / lists so they are
    JSON-serialisable and easy to log.
    """

    def __init__(self, record: RoutingRecord):
        self.rec = record
        self.E   = record.num_experts
        self.N   = record.gate_weights.shape[0]

        # Hard assignment: dominant expert per sample
        self.dominant_expert = record.gate_weights.argmax(dim=-1)  # (N,)

        # Unique values
        self.unique_gestures = sorted(record.gesture_labels.unique().tolist())
        self.unique_pids     = sorted(set(record.pids))

        self._pid_to_idx = {p: i for i, p in enumerate(self.unique_pids)}

    # ── Entropy ──────────────────────────────────────────────────────────────

    def routing_entropy(self) -> Dict[str, float]:
        """
        Per-sample routing entropy, then averaged.

        A perfectly flat distribution (all experts equal) has entropy log(E).
        A hard (one-hot) distribution has entropy 0.
        Values closer to 0 → sharper routing → experts more specialised.
        """
        w = self.rec.gate_weights.clamp(min=1e-9)
        h = -(w * w.log()).sum(dim=-1)        # (N,) — nats
        max_h = math.log(self.E)
        return {
            "mean_entropy_nats":      h.mean().item(),
            "mean_entropy_normalised": (h.mean() / max_h).item(),  # 0=sharp, 1=flat
            "std_entropy_nats":       h.std().item(),
        }

    # ── Load balance ─────────────────────────────────────────────────────────

    def load_balance(self) -> Dict[str, Any]:
        """
        Expert utilisation statistics.

        ideal_fraction = 1/E.  Values far from ideal → over/under-used experts.
        """
        # Fraction of samples where each expert is dominant
        #hard_frac = torch.zeros(self.E)
        #for e in range(self.E):
        #    hard_frac[e] = (self.dominant_expert == e).float().mean()
        hard_frac = torch.bincount(self.dominant_expert, minlength=self.E).float() / self.N


        # Mean soft weight per expert
        soft_frac = self.rec.gate_weights.mean(dim=0)

        ideal = 1.0 / self.E
        return {
            "expert_hard_fraction":  hard_frac.tolist(),
            "expert_soft_fraction":  soft_frac.tolist(),
            "ideal_fraction":        ideal,
            "hard_imbalance_ratio":  (hard_frac.max() / hard_frac.min().clamp(min=1e-9)).item(),
            "soft_imbalance_ratio":  (soft_frac.max() / soft_frac.min().clamp(min=1e-9)).item(),
        }

    # ── Gesture routing ──────────────────────────────────────────────────────

    def routing_by_gesture(self) -> Dict[str, Any]:
        """
        Mean gate weight matrix:  gesture × expert  (G × E).

        Tells you which expert(s) each gesture prefers.
        A row with a single large value → that gesture is routed sharply.
        """
        G = len(self.unique_gestures)
        weight_mat = np.zeros((G, self.E))
        dominant_mat = np.zeros((G, self.E))   # dominant expert frequency

        labels = self.rec.gesture_labels.numpy()
        gw     = self.rec.gate_weights.numpy()
        dom    = self.dominant_expert.numpy()

        for g_idx, g in enumerate(self.unique_gestures):
            mask = labels == g
            if mask.sum() == 0:
                continue
            weight_mat[g_idx]   = gw[mask].mean(axis=0)
            for e in range(self.E):
                dominant_mat[g_idx, e] = (dom[mask] == e).mean()

        return {
            "gesture_ids":          self.unique_gestures,
            "mean_weight_matrix":   weight_mat.tolist(),   # (G, E)
            "dominant_freq_matrix": dominant_mat.tolist(), # (G, E)
        }

    # ── Participant routing ───────────────────────────────────────────────────

    def routing_by_pid(self) -> Dict[str, Any]:
        """
        Mean gate weight per participant: pid × expert  (P × E).

        Tells you whether different users consistently land on different experts.
        High between-user variance → experts may specialise by user anatomy.
        """
        P = len(self.unique_pids)
        weight_mat = np.zeros((P, self.E))
        dominant_mat = np.zeros((P, self.E))

        pids_arr = np.array(self.rec.pids)
        gw       = self.rec.gate_weights.numpy()
        dom      = self.dominant_expert.numpy()

        for p_idx, pid in enumerate(self.unique_pids):
            mask = pids_arr == pid
            if mask.sum() == 0:
                continue
            weight_mat[p_idx]   = gw[mask].mean(axis=0)
            for e in range(self.E):
                dominant_mat[p_idx, e] = (dom[mask] == e).mean()

        return {
            "pid_list":             self.unique_pids,
            "mean_weight_matrix":   weight_mat.tolist(),   # (P, E)
            "dominant_freq_matrix": dominant_mat.tolist(), # (P, E)
        }

    # ── Demographics routing ─────────────────────────────────────────────────

    def routing_by_demographics(self,
                                demo_dim_labels: Optional[List[str]] = None
                                ) -> Optional[Dict[str, Any]]:
        """
        Pearson correlation between each demographics dimension and each expert's
        mean gate weight.  (D × E) correlation matrix.

        A large |correlation| for dimension d and expert e suggests that
        expert e specialises based on that demographic feature.

        Returns None if no demographics were collected.
        """
        if self.rec.demographics is None:
            return None

        demo = self.rec.demographics.float()   # (N, D)
        gw   = self.rec.gate_weights.float()   # (N, E)
        D    = demo.shape[1]

        corr = np.zeros((D, self.E))
        for d_idx in range(D):
            d_col = demo[:, d_idx].numpy()
            for e in range(self.E):
                w_col = gw[:, e].numpy()
                if d_col.std() < 1e-8 or w_col.std() < 1e-8:
                    corr[d_idx, e] = 0.0
                else:
                    corr[d_idx, e] = float(np.corrcoef(d_col, w_col)[0, 1])

        labels = demo_dim_labels or [f"demo_{i}" for i in range(D)]
        return {
            "demo_dim_labels": labels,
            "correlation_matrix": corr.tolist(),   # (D, E)
        }

    # ── Expert co-activation ─────────────────────────────────────────────────

    def expert_coactivation(self) -> Dict[str, Any]:
        """
        Pearson correlation matrix between expert weights (E × E).

        Large positive correlation → those experts tend to be activated together.
        Negative correlation → they anti-correlate (more specialised).
        """
        gw = self.rec.gate_weights.float()
        corr = np.corrcoef(gw.T.numpy())  # (E, E)
        return {"coactivation_matrix": corr.tolist()}

    # ── Full report ──────────────────────────────────────────────────────────

    def full_report(self, print_report: bool = True,
                    demo_dim_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run all analyses and return a unified dict.
        Optionally pretty-prints a summary.
        """
        report = {
            "model_name":           self.rec.model_name,
            "num_samples":          self.N,
            "num_experts":          self.E,
            "entropy":              self.routing_entropy(),
            "load_balance":         self.load_balance(),
            "routing_by_gesture":   self.routing_by_gesture(),
            "routing_by_pid":       self.routing_by_pid(),
            "expert_coactivation":  self.expert_coactivation(),
        }
        demo_info = self.routing_by_demographics(demo_dim_labels)
        if demo_info is not None:
            report["routing_by_demographics"] = demo_info

        if print_report:
            _print_report(report, self.E)

        return report

    # ── Plotting ─────────────────────────────────────────────────────────────

    def plot_all(self, save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate all diagnostic plots.  Returns a dict of {name: Figure}.
        If save_dir is given, also saves PNG files there.

        Requires matplotlib.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")   # non-interactive backend safe for HPC
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            print("[moe_analysis] matplotlib not found — skipping plots.")
            return {}

        figs: Dict[str, Any] = {}
        save_path = Path(save_dir) if save_dir else None
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        # ── 1. Load balance bar chart ────────────────────────────────────────
        lb  = self.load_balance()
        fig, ax = plt.subplots(figsize=(max(4, self.E), 4))
        x   = np.arange(self.E)
        ax.bar(x - 0.2, lb["expert_soft_fraction"], 0.35, label="Mean soft weight", color="#4C72B0")
        ax.bar(x + 0.2, lb["expert_hard_fraction"], 0.35, label="Dominant expert freq", color="#DD8452")
        ax.axhline(lb["ideal_fraction"], color="gray", linestyle="--", label=f"Ideal (1/E={lb['ideal_fraction']:.2f})")
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Fraction")
        ax.set_title(f"Expert load balance — {self.rec.model_name}")
        ax.set_xticks(x); ax.set_xticklabels([f"E{i}" for i in range(self.E)])
        ax.legend(); fig.tight_layout()
        figs["load_balance"] = fig
        if save_path:
            fig.savefig(save_path / "load_balance.png", dpi=150)
        plt.close(fig)

        # ── 2. Gesture × Expert heatmap ──────────────────────────────────────
        rg   = self.routing_by_gesture()
        mat  = np.array(rg["mean_weight_matrix"])
        fig, ax = plt.subplots(figsize=(max(4, self.E + 1), max(4, len(self.unique_gestures))))
        im = ax.imshow(mat, aspect="auto", vmin=0, cmap="YlOrRd")
        ax.set_xticks(range(self.E)); ax.set_xticklabels([f"E{i}" for i in range(self.E)])
        ax.set_yticks(range(len(self.unique_gestures)))
        ax.set_yticklabels([f"G{g}" for g in rg["gesture_ids"]])
        ax.set_xlabel("Expert"); ax.set_ylabel("Gesture class")
        ax.set_title(f"Mean gate weight — gesture × expert\n{self.rec.model_name}")
        plt.colorbar(im, ax=ax); fig.tight_layout()
        figs["gesture_expert_heatmap"] = fig
        if save_path:
            fig.savefig(save_path / "gesture_expert_heatmap.png", dpi=150)
        plt.close(fig)

        # ── 3. Participant × Expert heatmap ──────────────────────────────────
        rp   = self.routing_by_pid()
        mat  = np.array(rp["mean_weight_matrix"])
        fig, ax = plt.subplots(figsize=(max(4, self.E + 1), max(4, len(self.unique_pids))))
        im = ax.imshow(mat, aspect="auto", vmin=0, cmap="Blues")
        ax.set_xticks(range(self.E)); ax.set_xticklabels([f"E{i}" for i in range(self.E)])
        ax.set_yticks(range(len(self.unique_pids)))
        ax.set_yticklabels(rp["pid_list"], fontsize=8)
        ax.set_xlabel("Expert"); ax.set_ylabel("Participant")
        ax.set_title(f"Mean gate weight — participant × expert\n{self.rec.model_name}")
        plt.colorbar(im, ax=ax); fig.tight_layout()
        figs["pid_expert_heatmap"] = fig
        if save_path:
            fig.savefig(save_path / "pid_expert_heatmap.png", dpi=150)
        plt.close(fig)

        # ── 4. Demographics × Expert correlation (if available) ───────────────
        if self.rec.demographics is not None:
            rd  = self.routing_by_demographics()
            mat = np.array(rd["correlation_matrix"])
            fig, ax = plt.subplots(figsize=(max(4, self.E + 1), max(4, mat.shape[0])))
            vlim = max(abs(mat).max(), 0.1)
            im = ax.imshow(mat, aspect="auto", vmin=-vlim, vmax=vlim, cmap="RdBu_r")
            ax.set_xticks(range(self.E)); ax.set_xticklabels([f"E{i}" for i in range(self.E)])
            ax.set_yticks(range(mat.shape[0]))
            ax.set_yticklabels(rd["demo_dim_labels"], fontsize=8)
            ax.set_xlabel("Expert"); ax.set_ylabel("Demographics dim")
            ax.set_title(f"Demographics–expert correlation\n{self.rec.model_name}")
            plt.colorbar(im, ax=ax); fig.tight_layout()
            figs["demo_expert_correlation"] = fig
            if save_path:
                fig.savefig(save_path / "demo_expert_correlation.png", dpi=150)
            plt.close(fig)

        # ── 5. Per-sample entropy histogram ──────────────────────────────────
        w   = self.rec.gate_weights.clamp(min=1e-9)
        h   = -(w * w.log()).sum(dim=-1).numpy()
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(h, bins=30, color="#4C72B0", edgecolor="white")
        ax.axvline(math.log(self.E), color="red", linestyle="--",
                   label=f"Max entropy (log {self.E} = {math.log(self.E):.2f})")
        ax.set_xlabel("Routing entropy (nats)")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution of per-sample routing entropy\n{self.rec.model_name}")
        ax.legend(); fig.tight_layout()
        figs["entropy_histogram"] = fig
        if save_path:
            fig.savefig(save_path / "entropy_histogram.png", dpi=150)
        plt.close(fig)

        # ── 6. Expert co-activation matrix ───────────────────────────────────
        ca  = self.expert_coactivation()
        mat = np.array(ca["coactivation_matrix"])
        fig, ax = plt.subplots(figsize=(max(4, self.E + 1), max(4, self.E + 1)))
        im = ax.imshow(mat, aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(self.E)); ax.set_xticklabels([f"E{i}" for i in range(self.E)])
        ax.set_yticks(range(self.E)); ax.set_yticklabels([f"E{i}" for i in range(self.E)])
        ax.set_title(f"Expert co-activation (correlation)\n{self.rec.model_name}")
        for i in range(self.E):
            for j in range(self.E):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax); fig.tight_layout()
        figs["coactivation_matrix"] = fig
        if save_path:
            fig.savefig(save_path / "coactivation_matrix.png", dpi=150)
        plt.close(fig)

        if save_path and figs:
            print(f"[moe_analysis] Saved {len(figs)} plots to {save_path}")

        return figs


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run_routing_analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_routing_analysis(model: "nn.Module",
                         dataloader,
                         device: str,
                         num_experts: int,
                         model_name: str = "model",
                         demo_dim_labels: Optional[List[str]] = None,
                         save_dir: Optional[str] = None,
                         print_report: bool = True,
                         use_imu: bool = True) -> Tuple[Dict, Dict]:
    """
    One-shot wrapper: runs a full eval pass with routing collection,
    then returns (report_dict, figures_dict).

    The dataloader must yield dicts with keys:
      "emg"    : (B, C, T)
      "labels" : (B,)
      "pid"    : list[str]  (or "pids" — both checked)
      "imu"    : (B, C, T) [optional]
      "demographics" : (B, D) [optional]

    Example usage in mamlpp or pretrain eval loop:
        report, figs = run_routing_analysis(
            model         = model,
            dataloader    = val_dl,
            device        = "cuda",
            num_experts   = config["num_experts"],
            model_name    = config["model_type"],
            save_dir      = "routing_plots/",
        )
        print(f"Routing entropy: {report['entropy']['mean_entropy_normalised']:.3f}")
    """
    import torch

    collector = RoutingCollector(num_experts=num_experts, model_name=model_name)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            emg    = batch["emg"].to(device)
            labels = batch["labels"].cpu()
            imu    = batch.get("imu")
            if use_imu and imu is not None:
                imu = imu.to(device)
            else:
                imu = None

            pids = batch.get("user_id", batch.get("pid", batch.get("pids", ["unknown"] * emg.size(0))))
            # Also handle the case where user_id is a single string (episodic):
            if isinstance(pids, str):
                pids = [pids] * emg.size(0)
            demo = batch.get("demographics")

            # Support both model output styles
            out = model(emg, imu, return_routing=True)
            if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
                _logits, routing_info = out
                gate_w = routing_info["gate_weights"].cpu()
            else:
                print("[run_routing_analysis] Warning: model did not return routing info. "
                      "Make sure return_routing=True is handled in the model's forward().")
                break

            collector.add(
                gate_weights   = gate_w,
                gesture_labels = labels,
                pids           = pids,
                demographics   = demo.cpu() if demo is not None else None,
            )

    record   = collector.finalize()
    analyzer = RoutingAnalyzer(record)
    report   = analyzer.full_report(print_report=print_report,
                                    demo_dim_labels=demo_dim_labels)
    figs     = analyzer.plot_all(save_dir=save_dir)
    return report, figs


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_routing_record(record: RoutingRecord, path: str) -> None:
    """Save a RoutingRecord to a .pt file."""
    torch.save({
        "gate_weights":   record.gate_weights,
        "gesture_labels": record.gesture_labels,
        "pids":           record.pids,
        "demographics":   record.demographics,
        "model_name":     record.model_name,
        "num_experts":    record.num_experts,
    }, path)
    print(f"[moe_analysis] Saved RoutingRecord to {path}")


def load_routing_record(path: str) -> RoutingRecord:
    """Load a RoutingRecord from a .pt file."""
    d = torch.load(path, map_location="cpu")
    return RoutingRecord(
        gate_weights   = d["gate_weights"],
        gesture_labels = d["gesture_labels"],
        pids           = d["pids"],
        demographics   = d.get("demographics"),
        model_name     = d.get("model_name", "model"),
        num_experts    = d.get("num_experts", d["gate_weights"].shape[1]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optional: W&B logging
# ─────────────────────────────────────────────────────────────────────────────

def log_routing_to_wandb(report: Dict, step: Optional[int] = None) -> None:
    """
    Log a routing report dict to Weights & Biases.

    Logs flat scalars; matrices are skipped (log as images separately if needed).
    """
    try:
        import wandb
    except ImportError:
        print("[moe_analysis] wandb not installed — skipping wandb logging.")
        return

    flat = {}

    ent = report.get("entropy", {})
    flat["routing/entropy_mean"] = ent.get("mean_entropy_normalised", 0.0)
    flat["routing/entropy_std"]  = ent.get("std_entropy_nats", 0.0)

    lb = report.get("load_balance", {})
    flat["routing/imbalance_ratio_hard"] = lb.get("hard_imbalance_ratio", 1.0)
    flat["routing/imbalance_ratio_soft"] = lb.get("soft_imbalance_ratio", 1.0)
    for e_idx, frac in enumerate(lb.get("expert_hard_fraction", [])):
        flat[f"routing/expert_{e_idx}_dom_freq"] = frac

    wandb.log(flat, step=step)


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-printer
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(report: Dict, E: int) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  MoE Routing Analysis — {report['model_name']}")
    print(f"  N={report['num_samples']}  E={report['num_experts']}")
    print(sep)

    ent = report["entropy"]
    print(f"\n[Entropy]")
    print(f"  Mean (normalised 0→sharp, 1→flat): {ent['mean_entropy_normalised']:.4f}")
    print(f"  Mean (nats):                        {ent['mean_entropy_nats']:.4f}")
    print(f"  Std  (nats):                        {ent['std_entropy_nats']:.4f}")

    lb = report["load_balance"]
    print(f"\n[Load Balance]  (ideal = {lb['ideal_fraction']:.3f})")
    print(f"  Hard imbalance ratio (max/min dom freq): {lb['hard_imbalance_ratio']:.2f}x")
    print(f"  Soft imbalance ratio (max/min mean wgt): {lb['soft_imbalance_ratio']:.2f}x")
    for e_idx in range(E):
        hf = lb["expert_hard_fraction"][e_idx]
        sf = lb["expert_soft_fraction"][e_idx]
        print(f"    Expert {e_idx}: dom_freq={hf:.3f}  mean_wgt={sf:.3f}")

    rg = report["routing_by_gesture"]
    print(f"\n[Routing by Gesture]  (dominant expert per gesture)")
    mat = np.array(rg["dominant_freq_matrix"])
    for g_idx, g in enumerate(rg["gesture_ids"]):
        dom_e = int(np.argmax(mat[g_idx]))
        dom_f = mat[g_idx, dom_e]
        print(f"    Gesture {g:2d} → mostly Expert {dom_e} ({dom_f*100:.1f}% of samples)")

    rp = report["routing_by_pid"]
    print(f"\n[Routing by Participant]  (dominant expert per PID)")
    mat = np.array(rp["dominant_freq_matrix"])
    for p_idx, pid in enumerate(rp["pid_list"]):
        dom_e = int(np.argmax(mat[p_idx]))
        dom_f = mat[p_idx, dom_e]
        print(f"    {pid} → mostly Expert {dom_e} ({dom_f*100:.1f}%)")

    if "routing_by_demographics" in report:
        rd = report["routing_by_demographics"]
        print(f"\n[Routing by Demographics]  (Pearson |r| > 0.1)")
        corr = np.array(rd["correlation_matrix"])
        for d_idx, label in enumerate(rd["demo_dim_labels"]):
            for e_idx in range(E):
                r_val = corr[d_idx, e_idx]
                if abs(r_val) > 0.1:
                    print(f"    {label} ↔ Expert {e_idx}: r={r_val:+.3f}")

    ca = report["expert_coactivation"]
    print(f"\n[Expert Co-activation] (off-diagonal Pearson r, |r| > 0.3)")
    mat = np.array(ca["coactivation_matrix"])
    for i in range(E):
        for j in range(i + 1, E):
            r_val = mat[i, j]
            if abs(r_val) > 0.3:
                print(f"    Expert {i} ↔ Expert {j}: r={r_val:+.3f}")

    print(f"\n{sep}\n")