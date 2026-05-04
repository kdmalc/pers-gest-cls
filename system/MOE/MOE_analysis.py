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
        Per-sample routing entropy summary statistics.

        A perfectly flat distribution (all experts equal) has entropy log(E).
        A hard (one-hot) distribution has entropy 0.
        Values closer to 0 → sharper routing → experts more specialised.

        Note: the full per-sample entropy array is also returned by load_balance()
        under the key 'per_sample_entropy' for histogram plotting.
        """
        w     = self.rec.gate_weights.clamp(min=1e-9)
        h     = -(w * w.log()).sum(dim=-1)   # (N,) — nats
        max_h = math.log(self.E)
        return {
            "mean_entropy_nats":       h.mean().item(),
            "mean_entropy_normalised": (h.mean() / max_h).item(),  # 0=sharp, 1=flat
            "std_entropy_nats":        h.std().item(),
            "max_entropy_nats":        max_h,
        }

    # ── Load balance ─────────────────────────────────────────────────────────

    def load_balance(self, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Expert utilisation statistics.

        Four complementary metrics are returned:

        expert_soft_fraction   : mean gate weight per expert across all N samples.
                                 Ideal = top_k / E  (each selected expert shares
                                 weight equally).  Sensitive to weight magnitude,
                                 not just binary selection.

        expert_selection_freq  : fraction of samples in which each expert appears
                                 in the top-k set (gate_weight > 0).  This is the
                                 standard load-balance metric reported in the MoE
                                 literature (Shazeer 2017, Fedus 2022).
                                 Ideal = top_k / E.

        expert_dominant_freq   : fraction of samples where each expert has the
                                 single highest gate weight (argmax).  Useful for
                                 spotting a single "superstar" expert but can be
                                 misleading when weights are close — use as a
                                 diagnostic, not a primary metric.
                                 Ideal = 1 / E.

        per_sample_entropy     : per-sample routing entropy (nats), returned as a
                                 list so the caller can plot a histogram.
                                 max possible = log(E); 0 = perfectly hard routing.
        """
        gw = self.rec.gate_weights  # (N, E)

        # ── 1. Mean soft weight ──────────────────────────────────────────────
        soft_frac = gw.mean(dim=0)  # (E,)

        # ── 2. Selection frequency (top-k binary mask) ───────────────────────
        # gate_weights are already zeroed for non-selected experts by the router,
        # so any positive entry means "this expert was in the top-k set".
        selection_mask = (gw > 0).float()           # (N, E) — 1 if selected
        selection_freq = selection_mask.mean(dim=0)  # (E,)

        # Infer top_k from data if not supplied (median number of nonzero experts
        # per sample — robust to occasional all-zero rows from skipped episodes).
        if top_k is None:
            top_k = int(selection_mask.sum(dim=1).median().item())

        # ── 3. Dominant-expert frequency (argmax) ────────────────────────────
        dominant_freq = (
            torch.bincount(self.dominant_expert, minlength=self.E).float() / self.N
        )  # (E,)

        # ── 4. Per-sample entropy (nats) ─────────────────────────────────────
        w_clamped = gw.clamp(min=1e-9)
        per_sample_entropy = -(w_clamped * w_clamped.log()).sum(dim=-1)  # (N,)

        ideal_selection = top_k / self.E   # ideal for soft_frac & selection_freq
        ideal_dominant  = 1.0  / self.E   # ideal for dominant_freq

        return {
            # per-expert vectors
            "expert_soft_fraction":   soft_frac.tolist(),
            "expert_selection_freq":  selection_freq.tolist(),
            "expert_dominant_freq":   dominant_freq.tolist(),
            # per-sample entropy distribution
            "per_sample_entropy":     per_sample_entropy.tolist(),
            # ideals & summary scalars
            "ideal_selection_fraction": ideal_selection,
            "ideal_dominant_fraction":  ideal_dominant,
            "top_k":                    top_k,
            "selection_imbalance_ratio": (
                selection_freq.max() / selection_freq.clamp(min=1e-9).min()
            ).item(),
            "soft_imbalance_ratio": (
                soft_frac.max() / soft_frac.clamp(min=1e-9).min()
            ).item(),
            "dominant_imbalance_ratio": (
                dominant_freq.max() / dominant_freq.clamp(min=1e-9).min()
            ).item(),
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

        # ── 1a. Mean soft weight per expert ─────────────────────────────────
        lb  = self.load_balance()
        x   = np.arange(self.E)
        xlabels = [f"E{i}" for i in range(self.E)]
        fig_w = max(8, self.E * 0.55 + 2)

        fig, ax = plt.subplots(figsize=(fig_w, 4))
        bars = ax.bar(x, lb["expert_soft_fraction"], color="#4C72B0", edgecolor="white", linewidth=0.5)
        ax.axhline(
            lb["ideal_selection_fraction"], color="gray", linestyle="--",
            label=f"Ideal = top_k/E = {lb['top_k']}/{self.E} = {lb['ideal_selection_fraction']:.3f}",
        )
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Mean gate weight")
        ax.set_title(
            f"Mean Soft Gate Weight per Expert\n"
            f"N={self.N} samples | E={self.E} | top_k={lb['top_k']} | "
            f"imbalance ratio={lb['soft_imbalance_ratio']:.1f}x"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
        ax.legend()
        fig.tight_layout()
        figs["lb_soft_weight"] = fig
        if save_path:
            fig.savefig(save_path / "lb_soft_weight.png", dpi=150)
        plt.close(fig)

        # ── 1b. Expert selection frequency (top-k binary) ────────────────────
        fig, ax = plt.subplots(figsize=(fig_w, 4))
        ax.bar(x, lb["expert_selection_freq"], color="#55A868", edgecolor="white", linewidth=0.5)
        ax.axhline(
            lb["ideal_selection_fraction"], color="gray", linestyle="--",
            label=f"Ideal = top_k/E = {lb['top_k']}/{self.E} = {lb['ideal_selection_fraction']:.3f}",
        )
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Selection frequency")
        ax.set_title(
            f"Expert Selection Frequency (top-k binary)\n"
            f"N={self.N} samples | E={self.E} | top_k={lb['top_k']} | "
            f"imbalance ratio={lb['selection_imbalance_ratio']:.1f}x\n"
            f"Fraction of samples in which each expert appears in the top-{lb['top_k']} set"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
        ax.legend()
        fig.tight_layout()
        figs["lb_selection_freq"] = fig
        if save_path:
            fig.savefig(save_path / "lb_selection_freq.png", dpi=150)
        plt.close(fig)

        # ── 1c. Dominant expert frequency (argmax) ───────────────────────────
        fig, ax = plt.subplots(figsize=(fig_w, 4))
        ax.bar(x, lb["expert_dominant_freq"], color="#DD8452", edgecolor="white", linewidth=0.5)
        ax.axhline(
            lb["ideal_dominant_fraction"], color="gray", linestyle="--",
            label=f"Ideal = 1/E = 1/{self.E} = {lb['ideal_dominant_fraction']:.3f}",
        )
        ax.set_xlabel("Expert index")
        ax.set_ylabel("Dominant frequency")
        ax.set_title(
            f"Dominant Expert Frequency (argmax winner per sample)\n"
            f"N={self.N} samples | E={self.E} | top_k={lb['top_k']} | "
            f"imbalance ratio={lb['dominant_imbalance_ratio']:.1f}x\n"
            f"Fraction of samples where each expert has the single highest gate weight"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
        ax.legend()
        fig.tight_layout()
        figs["lb_dominant_freq"] = fig
        if save_path:
            fig.savefig(save_path / "lb_dominant_freq.png", dpi=150)
        plt.close(fig)

        # ── 1d. Per-sample routing entropy histogram ─────────────────────────
        ent      = self.routing_entropy()
        h_vals   = np.array(lb["per_sample_entropy"])
        max_h    = ent["max_entropy_nats"]
        mean_h   = ent["mean_entropy_nats"]
        norm_h   = ent["mean_entropy_normalised"]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(h_vals, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.5)
        ax.axvline(
            max_h, color="red", linestyle="--",
            label=f"Max entropy = log({self.E}) = {max_h:.2f} nats (uniform routing)",
        )
        ax.axvline(
            mean_h, color="orange", linestyle="-",
            label=f"Mean = {mean_h:.2f} nats (normalised = {norm_h:.3f})",
        )
        ax.set_xlabel("Routing entropy (nats)")
        ax.set_ylabel("Sample count")
        ax.set_title(
            f"Per-Sample Routing Entropy Distribution\n"
            f"N={self.N} | 0 = perfectly sharp (one expert), "
            f"{max_h:.2f} nats = perfectly uniform\n"
            f"Mean normalised entropy: {norm_h:.3f}  (0=sharp → 1=flat)"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        figs["lb_entropy_histogram"] = fig
        if save_path:
            fig.savefig(save_path / "lb_entropy_histogram.png", dpi=150)
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

        # ── 5. Expert co-activation matrix ───────────────────────────────────
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
    flat["routing/imbalance_ratio_dominant"]  = lb.get("dominant_imbalance_ratio", 1.0)
    flat["routing/imbalance_ratio_soft"]      = lb.get("soft_imbalance_ratio", 1.0)
    flat["routing/imbalance_ratio_selection"] = lb.get("selection_imbalance_ratio", 1.0)
    for e_idx, frac in enumerate(lb.get("expert_dominant_freq", [])):
        flat[f"routing/expert_{e_idx}_dominant_freq"] = frac
    for e_idx, frac in enumerate(lb.get("expert_selection_freq", [])):
        flat[f"routing/expert_{e_idx}_selection_freq"] = frac

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
    print(f"  Max possible (uniform):             {ent['max_entropy_nats']:.4f}")

    lb = report["load_balance"]
    top_k        = lb['top_k']
    ideal_sel    = lb['ideal_selection_fraction']
    ideal_dom    = lb['ideal_dominant_fraction']
    print(f"\n[Load Balance]  top_k={top_k} | ideal_selection=top_k/E={ideal_sel:.3f} | ideal_dominant=1/E={ideal_dom:.3f}")
    print(f"  Selection imbalance ratio (max/min selection_freq): {lb['selection_imbalance_ratio']:.2f}x")
    print(f"  Soft      imbalance ratio (max/min mean_weight):    {lb['soft_imbalance_ratio']:.2f}x")
    print(f"  Dominant  imbalance ratio (max/min dominant_freq):  {lb['dominant_imbalance_ratio']:.2f}x")
    print(f"  {'Expert':<10} {'selection_freq':>16} {'mean_soft_wgt':>15} {'dominant_freq':>15}")
    print(f"  {'-'*10} {'-'*16} {'-'*15} {'-'*15}")
    for e_idx in range(E):
        sf  = lb["expert_selection_freq"][e_idx]
        sw  = lb["expert_soft_fraction"][e_idx]
        df  = lb["expert_dominant_freq"][e_idx]
        print(f"  Expert {e_idx:<4} {sf:>16.3f} {sw:>15.3f} {df:>15.3f}")

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