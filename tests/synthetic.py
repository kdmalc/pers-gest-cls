"""
tests/synthetic.py
==================
Synthetic `tensor_dict` factory so the whole pipeline can be exercised with NO
cluster data.

`rebuttal/REBUTTAL_CODE_FINDINGS.md` §7 describes a synthetic harness that was
used to verify the modality-mask / q_query / gain-sweep changes and then never
committed. This module is that harness, committed, so nobody redoes the work.

On-disk schema (from `system/MAML/maml_data_pipeline.py:7-20`)::

    payload = {
        "data": {pid(str): {gesture_class(int): {
            "emg": Tensor (num_trials, seq_len, C_emg),   # (10, 64, 16)
            "imu": Tensor (num_trials, seq_len, C_imu) | None,
            "demo": Tensor (demo_in_dim,),
            "gest_ID": int,                # == the outer key, 0-indexed
            "rep_indices": [1..num_trials],  # 1-INDEXED rep numbers
        }}},
        "gesture_label_encoder": ..., "pid_encoder": ...,
        "gesture_feature_cols": ..., "pwod_pids": set(),
    }

Note the on-disk orientation is ``(trials, T, C)``.  ``reorient_tensor_dict``
flips it to ``(trials, C, T)`` in place and sets ``payload["_reoriented"]``.

--------------------------------------------------------------------------
TWO INVARIANTS THIS MODULE ENFORCES, BOTH LEARNED THE HARD WAY
--------------------------------------------------------------------------

1.  ``T`` must differ from BOTH channel counts.

    ``reorient_tensor_dict`` decides whether to permute by testing
    ``emg.shape[-1] == config['emg_in_ch']``.  With the real shapes
    (T=64, C_emg=16, C_imu=72) that test is unambiguous.  But a "small"
    fixture such as ``T=16, C_emg=16`` matches in BOTH orientations, so the
    idempotency guard is defeated and the tensor flips on every call.
    ``make_synthetic_tensor_dict`` therefore refuses such shapes.
    Shrink ``n_pids`` and ``n_trials`` for speed -- never ``T`` or ``C``.

2.  Content must be genuinely learnable AND physically structured.

    Learnable, so a 1-epoch CPU smoke test can assert the loss actually
    decreases (random noise makes such a test vacuous).  Physically
    structured, so the IMU rotation-augmentation tests are meaningful: the
    72 IMU channels are laid out as 12 sensors x 6 axes, where channels
    ``6s..6s+2`` are an accelerometer triad and ``6s+3..6s+5`` a gyroscope
    triad of the SAME physical sensor, related by a known constant factor.
    A correct per-sensor SO(3) augmentation applies the same rotation to
    both triads, so the relation survives; an incorrect one breaks it.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import torch

# Real-data dimensions. Defaults deliberately match ablation_config.py so a
# fixture-built config and a real config agree on every shape.
DEFAULT_T = 64
DEFAULT_C_EMG = 16
DEFAULT_C_IMU = 72
DEFAULT_DEMO_DIM = 12

# IMU channel layout assumed by the fixture, and the layout the rotation
# augmentation will assume once the real column names are dumped from the
# cluster pickle (see LIMITATIONS.md E7 -- the REAL layout is NOT yet verified,
# and `channel_visualization.ipynb` hints it may be non-contiguous).
IMU_N_SENSORS = 12
IMU_AXES_PER_SENSOR = 6          # accel xyz + gyro xyz
IMU_GYRO_OVER_ACCEL = 2.0        # the known relation the rotation tests check


def _pid_name(i: int) -> str:
    """PIDs that look like the real ones ('P004', 'P104') without colliding."""
    return f"S{i:03d}"


def _emg_for_class(g: int, n_trials: int, T: int, C: int, gen: torch.Generator):
    """
    Rectified, class-separable EMG envelope, shaped (n_trials, T, C).

    Class ``g`` gets a frozen random spatial mixing matrix over 4 temporal
    basis functions at class-specific frequencies. The result is rectified
    (matching the MAV-envelope character of the real 20 Hz input) and given a
    per-trial per-channel log-normal gain -- the same multiplicative nuisance
    that EMGGainJitter is designed to model, so a gain-jitter augmentation test
    has something realistic to act on.
    """
    n_basis = 4
    t = torch.linspace(0.0, 1.0, T)
    freqs = torch.tensor([1.0 + g, 2.0 + g, 3.5 + g, 5.0 + g])
    basis = torch.stack([torch.sin(2 * math.pi * f * t) for f in freqs])   # (n_basis, T)

    # Frozen per-class spatial pattern -> "which muscle is active".
    cls_gen = torch.Generator().manual_seed(1000 + g)
    A = torch.randn(C, n_basis, generator=cls_gen)                        # (C, n_basis)

    sig = (A @ basis).abs()                                               # (C, T), rectified
    out = sig.unsqueeze(0).repeat(n_trials, 1, 1)                         # (n_trials, C, T)

    # Per-trial, per-channel multiplicative gain (electrode-impedance analogue).
    gain = torch.exp(0.15 * torch.randn(n_trials, C, 1, generator=gen))
    out = out * gain
    out = out + 0.05 * torch.randn(out.shape, generator=gen).abs()

    return out.permute(0, 2, 1).contiguous()                              # (n_trials, T, C)


def _imu_for_class(g: int, n_trials: int, T: int, C: int, gen: torch.Generator):
    """
    IMU shaped (n_trials, T, C) with C == IMU_N_SENSORS * IMU_AXES_PER_SENSOR.

    Per sensor, a smooth 3-D accelerometer trajectory and a gyroscope triad
    equal to ``IMU_GYRO_OVER_ACCEL`` times it. That exact relation is what
    ``test_aug.py`` asserts survives a per-sensor rotation -- it holds only if
    both triads receive the SAME rotation matrix, which is the physically
    correct behaviour (they share a body frame).
    """
    assert C == IMU_N_SENSORS * IMU_AXES_PER_SENSOR, (
        f"synthetic IMU expects C == {IMU_N_SENSORS}*{IMU_AXES_PER_SENSOR}, got {C}"
    )
    t = torch.linspace(0.0, 1.0, T)
    out = torch.zeros(n_trials, C, T)

    for s in range(IMU_N_SENSORS):
        # Sensor- and class-specific smooth trajectory, frozen across trials.
        ph = torch.tensor([0.0, 2 * math.pi / 3, 4 * math.pi / 3]) + 0.37 * s
        f = 1.0 + 0.5 * g + 0.25 * s
        accel = torch.stack([torch.sin(2 * math.pi * f * t + p) for p in ph])  # (3, T)

        a0 = 6 * s
        out[:, a0 + 0 : a0 + 3, :] = accel.unsqueeze(0)
        out[:, a0 + 3 : a0 + 6, :] = IMU_GYRO_OVER_ACCEL * accel.unsqueeze(0)

    out = out + 0.02 * torch.randn(out.shape, generator=gen)
    return out.permute(0, 2, 1).contiguous()                               # (n_trials, T, C)


def make_synthetic_tensor_dict(
    n_pids: int = 6,
    n_classes: int = 10,
    n_trials: int = 10,
    T: int = DEFAULT_T,
    C_emg: int = DEFAULT_C_EMG,
    C_imu: int | None = DEFAULT_C_IMU,
    demo_dim: int = DEFAULT_DEMO_DIM,
    seed: int = 0,
    separable: bool = True,
) -> dict:
    """
    Build the full on-disk payload dict. Returns ``(trials, T, C)`` orientation.

    Args:
        C_imu: pass ``None`` to build an EMG-only fixture (``imu`` is ``None``,
            which the pipeline supports and several code paths special-case).
        separable: ``True`` gives class-separable signal so smoke tests can
            assert the loss decreases. ``False`` gives pure noise, useful for
            asserting a metric sits at chance.

    Raises:
        ValueError: if ``T`` equals either channel count. See invariant 1 in the
            module docstring -- this would silently defeat ``reorient_tensor_dict``'s
            idempotency guard.
    """
    bad = {C_emg} | ({C_imu} if C_imu is not None else set())
    if T in bad:
        raise ValueError(
            f"Ambiguous fixture shape: T={T} equals a channel count "
            f"(C_emg={C_emg}, C_imu={C_imu}). reorient_tensor_dict() decides "
            f"orientation by comparing shape[-1] to emg_in_ch/imu_in_ch, so a "
            f"square-ish fixture matches in BOTH orientations and the tensor "
            f"flips on every call. Shrink n_pids/n_trials instead of T/C."
        )

    gen = torch.Generator().manual_seed(seed)
    data: dict[str, dict[int, dict]] = {}

    for i in range(n_pids):
        pid = _pid_name(i)
        data[pid] = {}
        for g in range(n_classes):
            if separable:
                emg = _emg_for_class(g, n_trials, T, C_emg, gen)
                imu = (
                    _imu_for_class(g, n_trials, T, C_imu, gen)
                    if C_imu is not None else None
                )
            else:
                emg = torch.randn(n_trials, T, C_emg, generator=gen)
                imu = (
                    torch.randn(n_trials, T, C_imu, generator=gen)
                    if C_imu is not None else None
                )

            data[pid][g] = {
                "emg": emg,
                "imu": imu,
                "demo": torch.randn(demo_dim, generator=gen),
                "gest_ID": g,
                # 1-INDEXED rep numbers. trial_idx = rep_num - 1.
                "rep_indices": list(range(1, n_trials + 1)),
                "gesture_features": None,
            }

    return {
        "data": data,
        "gesture_label_encoder": None,
        "pid_encoder": None,
        "gesture_feature_cols": None,
        "pwod_pids": set(),
        # NB: deliberately absent rather than False, matching the real pickle --
        # reorient_tensor_dict uses .get("_reoriented", False).
    }


def write_synthetic_pkl(tmp_path: Path, filename: str = "synthetic_tensor_dict.pkl", **kw) -> str:
    """
    Pickle a synthetic payload and return its path.

    This is what lets ``get_maml_dataloaders`` / ``run_episodic_test_eval`` /
    ``run_supervised_test_eval`` run END TO END with no cluster data -- they are
    the only places that read a path rather than accepting a dict.
    """
    payload = make_synthetic_tensor_dict(**kw)
    p = Path(tmp_path) / filename
    with open(p, "wb") as f:
        pickle.dump(payload, f)
    return str(p)


def pid_names(n_pids: int = 6) -> list[str]:
    """The PID list a synthetic payload will contain, in construction order."""
    return [_pid_name(i) for i in range(n_pids)]


def hash_tensor_dict(tensor_dict: dict) -> str:
    """
    Content hash over every emg/imu tensor, for the mutation guard.

    Used by ``test_aug.py`` to assert that an augmented epoch leaves the shared
    ``tensor_dict`` untouched -- ``_build_episode`` hands out VIEWS into it
    (``emg_all[idx]``), so an in-place transform would silently corrupt every
    later episode. The ``zeros_like`` comment in ``_apply_modality_mask`` exists
    for exactly this reason.
    """
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    for pid in sorted(tensor_dict.keys()):
        for g in sorted(tensor_dict[pid].keys()):
            entry = tensor_dict[pid][g]
            for key in ("emg", "imu"):
                t = entry.get(key)
                if t is None:
                    h.update(b"None")
                    continue
                h.update(str(tuple(t.shape)).encode())
                h.update(t.detach().contiguous().numpy().tobytes())
    return h.hexdigest()
