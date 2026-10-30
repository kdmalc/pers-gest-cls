"""
body_regions.py
===============
Sensor index -> body location for the 16 EMG and 12 IMU sensors (same placement for every user).
Source: docs/emg-sensor-placements.png; derivation and verification in docs/specs/SPEC_body_regions.md.

All indices here are 0-based tensor indices. Pods 9-12 have two EMG leads each and take two recording
slots, so EMG channels 9-16 (1-based) are leads 9a 9b 10a 10b 11a 11b 12a 12b, and the IMUs of those
pods are recording slots 9, 11, 13, 15. Pods 10 and 12 sit on the distal forearm, so their IMUs belong
to the forearm regions and the hand regions are EMG-only.
"""
from __future__ import annotations

from dataclasses import dataclass

N_EMG = 16
N_IMU = 12
IMU_AXES = 6                     # ax ay az vx vy vz
FS64_FEATURES = 7                # channel-major: channel c, feature f is at c * 7 + f


@dataclass(frozen=True)
class Region:
    name: str
    side: str
    level: str
    emg: tuple                   # 0-based EMG channels
    imu: tuple                   # 0-based IMU sensors


REGIONS = (
    Region("R_upper_arm", "R", "upper_arm", (0, 1), (0, 1)),      # biceps, triceps
    Region("R_shoulder",  "R", "shoulder",  (2, 3), (2, 3)),      # deltoid, trapezius
    Region("L_upper_arm", "L", "upper_arm", (4, 5), (4, 5)),      # biceps, triceps
    Region("L_shoulder",  "L", "shoulder",  (6, 7), (6, 7)),      # deltoid, trapezius
    Region("R_forearm",   "R", "forearm",   (8, 9), (8, 9)),      # 9a extensor, 9b flexor; pods 9, 10
    Region("R_hand",      "R", "hand",      (10, 11), ()),        # 10a FDI, 10b APB
    Region("L_forearm",   "L", "forearm",   (12, 13), (10, 11)),  # 11a extensor, 11b flexor; pods 11, 12
    Region("L_hand",      "L", "hand",      (14, 15), ()),        # 12a FDI, 12b APB
)
REGION_NAMES = tuple(r.name for r in REGIONS)

EMG_LEADS = ("1", "2", "3", "4", "5", "6", "7", "8", "9a", "9b", "10a", "10b", "11a", "11b", "12a", "12b")
EMG_MUSCLES = ("R biceps", "R triceps", "R deltoid", "R trapezius",
               "L biceps", "L triceps", "L deltoid", "L trapezius",
               "R extensors", "R flexors", "R FDI", "R APB",
               "L extensors", "L flexors", "L FDI", "L APB")


def _check_partition(groups, n, what):
    seen = sorted(i for g in groups for i in g)
    if seen != list(range(n)):
        raise ValueError(f"{what} channels are not a partition of 0..{n - 1}: {seen}")


_check_partition([r.emg for r in REGIONS], N_EMG, "EMG")
_check_partition([r.imu for r in REGIONS], N_IMU, "IMU")


def emg_tensor_indices(channels, emg_repr: str) -> list:
    """EMG tensor columns of these channels: env20 has 1 column per channel, fs64 has 7."""
    if emg_repr == "env20":
        return list(channels)
    if emg_repr == "fs64":
        return [c * FS64_FEATURES + f for c in channels for f in range(FS64_FEATURES)]
    raise ValueError(f"emg_repr must be env20|fs64, got {emg_repr!r}")


def imu_tensor_indices(sensors) -> list:
    """IMU tensor columns of these sensors."""
    return [s * IMU_AXES + a for s in sensors for a in range(IMU_AXES)]
