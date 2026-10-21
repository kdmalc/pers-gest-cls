"""
Numpy adapter for cached episodes: what `$B`, kNN, ProtoNet, Riemannian, ROCKET and the
zero-shot centroid baseline consume (docs/SPEC_episode_cache.md).

tensor_dict must be reoriented to (reps, C, T) (maml_data_pipeline.reorient_tensor_dict).
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from episodes.episode_cache import Episode, materialize_check


@dataclass
class MaterializedEpisode:
    episode_id: str
    user: str
    n_way: int
    k_shot: int
    emg_support: np.ndarray            # (n_s, C_emg, T)
    imu_support: Optional[np.ndarray]  # (n_s, C_imu, T) or None
    y_support: np.ndarray              # (n_s,) local labels
    emg_query: np.ndarray
    imu_query: Optional[np.ndarray]
    y_query: np.ndarray
    rep_support: np.ndarray            # (n_s,) rep indices
    rep_query: np.ndarray
    cls_support: np.ndarray            # global gesture ids
    cls_query: np.ndarray


def materialize(ep: Episode, tensor_dict: dict) -> MaterializedEpisode:
    materialize_check(ep, tensor_dict)
    parts = {"s": ([], [], [], [], []), "q": ([], [], [], [], [])}  # emg, imu, y, rep, cls
    has_imu = tensor_dict[ep.user][ep.classes[0]]["imu"] is not None
    for label, cls in enumerate(ep.classes):
        slot = tensor_dict[ep.user][cls]
        if (slot["imu"] is not None) != has_imu:
            raise ValueError(f"{ep.user} class {cls}: inconsistent imu presence")
        for tag, reps in (("s", ep.support[label]), ("q", ep.query(label))):
            emg, imu, y, rep, c = parts[tag]
            for r in reps:
                emg.append(slot["emg"][r].numpy())
                if has_imu:
                    imu.append(slot["imu"][r].numpy())
                y.append(label)
                rep.append(r)
                c.append(cls)

    def pack(tag):
        emg, imu, y, rep, c = parts[tag]
        return (np.stack(emg), np.stack(imu) if has_imu else None, np.asarray(y),
                np.asarray(rep), np.asarray(c))

    es, is_, ys, rs, cs = pack("s")
    eq, iq, yq, rq, cq = pack("q")
    if set(rs.tolist()) & set(rq.tolist()) and _share_pair(cs, rs, cq, rq):
        raise AssertionError(f"{ep.id}: support/query share a (class, rep)")
    return MaterializedEpisode(ep.id, ep.user, ep.n_way, ep.k_shot, es, is_, ys, eq, iq, yq,
                               rs, rq, cs, cq)


def _share_pair(cs, rs, cq, rq):
    return bool(set(zip(cs.tolist(), rs.tolist())) & set(zip(cq.tolist(), rq.tolist())))
