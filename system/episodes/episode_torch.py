"""
Torch adapter for cached episodes (docs/SPEC_episode_cache.md).

CachedEpisodeDataset returns, per item, exactly the dict MetaGestureDataset._build_episode returns
(so maml_mm_collate, mamlpp_adapt_and_eval and the supervised finetune path work unchanged), plus
the cache's `episode_id`. tensor_dict must already be reoriented to (reps, C, T).
"""
import torch
from torch.utils.data import Dataset

from MAML.maml_data_pipeline import apply_modality_mask
from episodes.episode_cache import materialize_check


class CachedEpisodeDataset(Dataset):
    def __init__(self, tensor_dict, episodes, modality_mask):
        assert modality_mask in ("both", "emg_only", "imu_only"), modality_mask
        if not episodes:
            raise ValueError("no episodes given")
        for ep in episodes:
            materialize_check(ep, tensor_dict)
        self.data, self.episodes, self.modality_mask = tensor_dict, list(episodes), modality_mask
        self.episode_shape_log = []
        # the realised per-class query count is a function of k_shot only; record it up front
        for ep in self.episodes:
            self.episode_shape_log.append({
                "episode_id": ep.id, "user_id": ep.user, "n_classes_realised": ep.n_way,
                "n_support": ep.n_way * ep.k_shot,
                "n_query": ep.n_way * ep.n_query_per_class,
                "q_per_class": ep.n_query_per_class,
            })

    def __len__(self):
        return len(self.episodes)

    def _sample(self, user, cls, label, rep):
        slot = self.data[user][cls]
        imu_all = slot["imu"]
        emg, imu = apply_modality_mask(
            self.modality_mask, slot["emg"][rep], imu_all[rep] if imu_all is not None else None)
        return {"emg": emg, "imu": imu, "demo": slot["demo"], "label": label,
                "global_class": cls, "rep_idx": rep}

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        support, query = [], []
        for label, cls in enumerate(ep.classes):
            for rep in ep.support[label]:
                support.append(self._sample(ep.user, cls, label, rep))
            for rep in ep.query(label):
                query.append(self._sample(ep.user, cls, label, rep))
        return {"support": support, "query": query, "user_id": ep.user,
                "label_map": {cls: label for label, cls in enumerate(ep.classes)},
                "episode_id": ep.id}


def outcomes_to_per_episode(episodes, correct_counts):
    """Build the results-contract `per_episode` list from episodes and per-episode n_correct."""
    if len(episodes) != len(correct_counts):
        raise ValueError(f"{len(episodes)} episodes vs {len(correct_counts)} outcomes")
    out = []
    for ep, n_correct in zip(episodes, correct_counts):
        n_query = ep.n_way * ep.n_query_per_class
        if not 0 <= n_correct <= n_query:
            raise ValueError(f"{ep.id}: n_correct={n_correct} outside [0, {n_query}]")
        out.append({"id": ep.id, "user": ep.user, "n_way": ep.n_way, "k_shot": ep.k_shot,
                    "n_query": n_query, "n_correct": int(n_correct)})
    return out
