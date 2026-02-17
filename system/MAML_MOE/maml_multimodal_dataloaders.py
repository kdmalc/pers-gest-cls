##################################################################################################
# FOR META LEARNING --> Wrappers creating episodic dataloaders

import random
from collections import defaultdict
import torch
from torch.utils.data import IterableDataset

class UserClassIndex:
    """
    Builds user -> class -> [indices] map from a base dataset whose __getitem__
    returns dicts with at least keys {user_key, label_key}.
    """
    def __init__(self, base_ds, user_key="PIDs", label_key="labels"):
        groups = defaultdict(lambda: defaultdict(list))  # user -> class -> [idx]
        for idx in range(len(base_ds)):
            s = base_ds[idx]
            u = int(s[user_key])
            c = int(s[label_key])
            groups[u][c].append(idx)
        # freeze to regular dicts
        self.groups = {u: {c: idxs[:] for c, idxs in cls_map.items()}
                       for u, cls_map in groups.items()}
        self.users = list(self.groups.keys())

    def classes_with_min_count(self, user, min_count):
        return [c for c, idxs in self.groups[user].items() if len(idxs) >= min_count]


class EpisodicIterable(IterableDataset):
    """
    Meta-train episodic iterator.
    Yields dicts:
      {
        'support': collated_batch,   # collate_fn applied to list of sample dicts
        'query':   collated_batch,
        'user_id': int,
        'label_map': {global_class: local_id},
        'classes_global': [global_class ids in episode order]
      }

    Key behaviors:
      - Per-episode label remapping to local 0..(n_way-1).
      - Random class order per episode (can disable).
      - Optional meta-augmentation via 'task_augment' hook (train-only).
        * See 'task_augment' docstring below.
    """
    def __init__(
        self,
        base_ds,
        uc_index: UserClassIndex,
        users_subset,
        collate_fn,
        n_way=10,
        k_shot=1,
        q_query=9,
        episodes_per_epoch=1000,
        seed=0,
        shuffle_classes=True,
        # --- Optional meta-augmentation hook (TRAIN ONLY) ---
        # task_augment: callable or None
        # Signature:
        #   task_augment(
        #       user_id: int,
        #       classes: list[int],                # chosen global classes
        #       sup_idx_by_class: dict[int, list[int]],
        #       qry_idx_by_class: dict[int, list[int]],
        #       rng: random.Random
        #   ) -> tuple[
        #         list[int],                       # possibly modified classes
        #         dict[int, list[dict]],           # support samples by (global) class (pre-collate)
        #         dict[int, list[dict]]            # query samples by (global) class (pre-collate)
        #       ]
        #
        # Notes:
        #   - You can: (a) keep classes the same and only augment samples;
        #              (b) split/merge classes; or (c) synthesize new classes.
        #   - Return sample *dicts* (not indices) so you can inject fields
        #     (e.g., augmented tensors, flags). Labels will be remapped AFTER this.
        #
        task_augment=None,
    ):
        super().__init__()
        self.base_ds = base_ds
        self.uc = uc_index
        self.users = list(users_subset)
        self.collate_fn = collate_fn
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch
        self.seed = seed
        self.shuffle_classes = shuffle_classes
        self.task_augment = task_augment

    def _copy_and_set_label(self, sample_dict, new_label, orig_key="orig_label", label_key="labels"):
        s = dict(sample_dict)  # shallow copy; avoid mutating base dataset
        s[orig_key] = int(s[label_key])
        s[label_key] = int(new_label)
        return s

    def __iter__(self):
        # Fresh RNG per worker/epoch
        rng = random.Random(self.seed + int(torch.randint(0, 2**31-1, ()).item()))
        need = self.k_shot + self.q_query

        for _ in range(self.episodes_per_epoch):
            # --- pick a user with enough eligible classes
            ## So all my episodes are single-user? I think that is fine... prevents collisions certainly
            u = rng.choice(self.users)
            eligible = self.uc.classes_with_min_count(u, min_count=need)
            if len(eligible) < self.n_way:
                # fallback: choose another user who has enough
                candidates = [uu for uu in self.users
                              if len(self.uc.classes_with_min_count(uu, need)) >= self.n_way]
                if not candidates:
                    # If this ever happens, reduce n_way or adjust your splits
                    continue
                u = rng.choice(candidates)
                eligible = self.uc.classes_with_min_count(u, min_count=need)

            classes = rng.sample(eligible, self.n_way)  # N distinct classes

            # Within-episode label shuffling
            # randomize class order -> random local IDs
            if self.shuffle_classes:
                rng.shuffle(classes)

            # gather per-class indices (no overlap within episode)
            sup_idx_by_class, qry_idx_by_class = {}, {}
            for c in classes:
                idxs = self.uc.groups[u][c]
                choice = rng.sample(idxs, need)
                sup_idx_by_class[c] = choice[:self.k_shot]
                qry_idx_by_class[c] = choice[self.k_shot:]

            # materialize raw sample dicts (still using GLOBAL labels here)
            def idxs_to_samples(idx_list):
                return [self.base_ds[i] for i in idx_list]

            support_by_class = {c: idxs_to_samples(sup_idx_by_class[c]) for c in classes}
            query_by_class   = {c: idxs_to_samples(qry_idx_by_class[c]) for c in classes}

            # --- (optional) TASK-LEVEL META-AUGMENTATION (train only)
            # You can modify the class set and/or the actual samples here.
            # This is the code to augment in place
            ## It probably would be easier to just load in an augmented dataset
            ## Would be quicker on the cluster too (probably?) so we don't have to augment in place (do have to handle more data tho?)
            if self.task_augment is not None:
                classes, support_by_class, query_by_class = self.task_augment(
                    user_id=u,
                    classes=classes,
                    sup_idx_by_class=sup_idx_by_class,
                    qry_idx_by_class=qry_idx_by_class,
                    rng=rng
                )

            # local label map AFTER any augmentation/class edits
            label_map = {c: i for i, c in enumerate(classes)}

            # relabel samples to local 0..N-1 without touching base_ds
            support_samples = []
            query_samples   = []
            for c in classes:
                for s in support_by_class.get(c, []):
                    support_samples.append(self._copy_and_set_label(s, label_map[c]))
                for s in query_by_class.get(c, []):
                    query_samples.append(self._copy_and_set_label(s, label_map[c]))

            # collate into episode batches
            support = self.collate_fn(support_samples)
            query   = self.collate_fn(query_samples)

            yield {
                'support': support,
                'query': query,
                'user_id': u,
                'label_map': label_map,
                'classes_global': classes
            }


class FixedOneShotPerUserIterable(IterableDataset):
    """
    Meta-val/test iterable.
    For each user u, yields exactly ONE episode:
      - Support: exactly 1 example per class from support_ds (pre-split externally),
        using the first index per (u, c) deterministically.
      - Query: all remaining examples for those classes from query_ds.

    Behaviors:
      - Per-episode label remapping to local 0..(n_way-1), DETERMINISTIC order.
      - NO meta-augmentation (clean eval).

    TLDR: Iterate through all the withheld users deterministically, always returning the same 1-shot support set (make eval standard)
    """
    def __init__(self, support_ds, query_ds, users_subset, collate_fn, n_way=10):
        super().__init__()
        self.support_ds = support_ds
        self.query_ds = query_ds
        self.users = list(users_subset)
        self.collate_fn = collate_fn
        self.n_way = n_way

        # Build maps: user -> class -> idx / [idxs]
        self.sup_map = self._build_map(self.support_ds, require_one=True)
        self.qry_map = self._build_map(self.query_ds, require_one=False)

        # Check each user has at least n_way support classes
        for u in self.users:
            if u not in self.sup_map or len(self.sup_map[u]) < self.n_way:
                have = len(self.sup_map.get(u, {}))
                raise RuntimeError(
                    f"User {u} has only {have} support classes; need {self.n_way}."
                )

    @staticmethod
    def _build_map(ds, require_one):
        mp = {}
        for i in range(len(ds)):
            s = ds[i]
            u = int(s["PIDs"]); c = int(s["s"])
            if u not in mp: mp[u] = {}
            if c not in mp[u]: mp[u][c] = []
            mp[u][c].append(i)
        if require_one:
            # Keep exactly one deterministic index for each (u,c)
            for u in list(mp.keys()):
                for c in list(mp[u].keys()):
                    mp[u][c] = mp[u][c][0]  # first index deterministically
        return mp

    def _copy_and_set_label(self, sample_dict, new_label, orig_key="orig_label", label_key="labels"):
        s = dict(sample_dict)
        s[orig_key] = int(s[label_key])
        s[label_key] = int(new_label)
        return s

    def __iter__(self):
        for u in self.users:
            # Deterministic class order for evaluation
            classes = sorted(self.sup_map[u].keys())[:self.n_way]  # first n_way
            label_map = {c: i for i, c in enumerate(classes)}

            # Build support (exactly 1 per class)
            sup_idx = [self.sup_map[u][c] for c in classes]

            # Build query (all examples of those classes)
            qry_idx = []
            for c in classes:
                if u in self.qry_map and c in self.qry_map[u]:
                    qry_idx.extend(self.qry_map[u][c])

            # Materialize & relabel to local 0..N-1
            support_samples = [self._copy_and_set_label(self.support_ds[i], label_map[int(self.support_ds[i]["labels"])])
                               for i in sup_idx]

            query_samples = []
            for i in qry_idx:
                s = self.query_ds[i]
                gl = int(s["labels"])
                if gl in label_map:  # skip stray classes, just in case
                    query_samples.append(self._copy_and_set_label(s, label_map[gl]))

            # Collate & yield
            support = self.collate_fn(support_samples)
            query   = self.collate_fn(query_samples) if len(query_samples) else self.collate_fn([])

            yield {
                'support': support,
                'query': query,
                'user_id': u,
                'label_map': label_map,
                'classes_global': classes
            }

