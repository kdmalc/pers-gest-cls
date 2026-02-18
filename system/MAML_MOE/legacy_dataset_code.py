# -------------------- Notes on LSTMs / Attention --------------------
"""
Should experts be larger or add an MLP/LSTM after experts?
- Start simple: keep Expert as small MLP (as above). If you widen the encoder/fused dim (emb_dim), consider setting expert_bigger=True.
- If you want sequence-aware classification, expose emg_seq/imu_seq and add a [TEMP-HEAD] temporal head:
    class TemporalHead(nn.Module):
        def __init__(self, in_dim, hidden=128, num_layers=1, num_classes=10):
            super().__init__()
            self.rnn = nn.LSTM(in_dim, hidden, num_layers=num_layers, batch_first=True, bidirectional=True)
            self.cls = nn.Linear(hidden*2, num_classes)
        def forward(self, seq):  # (B, T, D)
            y, _ = self.rnn(seq)
            return self.cls(y.mean(dim=1))
- Where to hook it: replace `fused_h` with a fused sequence representation and feed to TemporalHead.

Should I investigate LSTMs?
- Only after you establish a clear win with EMG+IMU late fusion. LSTMs help when precise temporal order matters and windows are long. For short EMG windows (~300ms), TCNs often suffice. A tiny Transformer encoder layer (2 heads) at [ATTN-TEMP] is another strong option.

Where exactly to add attention?
- [ATTN-CROSS] insert a CrossAttention block consuming (emg_seq, imu_seq) and return updated sequences => pool => fused_h.
- [ATTN-TEMP] replace EMGEncoderTCN.block3 with TransformerEncoderLayer(d_model=emb_dim, nhead=2, dim_feedforward=2*emb_dim) on z^T.
"""

#########################################################################################################################

# -------------------- Data utilities (DataFrame -> Dataset -> DataLoader) --------------------

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ---- TwoDFSequenceDataset ----------------------------------------------------
class TwoDFSequenceDataset(Dataset):
    """
    Unified dataset that can be backed by:
      (A) DataFrames (your original behavior), OR
      (B) Prebuilt tensors.

    Emits one sample dict with:
      emg:   (C_emg, T)  float32
      imu:   (C_imu, T)  float32 or None
      demo:  (D_demo,)   float32
      labels: ()          int64   (class index)
      PIDs:  ()          int64   (numeric user index for embeddings)
    """
    # --------- Constructors for clarity (optional to use) ----------
    @classmethod
    def from_dataframes(cls, time_df, demo_df, **kwargs):
        return cls(time_df=time_df, demo_df=demo_df, **kwargs)

    @classmethod
    def from_tensors(cls, emg_t, labels_t, pids_t, *, imu_t=None, demo_t=None):
        # Note: window_len/emg_cols/etc. are irrelevant for tensor mode
        return cls(
            time_df=None,
            demo_df=None,
            emg_t=emg_t,
            imu_t=imu_t,
            demo_t=demo_t,
            labels_t=labels_t,
            pids_t=pids_t,
        )

    def __init__(
        self,
        # ---- DF mode (A) args, unchanged ----
        time_df: pd.DataFrame | None = None,
        demo_df: pd.DataFrame | None = None,
        *,
        window_len: int = 64,
        emg_cols=None,
        imu_cols=None,
        demo_cols=None,
        label_col: str = "Enc_Gesture_ID",
        user_id_col: str = "Enc_PID",

        # ---- Tensor mode (B) args (all tensors) ----
        emg_t: torch.Tensor | None = None,     # (N, C_emg, T)
        imu_t: torch.Tensor | None = None,     # (N, C_imu, T) or None
        demo_t: torch.Tensor | None = None,    # (N, D_demo) or None
        labels_t: torch.Tensor | None = None,  # (N,)
        pids_t: torch.Tensor | None = None,    # (N,)
    ):
        super().__init__()

        # Detect mode
        tensor_mode = emg_t is not None or labels_t is not None or pids_t is not None
        df_mode = (time_df is not None) and (demo_df is not None)

        if tensor_mode and df_mode:
            raise ValueError("Provide EITHER DataFrames (time_df & demo_df) OR tensor args (emg_t, labels_t, pids_t), not both.")
        if not tensor_mode and not df_mode:
            raise ValueError("You must provide DataFrames (DF mode) OR tensors (tensor mode).")

        self._mode = "tensor" if tensor_mode else "df"

        if self._mode == "df":
            # --------- Original DF-backed behavior (unchanged) ---------
            self.time_df = time_df.reset_index(drop=True)
            self.demo_df = demo_df.copy()

            if user_id_col not in self.demo_df.columns:
                raise KeyError(f"'{user_id_col}' must be a column in demo_df.")
            self.demo_df = self.demo_df.set_index(user_id_col, drop=False)

            self.window_len = int(window_len)
            self.emg_cols = list(emg_cols)
            self.imu_cols = list(imu_cols) if imu_cols is not None and len(imu_cols) > 0 else None
            self.demo_cols = list(demo_cols) if demo_cols is not None and len(demo_cols) > 0 else None
            self.label_col = label_col
            self.user_id_col = user_id_col

            unique_ids = pd.Index(self.demo_df.index.unique())
            self.pid_to_index = {pid: i for i, pid in enumerate(unique_ids)}
            self.index_to_pid = {i: pid for pid, i in self.pid_to_index.items()}

            n = len(self.time_df)
            if n < self.window_len:
                raise ValueError(f"time_df has {n} rows, smaller than window_len={self.window_len}.")
            self.n_full = (n // self.window_len) * self.window_len
            self.starts = np.arange(0, self.n_full, self.window_len, dtype=int)

        else:
            # --------- Tensor-backed behavior ---------
            # Basic presence & shape checks
            if emg_t is None or labels_t is None or pids_t is None:
                raise ValueError("Tensor mode requires emg_t, labels_t, and pids_t (imu_t/demo_t are optional).")

            if not torch.is_tensor(emg_t):    emg_t = torch.as_tensor(emg_t, dtype=torch.float32)
            if imu_t is not None and not torch.is_tensor(imu_t):   imu_t = torch.as_tensor(imu_t, dtype=torch.float32)
            if demo_t is not None and not torch.is_tensor(demo_t): demo_t = torch.as_tensor(demo_t, dtype=torch.float32)
            if not torch.is_tensor(labels_t): labels_t = torch.as_tensor(labels_t, dtype=torch.long)
            if not torch.is_tensor(pids_t):   pids_t   = torch.as_tensor(pids_t,   dtype=torch.long)

            if emg_t.ndim != 3:
                raise ValueError(f"emg_t must be (N,C,T); got {tuple(emg_t.shape)}")
            if imu_t is not None and imu_t.ndim != 3:
                raise ValueError(f"imu_t must be (N,C,T); got {tuple(imu_t.shape)}")
            if demo_t is not None and demo_t.ndim not in (1, 2):
                raise ValueError(f"demo_t must be (N,D) or (N,); got {tuple(demo_t.shape)}")

            if demo_t is not None and demo_t.ndim == 1:
                demo_t = demo_t.unsqueeze(1)  # (N,) -> (N,1)

            N = emg_t.shape[0]
            if labels_t.shape[0] != N: raise ValueError("labels_t length must match emg_t batch size")
            if pids_t.shape[0]   != N: raise ValueError("pids_t length must match emg_t batch size")
            if imu_t  is not None and imu_t.shape[0]  != N: raise ValueError("imu_t length must match emg_t batch size")
            if demo_t is not None and demo_t.shape[0] != N: raise ValueError("demo_t length must match emg_t batch size")

            # Store tensors
            self._emg_t    = emg_t.contiguous()
            self._imu_t    = None if imu_t is None else imu_t.contiguous()
            self._demo_t   = None if demo_t is None else demo_t.contiguous()
            self._labels_t = labels_t.contiguous()
            self._pids_t   = pids_t.contiguous()

    def __len__(self):
        if self._mode == "df":
            return len(self.starts)
        else:
            return self._emg_t.shape[0]

    def __getitem__(self, idx):
        if self._mode == "df":
            s = self.starts[idx]
            e = s + self.window_len
            block = self.time_df.iloc[s:e]

            # ---- EMG (C, T)
            emg_np = block[self.emg_cols].to_numpy(dtype=np.float32, copy=False)  # (T, C_emg)
            emg = torch.from_numpy(emg_np).T.contiguous()  # -> (C_emg, T)

            # ---- IMU (C, T) or None
            imu = None
            if self.imu_cols is not None:
                imu_np = block[self.imu_cols].to_numpy(dtype=np.float32, copy=False)  # (T, C_imu)
                imu = torch.from_numpy(imu_np).T.contiguous()  # -> (C_imu, T)

            # ---- Label (int64)
            label_val = block.iloc[0][self.label_col]
            if not np.issubdtype(np.asarray(label_val).dtype, np.integer):
                raise ValueError(
                    f"Label '{self.label_col}' must be integer-coded (got {label_val!r}). "
                    "Map classes to integers before using the dataset."
                )
            label = torch.tensor(int(label_val), dtype=torch.long)

            # ---- PID -> numeric index
            pid_val = block.iloc[0][self.user_id_col]
            if pid_val not in self.pid_to_index:
                raise KeyError(f"User {pid_val!r} not found in demo_df index.")
            PIDs = torch.tensor(self.pid_to_index[pid_val], dtype=torch.long)

            # ---- Demo (D,)
            demo_row = self.demo_df.loc[pid_val]
            if isinstance(demo_row, pd.DataFrame):
                demo_row = demo_row.iloc[0]  # if duplicates per PID, take the first
            if self.demo_cols is None:
                demo_vals = demo_row.drop(labels=[self.user_id_col], errors="ignore").to_numpy()
            else:
                demo_vals = demo_row[self.demo_cols].to_numpy()
            demo = torch.as_tensor(np.asarray(demo_vals, dtype=np.float32).reshape(-1), dtype=torch.float32)

            return {"emg": emg, "imu": imu, "demo": demo, "labels": label, "PIDs": PIDs}

        else:
            # Tensor-backed path
            emg   = self._emg_t[idx]
            imu   = None if self._imu_t is None else self._imu_t[idx]
            demo  = None if self._demo_t is None else self._demo_t[idx]
            label = self._labels_t[idx]
            pids  = self._pids_t[idx]
            return {"emg": emg, "imu": imu, "demo": demo, "labels": label, "PIDs": pids}


# ---- Collate (unimodal or multimodal) ---------------------------------------
def default_mm_collate_fixed(batch):
    """
    Stacks TwoDFSequenceDataset samples into model-ready tensors.

    Returns:
      emg:   (B, C_emg, T)
      imu:   (B, C_imu, T) or None
      demo:  (B, D_demo)
      labels: (B,)          int64
      PIDs:  (B,)          int64
    """
    # EMG (C,T) -> (B,C,T)
    emg = torch.stack([b["emg"] for b in batch], dim=0)
    if emg.dim() != 3:
        raise ValueError(f"EMG must be 3D, got {tuple(emg.shape)}")

    # IMU optional
    imu = None
    if all(("imu" in b) and (b["imu"] is not None) for b in batch):
        imu = torch.stack([b["imu"] for b in batch], dim=0)
        if imu.dim() != 3:
            raise ValueError(f"IMU must be 3D, got {tuple(imu.shape)}")

    # Demo (B, D)
    demo = torch.stack([b["demo"] for b in batch], dim=0).float()

    # Labels / PIDs: robust to ints / numpy scalars / 0-D tensors
    label = torch.as_tensor([int(b["labels"]) for b in batch], dtype=torch.long)
    PIDs  = torch.as_tensor([int(b["PIDs"])  for b in batch], dtype=torch.long)

    return {"emg": emg, "imu": imu, "demo": demo, "labels": label, "PIDs": PIDs}


# ---- Dataloader builder ------------------------------------------------------
def build_dataloader_from_two_dfs(
    time_df: pd.DataFrame,
    demo_df: pd.DataFrame,
    *,  # TODO: I hate that it uses *. Remove this if possible
    emg_cols=None,
    imu_cols=None,
    demo_cols=None,
    label_col="Enc_Gesture_ID",
    user_id_col="Enc_PID",
    window_len=64,
    batch_size=64, # TODO: I dont like that its using this with a default value, also idk what this is doing in terms of a meta-batch...
    shuffle=True,
    num_workers=0, # This ought to be pulled from the config...
    collate_fn=None,
):
    if emg_cols is None:
        raise ValueError("emg_cols must be provided (list of EMG column names).")
    if collate_fn is None:
        collate_fn = default_mm_collate_fixed

    ds = TwoDFSequenceDataset(
        time_df=time_df,
        demo_df=demo_df,
        window_len=window_len,
        emg_cols=emg_cols,
        imu_cols=imu_cols,
        demo_cols=demo_cols,
        label_col=label_col,
        user_id_col=user_id_col,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    return ds, dl


def ensure_tensor(x, dtype=None):
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(dtype) if dtype else x.clone().detach()
    return torch.tensor(x, dtype=dtype)

