from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader, TensorDataset

from MOE_multimodal_model_classes import *
from maml_multimodal_dataloaders import *


# TODO: This needs to be updated to use config instead of default values...
## And to accomodate k-fold cross validation...
def load_multimodal_data_loaders(config, 
    use_emg=True, use_imu=True, use_demographics=True, load_existing_dfs=False, save_dfs=False):
    """
    Returns:
      - If config['meta_learning'] is False/missing:
          (train_support_dl, train_query_dl, val_support_dl, val_query_dl, test_support_dl, test_query_dl)
      - If config['meta_learning'] is True:
          {'train': train_ep_dl, 'val': val_ep_dl, 'test': test_ep_dl, 'meta': True}
        where each loader yields episodic dicts: {'support', 'query', 'user_id'}.
    """
    if use_emg is False or use_imu is False or use_demographics is False:
        raise ValueError("Currently, emg, imu, and demographics all must be used.")

    emg_imu_pkl_full_path = config["emg_imu_pkl_full_path"]
    pwmd_xlsx_filepath = config["pwmd_xlsx_filepath"]
    pwoutmd_xlsx_filepath = config["pwoutmd_xlsx_filepath"]
    dfs_save_path = config["dfs_save_path"]
    dfs_load_path = config["dfs_load_path"]
    saved_df_timestamp = config["saved_df_timestamp"]
    
    train_PIDs = config["train_PIDs"]
    val_PIDs = config["val_PIDs"]
    test_PIDs = config["test_PIDs"]
    train_gesture_range = config["train_gesture_range"] 
    valtest_gesture_range = config["valtest_gesture_range"]

    # ----------------
    # Build / load DataFrames
    # ----------------
    if load_existing_dfs is False:
        data_df = pd.read_pickle(emg_imu_pkl_full_path)

        metadata_cols = ['Participant', 'Gesture_ID', 'Gesture_Num']
        metadata_cols_df = data_df[metadata_cols].rename(columns={"Participant": "PID"})
        metadata_cols_df['Gesture_Num'] = metadata_cols_df['Gesture_Num'].astype(int)

        # PID encoder
        all_PIDs = metadata_cols_df['PID']
        unique_PIDs = all_PIDs.unique()
        PID_encoder = LabelEncoder().fit(unique_PIDs)

        # Gesture encoder
        gesture_ID_label_encoder = LabelEncoder()
        metadata_cols_df['Enc_Gesture_ID'] = gesture_ID_label_encoder.fit_transform(metadata_cols_df['Gesture_ID'])
        metadata_cols_df['Enc_PID'] = PID_encoder.transform(metadata_cols_df['PID'])

        # Signals
        X_df = data_df.drop(metadata_cols, axis=1)
        ppd_B_X_df = preprocess_df_B_by_gesture(X_df)

        # Demographics (with & without disabilities)
        FULL_pwmd_demo_df = pd.read_excel(pwmd_xlsx_filepath)
        pwmd_demo_df = FULL_pwmd_demo_df[[
            "PID", "disability coding", "time disabled", "Actual handedness",
            "What is your age?", "What is your gender?", "BMI", "DASH score"
        ]][:-8]
        pwmd_demo_df["time disabled"] = pd.to_numeric(pwmd_demo_df["time disabled"].astype(str).strip(), errors='coerce')
        numeric_cols = pwmd_demo_df.select_dtypes(include='number').columns
        pwmd_demo_df[numeric_cols] = pwmd_demo_df[numeric_cols] / 100.0
        pwmd_demo_df["BMI"] = pwmd_demo_df["BMI"] / 70.0
        pwmd_demo_df['Enc_PID'] = PID_encoder.transform(pwmd_demo_df["PID"])

        FULL_pwoutmd_demo_df = pd.read_excel(pwoutmd_xlsx_filepath)
        pwoutmd_demo_df = FULL_pwoutmd_demo_df[[
            "PID", "disability coding", "time disabled", "Actual handedness",
            "What is your age?", "What is your gender?", "BMI", "DASH score"
        ]][:-5]
        pwoutmd_demo_df["time disabled"] = pd.to_numeric(pwoutmd_demo_df["time disabled"].astype(str).strip(), errors='coerce')
        numeric_cols2 = pwoutmd_demo_df.select_dtypes(include='number').columns
        pwoutmd_demo_df[numeric_cols2] = pwoutmd_demo_df[numeric_cols2] / 100.0
        pwoutmd_demo_df["BMI"] = pwoutmd_demo_df["BMI"] / 70.0
        pwoutmd_demo_df = pwoutmd_demo_df[~pwoutmd_demo_df['PID'].isin(['P001', 'P003'])]
        pwoutmd_demo_df['Enc_PID'] = PID_encoder.transform(pwoutmd_demo_df["PID"])

        combined_demo_df = pd.concat([pwmd_demo_df, pwoutmd_demo_df])
        demoENC_df = pd.get_dummies(
            combined_demo_df,
            columns=["disability coding", "Actual handedness", "What is your gender?"],
            drop_first=True
        )
        demoENC_df.drop(columns=["PID"], inplace=True)  # keep Enc_PID only

        full_yX_timeseries_df = pd.concat([metadata_cols_df, ppd_B_X_df], axis=1)

        # Classic fixed splits (still used either for classic loaders OR to form merged pools)
        train_support_df = full_yX_timeseries_df[
            (full_yX_timeseries_df['PID'].isin(train_PIDs)) &
            (full_yX_timeseries_df['Gesture_Num'].isin(train_gesture_range))
        ]
        train_query_df = full_yX_timeseries_df[
            (full_yX_timeseries_df['PID'].isin(train_PIDs)) &
            (full_yX_timeseries_df['Gesture_Num'] == 10)
        ]

        val_support_df = full_yX_timeseries_df[
            (full_yX_timeseries_df['PID'].isin(val_PIDs)) &
            (full_yX_timeseries_df['Gesture_Num'] == 1)
        ]
        val_query_df = full_yX_timeseries_df[
            (full_yX_timeseries_df['PID'].isin(val_PIDs)) &
            (full_yX_timeseries_df['Gesture_Num'].isin(valtest_gesture_range))
        ]

        test_support_df = full_yX_timeseries_df[
            (full_yX_timeseries_df['PID'].isin(test_PIDs)) &
            (full_yX_timeseries_df['Gesture_Num'] == 1)
        ]
        test_query_df = full_yX_timeseries_df[
            (full_yX_timeseries_df['PID'].isin(test_PIDs)) &
            (full_yX_timeseries_df['Gesture_Num'].isin(valtest_gesture_range))
        ]

        emg_cols = [c for c in ppd_B_X_df.columns if c.startswith("EMG")]
        imu_cols = [c for c in ppd_B_X_df.columns if c.startswith("IMU")]
        demo_cols = demoENC_df.columns

        if save_dfs:
            train_support_df.to_pickle(f"{dfs_save_path}{config['timestamp']}_train_support_df.pkl")
            train_query_df.to_pickle(f"{dfs_save_path}{config['timestamp']}_train_query_df.pkl")
            val_support_df.to_pickle(f"{dfs_save_path}{config['timestamp']}_val_support_df.pkl")
            val_query_df.to_pickle(f"{dfs_save_path}{config['timestamp']}_val_query_df.pkl")
            test_support_df.to_pickle(f"{dfs_save_path}{config['timestamp']}_test_support_df.pkl")
            test_query_df.to_pickle(f"{dfs_save_path}{config['timestamp']}_test_query_df.pkl")
            demoENC_df.to_pickle(f"{dfs_save_path}{config['timestamp']}_demoENC_df.pkl")
            with open(f"{dfs_save_path}{config['timestamp']}_columns.pkl", "wb") as f:
                pickle.dump([emg_cols, imu_cols, list(demo_cols)], f)
            print("Dataframes have been saved!")

    else:
        #print(f"dfs_load_path: {dfs_load_path}")
        train_support_df = pd.read_pickle(f"{dfs_load_path}{saved_df_timestamp}_train_support_df.pkl")
        train_query_df = pd.read_pickle(f"{dfs_load_path}{saved_df_timestamp}_train_query_df.pkl")
        val_support_df = pd.read_pickle(f"{dfs_load_path}{saved_df_timestamp}_val_support_df.pkl")
        val_query_df = pd.read_pickle(f"{dfs_load_path}{saved_df_timestamp}_val_query_df.pkl")
        test_support_df = pd.read_pickle(f"{dfs_load_path}{saved_df_timestamp}_test_support_df.pkl")
        test_query_df = pd.read_pickle(f"{dfs_load_path}{saved_df_timestamp}_test_query_df.pkl")
        demoENC_df = pd.read_pickle(f"{dfs_load_path}{saved_df_timestamp}_demoENC_df.pkl")
        with open(f"{dfs_load_path}{saved_df_timestamp}_columns.pkl", "rb") as f:
            emg_cols, imu_cols, demo_cols = pickle.load(f)

    # ----------------
    # Classic supervised loaders (original behavior)
    # ----------------
    if not config['meta_learning']:
        _, train_support_dl = build_dataloader_from_two_dfs(
            time_df=train_support_df, demo_df=demoENC_df,
            emg_cols=emg_cols, imu_cols=imu_cols, demo_cols=demo_cols,
            batch_size=config['batch_size']
        )
        _, train_query_dl = build_dataloader_from_two_dfs(
            time_df=train_query_df, demo_df=demoENC_df,
            emg_cols=emg_cols, imu_cols=imu_cols, demo_cols=demo_cols,
            batch_size=config['batch_size']
        )
        _, val_support_dl = build_dataloader_from_two_dfs(
            time_df=val_support_df, demo_df=demoENC_df,
            emg_cols=emg_cols, imu_cols=imu_cols, demo_cols=demo_cols,
            batch_size=config['ft_batch_size']
        )
        _, val_query_dl = build_dataloader_from_two_dfs(
            time_df=val_query_df, demo_df=demoENC_df,
            emg_cols=emg_cols, imu_cols=imu_cols, demo_cols=demo_cols,
            batch_size=config['ft_batch_size']
        )
        _, test_support_dl = build_dataloader_from_two_dfs(
            time_df=test_support_df, demo_df=demoENC_df,
            emg_cols=emg_cols, imu_cols=imu_cols, demo_cols=demo_cols,
            batch_size=config['ft_batch_size']
        )
        _, test_query_dl = build_dataloader_from_two_dfs(
            time_df=test_query_df, demo_df=demoENC_df,
            emg_cols=emg_cols, imu_cols=imu_cols, demo_cols=demo_cols,
            batch_size=config['ft_batch_size']
        )
        return (train_support_dl, train_query_dl, val_support_dl, val_query_dl, test_support_dl, test_query_dl)
    elif config["meta_learning"]:
        # ----------------
        # Meta-learning episodic loaders
        # ----------------
        # === TRAIN: merged pool; episodic, many tasks per user, overlap across episodes OK ===
        train_merged_df = pd.concat([train_support_df, train_query_df], ignore_index=True)
        train_ds, _ = build_dataloader_from_two_dfs(
            time_df=train_merged_df, demo_df=demoENC_df,
            emg_cols=emg_cols, imu_cols=imu_cols, demo_cols=demo_cols,
            batch_size=32, shuffle=False, num_workers=int(config['num_workers']),  # batch_size unused for episodic, allegedly...
            collate_fn=default_mm_collate_fixed
        )
        train_uc = UserClassIndex(train_ds, user_key="PIDs", label_key="label")
        train_users = train_uc.users

        # Meta config knobs
        n_way   = int(config['n_way'])  #10
        k_shot  = int(config['k_shot'])  #1
        q_query = int(config['q_query'])  #9
        episodes_per_epoch_train = int(config['episodes_per_epoch_train'])  #1000? Idk what is reasonable...
        num_workers = int(config['num_workers'])  # Probably 0 but we should raise this when on the cluster

        train_epi = EpisodicIterable(
            base_ds=train_ds, uc_index=train_uc, users_subset=train_users,
            collate_fn=default_mm_collate_fixed,
            n_way=n_way, k_shot=k_shot, q_query=q_query,
            episodes_per_epoch=episodes_per_epoch_train, seed=0
        )

        # === VAL: fixed one-shot per user (Gesture_Num==1 as support; 2..10 as query) ===
        val_support_ds, _ = build_dataloader_from_two_dfs(
            time_df=val_support_df, demo_df=demoENC_df,
            emg_cols=emg_cols, imu_cols=imu_cols, demo_cols=demo_cols,
            batch_size=32, shuffle=False, num_workers=num_workers, collate_fn=default_mm_collate_fixed
        )
        val_query_ds, _ = build_dataloader_from_two_dfs(
            time_df=val_query_df, demo_df=demoENC_df,
            emg_cols=emg_cols, imu_cols=imu_cols, demo_cols=demo_cols,
            batch_size=32, shuffle=False, num_workers=num_workers, collate_fn=default_mm_collate_fixed
        )
        val_uc_sup = UserClassIndex(val_support_ds, user_key="PIDs", label_key="label")
        val_users  = val_uc_sup.users

        val_epi = FixedOneShotPerUserIterable(
            support_ds=val_support_ds, query_ds=val_query_ds,
            users_subset=val_users, collate_fn=default_mm_collate_fixed,
            n_way=int(config.get('val_n_way', n_way))  # typically 10
        )

        # === TEST: fixed one-shot per user (Gesture_Num==1 as support; 2..10 as query) ===
        test_support_ds, _ = build_dataloader_from_two_dfs(
            time_df=test_support_df, demo_df=demoENC_df,
            emg_cols=emg_cols, imu_cols=imu_cols, demo_cols=demo_cols,
            batch_size=32, shuffle=False, num_workers=num_workers, collate_fn=default_mm_collate_fixed
        )
        test_query_ds, _ = build_dataloader_from_two_dfs(
            time_df=test_query_df, demo_df=demoENC_df,
            emg_cols=emg_cols, imu_cols=imu_cols, demo_cols=demo_cols,
            batch_size=32, shuffle=False, num_workers=num_workers, collate_fn=default_mm_collate_fixed
        )
        test_uc_sup = UserClassIndex(test_support_ds, user_key="PIDs", label_key="label")
        test_users  = test_uc_sup.users

        test_epi = FixedOneShotPerUserIterable(
            support_ds=test_support_ds, query_ds=test_query_ds,
            users_subset=test_users, collate_fn=default_mm_collate_fixed,
            n_way=int(config.get('test_n_way', n_way))  # typically 10
        )

        # DataLoaders for episodic iterables: batch_size=None because each yield is a full episode
        train_dl = DataLoader(train_epi, batch_size=None, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        val_dl   = DataLoader(val_epi,   batch_size=None, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        test_dl  = DataLoader(test_epi,  batch_size=None, num_workers=num_workers, pin_memory=torch.cuda.is_available())

        return train_dl, val_dl, test_dl


def _B_normalize_block(block_np, demean=True, eps=1e-8):
    """
    block_np: (T, D_block) numpy array for one biosignal (e.g., all IMU channels)
    Returns: (T, D_block) normalized per $B (demean per channel, divide by shared std over block)
    """
    if demean:
        block_np = block_np - block_np.mean(axis=0, keepdims=True)  # per-channel demean
    sigma = block_np.ravel().std(dtype=np.float64)
    if sigma < eps:
        return block_np  # flat signal; leave as-is
    return block_np / sigma

def preprocess_df_B_by_gesture(
    data_df: pd.DataFrame,
    biosignal_switch_ix: int = 72,   # [:switch) = IMU, [switch:] = EMG
    trial_length: int = 64,
    demean: bool = True,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Apply $B to every trial in the full dataframe.
    Assumptions:
      - data_df has ONLY sensor columns (no metadata), shape = (num_trials*trial_length, num_channels)
      - IMU columns come first, EMG columns follow
      - Each trial is a contiguous block of `trial_length` rows
    Returns: DataFrame with same shape/columns as input.
    """
    if data_df.isna().any().any():
        print("Warning: NaNs detected in input; consider cleaning first.")

    num_rows, num_cols = data_df.shape
    if num_rows % trial_length != 0:
        raise ValueError(f"Rows ({num_rows}) not divisible by trial_length ({trial_length}).")

    if not (0 < biosignal_switch_ix < num_cols):
        raise ValueError(f"biosignal_switch_ix {biosignal_switch_ix} must be in (0, {num_cols}).")

    num_trials = num_rows // trial_length
    cols = data_df.columns
    X = data_df.to_numpy(dtype=np.float64, copy=True)  # (N, D)

    for t in range(num_trials):
        s = t * trial_length
        e = s + trial_length
        trial = X[s:e, :]  # (T, D)

        imu_block = trial[:, :biosignal_switch_ix]
        emg_block = trial[:, biosignal_switch_ix:]

        imu_block = _B_normalize_block(imu_block, demean=demean, eps=eps)
        emg_block = _B_normalize_block(emg_block, demean=demean, eps=eps)

        X[s:e, :biosignal_switch_ix] = imu_block
        X[s:e, biosignal_switch_ix:] = emg_block

    out = pd.DataFrame(X, columns=cols, index=data_df.index)
    return out


# NOT USING THIS ONE FOR NOW
## This needs to be trained on the train set and then applied to the test set... too complicated to deal with rn

class PerChannelZScore:
    """
    Fit per-channel (column) scalers on training data.
    Use separate instances for IMU and EMG if you prefer, or pass the full frame if ranges are comparable.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, df):
        X = df.to_numpy().astype(np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0, ddof=0)
        # avoid zeros
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, df):
        X = df.to_numpy().astype(np.float64)
        X = (X - self.mean_) / self.std_
        return pd.DataFrame(X, columns=df.columns, index=df.index)


def make_user_loaders_from_dataloaders(ft_dl_all, nv_dl_all, config):
    """
    Creates user-specific dataloaders from the provided ft/val/test dataloaders.

    Axes handled:
      1. meta_learning vs non-meta_learning       (config['meta_learning'])
      2. multimodal vs non-multimodal             (config['multimodal'])
      3. FT episodic (support/query) vs flat      (config['use_supportquery_for_ft'])

    Semantics:

      NON-META (config['meta_learning'] == False)
      ----------------------------------------
      - Assumes ft_dl_all and nv_dl_all are SUPPORT and QUERY splits (of the same split, ie of the test split)
        over the SAME users (PIDs overlap).
      - Returns paired flat loaders:
            user_loaders[pid] = (ft_dl_for_pid, nv_dl_for_pid)
        Only PIDs that appear in BOTH dls are included.

      META (config['meta_learning'] == True)
      -----------------------------------
      - ft_dl_all and nv_dl_all may be DIFFERENT splits (e.g., val vs test),
        with DISJOINT user sets.
      - Returns a unified mapping over the UNION of users:
            user_loaders[pid] = (ft_dl_or_None, nv_dl_or_None)
        where:
          * If pid only in ft_dl_all: (ft_dl, None)
          * If pid only in nv_dl_all: (None, nv_dl)
          * If pid appears in both:   (ft_dl, nv_dl)

      Episodic vs flat FT:
        - config['use_supportquery_for_ft'] == False:
            * Flatten FT and NV loaders into per-user flat loaders
              (support+query combined within each split).
        - config['use_supportquery_for_ft'] == True AND meta_learning == True:
            * Expects ft_dl_all.dataset and nv_dl_all.dataset to be
              FixedOneShotPerUserIterable (or same interface).
            * Returns per-user episodic loaders:
                user_loaders[pid] = (ft_epi_dl_or_None, nv_epi_dl_or_None)
    """

    multimodal_version    = config['multimodal']
    metalearning_version  = config['meta_learning']
    use_sq_ft             = config['use_supportquery_for_ft']

    # Ensure we have some FT batch size for flat loaders
    ft_bs = int(config.get('ft_batch_size', config.get('batch_size', 32)))
    config['ft_batch_size'] = ft_bs

    # ----------------------------------------------------------------------
    # Helper: Flatten all samples out of a loader, grouped later by PID
    # ----------------------------------------------------------------------
    def _extract_all(dl, multimodal_version=False, metalearning_version=False):
        """
        Flattens all batches in `dl` into big tensors:
          - multimodal: (EMG, IMU, DEMO, LABEL, PIDs)
          - non-multimodal: (X, Y, PIDs)

        If metalearning_version=True and batches are episodic dicts with
        'support'/'query', this will flatten support + query into a global pool.
        """

        if multimodal_version:
            emgs, imus, demos, labels, pids = [], [], [], [], []
        else:
            xs, ys, pids = [], [], []

        has_dict = False

        for batch in dl:
            # --------------------------------------------------------
            # META-LEARNING EPISODIC FORMAT:
            #   batch = {"support": {...}, "query": {...}}
            # --------------------------------------------------------
            if (
                metalearning_version
                and isinstance(batch, dict)
                and "support" in batch
                and "query" in batch
            ):
                has_dict = True

                for split_name in ("support", "query"):
                    split = batch[split_name]

                    if multimodal_version:
                        # Expect multimodal dict: emg/imu/demo/label/PIDs
                        emg   = split["emg"]
                        imu   = split["imu"]
                        demo  = split["demo"]
                        label = split["label"]
                        pid   = split["PIDs"]

                        # If episodes are [N_tasks, K, ...], flatten to [N_tasks*K, ...]
                        if emg.dim() > 2 and hasattr(pid, "dim") and pid.dim() > 1:
                            B = emg.shape[0] * emg.shape[1]
                            emg   = emg.view(B, *emg.shape[2:])
                            imu   = imu.view(B, *imu.shape[2:])
                            demo  = demo.view(B, *demo.shape[2:])
                            label = label.view(-1)
                            pid   = pid.view(-1)

                        emgs.append(emg.cpu())
                        imus.append(imu.cpu())
                        demos.append(demo.cpu())
                        labels.append(label.cpu())
                        if pid is None:
                            raise RuntimeError(
                                "Missing participant_id in episodic batch; expected 'PIDs' key."
                            )
                        pids.append(pid.cpu() if torch.is_tensor(pid) else pid)

                    else:
                        # Non-multimodal meta-learning case (if ever used)
                        if not isinstance(split, dict):
                            raise RuntimeError(
                                "Expected dict splits for non-multimodal meta-learning; "
                                f"got {type(split)} instead."
                            )

                        # Try common key names
                        if "x" in split and "y" in split:
                            x = split["x"]
                            y = split["y"]
                        elif "inputs" in split and "labels" in split:
                            x = split["inputs"]
                            y = split["labels"]
                        else:
                            raise RuntimeError(
                                "Cannot find (x, y) or (inputs, labels) in meta-learning split."
                            )

                        pid = split.get("PIDs", split.get("pid", None))
                        if pid is None:
                            raise RuntimeError(
                                "Missing participant_id in meta-learning split; "
                                "expected 'PIDs' or 'pid'."
                            )

                        # Flatten episodes if [N_tasks, K, ...]
                        if x.dim() > 2 and hasattr(pid, "dim") and pid.dim() > 1:
                            B = x.shape[0] * x.shape[1]
                            x   = x.view(B, *x.shape[2:])
                            y   = y.view(-1)
                            pid = pid.view(-1)

                        xs.append(x.cpu())
                        ys.append(y.cpu())
                        pids.append(pid.cpu() if torch.is_tensor(pid) else pid)

            # --------------------------------------------------------
            # OLD MULTIMODAL (NON-META) FORMAT:
            #   batch = {"emg", "imu", "demo", "label", "PIDs"}
            # --------------------------------------------------------
            elif isinstance(batch, dict) and multimodal_version:
                has_dict = True

                emg   = batch['emg']
                imu   = batch['imu']
                demo  = batch['demo']
                label = batch['label']
                pid   = batch['PIDs']

                if pid is None:
                    raise RuntimeError("Missing participant_id; provide via 'PIDs' in batch dict.")

                emgs.append(emg.cpu())
                imus.append(imu.cpu())
                demos.append(demo.cpu())
                labels.append(label.cpu())
                pids.append(pid.cpu() if torch.is_tensor(pid) else pid)

            # --------------------------------------------------------
            # NON-MULTIMODAL TUPLE FORMAT:
            #   (x, y) or (x, y, pid)
            # --------------------------------------------------------
            else:
                # assume (x, y) or (x, y, pid)
                if len(batch) == 3:
                    x, y, pid = batch
                else:
                    x, y = batch
                    pid = None
                    # try attribute on dataset as last resort
                    if hasattr(dl.dataset, 'participant_ids'):
                        raise RuntimeError(
                            "participant_ids attribute fallback requires deterministic ordering; "
                            "set shuffle=False for this pass."
                        )
                    else:
                        raise RuntimeError("Cannot determine participant IDs from fallback path.")

                if pid is None:
                    raise RuntimeError("Missing participant_id in non-multimodal batch.")

                xs.append(x.cpu())
                ys.append(y.cpu())
                pids.append(pid.cpu() if torch.is_tensor(pid) else pid)

        # ------------------------------------------------------------
        # Concatenate across all batches
        # ------------------------------------------------------------
        if multimodal_version:
            EMG   = torch.cat(emgs)
            IMU   = torch.cat(imus)
            DEMO  = torch.cat(demos)
            LABEL = torch.cat(labels)

            if len(pids) and torch.is_tensor(pids[0]):
                P = torch.cat(pids)
            else:
                P = torch.tensor(pids)

            return EMG, IMU, DEMO, LABEL, P
        else:
            X = torch.cat(xs)
            Y = torch.cat(ys)

            if len(pids) and torch.is_tensor(pids[0]):
                P = torch.cat(pids)
            else:
                P = torch.tensor(pids)

            return X, Y, P

    # ----------------------------------------------------------------------
    # CASE 3: meta_learning == True AND use_supportquery_for_ft == True
    #         -> per-user episodic FT loaders using FixedOneShotPerUserIterable,
    #            allowing disjoint user sets between ft_dl_all and nv_dl_all.
    # ----------------------------------------------------------------------
    user_loaders = {}

    if metalearning_version and use_sq_ft:
        # We expect ft_dl_all.dataset and nv_dl_all.dataset to be FixedOneShotPerUserIterable
        ft_epi = ft_dl_all.dataset
        nv_epi = nv_dl_all.dataset

        if not isinstance(ft_epi, FixedOneShotPerUserIterable) or not isinstance(nv_epi, FixedOneShotPerUserIterable):
            raise RuntimeError(
                "use_supportquery_for_ft=True expects ft_dl_all.dataset and nv_dl_all.dataset "
                "to be FixedOneShotPerUserIterable (or at least to share its interface)."
            )

        ft_users = set(ft_epi.users)
        nv_users = set(nv_epi.users)
        all_pids = sorted(ft_users | nv_users)  # UNION (val ∪ test, etc.)

        if not all_pids:
            raise RuntimeError("No users found in episodic FT/NV datasets.")

        num_workers = int(config['num_workers'])

        for pid in all_pids:
            ft_dl = None
            nv_dl = None

            if pid in ft_users:
                ft_iter = FixedOneShotPerUserIterable(
                    support_ds=ft_epi.support_ds,
                    query_ds=ft_epi.query_ds,
                    users_subset=[pid],
                    collate_fn=ft_epi.collate_fn,
                    n_way=ft_epi.n_way
                )
                ft_dl = DataLoader(
                    ft_iter, batch_size=None, shuffle=False,
                    num_workers=num_workers, pin_memory=torch.cuda.is_available()
                )

            if pid in nv_users:
                nv_iter = FixedOneShotPerUserIterable(
                    support_ds=nv_epi.support_ds,
                    query_ds=nv_epi.query_ds,
                    users_subset=[pid],
                    collate_fn=nv_epi.collate_fn,
                    n_way=nv_epi.n_way
                )
                nv_dl = DataLoader(
                    nv_iter, batch_size=None, shuffle=False,
                    num_workers=num_workers, pin_memory=torch.cuda.is_available()
                )

            user_loaders[pid] = (ft_dl, nv_dl)

        return user_loaders

    # ----------------------------------------------------------------------
    # CASES 1 & 2: flat per-user loaders
    #
    #   - meta_learning == False (non-meta)
    #   - meta_learning == True, use_supportquery_for_ft == False
    #
    # These cases flatten everything and then group by PID.
    # Non-meta: only overlapping PIDs get paired.
    # Meta:     all PIDs in union get entries; halves may be None.
    # ----------------------------------------------------------------------
    ft_dl_all_shuffle  = getattr(ft_dl_all, 'shuffle', False)
    nv_dl_all_shuffle  = getattr(nv_dl_all, 'shuffle', False)

    if ft_dl_all_shuffle or nv_dl_all_shuffle:
        # Rewrap without shuffling to safely reconstruct per-user tensors
        ft_ds = ft_dl_all.dataset
        nv_ds = nv_dl_all.dataset
        ft_dl_all = DataLoader(ft_ds, batch_size=ft_bs, shuffle=False)
        nv_dl_all = DataLoader(nv_ds, batch_size=ft_bs, shuffle=False)

    ft_map = {}
    nv_map = {}

    if multimodal_version:
        EMG_ft, IMU_ft, DEMO_ft, LABEL_ft, PID_ft = _extract_all(
            ft_dl_all,
            multimodal_version=True,
            metalearning_version=metalearning_version
        )
        EMG_nv, IMU_nv, DEMO_nv, LABEL_nv, PID_nv = _extract_all(
            nv_dl_all,
            multimodal_version=True,
            metalearning_version=metalearning_version
        )

        # Build per-user FT datasets
        for pid in PID_ft.unique().tolist():
            mask = (PID_ft == pid)
            ft_ds = make_MOE_tensor_dataset(
                EMG_ft[mask], IMU_ft[mask], DEMO_ft[mask], LABEL_ft[mask],
                config, participant_ids=PID_ft[mask]
            )
            ft_dl = DataLoader(ft_ds, batch_size=ft_bs, shuffle=True)
            ft_map[pid] = ft_dl

        # Build per-user NV datasets
        for pid in PID_nv.unique().tolist():
            mask = (PID_nv == pid)
            nv_ds = make_MOE_tensor_dataset(
                EMG_nv[mask], IMU_nv[mask], DEMO_nv[mask], LABEL_nv[mask],
                config, participant_ids=PID_nv[mask]
            )
            nv_dl = DataLoader(nv_ds, batch_size=ft_bs, shuffle=False)
            nv_map[pid] = nv_dl

    else:
        X_ft, y_ft, pid_ft = _extract_all(
            ft_dl_all,
            multimodal_version=False,
            metalearning_version=metalearning_version
        )
        X_nv, y_nv, pid_nv = _extract_all(
            nv_dl_all,
            multimodal_version=False,
            metalearning_version=metalearning_version
        )

        for pid in pid_ft.unique().tolist():
            mask = (pid_ft == pid)
            ft_ds = make_tensor_dataset(X_ft[mask], y_ft[mask], config)
            ft_dl = DataLoader(ft_ds, batch_size=ft_bs, shuffle=True)
            ft_map[pid] = ft_dl

        for pid in pid_nv.unique().tolist():
            mask = (pid_nv == pid)
            nv_ds = make_tensor_dataset(X_nv[mask], y_nv[mask], config)
            nv_dl = DataLoader(nv_ds, batch_size=ft_bs, shuffle=False)
            nv_map[pid] = nv_dl

    # Pair or union depending on meta vs non-meta
    if not metalearning_version:
        # NON-META: only PIDs appearing in BOTH maps
        common_pids = sorted(set(ft_map.keys()) & set(nv_map.keys()))
        for pid in common_pids:
            user_loaders[pid] = (ft_map[pid], nv_map[pid])
    else:
        # META: union of PIDs; one side may be None (val vs test users)
        all_pids = sorted(set(ft_map.keys()) | set(nv_map.keys()))
        for pid in all_pids:
            ft_dl = ft_map.get(pid, None)
            nv_dl = nv_map.get(pid, None)
            user_loaders[pid] = (ft_dl, nv_dl)

    if not user_loaders:
        raise RuntimeError(
            "Could not build per-user loaders from provided dataloaders. "
            "Check PIDs or meta/non-meta settings."
        )

    return user_loaders


# This also appears in utils.gesture_dataset_classes. Should be the same here, I didn't edit it. Ought to sort that out...
## I guess this file actually has all the functions that that file did
def make_tensor_dataset(features, labels, config, reshape_2d_to_3d=True, participant_ids=None):
    """
    Converts features and labels to tensors and validates shape.
    If features are 2D and reshape_2d_to_3d is True, reshapes to (N, C, L).
    Optionally includes participant_ids in the returned TensorDataset.
    """
    features = ensure_tensor(features, dtype=torch.float32)
    labels = ensure_tensor(labels, dtype=torch.long)

    if features.ndim == 2 and reshape_2d_to_3d:
        num_channels = config["num_channels"]
        sequence_length = config["sequence_length"]
        features = features.view(-1, num_channels, sequence_length)

    assert features.ndim == 3, f"Expected 3D tensor (batch, channels, sequence), got {features.ndim}D with shape {features.shape}"
    assert features.shape[1] == config["num_channels"]
    assert features.shape[2] == config["sequence_length"]

    if participant_ids is not None:
        if isinstance(participant_ids[0], str):
            print("make_tensor_dataset(): Why is participants a list of unencoded strings?")
            try:
                participant_ids = [int(pid[1:]) for pid in participant_ids]
            except ValueError as e:
                raise ValueError(f"Failed to convert participant_ids to integers: {e}")

        participant_ids = ensure_tensor(participant_ids, dtype=torch.long)
        assert participant_ids.shape[0] == features.shape[0], (
            f"participant_ids length {participant_ids.shape[0]} does not match number of samples {features.shape[0]}"
        )
        return TensorDataset(features, labels, participant_ids)

    return TensorDataset(features, labels)