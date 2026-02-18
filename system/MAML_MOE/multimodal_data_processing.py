from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader, TensorDataset

from system.MAML_MOE.legacy_dataset_code import *
from system.MAML_MOE.maml_multimodal_dataloaders import *


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
    # NOTE: use_imu and such are part of the config now too, and they are not integrated with this
    ## There is some minor utility in allowing us to pass in False... I'm not sure when that would ever be helpful tho ngl

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

    if config["meta_learning"]:
        """So in this branch, we load in our separated support and query dfs, merge them together, create a ds and dl, toss the dl, then create two dls from the ds
        For the actual support/query breakdown, they are non-overlapping (good) but fully complementary (all samples are used, non-ideal)"""



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
        train_uc = UserClassIndex(train_ds, user_key="PIDs", label_key="labels")
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
        val_uc_sup = UserClassIndex(val_support_ds, user_key="PIDs", label_key="labels")
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
        test_uc_sup = UserClassIndex(test_support_ds, user_key="PIDs", label_key="labels")
        test_users  = test_uc_sup.users

        test_epi = FixedOneShotPerUserIterable(
            support_ds=test_support_ds, query_ds=test_query_ds,
            users_subset=test_users, collate_fn=default_mm_collate_fixed,
            n_way=int(config.get('test_n_way', n_way))  # typically 10
        )

        # DataLoaders for episodic iterables: batch_size=None because each yield is a full episode (eg 1. This is safest. Means we have to batch manually via looping if we want batches)
        train_dl = DataLoader(train_epi, batch_size=None, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        val_dl   = DataLoader(val_epi,   batch_size=None, num_workers=num_workers, pin_memory=torch.cuda.is_available())
        test_dl  = DataLoader(test_epi,  batch_size=None, num_workers=num_workers, pin_memory=torch.cuda.is_available())

        return train_dl, val_dl, test_dl
    else:
        raise ValueError("load_multimodal_data_loaders only supports Meta Learning")


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


def make_user_loaders_from_dataloaders(val_dl_all, test_dl_all, config):
    """
    Creates user-specific episodic dataloaders for Meta-Validation and Meta-Test.
    
    This function specifically handles the episodic structure (FixedOneShotPerUserIterable) 
    and returns two separate dictionaries mapping PID -> DataLoader.
    """
    # 1. Clean up and validate episodic datasets
    # We expect .dataset to be FixedOneShotPerUserIterable
    val_epi_base = val_dl_all.dataset
    test_epi_base = test_dl_all.dataset

    if not isinstance(val_epi_base, FixedOneShotPerUserIterable) or \
       not isinstance(test_epi_base, FixedOneShotPerUserIterable):
        raise RuntimeError(
            "Expected episodic datasets (FixedOneShotPerUserIterable) for val and test."
        )

    num_workers = int(config.get('num_workers', 0))
    pin_mem = torch.cuda.is_available()

    # --- Helper to create a dict of per-user loaders ---
    def _create_user_map(epi_dataset):
        user_map = {}
        for pid in epi_dataset.users:
            # Create a new iterator instance restricted to just this user
            user_specific_iter = FixedOneShotPerUserIterable(
                support_ds=epi_dataset.support_ds,
                query_ds=epi_dataset.query_ds,
                users_subset=[pid],
                collate_fn=epi_dataset.collate_fn,
                n_way=epi_dataset.n_way
            )
            # Each yield from this loader is a full 1-shot episode for this specific user
            user_map[pid] = DataLoader(
                user_specific_iter, 
                batch_size=None, 
                shuffle=False,
                num_workers=num_workers, 
                pin_memory=pin_mem
            )
        return user_map

    # 2. Generate separate dictionaries to avoid None-value collisions
    val_user_loaders = _create_user_map(val_epi_base)
    test_user_loaders = _create_user_map(test_epi_base)

    return val_user_loaders, test_user_loaders
