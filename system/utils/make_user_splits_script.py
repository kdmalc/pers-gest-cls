import numpy as np
import json
from sklearn.model_selection import KFold


SETUP_KFCV = True
FIXED_TRAIN_VAL_TEST = False
if SETUP_KFCV==True and FIXED_TRAIN_VAL_TEST==True:
    raise ValueError("Train/val/test with KFCV not supported yet.")
USE_NONDISABLED = True

K = 4  # Number of folds for cross-validation
NUM_TEST = 4

# NOT USED FOR KVAL!
NUM_VAL = 0 if SETUP_KFCV else 5

RANDOM_SEED = 17
np.random.seed(RANDOM_SEED)

# 32
all_users = ['P008', 'P119', 'P131', 'P122', 'P110', 'P111', 'P010', 'P132',
       'P115', 'P102', 'P106', 'P121', 'P107', 'P116', 'P114', 'P128',
       'P103', 'P104', 'P004', 'P105', 'P126', 'P005', 'P127', 'P123',
       'P011', 'P125', 'P109', 'P112', 'P118', 'P006', 'P124', 'P108']

# 32 - 6 = 26
disabled_users_only = ['P119', 'P131', 'P122', 'P110', 'P111', 'P132',
       'P115', 'P102', 'P106', 'P121', 'P107', 'P116', 'P114', 'P128',
       'P103', 'P104', 'P105', 'P126', 'P127', 'P123',
       'P125', 'P109', 'P112', 'P118', 'P124', 'P108']

if USE_NONDISABLED:
    my_users = all_users
else:
    my_users = disabled_users_only
NUM_TRAIN = len(my_users) - NUM_TEST - NUM_VAL

config_dict = {"SETUP_KFCV":SETUP_KFCV, "K":K, "FIXED_TRAIN_VAL_TEST":FIXED_TRAIN_VAL_TEST, "USE_NONDISABLED":USE_NONDISABLED, "NUM_TRAIN":NUM_TRAIN, "NUM_VAL":NUM_VAL, "NUM_TEST":NUM_TEST, "RANDOM_SEED":RANDOM_SEED}
print(config_dict)


if SETUP_KFCV==False:
    if FIXED_TRAIN_VAL_TEST==True:
        np.random.shuffle(my_users)
        # Train/test split (24 for training, 8 for testing)
        train_users = my_users[:NUM_TRAIN]
        val_users = my_users[NUM_TRAIN:NUM_TRAIN+NUM_VAL]
        test_users = my_users[NUM_TRAIN+NUM_VAL:]
        assert(len(test_users) == NUM_TEST)

        print(f"ACTUAL: {len(train_users)} train users")
        print(f"ACTUAL: {len(val_users)} val users")
        print(f"ACTUAL: {len(val_users)} test users")

        # Save to a JSON file
        split_dict = {"all_users": my_users, "train_users": train_users, "val_users": val_users, "test_users": test_users}
        with open("user_splits.json", "w") as f:
            json.dump(split_dict, f, indent=4)

        print("User splits saved to user_splits.json")
    elif FIXED_TRAIN_VAL_TEST==False:
        np.random.shuffle(my_users)
        # Train/test split (24 for training, 8 for testing)
        train_users = my_users[:24]
        test_users = my_users[24:]

        # Save to a JSON file
        split_dict = {"all_users": my_users, "train_users": train_users, "test_users": test_users}
        with open("user_splits.json", "w") as f:
            json.dump(split_dict, f, indent=4)

        print("User splits saved to user_splits.json")
else:
    # Shuffle and split
    np.random.shuffle(all_users)
    test_users = all_users[:NUM_TEST]
    remaining_users = all_users[NUM_TEST:]

    # K-Fold CV on remaining users
    kf = KFold(n_splits=K, shuffle=True, random_state=RANDOM_SEED)
    cv_splits = []

    for train_idx, val_idx in kf.split(remaining_users):
        train_users = [remaining_users[i] for i in train_idx]
        val_users = [remaining_users[i] for i in val_idx]
        cv_splits.append({
            "train": train_users,
            "val": val_users,
            "test": test_users
        })

    # Save to JSON
    with open("cv_splits.json", "w") as f:
        json.dump(cv_splits, f, indent=4)

    print(f"Saved {K}-fold cross-validation splits to cv_splits.json")

