# THIS TRAINS A SINGLE GENERIC CNN AND TESTS ON WITHHELD USERS
## NO CLUSTERING!!!

import numpy as np
import os
import sys

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to the system path
sys.path.insert(0, parent_dir)
#from moments_engr import *  # Just load the already preprocessed/engineered data
from DNN_FT_funcs import *
from configs.hyperparam_tuned_configs import *
cwd = os.getcwd()
print("Current Working Directory: ", cwd)

from global_seed import set_seed
set_seed()


MODEL_STR = "DynamicCNN" #"DynamicMomonaNet" "ELEC573Net" "OriginalELEC573CNN"
FEATENG = "FS"  # "moments" "FS" None
MY_CONFIG = determine_config(MODEL_STR, feateng=FEATENG)
MY_CONFIG["user_split_json_filepath"] = "April_25\\fixed_user_splits\\24_8_user_splits_RS17.json"

do_normal_logging = False

##################################################

expdef_df = load_expdef_gestures(feateng_method=MY_CONFIG["feature_engr"])
data_splits = make_data_split(expdef_df, MY_CONFIG)  # json splits are taken care of here!
# Fit LabelEncoder once on all participant IDs for consistency
# Why can't I just pull this directly from the loaded MY_CONFIG["user_split_json_filepath"]?
test_participants = list(np.unique(data_splits['novel_subject_test_dict']['participant_ids']))
all_participants = np.unique(data_splits['pretrain_dict']['participant_ids'] + data_splits['pretrain_subject_test_dict']['participant_ids'] + test_participants)

# Train base model
results = main_training_pipeline(
    data_splits, 
    all_participants=all_participants, 
    test_participants=test_participants,
    config=MY_CONFIG)

os.makedirs(MY_CONFIG["models_save_dir"])
print(f'Directory {MY_CONFIG["models_save_dir"]} created successfully!')
save_model(results["model"], MODEL_STR, MY_CONFIG["models_save_dir"], "pretrained", verbose=True, timestamp=MY_CONFIG["timestamp"])

if do_normal_logging:
    os.makedirs(MY_CONFIG["results_save_dir"])
    print(f'Directory {MY_CONFIG["results_save_dir"]} created successfully!')

    #visualize_model_performance(results)  # This isn't defined anymore... what happened...

    # TODO: This is broken right now, I removed train_performance and such from the saved results bc that is broken
    log_file = log_performance(results, base_filename=MODEL_STR, config=MY_CONFIG)
    print(f"Detailed performance log saved to: {log_file}")
