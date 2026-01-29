import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sys
import os
import matplotlib.pyplot as plt

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to the system path
sys.path.insert(0, parent_dir)

from global_seed import set_seed
set_seed()

from moments_engr import *
from agglo_model_clust import *
from DNN_FT_funcs import *
from DNN_AMC_funcs import *
from configs.hyperparam_tuned_configs import * 

SAVE_RESULTS = True
MODEL_STR = "DynamicCNN"  #"DynamicMomonaNet" "CTRLNet" "DynamicCNN" "CNNModel3Layer"
FEATENG = "moments"  # "moments" "FS" None
MY_CONFIG = determine_config(MODEL_STR, feateng=FEATENG)
MY_CONFIG["user_split_json_filepath"] = "April_25\\fixed_user_splits\\16_5_5_trainvaltest_user_splits_RS17.json"
SPLIT_INDEX = None  # If not using kfcv, this should be None

expdef_df = load_expdef_gestures(feateng_method=MY_CONFIG["feature_engr"])
data_splits = make_data_split(expdef_df, MY_CONFIG, split_index=SPLIT_INDEX)  # json splits are taken care of here!
# Fit LabelEncoder once on all participant IDs for consistency
# Why can't I just pull this directly from the loaded MY_CONFIG["user_split_json_filepath"]?
test_participants = list(np.unique(data_splits['novel_subject_test_dict']['participant_ids']))
all_participants = np.unique(data_splits['pretrain_dict']['participant_ids'] + data_splits['pretrain_subject_test_dict']['participant_ids'] + test_participants)
# Still have to apply labelencoder here (I think) because we apply it to the participant_ids (for clustering). 
## The gesture_IDs have already been encoded in load_expdef!
label_encoder = LabelEncoder()
label_encoder.fit(all_participants)
# Process train and test sets
train_df = process_split(data_splits, 'pretrain_dict', label_encoder)
intra_test_df = process_split(data_splits, 'pretrain_subject_test_dict', label_encoder)
cross_test_df = process_split(data_splits, 'novel_subject_test_dict', label_encoder)
data_dfs_dict = {'pretrain_df':train_df, 'pretrain_subject_test_df':intra_test_df} #, 'novel_subject_test_df':cross_test_df}

# Only clustering wrt intra_test results, not cross_test results, for now...
merge_log, intra_cluster_performance, cross_cluster_performance, nested_clus_model_dict, all_clus_logs_dict = DNN_agglo_merge_procedure(data_dfs_dict, MY_CONFIG)

# Save the data to a file
if SAVE_RESULTS:
    print(f'{MY_CONFIG["results_save_dir"]}')
    # Create log directory if it doesn't exist
    os.makedirs(MY_CONFIG["results_save_dir"], exist_ok=True)
    # Include timestamp in file name? I think it is already included in results_save_dir?
    #with open(f'{MY_CONFIG["results_save_dir"]}\\{MY_CONFIG["timestamp"]}_{MODEL_STR}_agglo_merge_res.pkl', 'wb') as f:
    with open(f'{MY_CONFIG["results_save_dir"]}\\{MODEL_STR}_agglo_merge_res.pkl', 'wb') as f:
        pickle.dump(merge_log, f)
        pickle.dump(intra_cluster_performance, f)
        pickle.dump(cross_cluster_performance, f)
        pickle.dump(nested_clus_model_dict, f)
    print("Data has been saved successfully!")

#print("Complete")

# Loop through selected cluster IDs
my_cluster_model_IDs = nested_clus_model_dict['Iter18'].keys()
for selected_cluster_ID in my_cluster_model_IDs:
    if selected_cluster_ID < 24:
        # Original single-user models: use iteration 0
        loss_logs_dict = all_clus_logs_dict[0][selected_cluster_ID]
    else:
        iter_number = selected_cluster_ID - list(all_clus_logs_dict[1].keys())[0] + 1
        loss_logs_dict = all_clus_logs_dict[iter_number][selected_cluster_ID]
    # Extract logs
    train_log = loss_logs_dict.get("train_loss_log")
    test_log = loss_logs_dict.get("val_loss_log")
    # Skip if both logs are empty
    if not train_log and not test_log:
        print(f"No logs found for cluster {selected_cluster_ID}, skipping.")
        continue
    # Unzip into (iteration, loss) lists
    train_iters, train_losses = zip(*[(i, val) for i, val in enumerate(train_log)])
    test_iters, test_losses = zip(*[(i, val) for i, val in enumerate(test_log)])

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_iters, train_losses, label="Train Loss", marker='o')
    plt.plot(test_iters, test_losses, label="Test Loss", marker='s')
    plt.title(f"Train vs Test Loss for Cluster {selected_cluster_ID}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
# Need to parse all_clus_logs_dict (keys are iter numbers, keys of nested dict are cluster_IDs) to select all the model used in Iter18 (currently 18, 43, 45, 46, 47, 49)
# - This will give me the train/test logs, then I can choose which cluster model to plot

# - Can also plot a few train/test logs from the early single subject cluster models



