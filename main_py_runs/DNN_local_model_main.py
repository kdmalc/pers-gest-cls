import sys
import os

# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the parent directory to the system path
sys.path.insert(0, parent_dir)
# Now you can import your global_seed module
from global_seed import *
set_seed()

from DNN_FT_funcs import *
#from DNN_AMC_funcs import *
from configs.hyperparam_tuned_configs import *
from full_study_funcs import prepare_data_for_local_models
from viz_utils.quick_analysis_plots import plot_train_test_loss_resdictv


#MODEL_STR = "DynamicCNN"
MY_CONFIG = DCNN_moments_config
#MY_CONFIG["use_earlystopping"] = False
#MY_CONFIG["num_epochs"] = 20
#MY_CONFIG["model_str"] = MODEL_STR

MY_CONFIG["num_epochs"] = 200
MY_CONFIG["learning_rate"] = 0.002
MY_CONFIG["conv_layers"] = [[32, 1, 1]]  # Kernel size isn't saved!!!!
MY_CONFIG["fc_layers"] = [16] 
MY_CONFIG["weight_decay"] = 0.001
MY_CONFIG["fc_dropout"] = 0.5
MY_CONFIG["cnn_dropout"] = 0.0

NUM_LOCAL_MODELS = 2 
NUM_LOCAL_TRAIN_GESTURE_SAMPLES = 1

train_key = "novel_trainFT_dict"

expdef_df = load_expdef_gestures(feateng_method=MY_CONFIG["feature_engr"])
all_participants = list(expdef_df['Participant'].unique())

# Choosing NUM_LOCAL_MODELS somewhat arbitrarily, don't need to look at all 32 results... save some computation ig
# Setting num_gesture_training_trials=3 so that this replicates finetuning data used for local
# num_gesture_ft_trials should not be used I don't think, not in main_training_pipeline anyways
data_splits = make_data_split(expdef_df, MY_CONFIG, num_train_tpg=NUM_LOCAL_TRAIN_GESTURE_SAMPLES, num_ft_tpg=NUM_LOCAL_TRAIN_GESTURE_SAMPLES)#, use_only_these_encoded_gestures=[3])

# Need to make a dictionary of train/test loaders (guess I have to add cross as well for reusability)
unique_gestures = np.unique(data_splits[train_key]['labels'])
num_classes = len(unique_gestures)
input_dim = data_splits[train_key]['feature'].shape[1]

user_dict = prepare_data_for_local_models(data_splits, MY_CONFIG)

res_dict_lst = []
participants_to_train = list(set(data_splits[train_key]['participant_ids'][:NUM_LOCAL_MODELS+1]))
for p_idx, pid in enumerate(participants_to_train):
    print(f"Training local model for {pid} ({p_idx+1}/{NUM_LOCAL_MODELS})")

    all_test_pids = data_splits['novel_subject_test_dict']['participant_ids']
    excluding_self_test_pids = [test_pid for test_pid in all_test_pids if test_pid!=pid]

    res_dict_lst.append(main_training_pipeline(data_splits=None, all_participants=all_participants, 
                                               test_participants=excluding_self_test_pids, 
                                               config=MY_CONFIG, 
                                               train_intra_cross_loaders=user_dict[pid]))
    print(f'PID {pid}: Train acc: {res_dict_lst[-1]["train_accuracy"]*100:.2f}%, Intra acc: {res_dict_lst[-1]["intra_test_accuracy"]*100:.2f}%, Cross acc: {res_dict_lst[-1]["cross_test_accuracy"]*100:.2f}%')

# res_dict_lst would either need to be saved, or extracted in a iPYNB so that we can plot train/test logs...

print("Complete")

# Save the data to a file
## TODO: Ensure this is saving to the correct place
## TODO: This is the saving for the AMC, not for local.
# Need to save the results from local tho...
#print(f'{MY_CONFIG["results_save_dir"]}\\{MY_CONFIG["timestamp"]}')
#print()
#with open(f'{MY_CONFIG["results_save_dir"]}\\{MY_CONFIG["timestamp"]}_{MODEL_STR}_agglo_merge_res.pkl', 'wb') as f:
#    pickle.dump(merge_log, f)
#    pickle.dump(intra_cluster_performance, f)
#    pickle.dump(cross_cluster_performance, f)
#    pickle.dump(nested_clus_model_dict, f)
#print("Data has been saved successfully!")

plot_train_test_loss_resdictv(res_dict_lst[0], f'Local, PID {participants_to_train[0]}', print_acc=True, acc_keys=None, log_keys=None, use_cross=False)
