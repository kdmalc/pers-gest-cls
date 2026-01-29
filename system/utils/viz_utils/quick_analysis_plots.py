import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import datetime
import numpy as np


def plot_model_acc_boxplots(data_dict, my_title=None, save_fig=False, plot_save_name=None, print_stats_sum=True, 
                            save_dir="C:\\Users\\kdmen\\Repos\\fl-gestures\\April_25\\results\\plots", enforce_xtick_newline=True,
                            tick_fontsize=12, title_fontsize=12, label_fontsize=12, figsize_tuple=(6, 4), 
                            data_keys=['global_acc_data', 'pretrained_cluster_acc_data', 'local_acc_data', 'ft_global_acc_data', 'ft_cluster_acc_data'],
                            labels=['Generic\nGlobal', 'Pretrained\nCluster', 'Local', 'Fine-Tuned\nGlobal', 'Fine-Tuned\nCluster']):
    
    data = [data_dict[key] for key in data_keys]

    # Colors: all boxes are light blue
    box_color = '#C7DDF0'
    colors_lst = [box_color] * len(data_keys)

    # Navy blue for all datapoints
    #point_color = '#0A2540'  # --> Navy was too close to black
    point_color = '#3330e3'  # Saturated blue

    #if my_title is None and model_str is not None:  # model_str should/will never be None
    #    my_title = f"{model_str} Per-User Accuracies"

    fig, ax = plt.subplots(figsize=figsize_tuple)

    # Create boxplots
    bp = ax.boxplot(data, patch_artist=True, widths=0.5)
    if enforce_xtick_newline:
        new_xtick_labels = [label.replace(' ', '\n') for label in labels]
    else:
        new_xtick_labels = labels
    ax.set_xticklabels(new_xtick_labels, ha='center', rotation=0, fontsize=tick_fontsize, linespacing=1.5)

    # Customize box colors
    for patch, color in zip(bp['boxes'], colors_lst):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay uniform scatter points (navy blue)
    for i, acc_list in enumerate(data):
        for acc in acc_list:
            ax.scatter(i + 1, acc, color=point_color, edgecolor='black', s=40, marker='o', alpha=0.9)
        if print_stats_sum:
            acc_npy = np.array(acc_list)
            # The first part of this code is replacing the newline character with a space
            print(f"{labels[i].replace(chr(10), ' ')}: mean accuracy {(np.mean(acc_npy)*100):.2f}%, std {(np.std(acc_npy)*100):.2f}%")

    # Aesthetic tweaks
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Y-axis gridlines at every 10%
    #yticks = [round(val, 1) for val in np.arange(0, 1.1, 0.1)]
    #ax.set_yticks(yticks)
    #ax.set_yticklabels(yticks, rotation=0, fontsize=tick_fontsize)
    #ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    # Y-axis gridlines and labels at every 20%
    yticks = np.arange(0.2, 1.01, 0.2)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{int(y * 100)}" for y in yticks], fontsize=tick_fontsize)  # "20", "40", ...
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

    ax.set_ylabel('Accuracy (%)', fontsize=label_fontsize)
    if my_title is None:
        pass
    else:
        ax.set_title(my_title, fontsize=title_fontsize)
    #ax.tick_params(axis='both', labelsize=14)

    plt.tight_layout()

    if save_fig:
        plt.savefig(f"{save_dir}\\{plot_save_name}.png", dpi=500, bbox_inches='tight')

    plt.show()


def plot_model_acc_boxplots_OLD_uniqueUserMarkers(data_dict, model_str=None, colors_lst=None, my_title=None, save_fig=False, plot_save_name=None, print_stats_sum=True, 
                            save_dir="C:\\Users\\kdmen\\Repos\\fl-gestures\\ELEC573_Proj\\results",
                            data_keys=['global_acc_data', 'pretrained_cluster_acc_data', 'local_acc_data', 'ft_global_acc_data', 'ft_cluster_acc_data'],
                            labels=['Generic Global', 'Pretrained Cluster', 'Local', 'Fine-Tuned Global', 'Fine-Tuned Cluster']):
    
    data = [data_dict[key] for key in data_keys]

    # Generate default colors if not provided
    if colors_lst is None:
        colors_lst = ['lightgray'] * len(data_keys)  # Subtle box colors

    # Assign unique colors and markers per user (assuming user index)
    num_users = max(len(d) for d in data)
    user_colors = plt.cm.get_cmap("tab10", num_users)  # Distinct colors
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X', '<', '>']  # Unique markers

    if my_title is None and model_str is not None:
        my_title = f"{model_str} Per-User Accuracies"

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create boxplots
    bp = ax.boxplot(data, patch_artist=True, labels=labels)

    # Customize box colors
    for patch, color in zip(bp['boxes'], colors_lst):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)  # Slight transparency

    # Overlay scatter points per user with consistent colors and markers
    for i, acc_list in enumerate(data):
        for user_idx, acc in enumerate(acc_list):
            marker = markers[user_idx % len(markers)]  # Cycle through markers
            ax.scatter(i + 1, acc, color=user_colors(user_idx), edgecolor='black', s=50, marker=marker, alpha=0.8)
        if print_stats_sum:
            acc_npy = np.array(acc_list)
            print(f"{labels[i]}: mean accuracy {(np.mean(acc_npy)*100):.2f}%, std {(np.std(acc_npy)*100):.2f}%")

    # Remove top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Increase font sizes
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_title(my_title, fontsize=20)
    ax.tick_params(axis='both', labelsize=12)

    # Set y-axis limits
    ax.set_ylim(0, 1.0)

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_fig:
        plt.savefig(f"{save_dir}\\{plot_save_name}.png", dpi=500, bbox_inches='tight')

    plt.show()


def plot_train_test_loss_acc_direct(title, train_loss_log, test_loss_log, train_acc_log=None, test_acc_log=None):
    """Plots training/test loss and accuracy curves."""
    fig, ax1 = plt.subplots()
    
    # Plot Loss
    ax1.plot(train_loss_log, color='blue', linestyle='-', label='Train Loss')
    ax1.plot(test_loss_log, color='orange', linestyle='-', label='Test Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Plot Accuracy if available
    if train_acc_log is not None and test_acc_log is not None:
        ax2 = ax1.twinx()
        ax2.plot(train_acc_log, color='blue', linestyle='--', label='Train Accuracy')
        ax2.plot(test_acc_log, color='orange', linestyle='--', label='Test Accuracy')
        ax2.set_ylabel('Accuracy', color='black')
        ax2.tick_params(axis='y', labelcolor='black')
        
        # Combine legends
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='best')
    else:
        ax1.legend(loc='best')
    
    plt.title(title)
    plt.show()


def plot_train_test_loss_resdictv(res_dict, my_title, print_acc=True, acc_keys=None, log_keys=None, use_cross=True):
    if acc_keys is None:
        train_acc_key = 'train_accuracy'
        intra_acc_key = 'intra_test_accuracy'
        cross_acc_key = 'cross_test_accuracy'
    else:
        train_acc_key = log_keys[0]
        intra_acc_key = log_keys[1]
        cross_acc_key = log_keys[2]
    
    if log_keys is None:
        train_key = 'train_loss_log'
        intra_key = 'intra_test_loss_log'
        cross_key = 'cross_test_loss_log'
    else:
        train_key = log_keys[0]
        intra_key = log_keys[1]
        cross_key = log_keys[2]

    if print_acc:
        print("Final Accuracies (averaged across users and gestures)")
        if train_acc_key is not None:
            print(f"Train accuracy: {res_dict[train_acc_key]*100:.2f}%")
        if intra_acc_key is not None:
            print(f"Intra test accuracy: {res_dict[intra_acc_key]*100:.2f}%")
        if cross_acc_key is not None and use_cross==True:
            print(f"Cross test accuracy: {res_dict[cross_acc_key]*100:.2f}%")

    if train_key is not None:
        train_log = res_dict[train_key]
    if intra_key is not None:
        intra_log = res_dict[intra_key]
    if cross_key is not None and use_cross==True:
        cross_log = res_dict[cross_key]

    plt.plot(train_log, label="Train")
    plt.plot(intra_log, label="Intra Test")
    if use_cross==True:
        plt.plot(cross_log, label="Cross Test")
    plt.title(my_title)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


def visualize_train_test_loss_curves(results, config, train_loss_log=None, test_loss_log=None, my_title=None, ft=False):
    if my_title is None:
        my_title = "Train-Test Loss Curves"
    
    if ft:
        fig_filename = f"ft_train_test_loss_curves.png"
    else:
        fig_filename = f"train_test_loss_curves.png"

    if results is not None:
        plt.plot(results["train_loss_log"], label="Train")
        plt.plot(results["intra_test_loss_log"], label="Intra Test")
        plt.title("Train-Test Loss Curves")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
    elif results is None and train_loss_log is not None and test_loss_log is not None:
        plt.plot(train_loss_log, label="Train")
        plt.plot(test_loss_log, label="Intra Test")
        plt.title(my_title)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

    plt.legend()
    plt.tight_layout()
    # Generate unique filename with timestamp
    fig_dir = config["results_save_dir"]
    os.makedirs(fig_dir, exist_ok=True)
    fig_save_path = os.path.join(fig_dir, fig_filename)
    plt.savefig(fig_save_path)

    plt.show()


def plot_gesture_performance(performance_dict, title, save_filename, save_path="ELEC573_Proj\\results\\heatmaps"):
    """
    Create a heatmap of gesture performance for each participant
    
    Args:
    - performance_dict: Dictionary of performance metrics
    - title: Title for the plot
    - save_path: Optional path to save the plot
    """
    # Convert performance dict to a DataFrame
    data = []
    for participant, gesture_performance in performance_dict.items():
        for gesture, accuracy in gesture_performance.items():
            data.append({
                'Participant': participant, 
                'Gesture': gesture, 
                'Accuracy': accuracy
            })
    
    performance_df = pd.DataFrame(data)
    
    # Create a pivot table for heatmap
    performance_pivot = performance_df.pivot(
        index='Participant', 
        columns='Gesture', 
        values='Accuracy'
    )
    
    # Set up the matplotlib figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap using seaborn
    sns.heatmap(
        performance_pivot, 
        annot=True, 
        cmap='YlGnBu', 
        center=0.5, 
        vmin=0, 
        vmax=1,
        fmt='.2f'
    )
    
    plt.title(title)
    plt.tight_layout()
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    fig_filename = f"{save_filename}.png"
    fig_dir = os.path.join(save_path, f"{timestamp}")
    os.makedirs(fig_dir, exist_ok=True)
    fig_save_path = os.path.join(fig_dir, fig_filename)
    plt.savefig(fig_save_path)
    return performance_pivot


def print_detailed_performance(performance_dict, set_type):
    """
    Print detailed performance metrics for each participant and gesture
    
    Args:
    - performance_dict: Dictionary of performance metrics
    - set_type: String describing the dataset (e.g., 'Training', 'Testing')
    """
    print(f"\n--- {set_type} Set Performance ---")
    
    # Overall statistics
    all_accuracies = []
    
    for participant, gesture_performance in performance_dict.items():
        print(f"\nParticipant {participant}:")
        participant_accuracies = []
        
        for gesture, accuracy in sorted(gesture_performance.items()):
            print(f"  Gesture {gesture}: {accuracy:.2%}")
            participant_accuracies.append(accuracy)
            all_accuracies.append(accuracy)
        
        # Participant-level summary
        print(f"  Average Accuracy: {np.mean(participant_accuracies):.2%}")
    
    # Overall summary
    print(f"\nOverall {set_type} Set Summary:")
    print(f"Mean Accuracy: {np.mean(all_accuracies):.2%}")
    print(f"Accuracy Standard Deviation: {np.std(all_accuracies):.2%}")
    print(f"Minimum Accuracy: {np.min(all_accuracies):.2%}")
    print(f"Maximum Accuracy: {np.max(all_accuracies):.2%}")


def visualize_model_acc_heatmap(results, print_results=False):
    """
    Comprehensive visualization and printing of model performance
    
    Args:
    - results: Dictionary containing training and testing performance
    """
    # Visualize Training Set Performance
    train_performance_heatmap = plot_gesture_performance(
        results['train_performance'], 
        'Training Set - Gesture Performance by Participant',
        'training_performance_heatmap'
    )
    
    # Visualize Testing Set Performance
    intra_test_performance_heatmap = plot_gesture_performance(
        results['intra_test_performance'], 
        'Intra Testing Set - Gesture Performance by Participant',
        'intra_testing_performance_heatmap'
    )

    # Visualize Testing Set Performance
    cross_test_performance_heatmap = plot_gesture_performance(
        results['cross_test_performance'], 
        'Cross Testing Set - Gesture Performance by Participant',
        'cross_testing_performance_heatmap'
    )
    
    if print_results:
        # Print detailed performance
        print_detailed_performance(results['train_performance'], 'Training')
        print_detailed_performance(results['intra_test_performance'], 'Intra Testing')
        print_detailed_performance(results['cross_test_performance'], 'Cross Testing')
        
        # Additional overall metrics
        print("\nOverall Model Performance:")
        print(f"Training Accuracy: {results['train_accuracy']:.2%}")
        print(f"Intra Testing Accuracy: {results['intra_test_accuracy']:.2%}")
        print(f"Cross Testing Accuracy: {results['cross_test_accuracy']:.2%}")