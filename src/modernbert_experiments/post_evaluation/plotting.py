import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
import seaborn as sns

import src.settings as sts

def generate_metrics_line_plots_for_experiment_all_runs(dir_experiment: Path):

    # Define dictionary to hold the summaries of all best runs
    dict_summaries: dict = {}

    # Define path of the artifacts folder of given experiment
    dir_artifacts: Path = dir_experiment / sts.DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACTS

    # Iterate content of experiment specific artifacts folder
    for path_run_artifact in dir_artifacts.iterdir():
        # Ensure that we have a directory of a run
        if (path_run_artifact.is_dir()) and (dir_experiment.name in path_run_artifact.name):
            # Build filepath of the summary json
            filepath_run_summary_json: Path = path_run_artifact / sts.FILENAME_RESULTS_EXPERIMENTS_ARTIFACT_RUN_SUMMARY
            # Ensure that the json file exists
            if filepath_run_summary_json.is_file():
                # Open it and add its content mapped to the experiment run name
                with open(filepath_run_summary_json, mode="r") as f:
                    dict_summaries.update({path_run_artifact: json.load(f)})

    # Extract names of available metrics
    first_summary: dict = dict_summaries[list(dict_summaries.keys())[0]]
    available_metrics: list[str] = list(first_summary["metrics"].keys())

    # Create plots folder in experiment folder
    dir_plots: Path = dir_experiment / sts.DIR_NAME_RESULTS_EVALUATION / sts.DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACT_PLOTS
    dir_plots.mkdir(parents=True, exist_ok=True)

    # Iterate available metrics
    for metric in available_metrics:
        # Create figure for current metric
        plt.figure(figsize=(5, 3))

        # Iterate summaries
        for run_name, dict_summary in dict_summaries.items():
            dict_config: dict = dict_summary["config"]
            dict_metrics: dict = dict_summary["metrics"]
            lst_metrics: list = dict_metrics[metric]
            learning_rate: float = float(dict_config["learning_rate"])
            weight_decay: float = float(dict_config["weight_decay"])
            warmup_ratio: float = float(dict_config["warmup_ratio"])
            # TODO: Add line plot to plotly figure

            # Add a line plot for the run's metric
            plt.plot(
                lst_metrics,
                #label=f"(lr={learning_rate:.3e}, wd={weight_decay:.3e}, wr={warmup_ratio:.3e})"
            )

        # Set plot title and labels
        plt.xlabel("Epoch")
        plt.ylim((0, 1))
        plt.xlim((0, 20))
        plt.ylabel(metric)
        #plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the figure in the experiment specific plots directory
        save_path = dir_plots / f"{metric}.png"
        plt.savefig(save_path, format='png')
        plt.close()


def generate_gpu_watt_usage_f1_scatterplot(dict_watt_usage_per_experiment: dict):
    # Mapping of experiment names to display labels
    dict_exp_names = {
        "1_fix_encoder_and_train_classifier": "Approach 1",
        "2_finetune_encoder_and_train_classifier": "Approach 2",
        "3_finetune_encoder_and_train_classifier_augmented": "Approach 3",
        "4_finetune_encoder_and_train_classifier_process_mwe": "Approach 4",
        "5_finetune_encoder_and_train_classifier_process_mwe_custom_loss_1": "Approach 5"
    }

    plt.figure(figsize=(5, 3))

    # Use a colormap to obtain a set of colors
    cmap = plt.get_cmap('tab10')
    # Some versions of matplotlib support cmap.colors; otherwise, generate a list of colors.
    colors = cmap.colors if hasattr(cmap, 'colors') else [cmap(i) for i in range(10)]

    # Plot each experiment as an individual scatter point with a different color
    for i, (exp_name, metrics) in enumerate(dict_watt_usage_per_experiment.items()):
        if metrics.get("gpu_kWh") is not None and metrics.get("f1_macro") is not None:
            # Create a label with the mapped experiment name and epoch number
            label = f"{dict_exp_names.get(exp_name, exp_name)} (Epoch {metrics.get('num_epochs', '')})"
            plt.scatter(metrics["gpu_kWh"], metrics["f1_macro"],
                        color=colors[i % len(colors)], label=label)

    plt.xlabel("GPU Energy Consumption (kWh)")
    plt.ylabel("F1 Macro Score")
    plt.grid(True, zorder=0)

    # Place legend in the upper left corner
    plt.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(
        sts.DIR_PATH_RESULTS / f"{sts.FILENAME_GPU_ENERGY_CONSUMPTION_EXPERIMENTS_COMPARISON_PLOT}.png",
        format='png'
    )
    plt.savefig(
        sts.DIR_PATH_RESULTS / f"{sts.FILENAME_GPU_ENERGY_CONSUMPTION_EXPERIMENTS_COMPARISON_PLOT}.pdf",
        format='pdf'
    )
    plt.show()


def generate_confusion_matrix_for_best_model_predictions(
        predictions_output,
        dir_output: Path
):
    def plot_confusion_matrices(y_true, y_pred, labels=None, label_names=None):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

        if labels is None:
            labels = np.unique(np.concatenate((y_true, y_pred)))

        if label_names is None:
            label_names = labels  # fallback to numbers

        fig, axs = plt.subplots(1, 2, figsize=(14, 5))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axs[0],
                    xticklabels=label_names, yticklabels=label_names)
        axs[0].set_title("Confusion Matrix (Counts)")
        axs[0].set_xlabel("Predicted")
        axs[0].set_ylabel("Actual")

        sns.heatmap(cm_norm * 100, annot=True, fmt='.1f', cmap='Greens', ax=axs[1],
                    xticklabels=label_names, yticklabels=label_names)
        axs[1].set_title("Confusion Matrix (Row %)")
        axs[1].set_xlabel("Predicted")
        axs[1].set_ylabel("Actual")

        plt.tight_layout()
        plt.savefig(
            dir_output / f"{sts.FILENAME_RESULTS_EVALUATION_CONFUSION_MATRIX_PLOT}.png"
        )
        plt.savefig(
            dir_output / f"{sts.FILENAME_RESULTS_EVALUATION_CONFUSION_MATRIX_PLOT}.pdf"
        )
        plt.show()

    y_true = predictions_output.label_ids
    y_pred = np.argmax(predictions_output.predictions, axis=1)

    plot_confusion_matrices(
        y_true=y_true,
        y_pred=y_pred,
        labels=[0,1],
        label_names=['idiomatic', 'literal']
    )


def generate_loss_curves_for_best_models(dict_loss_curves: dict):
    # Determine the default legend font size and reduce by 2 points if it's a number.
    default_legend_size = mpl.rcParams.get('legend.fontsize', 10)
    try:
        legend_fontsize = default_legend_size - 2
    except TypeError:
        # If the default is not numeric, fall back to 'small'
        legend_fontsize = 'small'

    # Legend mapping dictionary
    legend_mapping = {
        "1_fix_encoder_and_train_classifier": "Approach 1",
        "2_finetune_encoder_and_train_classifier": "Approach 2",
        "3_finetune_encoder_and_train_classifier_augmented": "Approach 3",
        "4_finetune_encoder_and_train_classifier_process_mwe": "Approach 4",
        "5_finetune_encoder_and_train_classifier_process_mwe_custom_loss_1": "Approach 5",
    }

    # Set the desired uniform font size for axis labels
    label_fontsize = 10

    # Create figure with 2 vertically stacked subplots sharing the same x-axis.
    # Using gridspec_kw to remove vertical space between subplots.
    fig, (ax_train, ax_eval) = plt.subplots(
        2, 1, figsize=(4, 4.5), sharex=True, gridspec_kw={'hspace': 0}
    )

    # Plot training and evaluation loss for each experiment with smaller dots.
    for exp_key, metrics in dict_loss_curves.items():
        epochs_train = list(range(len(metrics["train_loss"])))
        epochs_eval = list(range(len(metrics["eval_loss"])))
        label = legend_mapping.get(exp_key, exp_key)

        ax_train.plot(epochs_train, metrics["train_loss"],
                      marker='o', markersize=3, label=label)
        ax_eval.plot(epochs_eval, metrics["eval_loss"],
                     marker='o', markersize=3, label=label)

    # Customize training loss subplot.
    ax_train.set_ylabel("Training Loss", fontsize=label_fontsize)
    ax_train.set_xlim(0, 20)
    # Limit the y-axis so its top is at 4.
    ax_train.set_ylim(-0.6, 0.8)
    # Manually set y-ticks to omit 4.
    ax_train.set_yticks([-0.3, 0, 0.3, 0.6])
    ax_train.legend(loc='upper right', fontsize=legend_fontsize)
    # Set the y-label position manually for consistent horizontal alignment.
    ax_train.yaxis.set_label_coords(-0.15, 0.5)
    # Enable grid on the training subplot.
    ax_train.grid(True)

    # Customize evaluation loss subplot.
    ax_eval.set_ylabel("Evaluation Loss", fontsize=label_fontsize)
    ax_eval.set_xlim(0, 20)
    # Limit the y-axis so its top is at 4.
    ax_eval.set_ylim(0.5, 4)
    # Manually set y-ticks to omit 4.
    ax_eval.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax_eval.legend(loc='lower right', fontsize=legend_fontsize)
    # Set the y-label position manually for consistent horizontal alignment.
    ax_eval.yaxis.set_label_coords(-0.15, 0.5)
    # Enable grid on the evaluation subplot.
    ax_eval.grid(True)


    # Set the x-axis label on the bottom subplot with the same font size.
    ax_eval.set_xlabel("Epoch", fontsize=label_fontsize, labelpad=8)

    plt.tight_layout(pad=0.4)

    plt.savefig(
        sts.DIR_PATH_RESULTS  / f"{sts.FILENAME_GPU_ENERGY_CONSUMPTION_LOSS_COMPARISON_PLOT}.png"
    )
    plt.savefig(
        sts.DIR_PATH_RESULTS  / f"{sts.FILENAME_GPU_ENERGY_CONSUMPTION_LOSS_COMPARISON_PLOT}.pdf"
    )

    plt.show()
