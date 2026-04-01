import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments
)

from src.modernbert_experiments.dataset import get_split_datasets
from src.modernbert_experiments.post_evaluation.plotting import (
    generate_metrics_line_plots_for_experiment_all_runs,
    generate_confusion_matrix_for_best_model_predictions,
    generate_loss_curves_for_best_models,
    generate_gpu_watt_usage_f1_scatterplot
)
from src.modernbert_experiments.trainer import SlidTrainer
import src.settings as sts



def extract_examples_per_tp_tn_fp_fn(
        predictions_output,
        tokenizer,
        ds_val,  # assumed to be a Hugging Face Dataset with an "id" column
        num_export_examples_per_tp_tn_fp_fn: int,
        dir_output: Path
):
    """"""
    # Extract predicted classes similar to compute_metrics
    logits = predictions_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    if logits.ndim == 1 or logits.shape[-1] == 1:
        pred_classes = (logits >= 0).astype(int)
    else:
        pred_classes = np.argmax(logits, axis=1)
    label_ids = predictions_output.label_ids

    # Prepare containers for examples
    true_positives = []
    true_negatives = []
    false_positives = []
    false_negatives = []

    # Iterate examples and log as TP, TN, FP, FN using the unique "id" field
    for i, example in enumerate(ds_val):
        actual = label_ids[i]
        pred = pred_classes[i]
        unique_id = example["id"]  # Unique identifier from CSV
        input_ids = example["input_ids"]
        text_decoded = tokenizer.decode(input_ids, skip_special_tokens=True)
        exported_example = {
            "id": unique_id,
            "text": text_decoded,
            "prediction": int(pred),
            "label": int(actual)
        }

        if pred == 1 and actual == 1:
            true_positives.append(exported_example)
        elif pred == 0 and actual == 0:
            true_negatives.append(exported_example)
        elif pred == 1 and actual == 0:
            false_positives.append(exported_example)
        elif pred == 0 and actual == 1:
            false_negatives.append(exported_example)

    # Helper function to sample examples if more than requested
    def sample_examples(examples, n):
        if n is not None and len(examples) > n:
            return random.sample(examples, n)
        return examples

    tp_sampled = sample_examples(true_positives, num_export_examples_per_tp_tn_fp_fn)
    tn_sampled = sample_examples(true_negatives, num_export_examples_per_tp_tn_fp_fn)
    fp_sampled = sample_examples(false_positives, num_export_examples_per_tp_tn_fp_fn)
    fn_sampled = sample_examples(false_negatives, num_export_examples_per_tp_tn_fp_fn)

    export_dict = {
        "true_positive": tp_sampled,
        "true_negative": tn_sampled,
        "false_positive": fp_sampled,
        "false_negative": fn_sampled,
    }

    # Save the examples in JSON format.
    export_filepath: Path = dir_output / sts.FILENAME_RESULTS_EVALUATION_PREDICTION_EXAMPLES
    with open(export_filepath, "w") as f:
        json.dump(export_dict, f, indent=sts.JSON_INDENT)

    print(f"Exported evaluation prediction examples: {export_filepath}")


def get_watt_usage_dict_for_experiment(dir_experiment: Path) -> dict:

    # Define path of the best content
    filepath_gpu_watt_usage_file: Path =\
            dir_experiment / sts.DIR_NAME_RESULTS_BEST / sts.FILENAME_GPU_WATT_USAGE_FILE

    # Read watt usage as list
    df_watt_usage: pd.DataFrame = pd.read_csv(
        filepath_or_buffer=filepath_gpu_watt_usage_file,
        sep=","
    )

    filepath_ray_result_json: Path = (
            dir_experiment
            / sts.DIR_NAME_RESULTS_BEST
            / sts.FILENAME_RESULTS_EXPERIMENT_ARTIFACT_RAY_RESULT_JSON_FILE
    )
    if filepath_ray_result_json.is_file():
        # Open it and add its content mapped to the experiment run name
        with open(filepath_ray_result_json, mode="r") as f:
            dict_ray_result: dict = json.load(f)

    filepath_run_summary_json: Path = (
            dir_experiment
            / sts.DIR_NAME_RESULTS_BEST
            / sts.FILENAME_RESULTS_EXPERIMENTS_ARTIFACT_RUN_SUMMARY
    )
    if filepath_run_summary_json.is_file():
        with open(filepath_run_summary_json, mode="r") as f:
            dict_run_summary: dict = json.load(f)

    # The first column is renamed to "time", and the second to "power"
    new_columns = []
    for i, col in enumerate(df_watt_usage.columns):
        if i == 0:
            new_columns.append("time")
        elif i == 1:
            new_columns.append("power")
        else:
            new_columns.append(col)
    df_watt_usage.columns = new_columns

    # Ensure that the "time" and "power" columns are numeric
    df_watt_usage["time"] = pd.to_numeric(df_watt_usage["time"], errors="coerce")
    df_watt_usage["power"] = pd.to_numeric(df_watt_usage["power"], errors="coerce")

    # Calculate the time differences (in seconds) between consecutive measurements
    df_watt_usage["delta_time"] = df_watt_usage["time"].diff().fillna(0)

    # Compute the energy for each interval (Joules), as 1 Watt * 1 second equals 1 Joule
    df_watt_usage["energy"] = df_watt_usage["delta_time"] * df_watt_usage["power"]

    # Calculate the cumulative sum of energy (in Joules)
    df_watt_usage["cumulative_energy"] = df_watt_usage["energy"].cumsum()

    # Optionally, convert cumulative energy to kilowatt-hours (1 kWh = 3,600,000 Joules)
    df_watt_usage["cumulative_energy_kWh"] = df_watt_usage["cumulative_energy"] / 3.6e6

    # Update a
    return {
        "accuracy": dict_ray_result["eval_accuracy"],
        "f1_macro": dict_ray_result["eval_f1_macro"],
        "num_epochs": len(dict_run_summary["metrics"]["train_loss"]),
        "gpu_kWh": df_watt_usage["cumulative_energy_kWh"].iloc[-1]
    }

def get_loss_curves_dict_for_experiment(dir_experiment: Path) -> dict:
    filepath_run_summary_json: Path = (
            dir_experiment
            / sts.DIR_NAME_RESULTS_BEST
            / sts.FILENAME_RESULTS_EXPERIMENTS_ARTIFACT_RUN_SUMMARY
    )

    with open(filepath_run_summary_json, mode="r") as f:
        dict_summary: dict = json.load(f)

    dict_metrics: dict = dict_summary["metrics"]

    return {
        "train_loss": dict_metrics["train_loss"],
        "eval_loss": dict_metrics["eval_loss"],
    }

def evaluate_experiment_based_on_inference_predictions_using_best_model(
        output_dir: Path,
        dir_best_model: Path,
        fp_ray_result: Path,
        num_export_examples_per_tp_tn_fp_fn: int
):
    with open(fp_ray_result, mode="r") as f:
        dict_ray_result: dict = json.load(f)

    dict_config: dict = dict_ray_result["config"]

    # Load data stuff
    tokenizer, ds_train, ds_val = get_split_datasets(
        config=dict_config
    )

    # Load saved model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(
        dir_best_model,
        output_hidden_states=True
    )
    model.resize_token_embeddings(len(tokenizer))

    # Define training arguments for evaluation (only required by the Trainer)
    eval_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=dict_config["batch_size"],
        report_to=[],  # Disable reporting (e.g., to W&B).
    )

    # Create the trainer. Here, SlidTrainer is used which already implements compute_metrics
    trainer = SlidTrainer(
        model=model,
        args=eval_args,
        tokenizer=tokenizer,
        eval_dataset=ds_val
    )

    # Generate predictions.
    predictions_output = trainer.predict(ds_val)

    ### Evaluation function calls
    extract_examples_per_tp_tn_fp_fn(
        predictions_output=predictions_output,
        tokenizer=tokenizer,
        ds_val=ds_val,
        num_export_examples_per_tp_tn_fp_fn=num_export_examples_per_tp_tn_fp_fn,
        dir_output=output_dir
    )
    generate_confusion_matrix_for_best_model_predictions(
        predictions_output=predictions_output,
        dir_output=output_dir
    )

    # The compute_metrics method expects a tuple (logits, labels)
    metrics = trainer.compute_metrics((predictions_output.predictions, predictions_output.label_ids))

    # Optionally print the metrics.
    print(f"F1 Macro Score: {metrics["f1_macro"]}")
    # for key, value in metrics.items():
    #     print(f"{key}: {value}")

    return metrics


def evaluate_experiments(evaluate_log: bool = True, evaluate_model: bool = True):
    dir_experiment_names: list[str] = [
        "1_fix_encoder_and_train_classifier",
        "2_finetune_encoder_and_train_classifier",
        "3_finetune_encoder_and_train_classifier_augmented",
        "4_finetune_encoder_and_train_classifier_process_mwe",
        "5_finetune_encoder_and_train_classifier_process_mwe_custom_loss_1",
    ]

    dict_watt_usage_per_experiment: dict = {}
    dict_loss_curves: dict = {}

    for exp_name in tqdm(dir_experiment_names, desc="Evaluating experiments: "):
        # Define required paths
        dir_exp: Path = sts.DIR_PATH_RESULTS_EXPERIMENTS / exp_name
        dir_exp_best: Path = dir_exp / sts.DIR_NAME_RESULTS_BEST
        dir_best_model: Path = dir_exp_best / sts.DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACT_MODEL
        dir_evaluation: Path = dir_exp / sts.DIR_NAME_RESULTS_EVALUATION
        fp_ray_result: Path = dir_exp_best / sts.FILENAME_RESULTS_EXPERIMENT_ARTIFACT_RAY_RESULT_JSON_FILE

        # Evaluate the experiment based on the log files
        if evaluate_log:
            # Create plotting results
            generate_metrics_line_plots_for_experiment_all_runs(
                dir_experiment=dir_exp
            )
            dict_watt_usage_per_experiment.update(
                {
                    exp_name: get_watt_usage_dict_for_experiment(dir_experiment=dir_exp)
                }
            )
            dict_loss_curves.update(
                {
                    exp_name: get_loss_curves_dict_for_experiment(dir_experiment=dir_exp)
                }
            )

        # Evaluate the experiment by generating predictions
        if evaluate_model:
            evaluate_experiment_based_on_inference_predictions_using_best_model(
                output_dir=dir_evaluation,
                dir_best_model=dir_best_model,
                fp_ray_result=fp_ray_result,
                num_export_examples_per_tp_tn_fp_fn=100
            )

    if evaluate_log:
        # generate_gpu_watt_usage_f1_scatterplot(dict_watt_usage_per_experiment)
        generate_loss_curves_for_best_models(dict_loss_curves)


if __name__ == "__main__":
    evaluate_experiments(
        evaluate_log=True,
        evaluate_model=False
    )
