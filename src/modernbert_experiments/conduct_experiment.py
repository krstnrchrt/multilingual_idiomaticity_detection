import shutil
from datetime import datetime
from pathlib import Path

import ray
import wandb
from ray import tune
from ray.tune.search.bayesopt import BayesOptSearch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments, EarlyStoppingCallback
)

from src.modernbert_experiments.dataset import get_split_datasets
import src.settings as sts
from src.modernbert_experiments.loss_functions import (
    VanillaCrossEntropyLoss,
    AbstractSimilarityLossFunction,
)
from src.modernbert_experiments.trainer import (
    SlidTrainer,
    LogCallback
)
import src.utils as uts


def make_model_init(config: dict, tokenizer):
    """Creates a model initialization function with the given configuration and tokenizer."""
    def model_init():
        """Initializes a model with the given configuration."""
        model_name = config["model_name"]
        if model_name == sts.MODERN_BERT_NAME:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=sts.NUM_LABELS,
                reference_compile=False,
                output_hidden_states=True  # Enable hidden states output
            )
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=sts.NUM_LABELS,
                output_hidden_states=True  # Enable hidden states output
            )

        model.resize_token_embeddings(len(tokenizer))

        # Fix the weights of ModernBERT (encoder)
        if config["fix_encoder"]:
            if model_name != sts.MODERN_BERT_NAME:
                raise ValueError("Fixing the encoder only supported for ModernBERT!")

            if hasattr(model, "model"):
                for param in model.model.parameters():
                    param.requires_grad = False
            else:
                raise ValueError("ModernBERT model must have 'model' attribute!")

        return model

    return model_init


def tune_train(config: dict, ds_train=None, ds_val=None, tokenizer=None):
    """Trains a model with the given configuration and reports the evaluation metrics to Ray Tune."""
    # Define run name
    run_name: str = f"{config['experiment_name']}_{datetime.now().strftime("%Y%m%d_%H%M%S")}"
    config["run_name"] = run_name

    # Define artifacts folder for this run
    dir_artifacts: Path = config["dir_artifacts"] / run_name
    config["dir_artifacts_run_specific"] = dir_artifacts

    # Start a new W&B run for each trial
    wandb.init(
        project=sts.WANDB_PROJECT,
        entity=sts.WANDB_ENTITY,
        name=run_name,
        config=config
    )

    # Define arguments for the trainer
    training_args = TrainingArguments(
        run_name=run_name,
        output_dir=str(dir_artifacts),
        metric_for_best_model=config["metric_for_best_model"],
        greater_is_better=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        remove_unused_columns=False,
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        optim=config["optimizer"],
        learning_rate=config["learning_rate"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        fp16=True,
        logging_dir=str(sts.DIR_PATH_HUGGING_FACE_LOGS),
        load_best_model_at_end=True,
        report_to="wandb",
    )

    if "loss_function" not in config:
        # Allow default loss function
        loss_function = None
    elif config["loss_function"] == VanillaCrossEntropyLoss:
        # Use base loss function of custom ones
        loss_function = VanillaCrossEntropyLoss()
    elif issubclass((cls := config["loss_function"]), AbstractSimilarityLossFunction):
        # Allow similarity loss implementation
        similarity_alpha = config["similarity_alpha"]
        loss_function = cls(alpha=similarity_alpha)
    else:
        raise TypeError("Not supported loss function type!")

    trainer = SlidTrainer(
        model_init=make_model_init(config, tokenizer),
        args=training_args,
        loss_function=loss_function,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"]),
            LogCallback(dir_artifacts=dir_artifacts)
        ]
    )

    # Train model and evaluate using loaded best model
    trainer.train()
    eval_results = trainer.evaluate()

    # Save best model and checkpoint
    dir_best_checkpoint: str = trainer.state.best_model_checkpoint
    dir_best_model: str = str(
        dir_artifacts /
        sts.DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACT_BEST_MODEL
        / Path(dir_best_checkpoint).name
    )
    trainer.save_model(dir_best_model)
    tokenizer.save_pretrained(dir_best_model)
    print(f"Best model and tokenizer saved to: {dir_best_model}")

    # Define a destination folder for the final best checkpoint
    final_checkpoint_dir = (
            dir_artifacts
            / sts.DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACT_BEST_CHECKPOINT
            / Path(dir_best_checkpoint).name
    )
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Copy the entire checkpoint folder (including optimizer and scheduler state)
    shutil.copytree(dir_best_checkpoint, final_checkpoint_dir, dirs_exist_ok=True)
    print(f"Best checkpoint saved to: {final_checkpoint_dir}")

    # Remove checkpoints that do not belong to the best state of the model
    if sts.CLEANUP_CHECKPOINTS_AFTER_TRAINING:
        print("Removing non-best checkpoints...")
        for dir_checkpoint in dir_artifacts.glob("checkpoint-*"):
            if dir_checkpoint.is_dir():
                shutil.rmtree(dir_checkpoint)

    # Report the evaluation metrics and the trial's checkpoint to Ray Tune to select best model
    tune.report(
        metrics={
            "eval_loss": eval_results["eval_loss"],

            "eval_accuracy": eval_results["eval_accuracy"],
            "eval_accuracy_balanced": eval_results["eval_accuracy_balanced"],

            "eval_f1": eval_results["eval_f1"],
            "eval_f1_micro": eval_results["eval_f1_micro"],
            "eval_f1_macro": eval_results["eval_f1_macro"],
            "eval_f1_weighted": eval_results["eval_f1_weighted"],

            "eval_f2": eval_results["eval_f2"],
            "eval_f2_micro": eval_results["eval_f2_micro"],
            "eval_f2_macro": eval_results["eval_f2_macro"],
            "eval_f2_weighted": eval_results["eval_f2_weighted"],

            "eval_precision": eval_results["eval_precision"],
            "eval_precision_micro": eval_results["eval_precision_micro"],
            "eval_precision_macro": eval_results["eval_precision_macro"],
            "eval_precision_weighted": eval_results["eval_precision_weighted"],

            "eval_recall": eval_results["eval_recall"],
            "eval_recall_micro": eval_results["eval_recall_micro"],
            "eval_recall_macro": eval_results["eval_recall_macro"],
            "eval_recall_weighted": eval_results["eval_recall_weighted"],

            "eval_roc_auc": eval_results["eval_roc_auc"],
            "eval_auprc": eval_results["eval_auprc"],
            "eval_mcc": eval_results["eval_mcc"],

            "eval_tpr": eval_results["eval_tpr"],
            "eval_tnr": eval_results["eval_tnr"],
            "eval_fnr": eval_results["eval_fnr"],

            "eval_tp": eval_results["eval_tp"],
            "eval_tn": eval_results["eval_tn"],
            "eval_fp": eval_results["eval_fp"],
            "eval_fn": eval_results["eval_fn"]
        }
    )

    wandb.finish()


def conduct_experiment(config: dict):
    """"Executes on experiment with the given configuration."""

    dir_experiment: Path = config["dir_experiment"]
    dir_artifacts: Path = config["dir_artifacts"]
    dir_ray: Path = config["dir_ray"]
    dir_best: Path = config["dir_best"]

    dir_experiment.mkdir(parents=True, exist_ok=True)
    dir_artifacts.mkdir(parents=True, exist_ok=True)
    dir_ray.mkdir(parents=True, exist_ok=True)

    # Prepare data
    tokenizer, ds_train, ds_val = get_split_datasets(config=config)
    ds_train_ref = ray.put(ds_train)
    ds_val_ref = ray.put(ds_val)

    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=sts.NUM_CPUs,
            num_gpus=sts.NUM_GPUs
        )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(
                tune_train,
                ds_train=ds_train_ref,
                ds_val=ds_val_ref,
                tokenizer=tokenizer
            ),
            resources={
                "cpu": sts.NUM_CPUs,
                "gpu": sts.NUM_GPUs
            }
        ),
        tune_config=tune.TuneConfig(
            metric=config["metric_for_best_model"],
            mode=config["metric_mode_for_best_model"],
            search_alg=BayesOptSearch(),
            num_samples=config["num_samples"],
            max_concurrent_trials=1
        ),
        param_space=config,
        run_config=tune.RunConfig(
            name=config["experiment_name"],
            storage_path=str(config["dir_ray"])
        )
    )

    # Search...
    results = tuner.fit()

    # Handle best model
    best_trial = results.get_best_result(metric="eval_f1_macro", mode="max")
    print(f"Tune Results Experiment Directory: {results.experiment_path}")
    best_run_name = best_trial.config.get("run_name")
    print(f"Best run name: {best_run_name}")
    print(f"Best model config: {best_trial.config}")

    # Copy best model, respective checkpoint and summary JSON file
    dir_best_run_artifacts: Path = dir_artifacts / best_run_name
    dir_best_checkpoint: Path = uts.retrieve_artifact_path_from_nested_best_artifact_directory(
        dir_best_artifact=dir_best_run_artifacts / sts.DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACT_BEST_CHECKPOINT
    )
    dir_best_model: Path = uts.retrieve_artifact_path_from_nested_best_artifact_directory(
        dir_best_artifact=dir_best_run_artifacts / sts.DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACT_BEST_MODEL
    )
    dir_best.mkdir(parents=True, exist_ok=True)
    uts.copy_dir_contents(dir_best_checkpoint, dir_best / sts.DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACT_CHECKPOINT)
    uts.copy_dir_contents(dir_best_model, dir_best / sts.DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACT_MODEL)
    filepath_best_run_summary: Path = dir_best_run_artifacts / sts.FILENAME_RESULTS_EXPERIMENTS_ARTIFACT_RUN_SUMMARY
    shutil.copy(filepath_best_run_summary, dir_best / filepath_best_run_summary.name)

    # Copy ray result JSON file of the best trial to the artifacts folder of the best trial
    filepath_ray_result: Path = Path(best_trial.path) / sts.FILENAME_RAY_RESULT_JSON_FILE
    shutil.copy(filepath_ray_result, dir_best / sts.FILENAME_RESULTS_EXPERIMENT_ARTIFACT_RAY_RESULT_JSON_FILE)

    # Finish W&B run for this HP-search
    wandb.finish()
