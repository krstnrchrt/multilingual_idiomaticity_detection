from typing import Final
from pathlib import Path
import traceback

import torch.cuda
from ray import tune

import src.settings as sts
import src.utils as uts
from src.modernbert_experiments.conduct_experiment import conduct_experiment
from src.modernbert_experiments.loss_functions import (
    VanillaCrossEntropyLoss,
    CosineSimilarityMWEWithTargetSentenceLoss,
    LossFunction
)


num_train_epochs: Final[int] = 20
num_samples: Final[int] = 18
batch_size: Final[int] = 32
metric_for_best_model: Final[str] = "eval_f1_macro"
metric_mode_for_best_model: Final[str] = "max"
early_stopping_patience: Final[int] = 5
warmup_ratio: Final = tune.uniform(1e-1, 3e-1)
learning_rate: Final = tune.uniform(1e-6, 1e-4)
weight_decay: Final = tune.uniform(1e-3, 1e-1)
similarity_alpha = tune.uniform(1e-1, 5e-1)


def define_test_experiment() -> dict:
    experiment_name: str = "0_test_experiment"
    dir_experiment, dir_artifacts, dir_ray, dir_best = uts.define_experiment_directory_names(
        experiment_name=experiment_name
    )
    config = {
        "experiment_name": experiment_name,
        "dir_experiment": dir_experiment,
        "dir_artifacts": dir_artifacts,
        "dir_ray": dir_ray,
        "dir_best": dir_best,
        "use_augmented_train_zero_shot_data": False,
        "scenario": sts.Scenario.ZERO_SHOT,
        "model_name": sts.MODERN_BERT_NAME,
        "fix_encoder": False,
        "process_mwe": True,
        "loss_function": VanillaCrossEntropyLoss,
        "num_train_epochs": 2,
        "num_samples": 1,
        "batch_size": 32,
        "early_stopping_patience" : early_stopping_patience,
        "metric_for_best_model": metric_for_best_model,
        "metric_mode_for_best_model": metric_mode_for_best_model,
        "optimizer": "adamw_torch_fused",
        "lr_scheduler": "linear",
        "warmup_ratio": 0.1,
        "learning_rate": 0.0001,
        "weight_decay": 0.1
    }

    return config


def define_experiment_1() -> dict:
    experiment_name: str = "1_fix_encoder_and_train_classifier"
    dir_experiment, dir_artifacts, dir_ray, dir_best = uts.define_experiment_directory_names(
        experiment_name=experiment_name
    )
    config = {
        "experiment_name": experiment_name,
        "dir_experiment": dir_experiment,
        "dir_artifacts": dir_artifacts,
        "dir_ray": dir_ray,
        "dir_best": dir_best,
        "use_augmented_train_zero_shot_data": False,
        "scenario": sts.Scenario.ZERO_SHOT,
        "model_name": sts.MODERN_BERT_NAME,
        "fix_encoder": True,
        "process_mwe": False,
        "loss_function": VanillaCrossEntropyLoss,
        "num_train_epochs": num_train_epochs,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "early_stopping_patience" : early_stopping_patience,
        "metric_for_best_model": metric_for_best_model,
        "metric_mode_for_best_model": metric_mode_for_best_model,
        "optimizer": "adamw_torch_fused",
        "lr_scheduler": "linear",
        "warmup_ratio": warmup_ratio,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay
    }

    return config


def define_experiment_2() -> dict:
    experiment_name: str = "2_finetune_encoder_and_train_classifier"
    dir_experiment, dir_artifacts, dir_ray, dir_best = uts.define_experiment_directory_names(
        experiment_name=experiment_name
    )
    config = {
        "experiment_name": experiment_name,
        "dir_experiment": dir_experiment,
        "dir_artifacts": dir_artifacts,
        "dir_ray": dir_ray,
        "dir_best": dir_best,
        "use_augmented_train_zero_shot_data": False,
        "scenario": sts.Scenario.ZERO_SHOT,
        "model_name": sts.MODERN_BERT_NAME,
        "fix_encoder": False,
        "process_mwe": False,
        "loss_function": VanillaCrossEntropyLoss,
        "num_train_epochs": num_train_epochs,
        "num_samples": num_samples,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "metric_for_best_model": metric_for_best_model,
        "metric_mode_for_best_model": metric_mode_for_best_model,
        "optimizer": "adamw_torch_fused",
        "lr_scheduler": "linear",
        "warmup_ratio": warmup_ratio,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
    }

    return config


def define_experiment_3() -> dict:
    experiment_name: str = "3_finetune_encoder_and_train_classifier_augmented"
    dir_experiment, dir_artifacts, dir_ray, dir_best = uts.define_experiment_directory_names(
        experiment_name=experiment_name
    )
    config = {
        "experiment_name": experiment_name,
        "dir_experiment": dir_experiment,
        "dir_artifacts": dir_artifacts,
        "dir_ray": dir_ray,
        "dir_best": dir_best,
        "use_augmented_train_zero_shot_data": True,
        "scenario": sts.Scenario.ZERO_SHOT,
        "model_name": sts.MODERN_BERT_NAME,
        "fix_encoder": False,
        "process_mwe": False,
        "loss_function": VanillaCrossEntropyLoss,
        "num_train_epochs": num_train_epochs,
        "num_samples": 1,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "metric_for_best_model": metric_for_best_model,
        "metric_mode_for_best_model": metric_mode_for_best_model,
        "optimizer": "adamw_torch_fused",
        "lr_scheduler": "linear",
        "warmup_ratio": 0.1312037280884873,
        "learning_rate": 6.0267189935506624e-05,
        "weight_decay": 0.016443457513284063,
    }

    return config


def define_experiment_4() -> dict:
    experiment_name: str = "4_finetune_encoder_and_train_classifier_process_mwe"
    dir_experiment, dir_artifacts, dir_ray, dir_best = uts.define_experiment_directory_names(
        experiment_name=experiment_name
    )
    config = {
        "experiment_name": experiment_name,
        "dir_experiment": dir_experiment,
        "dir_artifacts": dir_artifacts,
        "dir_ray": dir_ray,
        "dir_best": dir_best,
        "use_augmented_train_zero_shot_data": False,
        "scenario": sts.Scenario.ZERO_SHOT,
        "model_name": sts.MODERN_BERT_NAME,
        "fix_encoder": False,
        "process_mwe": True,
        "loss_function": VanillaCrossEntropyLoss,
        "num_train_epochs": num_train_epochs,
        "num_samples": 1,
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "metric_for_best_model": metric_for_best_model,
        "metric_mode_for_best_model": metric_mode_for_best_model,
        "optimizer": "adamw_torch_fused",
        "lr_scheduler": "linear",
        "warmup_ratio": 0.1312037280884873,
        "learning_rate": 6.0267189935506624e-05,
        "weight_decay": 0.016443457513284063,
    }

    return config


def define_experiment_5() -> dict:
    loss_function: type[LossFunction] = CosineSimilarityMWEWithTargetSentenceLoss
    experiment_name: str = f"5_finetune_encoder_and_train_classifier_process_mwe_{loss_function.get_identifier()}"
    dir_experiment, dir_artifacts, dir_ray, dir_best = uts.define_experiment_directory_names(
        experiment_name=experiment_name
    )
    config = {
        "experiment_name": experiment_name,
        "dir_experiment": dir_experiment,
        "dir_artifacts": dir_artifacts,
        "dir_ray": dir_ray,
        "dir_best": dir_best,
        "use_augmented_train_zero_shot_data": False,
        "scenario": sts.Scenario.ZERO_SHOT,
        "model_name": sts.MODERN_BERT_NAME,
        "fix_encoder": False,
        "process_mwe": True,
        "loss_function": loss_function,
        "num_train_epochs": num_train_epochs,
        "num_samples": 12,  # Only twelve because we reduced the number of tunable hyperparameters from 3 to 2
        "batch_size": batch_size,
        "early_stopping_patience": early_stopping_patience,
        "metric_for_best_model": metric_for_best_model,
        "metric_mode_for_best_model": metric_mode_for_best_model,
        "optimizer": "adamw_torch_fused",
        "lr_scheduler": "linear",
        "warmup_ratio": 0.1312037280884873,
        "learning_rate": learning_rate,
        "weight_decay": 0.016443457513284063,
        "similarity_alpha": similarity_alpha
    }

    return config


# Entrypoint of this script to execute the experiments
if __name__ == "__main__":
    print(f"Cuda is available {torch.cuda.is_available()}")

    experiment_configs: list[dict] = [
        # define_test_experiment()
        # define_experiment_1(),
        # define_experiment_2(),
        # define_experiment_3(),
        # define_experiment_4(),
        # define_experiment_5()
    ]

    for ind, experiment_config in enumerate(experiment_configs):
        exp_name: str = experiment_config["experiment_name"]
        exp_dir: Path = experiment_config["dir_experiment"]
        try:
            print(f"({ind+1}/{len(experiment_configs)}) Starting experiment: {exp_name}")
            conduct_experiment(config=experiment_config)
            print(f"Finished experiment: {exp_name}")
        except (Exception,) as e:
            print(
                f"\n\n"
                f"An Error occurred while conducting experiment {exp_name}!\n"
                f"Error: {e}"
                f"\n\n"
            )
            # Write the error message and traceback to the file.
            if not exp_dir.is_dir():
                exp_dir.mkdir(parents=True, exist_ok=True)
            with open(
                    experiment_config["dir_experiment"] / sts.FILENAME_RESULTS_EXPERIMENTS_ARTIFACT_RUN_ERROR,
                    mode="w"
            ) as f:
                f.write(f"An error occurred while conducting experiment {exp_name}:\n")
                f.write(str(e) + "\n")
                f.write(traceback.format_exc())

    print("Finished all experiments!")
