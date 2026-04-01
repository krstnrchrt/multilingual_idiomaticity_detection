import os
from enum import Enum
from pathlib import Path
from typing import Final
from transformers import set_seed

from src.utils import solve_secrets_related_env


class Scenario(Enum):
    ZERO_SHOT = "zero_shot"
    ONE_SHOT = "one_shot"


""""Remove all checkpoints that are not linked to the best state of a model in its training."""
CLEANUP_CHECKPOINTS_AFTER_TRAINING: Final[bool] = True


"""Define constant numbers."""
SEED: Final[int] = 42
CLASS_POSITIVE: Final[int] = 0
CLASS_NEGATIVE: Final[int] = 1
NUM_LABELS: Final[int] = 2
NUM_GPUs: Final[int] = 1
NUM_CPUs: Final[int] = 24

JSON_INDENT: Final[int] = 2

# Per split max length of tokenized sequences (unpadded) with context sentences
TRAIN_SPLIT_ZERO_SHOT_MAX_LENGTH_TOKENIZED_SEQUENCE: Final[int] = 387
TRAIN_SPLIT_ONE_SHOT_MAX_LENGTH_TOKENIZED_SEQUENCE: Final[int] = 343
TEST_SPLIT_MAX_LENGTH_TOKENIZED_SEQUENCE: Final[int] = 336

SPLITS_MAX_LENGTH_TOKENIZED_SEQUENCE: Final[int] = max(
    TRAIN_SPLIT_ZERO_SHOT_MAX_LENGTH_TOKENIZED_SEQUENCE,
    TRAIN_SPLIT_ONE_SHOT_MAX_LENGTH_TOKENIZED_SEQUENCE,
    TEST_SPLIT_MAX_LENGTH_TOKENIZED_SEQUENCE
)


"""Set the seed for reproducibility."""
set_seed(SEED)


"""Define constant filenames and paths."""
# Filenames of CSV files
FILENAME_TRAIN_ZERO_SHOT_CSV: Final[str] = "train_zero_shot.csv"
FILENAME_TRAIN_ZERO_SHOT_AUGMENTED_CSV: Final[str] = "train_zero_shot_augmented.csv"
FILENAME_TRAIN_ONE_SHOT_CSV: Final[str] = "train_one_shot.csv"
FILENAME_TEST_CSV: Final[str] = "test.csv"
FILENAME_TEST_GOLD: Final[str] = "test_gold.csv"

# Path of the project root
PROJECT_ROOT_DIR: Final[Path] = Path(__file__).parent.parent

# Path of the mounted (mounted) folder
DIR_PATH_RESULTS: Final[Path] = PROJECT_ROOT_DIR / "results"
DIR_PATH_RESULTS_EXPERIMENTS: Final[Path] = DIR_PATH_RESULTS / "experiments"
DIR_NAME_RESULTS_EXPERIMENTS_RAY: Final[str] = "ray"
DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACTS: Final[str] = "artifacts"
DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACT_PLOTS: Final[str] = "plots"
DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACT_CHECKPOINT: Final[str] = "checkpoint"
DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACT_MODEL: Final[str] = "model"
DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACT_BEST_CHECKPOINT: Final[str] = "best_checkpoint"
DIR_NAME_RESULTS_EXPERIMENTS_ARTIFACT_BEST_MODEL: Final[str] = "best_model"
FILENAME_RESULTS_EXPERIMENTS_ARTIFACT_RUN_SUMMARY: Final[str] = "run_summary.json"
FILENAME_RESULTS_EXPERIMENTS_ARTIFACT_RUN_ERROR: Final[str] = "error.txt"
FILENAME_RESULTS_EXPERIMENT_ARTIFACT_RAY_RESULT_JSON_FILE: Final[str] = "ray_result.json"
DIR_NAME_RESULTS_BEST: Final[str] = "best"
DIR_NAME_RESULTS_EVALUATION: Final[str] = "evaluation"
FILENAME_RESULTS_EVALUATION_PREDICTION_EXAMPLES: Final[str] = "prediction_examples.json"
FILENAME_RESULTS_EVALUATION_CONFUSION_MATRIX_PLOT: Final[str] = "confusion_matrix"
FILENAME_RAY_RESULT_JSON_FILE: Final[str] = "result.json"
FILENAME_GPU_WATT_USAGE_FILE: Final[str] = "GPU_watt_usage.csv"
FILENAME_GPU_ENERGY_CONSUMPTION_EXPERIMENTS_COMPARISON_PLOT: Final[str] = "energy_consumption_plot"
FILENAME_GPU_ENERGY_CONSUMPTION_LOSS_COMPARISON_PLOT: Final[str] = "loss_comparison_plot"


# Path of the data folder
DIR_PATH_DATA: Final[Path] = PROJECT_ROOT_DIR / "data"

# Paths of files in the original dataset
DIR_PATH_DATA_ORIGINAL: Final[Path] = DIR_PATH_DATA / "original"
FILEPATH_DATA_ORIGINAL_TRAIN_ZEROSHOT_CSV: Final[Path] = DIR_PATH_DATA_ORIGINAL / FILENAME_TRAIN_ZERO_SHOT_CSV
FILEPATH_DATA_ORIGINAL_TRAIN_ONESHOT_CSV: Final[Path] =  DIR_PATH_DATA_ORIGINAL / FILENAME_TRAIN_ONE_SHOT_CSV
FILEPATH_DATA_ORIGINAL_TEST_CSV: Final[Path] = DIR_PATH_DATA_ORIGINAL / FILENAME_TEST_CSV
FILEPATH_DATA_ORIGINAL_TEST_GOLD_CSV: Final[Path] = DIR_PATH_DATA_ORIGINAL / FILENAME_TEST_GOLD

# Paths of files in the final dataset
DIR_PATH_DATA_FINAL: Final[Path] = DIR_PATH_DATA / "final"
FILEPATH_DATA_FINAL_TRAIN_ZEROSHOT_CSV: Final[Path] = DIR_PATH_DATA_FINAL / FILENAME_TRAIN_ZERO_SHOT_CSV
FILEPATH_DATA_FINAL_TRAIN_ZEROSHOT_AUGMENTED_CSV: Final[Path] = DIR_PATH_DATA_FINAL / FILENAME_TRAIN_ZERO_SHOT_AUGMENTED_CSV
FILEPATH_DATA_FINAL_TRAIN_ONESHOT_CSV: Final[Path] =  DIR_PATH_DATA_FINAL / FILENAME_TRAIN_ONE_SHOT_CSV
FILEPATH_DATA_FINAL_TEST_CSV: Final[Path] = DIR_PATH_DATA_FINAL / FILENAME_TEST_CSV

# Paths of Weights & Biases
DIR_PATH_WEIGHTS_AND_BIASES: Final[Path] = PROJECT_ROOT_DIR / "wandb"
DIR_PATH_WEIGHTS_AND_BIASES_LOGS: Final[Path] = DIR_PATH_WEIGHTS_AND_BIASES / "logs"
DIR_PATH_WEIGHTS_AND_BIASES_RESULTS: Final[Path] = DIR_PATH_WEIGHTS_AND_BIASES / "results"

# Paths of Hugging Face
DIR_PATH_HUGGING_FACE: Final[Path] = PROJECT_ROOT_DIR / "huggingface"
DIR_PATH_HUGGING_FACE_LOGS: Final[Path] = DIR_PATH_HUGGING_FACE / "logs"

# Paths of secrets
DIR_PATH_SECRETS: Final[Path] = PROJECT_ROOT_DIR / "secrets"
FILEPATH_WANDB_API_KEY: Final[Path] = DIR_PATH_SECRETS / ".wandb_api_key"
FILEPATH_WANDB_USERNAME: Final[Path] = DIR_PATH_SECRETS / ".wandb_username"

# Model names
MODERN_BERT_NAME: Final[str] = "answerdotai/ModernBERT-base"
BERT_BASE_MULTILINGUAL_CASED: Final[str] = "bert-base-multilingual-cased"


"""Get and set values of environment variables with default values and secret files."""
# Personal W&B API key
WANDB_API_KEY : Final[str] = solve_secrets_related_env(
    env_name=(env_name := "WANDB_API_KEY"),
    env_secrets_filepath=FILEPATH_WANDB_API_KEY
)
os.environ[env_name] = WANDB_API_KEY

# Personal W&B API username
WANDB_USERNAME : Final[str] = solve_secrets_related_env(
    env_name=(env_name := "WANDB_USERNAME"),
    env_secrets_filepath=FILEPATH_WANDB_USERNAME
)
os.environ[env_name] = WANDB_USERNAME

# Entity (the team)
WANDB_ENTITY : Final[str] = solve_secrets_related_env(
    env_name=(env_name := "WANDB_ENTITY"),
    default_value="anlp-slid"
)
os.environ[env_name] = WANDB_ENTITY

# Project name
WANDB_PROJECT : Final[str] = solve_secrets_related_env(
    env_name=(env_name := "WANDB_PROJECT"),
    default_value="slid"
)
os.environ[env_name] = WANDB_PROJECT

# Allow anonymous runs
WANDB_ANONYMOUS : Final[str] = solve_secrets_related_env(
    env_name=(env_name := "WANDB_ANONYMOUS"),
    default_value="never"
)
os.environ[env_name] = WANDB_ANONYMOUS

# Default name of the run (aka experiment according to our nomenclature)
WANDB_NAME : Final[str] = solve_secrets_related_env(
    env_name=(env_name := "WANDB_NAME"),
    default_value="please-define-a-name"
)
os.environ[env_name] = WANDB_NAME

# Hugging Face related
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
