# Multilingual Idiomaticity Detection (SLID)


A group research project for **SemEval 2022 Task 2, Subtask A** — binary classification of multi-word expressions (MWEs) as **idiomatic** or **literal** in context. The system fine-tunes [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) with Bayesian hyperparameter search via Ray Tune and tracks experiments with Weights & Biases.


---


## Table of Contents


- [What the project does](#what-the-project-does)
- [Project structure](#project-structure)
- [Getting started](#getting-started)
 - [Prerequisites](#prerequisites)
 - [Environment setup](#environment-setup)
 - [Data](#data)
 - [Running experiments](#running-experiments)
- [Experiments](#experiments)
- [Docker](#docker)
- [Weights & Biases](#weights--biases)
- [Notebooks](#notebooks)
- [Who maintains and contributes](#who-maintains-and-contributes)


---


## What the project does


This project addresses **SemEval 2022 Task 2 – Subtask A: Binary Idiomaticity Detection**. Given a sentence containing a potential MWE (multi-word expression) along with its surrounding context (previous and next sentences), the task is to classify the usage as either:


- **Idiomatic** (the MWE does not carry its literal meaning, e.g. *"he spilled the beans"*)
- **Literal** (the words should be read compositionally)


The dataset covers **multiple languages** (English, Portuguese, and Galician). The model operates in the **zero-shot** and **one-shot** scenarios defined by the shared task.


**Key features:**


- Fine-tunes **ModernBERT-base** for sequence classification
- Custom **MWE-aware text preprocessing**: MWEs in the target sentence are wrapped with `[E] … [/E]` markers so the model can attend to their span explicitly
- Custom **cosine-similarity loss** that pulls the MWE representation towards its sentential context for idiomatic examples
- **Bayesian hyperparameter optimisation** over learning rate, weight decay, and warmup ratio using Ray Tune + HyperOpt
- **Data augmentation** pipeline for the zero-shot training split
- Full **experiment tracking** with Weights & Biases
- Reproducible training via **Docker** (CPU and GPU images)


---


## Project structure


```
multilingual_idiomaticity_detection/
├── src/
│   ├── main.py                        # Experiment entry-point; defines & launches experiment configs
│   ├── settings.py                    # Global constants, paths, seeds, model name
│   ├── utils.py                       # Helper utilities (secrets, paths, directory helpers)
│   ├── modernbert_experiments/
│   │   ├── conduct_experiment.py      # Ray Tune training loop, W&B integration
│   │   ├── dataset.py                 # Dataset loading, tokenisation, MWE position tracking
│   │   ├── loss_functions.py          # VanillaCrossEntropyLoss + CosineSimilarityMWEWithTargetSentenceLoss
│   │   ├── trainer.py                 # SlidTrainer (HuggingFace Trainer subclass) + custom metrics
│   │   └── post_evaluation/
│   │       ├── evaluate.py            # Post-hoc evaluation: TP/TN/FP/FN examples, GPU energy stats
│   │       └── plotting.py            # Confusion matrices, metric curves, energy-consumption plots
│   ├── preprocessing/
│   │   ├── text_preprocessing_pipeline.py  # Combines Previous/Target/Next; inserts MWE markers
│   │   ├── augment_dataset.py              # (WIP) Training-data augmentation
│   │   ├── merge_test_dataset.py           # Merges dev.csv + dev_gold.csv into test.csv
│   │   └── determine_sequence_lengths.py   # Utility to compute max tokenised length per split
│   └── notebooks/
│       ├── official_baseline/         # SemEval 2022 official baseline notebook
│       ├── bilstm_baseline/           # BiLSTM baseline (preprocessing + training)
│       ├── group_baselines/           # Naive majority / random baselines
│       ├── data_analysis/             # Exploratory data analysis
│       ├── preprocessing_pipeline/    # Interactive preprocessing pipeline notebooks
│       ├── data_augmentation/         # Data augmentation experiments
│       ├── linguistic_analysis/       # Error analysis by MWE pattern and language
│       └── final_analysis/            # Final cross-experiment results analysis
├── data/
│   ├── original/                      # Raw SemEval data (downloaded at Docker build time)
│   └── final/                         # Pre-processed CSVs consumed by the training pipeline
├── results/
│   └── experiments/                   # Per-experiment output: checkpoints, models, Ray artefacts
├── docker/
│   ├── build/                         # Docker build scripts (data download, env setup)
│   └── prod/                          # Production runtime scripts (experiment runner)
├── secrets/                           # API keys (not committed; mounted at runtime)
├── Dockerfile                         # Multi-stage Docker build (pixi-based)
└── pixi.toml                          # Dependency manifest (CPU + GPU feature sets)
```


---


## Getting started


### Prerequisites


| Tool | Version |
|------|---------|
| Python | ≥ 3.12 |
| [pixi](https://pixi.sh) | latest |
| Docker (optional) | latest |
| CUDA (for GPU training) | ≥ 12.0 |


### Environment setup


This project uses [pixi](https://pixi.sh) to manage environments. After installing pixi:


```bash
# Install the CPU environment (works on any machine)
pixi install --environment slid-prod-cpu


# — OR — install the GPU environment (Linux + CUDA 12 required)
pixi install --environment slid-prod-gpu
```


Activate a shell with the environment:


```bash
pixi shell --environment slid-prod-cpu
```


### Data


The SemEval 2022 Task 2 dataset is sourced from the [official repository](https://github.com/H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity).


**Automatic (via Docker build):** The Dockerfile clones the repository and copies the relevant CSVs to `data/original/` automatically.


**Manual setup:**


```bash
# Clone the SemEval repo and copy files
git clone https://github.com/H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity.git /tmp/sem_eval_repo


mkdir -p data/original
cp /tmp/sem_eval_repo/SubTaskA/Data/train_one_shot.csv  data/original/
cp /tmp/sem_eval_repo/SubTaskA/Data/train_zero_shot.csv data/original/
cp /tmp/sem_eval_repo/SubTaskA/Data/dev.csv             data/original/test.csv
cp /tmp/sem_eval_repo/SubTaskA/Data/dev_gold.csv        data/original/test_gold.csv
```


Merge the test split (adds gold labels):


```bash
python -m src.preprocessing.merge_test_dataset
```


Copy or symlink the final pre-processed CSVs to `data/final/` before training.


### Weights & Biases secrets


Create the secrets directory and add your credentials (these are mounted into the container at runtime, never committed):


```bash
mkdir -p secrets
echo "your_wandb_api_key"  > secrets/.wandb_api_key
echo "your_wandb_username" > secrets/.wandb_username
```


Alternatively, export the environment variables directly:


```bash
export WANDB_API_KEY=your_key
export WANDB_ENTITY=anlp-slid
export WANDB_PROJECT=slid
```


### Running experiments


Uncomment the desired experiment(s) in `src/main.py` and run:


```bash
python -m src.main
```


---


## Experiments


Five main experiments are defined in `src/main.py`, each exploring a different modelling choice:


| # | Name | Encoder | MWE processing | Loss function | HPO samples |
|---|------|---------|---------------|---------------|-------------|
| 1 | `1_fix_encoder_and_train_classifier` | Frozen | No | Cross-entropy | 18 |
| 2 | `2_finetune_encoder_and_train_classifier` | Fine-tuned | No | Cross-entropy | 18 |
| 3 | `3_finetune_encoder_and_train_classifier_augmented` | Fine-tuned | No | Cross-entropy | 1 (best HPs from exp 2) |
| 4 | `4_finetune_encoder_and_train_classifier_process_mwe` | Fine-tuned | `[E]…[/E]` markers | Cross-entropy | 1 (best HPs from exp 2) |
| 5 | `5_finetune_encoder_…_custom_loss` | Fine-tuned | `[E]…[/E]` markers | Cosine-similarity MWE loss | 12 |


**MWE text format (experiments 4 & 5):**


```
<previous sentence> [SEP] <context…[E] MWE [/E]…context> [SEP] <next sentence>
```


**Custom loss (experiment 5):** A weighted combination of cross-entropy and a cosine-similarity term that encourages the MWE token representation to be similar to its surrounding sentence context for idiomatic examples (and dissimilar for literal ones), controlled by hyperparameter `alpha`.


All experiments use:
- **Model:** `answerdotai/ModernBERT-base`
- **Optimiser:** AdamW (fused)
- **Scheduler:** Linear with warmup
- **Early stopping patience:** 5 epochs
- **Best-model metric:** macro F1


---


## Docker


Docker images are built and run via pixi tasks. Both CPU and GPU variants are available.


```bash
# Build CPU image
pixi run build_cpu_container


# Build GPU image (Linux + NVIDIA GPU required)
pixi run build_gpu_container


# Run CPU container (results and W&B output are volume-mounted)
pixi run run_cpu_container


# Run GPU container
pixi run run_gpu_container
```


Results are written to `./results/` and W&B artefacts to `./wandb/` on the host via Docker volume mounts.


---


## Weights & Biases


All training runs are logged to the W&B project **`slid`** (entity **`anlp-slid`**). Each Ray Tune trial creates an individual W&B run. Metrics tracked include accuracy, balanced accuracy, F1 (binary / micro / macro / weighted), F2, precision, recall, ROC-AUC, MCC, and PR-AUC.


---


## Notebooks


Exploratory and baseline notebooks live under `src/notebooks/`:


| Notebook | Purpose |
|----------|---------|
| `official_baseline/SemEval_2022_Task_2_Subtask_A_BASELINE.ipynb` | Reproduces the shared-task official baseline |
| `bilstm_baseline/bilstm_train.ipynb` | BiLSTM sequence classifier baseline |
| `group_baselines/naive_baselines.ipynb` | Majority-class and random baselines |
| `data_analysis/explorative_data_analysis.ipynb` | Dataset statistics and class distributions |
| `preprocessing_pipeline/preprocessing_pipeline.ipynb` | Step-by-step preprocessing pipeline walkthrough |
| `data_augmentation/data_aug.ipynb` | Data augmentation experiments |
| `linguistic_analysis/linguistic_analysis.ipynb` | Error analysis by MWE pattern and language |
| `final_analysis/final_analysis.ipynb` | Cross-experiment results comparison |


---


## Authors & Contributors

This project was developed as part of the **Applied Natural Language Processing (ANLP)** course at the **University of Potsdam**.

| Name | Contact |
|------|---------|
| Maria Manina | maria.manina@uni-potsdam.de |
| Arne Schernich | arne.schernich@uni-potsdam.de |
| Kristina Richert | kristina.richert@uni-potsdam.de |




