#!/bin/sh

# Ensure that we are in the right folder
cd /slid/data || exit 1

# Clone the repository into the container
git clone https://github.com/H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity.git sem_eval_repo

# Make directory to place original data in
mkdir original

# Copy out all relevant files
cp /slid/data/sem_eval_repo/SubTaskA/Data/train_one_shot.csv /slid/data/original
cp /slid/data/sem_eval_repo/SubTaskA/Data/train_zero_shot.csv /slid/data/original
cp /slid/data/sem_eval_repo/SubTaskA/Data/dev.csv /slid/data/original/test.csv
cp /slid/data/sem_eval_repo/SubTaskA/Data/dev_gold.csv /slid/data/original/test_gold.csv

# Go back into project root directory
cd /slid || exit 1
