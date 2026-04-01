#!/bin/sh

echo "Preprocessing data...."

# Create folder for final dataset
mkdir /slid/data/final

# Go into the folder containing the original data
cd /slid/data/original || exit 1

# Copy data that has the correct format into the final data folder
cp train_zero_shot.csv ../final/
cp train_one_shot.csv ../final/

# Go into the source-code folder
cd /slid || exit 1

# Preprocess original test dataset to be usable
python -m src.preprocessing.merge_test_dataset
