from functools import partial
from pathlib import Path
import hashlib

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets
from pandas import DataFrame
from transformers import AutoTokenizer

import src.settings as sts
from src.preprocessing.text_preprocessing_pipeline import apply_text_preprocessing_pipeline


def read_csv_as_dataset(filepath_csv: Path, process_mwe: bool) -> Dataset:
    df: DataFrame = pd.read_csv(filepath_csv)

    # Determine which column to use as the unique ID
    if "ID" in df.columns:
        id_column = "ID"
    elif "DataID" in df.columns:
        id_column = "DataID"
    else:
        raise KeyError("A ID column is required in the CSV file!")

    # For debugging
    # df = df.head(200)

    df = apply_text_preprocessing_pipeline(df, process_mwe)

    ds: Dataset = Dataset.from_pandas(df[["text", "Label", id_column]])
    ds = ds.rename_column("Label", "labels")
    ds = ds.rename_column(id_column, "id")

    return ds


def tokenize_function(examples, tokenizer, config):
    if isinstance(examples["text"], list):
        examples["text"] = [str(text) for text in examples["text"]]
    else:
        examples["text"] = str(examples["text"])

    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding='max_length',
        max_length=sts.SPLITS_MAX_LENGTH_TOKENIZED_SEQUENCE
    )

    # Keep identifier column
    tokenized["id"] = examples["id"]

    if not config["process_mwe"]:
        return tokenized

    # Extract IDs for the special token
    sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    e_start_token_id = tokenizer.convert_tokens_to_ids("[E]")
    e_end_token_id = tokenizer.convert_tokens_to_ids("[/E]")

    # Initialize lists to store start and end positions
    mwe_start_positions = []
    mwe_end_positions = []
    target_start_positions = []
    target_end_positions = []

    # Process each sequence of token IDs
    for input_ids in tokenized["input_ids"]:
        # Find [SEP] token positions
        sep_positions = [i for i, token_id in enumerate(input_ids) if token_id == sep_token_id]

        # Find target sentence boundaries
        if len(sep_positions) >= 2:
            target_start = sep_positions[0] + 1
            target_end = sep_positions[1] - 1
        else:
            # Fallback if we don't find enough SEP tokens
            target_start = 1  # Skip CLS token
            target_end = len(input_ids) - 2  # Leave room for final SEP

        # Find MWE marker positions
        try:
            e_start_pos = input_ids.index(e_start_token_id)
            e_end_pos = input_ids.index(e_end_token_id)

            # MWE is between markers (exclude the markers themselves)
            mwe_start = e_start_pos + 1
            mwe_end = e_end_pos - 1
        except ValueError:
            # Fallback if markers aren't found
            # Place in middle of target section
            span_length = (target_end - target_start) // 3
            mwe_start = target_start + span_length
            mwe_end = mwe_start + span_length

        #Store positions
        mwe_start_positions.append(mwe_start)
        mwe_end_positions.append(mwe_end)
        target_start_positions.append(target_start)
        target_end_positions.append(target_end)

    # Add MWE positions to tokenized outputs
    tokenized["mwe_start_positions"] = mwe_start_positions
    tokenized["mwe_end_positions"] = mwe_end_positions
    tokenized["target_start_positions"] = target_start_positions
    tokenized["target_end_positions"] = target_end_positions

    return tokenized


def get_split_datasets(config: dict):
    # Extract setting if MWEs should be handled
    process_mwe = config["process_mwe"]

    if config["use_augmented_train_zero_shot_data"]:
        train_filepath_csv: Path = sts.FILEPATH_DATA_FINAL_TRAIN_ZEROSHOT_AUGMENTED_CSV
    else:
        train_filepath_csv: Path = sts.FILEPATH_DATA_FINAL_TRAIN_ZEROSHOT_CSV

    # Prepare train and validation dataset
    ds_zero_shot: Dataset = read_csv_as_dataset(
        filepath_csv=train_filepath_csv,
        process_mwe=process_mwe
    )

    if config["scenario"] == sts.Scenario.ONE_SHOT:
        ds_one_shot: Dataset = read_csv_as_dataset(
            filepath_csv=sts.FILEPATH_DATA_FINAL_TRAIN_ONESHOT_CSV,
            process_mwe=process_mwe
        )
        ds_train = concatenate_datasets([ds_zero_shot, ds_one_shot])
    else:
        ds_train: Dataset = ds_zero_shot

    # Prepare evaluation dataset
    ds_val: Dataset = read_csv_as_dataset(
        filepath_csv=sts.FILEPATH_DATA_FINAL_TEST_CSV,
        process_mwe=process_mwe
    )

    # Load tokenizer
    if Path(config["model_name"]).is_dir():
        tokenizer = AutoTokenizer.from_pretrained(Path(config["model_name"]))
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            use_fast=True
        )

    if process_mwe:
        # Add the markers as additional special tokens
        special_tokens_dict = {'additional_special_tokens': ['[E]', '[/E]']}
        tokenizer.add_special_tokens(special_tokens_dict)

    # Pass tokenizer using partial
    tokenize_with_tokenizer = partial(tokenize_function, tokenizer=tokenizer, config=config)

    datasets = DatasetDict({"train": ds_train, "val": ds_val})

    tokenized_datasets = datasets.map(tokenize_with_tokenizer, batched=True)

    # Return tokenized datasets
    return tokenizer, tokenized_datasets["train"], tokenized_datasets["val"]
