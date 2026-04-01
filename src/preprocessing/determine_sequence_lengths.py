from transformers import AutoTokenizer
import pandas as pd

import src.settings as sts
from src.preprocessing.text_preprocessing_pipeline import apply_text_preprocessing_pipeline

def determine_max_sequence_length_of_tokenized_datasets():
    # Load tokenizer of ModernBERT to determine the maximum sequence length of tokens
    tokenizer = AutoTokenizer.from_pretrained(sts.MODERN_BERT_NAME, use_fast=True)
    special_tokens_dict = {'additional_special_tokens': ['[E]', '[/E]']}
    tokenizer.add_special_tokens(special_tokens_dict)

    df_zero = pd.read_csv(sts.FILEPATH_DATA_FINAL_TRAIN_ZEROSHOT_CSV)
    df_one = pd.read_csv(sts.FILEPATH_DATA_FINAL_TRAIN_ONESHOT_CSV)
    df_test = pd.read_csv(sts.FILEPATH_DATA_FINAL_TEST_CSV)

    df_zero = apply_text_preprocessing_pipeline(df_zero, process_mwe=True)
    df_one = apply_text_preprocessing_pipeline(df_one, process_mwe=True)
    df_test = apply_text_preprocessing_pipeline(df_test, process_mwe=True)

    max_length_zero = tokenizer(str(df_zero.loc[df_zero["text"].str.len().idxmax(), "text"]))["input_ids"]
    max_length_one = tokenizer(str(df_one.loc[df_one["text"].str.len().idxmax(), "text"]))["input_ids"]
    max_length_test = tokenizer(str(df_test.loc[df_test["text"].str.len().idxmax(), "text"]))["input_ids"]

    print(f"max_length_zero: {len(max_length_zero)}")
    print(f"max_length_one: {len(max_length_one)}")
    print(f"max_length_test: {len(max_length_test)}")


# Entrypoint of this script
if __name__ == "__main__":
    determine_max_sequence_length_of_tokenized_datasets()
