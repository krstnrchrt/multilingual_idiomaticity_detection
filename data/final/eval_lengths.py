from transformers import AutoTokenizer
import pandas as pd


def mark(df):
    # Create a new column where the MWE is marked in the target sentence
    marked_targets = []
    for i, row in df.iterrows():
        target = row["Target"]
        mwe = row["MWE"]

        if mwe in target:
            marked_target = target.replace(mwe, f"[E] {mwe} [/E]")
        else:
            marked_target = target

        marked_targets.append(marked_target)

    # Store the marked target in the df
    df["marked_target"] = marked_targets

    df["text"] = (
            df["Previous"]
            + "[SEP]"
            + df["marked_target"]
            + "[SEP]"
            + df["Next"]
    )
    return df

fp_train_one = "train_zero_shot.csv"
fp_train_zero = "train_one_shot.csv"
fp_test = "test.csv"

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base", use_fast=True)
special_tokens_dict = {'additional_special_tokens': ['[E]', '[/E]']}
tokenizer.add_special_tokens(special_tokens_dict)


df_zero = pd.read_csv("train_zero_shot.csv")
df_one = pd.read_csv("train_one_shot.csv")
df_test = pd.read_csv("test.csv")

df_zero = mark(df_zero)
df_one = mark(df_one)
df_test = mark(df_test)

max_length_zero = tokenizer(str(df_zero.loc[df_zero["text"].str.len().idxmax(), "text"]))["input_ids"]
max_length_one = tokenizer(str(df_one.loc[df_one["text"].str.len().idxmax(), "text"]))["input_ids"]
max_length_test = tokenizer(str(df_test.loc[df_test["text"].str.len().idxmax(), "text"]))["input_ids"]

print(f"max_length_zero: {len(max_length_zero)}")
print(f"max_length_one: {len(max_length_one)}")
print(f"max_length_test: {len(max_length_test)}")
