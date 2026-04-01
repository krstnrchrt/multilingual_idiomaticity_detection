import pandas as pd


def apply_text_preprocessing_pipeline(df: pd.DataFrame, process_mwe: bool) -> pd.DataFrame:
    if process_mwe:
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

        # Combine content
        df["text"] = (
                df["Previous"]
                + "[SEP]"
                + df["marked_target"]
                + "[SEP]"
                + df["Next"]
        )
    else:
        # Combine content
        df["text"] = (
                df["Previous"]
                + " "
                + df["Target"]
                + " "
                + df["Next"]
        )

    return df
