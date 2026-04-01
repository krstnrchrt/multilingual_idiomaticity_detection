import pandas as pd

import src.settings as sts

def merge_test_dataset():
    # Load both CSV files of the test dataset
    df_test_csv: pd.DataFrame = pd.read_csv(sts.FILEPATH_DATA_ORIGINAL_TEST_CSV, sep=",")
    df_test_gold_csv: pd.DataFrame = pd.read_csv(sts.FILEPATH_DATA_ORIGINAL_TEST_GOLD_CSV, sep=",")

    # Sort both dataframe according to the ID column
    df_test_csv = df_test_csv.sort_values(by="ID").reset_index(drop=True)
    df_test_gold_csv = df_test_gold_csv.sort_values(by="ID").reset_index(drop=True)

    # Ensure that both ID columns are identical
    if not df_test_csv["ID"].equals(df_test_gold_csv["ID"]):
        raise ValueError("The ID columns of the test and the test gold datasets are not identical!")

    # Merge th two datasets
    df_test_csv["Label"] = pd.Series(df_test_gold_csv["Label"])
    df_test_csv["DataID"] = pd.Series(df_test_gold_csv["DataID"])

    # Save merged DataFrame as the test.csv file
    df_test_csv.to_csv(sts.FILEPATH_DATA_FINAL_TEST_CSV, index=False)


# Entrypoint of this script
if __name__ == "__main__":
    merge_test_dataset()
