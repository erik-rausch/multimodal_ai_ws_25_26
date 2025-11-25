import pandas as pd
import json

wavs = "dataset/wavs/"

input_file = "dataset-creation/out/validate.parquet"
output_file = "dataset/output.jsonl"


def extract_parquet(parquet_file):
    df = pd.read_parquet(parquet_file)



def parquet_to_jsonl(parquet_file, jsonl_file, exclude_columns=None):
    df = pd.read_parquet(parquet_file)

    # exclude binary columns
    if exclude_columns:
        existing_excludes = set(exclude_columns) & set(df.columns)
        non_existing = set(exclude_columns) - set(df.columns)

        if non_existing:
            print(f"Spalten existieren nicht: {non_existing}")

        df = df.drop(columns=list(existing_excludes))

    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            json.dump(row.to_dict(), f, ensure_ascii=False)
            f.write('\n')


parquet_to_jsonl(input_file, output_file, ["context_audio", "q1_audio", "q2_audio", "q3_audio"])