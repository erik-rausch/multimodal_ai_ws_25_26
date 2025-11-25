from pathlib import Path

import pandas as pd
import json

wavs = "dataset/wavs/"

input_file = "../dataset-creation/out/validate.parquet"
output_file = "dataset/output1.jsonl"


def extract_parquet(parquet_file):
    df = pd.read_parquet(parquet_file)



def parquet_to_jsonl(parquet_file, jsonl_file):
    df = pd.read_parquet(parquet_file)

    # create dir if not existing
    output_file_path = Path(jsonl_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            base_path = f"{wavs}/{row.get('transcript_id')}"
            target_dir = Path(base_path)
            target_dir.mkdir(parents=True, exist_ok=True)

            # context audio
            with open(f"{base_path}/context.wav", 'wb') as af:
                af.write(row.get("context_audio"))

            # q1
            json.dump({
                'context_id': row.get("transcript_id"),
                'context_text': row.get("context_text"),
                'context_audio': f"{base_path}/context.wav",
                'question_text': row.get("q1_text"),
                'question_audio': f"{base_path}/q1.wav",
                'answer_text': row.get("a1_text"),
                'level': 1
            }, f, ensure_ascii=False)

            with open(f"{base_path}/q1.wav", 'wb') as af:
                af.write(row.get("q1_audio"))

            # q2
            json.dump({
                'context_id': row.get("transcript_id"),
                'context_text': row.get("context_text"),
                'context_audio': f"{base_path}/context.wav",
                'question_text': row.get("q2_text"),
                'question_audio': f"{base_path}/q2.wav",
                'answer_text': row.get("a2_text"),
                'level': 2
            }, f, ensure_ascii=False)

            with open(f"{base_path}/q2.wav", 'wb') as af:
                af.write(row.get("q2_audio"))

            # q3
            json.dump({
                'context_id': row.get("transcript_id"),
                'context_text': row.get("context_text"),
                'context_audio': f"{base_path}/context.wav",
                'question_text': row.get("q3_text"),
                'question_audio': f"{base_path}/q3.wav",
                'answer_text': row.get("a3_text"),
                'level': 3
            }, f, ensure_ascii=False)

            with open(f"{base_path}/q3.wav", 'wb') as af:
                af.write(row.get("q3_audio"))

            f.write('\n')


parquet_to_jsonl(input_file, output_file)