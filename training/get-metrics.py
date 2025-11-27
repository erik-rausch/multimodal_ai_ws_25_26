import pandas as pd
import json
from ydata_profiling import ProfileReport

evaluation_sets = {
    "ac-aq": "evaluation_results/eval-ac-aq-results.jsonl",
    "ac-tq": "evaluation_results/eval-ac-tq-results.jsonl",
    "tc-aq": "evaluation_results/eval-tc-aq-results.jsonl",
}

def generate_word_count_report(file_path: str, name: str):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))

    df = pd.DataFrame(data)

    df['word_count'] = df['generated_answer'].apply(lambda x: len(x.split()) if pd.notna(x) else 0)

    result_df = df[['context_id', 'level', 'word_count']].copy()
    result_df.columns = ['transcript_id', 'level', 'word_count']

    print(result_df.head())

    result_df.to_csv(f'metrics/{name}.csv', index=False)

    # Pandas Profiling Report erstellen
    profile = ProfileReport(result_df, title="Transcript Profiling Report", explorative=True)
    profile.to_file(f"metrics/report_{name}.html")

for k, v in evaluation_sets.items():
    generate_word_count_report(v, k)