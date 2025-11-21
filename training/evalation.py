from ask_lisa import ask_lisa
import json

eval_prompt_path = "../../system_prompts/eval_answer_prompt.txt"
test_partition = "dataset/test.jsonl"

out = "evaluation_results/"
out_file = f"{out}test-results.jsonl"
out_compact_file = f"{out}test-results.compact.json"

with open(eval_prompt_path, "r") as f:
    eval_prompt = f.read()


def evaluate(context: str, question: str, expected_answer: str, generated_answer: str) -> int:
    res = ask_lisa(eval_prompt, f"""
        Kontext: {context}
        Frage: {question}
        Ground Truth Antwort: {expected_answer}
        Generierte Antwort: {generated_answer}
    """)
    if res is None or len(res) > 1:
        return -1
    return int(res)


dataset_entries = []

with open(test_partition, encoding="UTF-8") as rf:
    for line in rf:
        entry = json.loads(line.strip())
        dataset_entries.append(entry)

results = {
    1: 0,
    2: 0,
    3: 0,
    4: 0
}

for entry in dataset_entries:
    eval_value = evaluate(entry["context"], entry["question"], entry["answer"], entry["generated_answer"])
    if eval_value == -1:
        raise ValueError("The Lisa result had the wrong format.")

    entry["eval"] = eval_value
    results[eval_value] += 1

with open(out_file, "w") as f:
    json.dump(dataset_entries, f)

with open(out_compact_file, "w") as f:
    json.dump(results, f)

print("Evaluation results: ")
print(results)
entry_count = results[1] + results[2] + results[3] + results[4]
print(f"Accuracy: {results[1] / entry_count}")
print(f"Real answers: {1 - results[4] / entry_count}")
