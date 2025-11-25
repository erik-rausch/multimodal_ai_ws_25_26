from ask_lisa import ask_lisa
import json
import os
from inference_sqa_granite_onestep import infer_audio_context__text_question, infer_audio_context__audio_question, \
    infer_text_context__audio_question

eval_prompt_path = "../system_prompts/eval_answer_prompt.txt"
test_partition = "dataset/test.jsonl"
out = "evaluation_results/"
evaluation_id = "untrained"
out_path = f"{out}{evaluation_id}/"
# t = text, a = audio, c = context, q = question
modes = {
    "ac-tq": infer_audio_context__text_question,
    "tc-aq": infer_text_context__audio_question,
    "ac-aq": infer_audio_context__audio_question
}

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
        print(f"Wrong format: {res}")
        return -1
    return int(res)


dataset_entries = []

with open(test_partition, encoding="UTF-8") as rf:
    for line in rf:
        entry = json.loads(line.strip())
        dataset_entries.append(entry)

for mode, inference in modes.items():
    results = {
        "items": len(dataset_entries),
        "overall": {
            1: 0,
            2: 0,
            3: 0,
            4: 0
        },
        "difficulties": {
            1: {
                1: 0,
                2: 0,
                3: 0,
                4: 0
            },
            2: {
                1: 0,
                2: 0,
                3: 0,
                4: 0
            },
            3: {
                1: 0,
                2: 0,
                3: 0,
                4: 0
            }
        }
    }
    difficulties = results["difficulties"]
    overall = results["overall"]
    item_count = len(dataset_entries)
    results["items"] = item_count

    for index, entry in enumerate(dataset_entries):
        print(f"[{mode}] Processing entry {index + 1}/{item_count}")
        generated_answer = inference(entry)
        print(f"--> {generated_answer}")
        eval_value = evaluate(entry["context_text"], entry["question_text"], entry["answer_text"], generated_answer)
        print(f"--> Evaluation: {eval_value}")
        if eval_value == -1:
            raise ValueError("The Lisa result had the wrong format.")

        entry["eval"] = eval_value
        overall[eval_value] += 1
        difficulties[entry["level"]][eval_value] += 1

    os.makedirs(out_path, exist_ok=True)
    with open(f"{out_path}eval-{mode}-results.jsonl", "w", encoding="utf-8") as f:
        for entry in dataset_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    with open(f"{out_path}eval-{mode}-results-compact.json", "w") as f:
        json.dump(results, f)

    print(f"""
    ==================================
    ==================================
    Evaluation results for mode {mode}:
    {results}
    ==================================
    ==================================
    """)
