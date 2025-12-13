import json
import os
from pathlib import Path
from train_utils import evaluate, ac_tq_instruction, tc_aq_instruction, ac_aq_instruction, load_audio, combine_audios
import librosa
import torch
from huggingface_hub import hf_hub_download
from scipy.io import wavfile
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

test_partition = "dataset/test.jsonl"
out = "evaluation_results/"
evaluation_id = "ac-aq-2"
out_path = f"{out}{evaluation_id}/"
model_base_path = f"logs/{evaluation_id}/checkpoint-675"
# model_base_path = "/training-1/modelhub/granite-speech-3.3-2b"

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_base_path)
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_base_path,
    device_map=device,
    torch_dtype=torch.bfloat16,
)


def run_inference(prompt: str, audio_signal):
    model_inputs = processor(prompt, audio_signal, device=device, return_tensors="pt").to(device)

    # generate
    model_outputs = model.generate(
        **model_inputs,
        max_new_tokens=200,
        do_sample=False,
        num_beams=1
    )

    # extract only new tokens (strip the prompt)
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
    output = tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)

    return output[0]


def infer_audio_context__text_question(entry) -> str:
    return run_inference(ac_tq_instruction(tokenizer, entry), load_audio(entry["context_audio"]))

def infer_text_context__audio_question(entry) -> str:
    return run_inference(tc_aq_instruction(tokenizer, entry), load_audio(entry["question_audio"]))

def infer_audio_context__audio_question(entry) -> str:
    audio, _ = combine_audios(entry["context_audio"], entry["question_audio"])
    return run_inference(ac_aq_instruction(tokenizer, entry), audio)


# t = text, a = audio, c = context, q = question
modes = {
    "ac-tq": infer_audio_context__text_question,
    "tc-aq": infer_text_context__audio_question,
    "ac-aq": infer_audio_context__audio_question
}

dataset_entries = []

with open(test_partition, encoding="UTF-8") as rf:
    for line in rf:
        entry = json.loads(line.strip())
        dataset_entries.append(entry)

for mode, inference in modes.items():
    results = {
        "items": len(dataset_entries),
        "overall": {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        },
        "difficulties": {
            1: {
                0: 0,
                1: 0,
                2: 0,
                3: 0
            },
            2: {
                0: 0,
                1: 0,
                2: 0,
                3: 0
            },
            3: {
                0: 0,
                1: 0,
                2: 0,
                3: 0
            }
        }
    }
    difficulties = results["difficulties"]
    overall = results["overall"]
    item_count = len(dataset_entries)
    results["items"] = item_count
    out_file = f"{out_path}eval-{mode}-results.jsonl"
    out_compact_file = f"{out_path}eval-{mode}-results-compact.json"

    if Path(out_file).exists() and Path(out_compact_file).exists():
        print(f"Skipping {mode} because results already exist.")
        continue

    for index, entry in enumerate(dataset_entries):
        print(f"[{mode}] Processing entry {index + 1}/{item_count}")
        generated_answer = inference(entry)
        print(f"--> {generated_answer}")

        eval_value = evaluate(entry["context_text"], entry["question_text"], entry["answer_text"], generated_answer)
        entry["generated_answer"] = generated_answer
        print(f"--> Evaluation: {eval_value}")
        if eval_value == -1:
            raise ValueError("The Lisa result had the wrong format.")

        entry["eval"] = eval_value
        overall[eval_value] += 1
        difficulties[entry["level"]][eval_value] += 1

    os.makedirs(out_path, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for entry in dataset_entries:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    with open(out_compact_file, "w") as f:
        json.dump(results, f)

    print(f"""
    ==================================
    ==================================
    Evaluation results for mode {mode}:
    {results}
    ==================================
    ==================================
    """)
