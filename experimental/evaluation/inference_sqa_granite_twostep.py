import torch
import torchaudio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub import hf_hub_download
import json
import librosa
import jiwer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "/training-1/modelhub/granite-speech-3.3-2b"
#model_name = "logs/checkpoint-195"
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name, device_map=device, torch_dtype=torch.bfloat16
)

def transcribe(audio) -> str:
    system_prompt = "Knowledge Cutoff Date: April 2024.\nToday's Date: April 9, 2025.\nYou are Granite, developed by IBM. You are a helpful AI assistant"
    instruction = "Please transcribe the following audio to text<|audio|>"
    chat = [
        dict(role="system", content=system_prompt),
        dict(role="user", content=instruction)
    ]
    prompt = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False,
    )
    model_inputs = processor(prompt, audio, device=device, return_tensors="pt").to(device)
    model_outputs = model.generate(**model_inputs, max_new_tokens=200)
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = model_outputs[:, num_input_tokens:]
    output_text = tokenizer.batch_decode(
        new_tokens, add_special_tokens=False, skip_special_tokens=True
    )
    return output_text


def llm_response(query):
    chat = [dict(role="user", content=query)]
    prompt = tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        tokenize=False
    )
    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    # calling the base LLM and disabling the LoRA adaptors
    model_outputs = model.generate(**model_inputs, max_new_tokens=200)
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = model_outputs[:, num_input_tokens:]

    output_text = tokenizer.batch_decode(
        new_tokens, add_special_tokens=False, skip_special_tokens=True
    )
    return output_text


dataset_entries = []
with open("dataset/test.jsonl", encoding="UTF-8") as rf:
    for line in rf:
        entry = json.loads(line.strip())
        dataset_entries.append(entry)
for entry in dataset_entries:
    y, sr = librosa.load(entry["audio"], sr=16000)

    transcription = transcribe(y)[0]
    wer = jiwer.wer(entry["text"], transcription)
    instruction = f"Beantworte die Frage '{entry["question"]}' aus dem Inhalt des folgenden Texts: {transcription}"
    response = llm_response(instruction)[0]

    print(f"Transcription Ground Truth: {entry["text"]}")
    print(f"ALM Transcription: {entry["text"]}")
    print(f"Word Error Rate: {wer}")
    print(f"Question: {entry["question"]}")
    print(f"Expected Answer: {entry["answer"]}")
    print(f"ALM Answer: {response}")
    print("-"*30)