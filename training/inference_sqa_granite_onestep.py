import librosa
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from huggingface_hub import hf_hub_download

# -----------------------------------------------------------
# Model initialisation
# -----------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "/training-1/modelhub/granite-speech-3.3-2b"
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name,
    device_map=device,
    torch_dtype=torch.bfloat16,
)

# -----------------------------------------------------------
# Utility: load audio at 16kHz
# -----------------------------------------------------------
def load_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    return audio


# -----------------------------------------------------------
# Utility: run model inference
# -----------------------------------------------------------
def run_inference(prompt: str, audio_signal):
    # prepare model inputs
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


# ===========================================================
# 1) Context = Audio, Question = Text
# ===========================================================
def infer_audio_context__text_question(entry) -> str:
    question = entry["question_text"]
    instruction = f"Beantworte die Frage '{question}' aus dem Inhalt des folgenden Audios <|audio|>"

    chat = [dict(role="user", content=instruction)]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    audio = load_audio(entry["context_audio"])
    return run_inference(prompt, audio)


# ===========================================================
# 2) Context = Text, Question = Audio
# ===========================================================
def infer_text_context__audio_question(entry) -> str:
    context = entry["context_text"]
    instruction = (
        f"Der folgende Text bildet den Kontext:\n\n"
        f"{context}\n\n"
        f"Beantworte die Frage aus dem folgenden Audio <|audio|> basierend auf diesem Kontext."
    )

    chat = [dict(role="user", content=instruction)]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    audio_question = load_audio(entry["question_audio"])
    return run_inference(prompt, audio_question)


# ===========================================================
# 3) Context + Question = Both Audio (concatenated with silence)
# ===========================================================
def infer_audio_context__audio_question(entry) -> str:
    context_audio = load_audio(entry["context_audio"])
    question_audio = load_audio(entry["question_audio"])

    # 2 second of silence at 16kHz
    silence = torch.zeros(int(32000))

    # concatenate context + silence + question
    combined = torch.cat([torch.tensor(context_audio), silence, torch.tensor(question_audio)], dim=0)

    instruction = (
        "Beantworte die Frage im zweiten Audioteil basierend auf dem Inhalt des ersten Audioteils. "
        "Die beiden sind durch Stille getrennt."
    )

    chat = [dict(role="user", content=instruction)]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    return run_inference(prompt, combined.cpu().numpy())
