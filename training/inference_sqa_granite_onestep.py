import librosa
import torch
from huggingface_hub import hf_hub_download
from scipy.io import wavfile
from train_utils import ac_tq_instruction, tc_aq_instruction, ac_aq_instruction, load_audio, combine_audios
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

# -----------------------------------------------------------
# Model initialisation
# -----------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# model_base_path = "/training-1/modelhub/granite-speech-3.3-2b"
model_base_path = "logs/ac-tq-4/checkpoint-5400"

processor = AutoProcessor.from_pretrained(model_base_path)
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_base_path,
    device_map=device,
    torch_dtype=torch.bfloat16,
)


# -----------------------------------------------------------
# Utility: run model inference
# -----------------------------------------------------------
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


# ===========================================================
# 1) Context = Audio, Question = Text
# ===========================================================
def infer_audio_context__text_question(entry) -> str:
    return run_inference(ac_tq_instruction(tokenizer, entry), load_audio(entry["context_audio"]))


# ===========================================================
# 2) Context = Text, Question = Audio
# ===========================================================
def infer_text_context__audio_question(entry) -> str:
    return run_inference(tc_aq_instruction(tokenizer, entry), load_audio(entry["question_audio"]))


# ===========================================================
# 3) Context + Question = Both Audio (concatenated with silence)
# ===========================================================
def infer_audio_context__audio_question(entry) -> str:
    _, audio = combine_audios(entry["context_audio"], entry["question_audio"])
    return run_inference(ac_aq_instruction(tokenizer, entry), audio)
