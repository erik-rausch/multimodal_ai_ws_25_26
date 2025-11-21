import jiwer
import librosa
import torch
import torchaudio
from huggingface_hub import hf_hub_download
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "/training-1/modelhub/granite-speech-3.3-2b"
# model_name = "logs/checkpoint-195"
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = processor.tokenizer
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_name, device_map=device, torch_dtype=torch.bfloat16
)

def inference(entry) -> str:
    question = entry["question"]
    instruction = f"Beantworte die Frage '{question}' aus dem Inhalt des folgenden Audios <|audio|>"

    chat = [
        # dict(role="system", content=system_prompt),
        dict(role="user", content=instruction),
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    y, sr = librosa.load(entry["audio"], sr=16000)
    # run the processor+model
    model_inputs = processor(prompt, y, device=device, return_tensors="pt").to(device)
    model_outputs = model.generate(**model_inputs, max_new_tokens=200, do_sample=False, num_beams=1)

    # Transformers includes the input IDs in the response.
    num_input_tokens = model_inputs["input_ids"].shape[-1]
    new_tokens = torch.unsqueeze(model_outputs[0, num_input_tokens:], dim=0)
    output_text = tokenizer.batch_decode(new_tokens, add_special_tokens=False, skip_special_tokens=True)
    return output_text[0]
