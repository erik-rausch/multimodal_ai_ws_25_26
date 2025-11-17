from io import BytesIO

import torch
import torchaudio.transforms as T
import os
import torch
from snac import SNAC

from peft import PeftModel
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer

# load voices from voices.txt
with open("voices.txt", "r", encoding="utf-8") as vf:
    voices = [line.strip() for line in vf.readlines() if line.strip()]

model = AutoModelForCausalLM.from_pretrained(
    "SebastianBodza/Kartoffel_Orpheus-3B_german_synthetic-v0.1",
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    "SebastianBodza/Kartoffel_Orpheus-3B_german_synthetic-v0.1",
)

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to("cuda")

chosen_voice = "Martin"

prompts = [
    'Tief im verwunschenen Wald, wo die Bäume uralte Geheimnisse flüsterten, lebte ein kleiner Gnom namens Fips, der die Sprache der Tiere verstand.',
]

def process_single_prompt(prompt, chosen_voice):
    if chosen_voice == "in_prompt" or chosen_voice == "":
        full_prompt = prompt
    else:
        full_prompt = f"{chosen_voice}: {prompt}"
    start_token = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64)

    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids
    modified_input_ids = torch.cat([start_token, input_ids, end_tokens], dim=1)

    input_ids = modified_input_ids.to("cuda")
    attention_mask = torch.ones_like(input_ids)

    generated_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=4000,
        do_sample=True,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
        num_return_sequences=1,
        eos_token_id=128258,
        use_cache=True,
    )

    token_to_find = 128257
    token_to_remove = 128258

    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx + 1 :]
    else:
        cropped_tensor = generated_ids

    masked_row = cropped_tensor[0][cropped_tensor[0] != token_to_remove]
    row_length = masked_row.size(0)
    new_length = (row_length // 7) * 7
    trimmed_row = masked_row[:new_length]
    code_list = [t - 128266 for t in trimmed_row]

    return code_list


def redistribute_codes(code_list):
    layer_1 = []
    layer_2 = []
    layer_3 = []
    for i in range((len(code_list) + 1) // 7):
        layer_1.append(code_list[7 * i])
        layer_2.append(code_list[7 * i + 1] - 4096)
        layer_3.append(code_list[7 * i + 2] - (2 * 4096))
        layer_3.append(code_list[7 * i + 3] - (3 * 4096))
        layer_2.append(code_list[7 * i + 4] - (4 * 4096))
        layer_3.append(code_list[7 * i + 5] - (5 * 4096))
        layer_3.append(code_list[7 * i + 6] - (6 * 4096))

    codes = [
        torch.tensor(layer_1).unsqueeze(0),
        torch.tensor(layer_2).unsqueeze(0),
        torch.tensor(layer_3).unsqueeze(0),
    ]
    codes = [c.to("cuda") for c in codes]

    audio_hat = snac_model.decode(codes)
    return audio_hat


def to_speech(text):
    # random voice
    random_voice = voices[os.urandom(1)[0] % len(voices)]

    with torch.no_grad():
        code_list = process_single_prompt(text, random_voice)
        samples = redistribute_codes(code_list)

    audio_numpy = samples.detach().squeeze().to("cpu").numpy()

    # Audio in BytesIO Buffer schreiben, statt in Datei auszugeben
    buffer = BytesIO()
    sf.write(buffer, audio_numpy, 24000, format='WAV')
    buffer.seek(0)  # Zurück zum Anfang des Buffers

    return buffer.read()