import torch
import os
from snac import SNAC
from io import BytesIO
import soundfile as sf
from transformers import AutoModelForCausalLM, AutoTokenizer


class OptimizedTTS:
    def __init__(self, voice="Julian"):
        """Modelle werden EINMAL beim Initialisieren geladen"""
        print("Loading models... (only happens once)")

        self.tokenizer = AutoTokenizer.from_pretrained(
            "SebastianBodza/Kartoffel_Orpheus-3B_german_synthetic-v0.1"
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            "SebastianBodza/Kartoffel_Orpheus-3B_german_synthetic-v0.1",
            device_map="auto",
            torch_dtype=torch.float16,  # Schneller!
        )
        self.model.eval()  # Inference-Modus

        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
        self.snac_model = self.snac_model.to("cuda")
        self.snac_model.eval()

        self.voice = voice

        # Pre-compute konstante Tensoren
        self.start_token = torch.tensor([[128259]], dtype=torch.int64, device="cuda")
        self.end_tokens = torch.tensor([[128009, 128260]], dtype=torch.int64, device="cuda")

        print("Models loaded!")

    @torch.inference_mode()  # Schneller als no_grad()
    def generate(self, text):
        """Generiert Audio aus Text"""
        # Prompt vorbereiten
        full_prompt = f"{self.voice}: {text}"
        input_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids.to("cuda")

        # Tokens hinzufügen
        modified_input_ids = torch.cat([self.start_token, input_ids, self.end_tokens], dim=1)

        # Audio-Tokens generieren
        generated_ids = self.model.generate(
            input_ids=modified_input_ids,
            attention_mask=torch.ones_like(modified_input_ids),
            max_new_tokens=4000,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=128258,
            use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Token-Processing
        code_list = self._extract_codes(generated_ids)

        # Audio dekodieren
        audio_hat = self._decode_audio(code_list)

        return audio_hat.detach().squeeze().cpu().numpy()

    def _extract_codes(self, generated_ids):
        """Extrahiert Audio-Codes aus generierten Tokens"""
        token_indices = (generated_ids == 128257).nonzero(as_tuple=True)

        if len(token_indices[1]) > 0:
            last_idx = token_indices[1][-1].item()
            cropped = generated_ids[:, last_idx + 1:]
        else:
            cropped = generated_ids

        masked = cropped[0][cropped[0] != 128258]
        new_length = (masked.size(0) // 7) * 7
        trimmed = masked[:new_length]

        return [t.item() - 128266 for t in trimmed]

    def _decode_audio(self, code_list):
        """Konvertiert Codes zu Audio"""
        layer_1, layer_2, layer_3 = [], [], []

        for i in range(len(code_list) // 7):
            idx = 7 * i
            layer_1.append(code_list[idx])
            layer_2.append(code_list[idx + 1] - 4096)
            layer_3.extend([
                code_list[idx + 2] - 8192,
                code_list[idx + 3] - 12288,
            ])
            layer_2.append(code_list[idx + 4] - 16384)
            layer_3.extend([
                code_list[idx + 5] - 20480,
                code_list[idx + 6] - 24576,
            ])

        codes = [
            torch.tensor(layer_1, device="cuda").unsqueeze(0),
            torch.tensor(layer_2, device="cuda").unsqueeze(0),
            torch.tensor(layer_3, device="cuda").unsqueeze(0),
        ]

        return self.snac_model.decode(codes)

    def to_wav_bytes(self, text):
        """Generiert WAV-Bytes (für API-Responses etc.)"""
        audio_numpy = self.generate(text)
        buffer = BytesIO()
        sf.write(buffer, audio_numpy, 24000, format='WAV')
        buffer.seek(0)
        return buffer.read()