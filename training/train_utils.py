import librosa
from ask_lisa import ask_lisa
import numpy as np

eval_prompt_path = "../system_prompts/eval_answer_prompt.txt"

with open(eval_prompt_path, "r") as f:
    eval_prompt = f.read()


def load_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    return audio


def combine_audios(path_1, path_2, pause_sec=1.5):
    audio1, sr1 = librosa.load(path_1, sr=16000)
    audio2, sr2 = librosa.load(path_2, sr=16000)
    assert sr1 == sr2 == 16000, "Beide Dateien müssen 16 kHz haben!"
    pause = np.zeros(int(sr1 * pause_sec), dtype=np.float32)
    combined_audio = np.concatenate([audio1, pause, audio2])
    return sr1, combined_audio


def evaluate(context: str, question: str, expected_answer: str, generated_answer: str) -> int:
    try:
        res = ask_lisa(eval_prompt, f"""
            Kontext: {context}
            Frage: {question}
            Ground Truth Antwort: {expected_answer}
            Generierte Antwort: {generated_answer}
        """)
        if res is None:
            print(f"Received None from LISA, returning -1")
            return -1

        res = res.strip()
        # Falls LISA mehr als 1 Zeichen liefert oder kein Digit → -1 zurückgeben
        if len(res) != 1 or not res.isdigit():
            print(f"Wrong format from LISA: '{res}', returning -1")
            return -1

        return int(res)
    except Exception as e:
        print(f"Exception in evaluate: {e}, returning -1")
        return -1


def ac_tq_instruction(tokenizer, entry):
    question = entry["question_text"]
    instruction = f"Beantworte die Frage '{question}' aus dem Inhalt des folgenden Audios <|audio|>"
    chat = [dict(role="user", content=instruction)]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def tc_aq_instruction(tokenizer, entry):
    context = entry["context_text"]
    instruction = (
        f"Der folgende Text bildet den Kontext:\n\n"
        f"{context}\n\n"
        f"Beantworte die Frage aus dem folgenden Audio <|audio|> basierend auf diesem Kontext."
    )
    chat = [dict(role="user", content=instruction)]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def ac_aq_instruction(tokenizer, _):
    instruction = (
        "Das Audio <|audio|> besteht aus einem Kontext und einer Frage zum Kontext."
        "Zuerst kommt der Kontext, dann Stille, dann die Frage.\n\n"
        "Beantworte diese Frage basierend auf diesem Kontext."
    )
    chat = [dict(role="user", content=instruction)]
    return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


def ac_tq_audio(entry):
    return librosa.load(entry["context_audio"]["path"], sr=16000)


def tc_aq_audio(entry):
    return librosa.load(entry["question_audio"]["path"], sr=16000)


def ac_aq_audio(entry):
    return combine_audios(entry["context_audio"]["path"], entry["question_audio"]["path"])
