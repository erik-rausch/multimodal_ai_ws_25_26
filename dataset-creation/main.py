import json
import os
from pathlib import Path

from dotenv import load_dotenv
from sqa_types import Transcript, SpokenQuestionAnswerPair
from generate_questions import generate_qa
from index_files import load_transcript
from generate_tts import to_speech
from datetime import datetime
from combine_parquet import export_parquet
from fail_handeling import add_failed_log, remove_failed_samples

base_dir = "/training-1/asr_dataset_files/asr_bundestag"
train = base_dir + "/train_nodev"
validate = base_dir + "/train_dev"
test = base_dir + "/test"
wav_dir = "/training-1/asr/bundestag/dataset/wavs"

load_dotenv(dotenv_path="../.env")

failed_counter = 0
remove_failed_samples()

datasets = {
    "train": load_transcript(train, "train", 60, 1200),
    "validate": load_transcript(validate, "validate", 60, 150),
    "test": load_transcript(test, "test", 60, 150),
    "dev": load_transcript(train, "example", 60, 5)
}

none_keys = [k for k, v in datasets.items() if v is None]

if none_keys:
    print("Nicht genug Transkripte bei:", none_keys)

# first load transcripts
current_partition = 'train'
transcripts: list[Transcript] = datasets[current_partition]
question_audios: dict[str, list[SpokenQuestionAnswerPair]] = {}

total_context_size = 0

# generate questions + answers for transcripts using AI
for (index, transcript) in enumerate(transcripts):
    print(f"Handling transcript {transcript['id']} - {index+1}/{len(transcripts)}")

    json_file_path = Path(os.path.join("tts-audios", transcript['id'], "question_set.json"))
    json_file_path.parent.mkdir(parents=True, exist_ok=True)

    if json_file_path.exists():
        print("-> Checkpoint found: loading questions from JSON")
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            questions = data["questions"]
    else:
        print("-> No checkpoint found: generating questions...")
        questions = generate_qa(transcript['text'])
        if not questions or len(questions) != 3:
            add_failed_log(transcript['id'])
            failed_counter += 1
            continue
        else:
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump({
                    "questions": questions,
                    **transcript,
                }, f, indent=2, ensure_ascii=False)


    total_context_size += os.path.getsize(transcript['filepath'])

    spoken_questions: list[SpokenQuestionAnswerPair] = []

    for question in questions:
        wav_path = Path(f"tts-audios/{transcript['id']}/{question['difficulty']}.wav")

        # --- WAV-Datei prüfen ---
        if wav_path.exists():
            print(f"--> WAV is already existing: {wav_path.name}")
            with open(wav_path, "rb") as f:
                wav = f.read()
        else:
            print(f"--> Generate TTS for question ({question['difficulty']})")
            start_time = datetime.now()

            wav = to_speech(question['question'])

            with open(wav_path, "wb") as f:
                f.write(wav)

            print(f"--> HANDLED in {datetime.now() - start_time}")

        total_context_size += os.path.getsize(wav_path)

        spoken_questions.append({
            **question,
            "audio_buffer": wav
        })

    question_audios[transcript['id']] = spoken_questions

# Generate parquet
if failed_counter > 0:
    print(f"Some samples ({failed_counter}) had fails. No Parquet file will be generated. Restart main.py to retry failed samples.")
else:
    export_parquet(transcripts, question_audios, out_path=f"out/{current_partition}.parquet")
    print(f"Total context size: {total_context_size/(2**20)}")