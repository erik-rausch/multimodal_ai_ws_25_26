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

base_dir = "/training-1/asr_dataset_files/asr_bundestag"
train = base_dir + "/train_nodev"
validate = base_dir + "/train_dev"
test = base_dir + "/test"
wav_dir = "/training-1/asr/bundestag/dataset/wavs"

load_dotenv(dotenv_path="../.env")

datasets = {
    "train": load_transcript(train, 70, 1200),
    "validate": load_transcript(validate, 70, 150),
    "test": load_transcript(test, 70, 150),
    "dev": load_transcript(train, 70, 5)
}

# first load transcripts
transcripts: list[Transcript] = datasets['dev']
question_audios: dict[str, list[SpokenQuestionAnswerPair]] = {}

total_context_size = 0

# generate questions + answers for transcripts using AI
for (index, transcript) in enumerate(transcripts):
    print(f"Handling transcript {transcript['id']} - {index+1}/{len(transcripts)}")
    questions = generate_qa(transcript['text'])

    total_context_size += os.path.getsize(transcript['filepath'])

    json_file_path = Path(os.path.join("tts-audios", transcript['id'], "question_set.json"))
    json_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump({
            "questions": questions,
            **transcript,
        }, f, indent=2, ensure_ascii=False)

    spoken_questions: list[SpokenQuestionAnswerPair] = []

    for question in questions:
        print(f"--> handling question {question['difficulty']}")
        start_time = datetime.now()
        # generate tts audio
        wav = to_speech(question['question'])

        file_path = "tts-audios/" + transcript['id'] + f"/{question['difficulty']}.wav"
        with open(file_path, "wb") as f:
            f.write(wav)

        total_context_size += os.path.getsize(file_path)

        finished_time = datetime.now()

        spoken_questions.append({
            **question,
            "audio_buffer": wav
        })

        print(f"--> HANDLED in {finished_time - start_time}")

    question_audios[transcript['id']] = spoken_questions

# Generate parquet
export_parquet(transcripts, question_audios, out_path="out/dataset.parquet")

print(f"Total context size: {total_context_size/(2**20)}")