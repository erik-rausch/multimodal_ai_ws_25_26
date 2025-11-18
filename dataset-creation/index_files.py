import json

from generate_questions import check_context
from sqa_types import Transcript
from pathlib import Path


def save_relevant_transcripts(path: str, transcripts: list[Transcript]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: str) -> list[Transcript]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_transcript(base_path: str, name: str, minimum_word_count=70, limit=5) -> list[Transcript] | None:
    store_path = f"out/{name}-transcripts.json"

    if Path(store_path).exists():
        print(f"checkpoint found for {name}. Skipped context check.")
        return load_checkpoint(store_path)

    # mapping of id => path
    transcript_mappings: dict[str, str] = {}

    # array of all transcript classes
    transcripts: list[Transcript] = []

    with open(base_path + "/wav.scp", "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            words = line.strip().split()
            transcript_mappings.update({words[0]: words[1]})

    with open(base_path + "/text", "r", encoding="utf-8") as f:
        for index, line in enumerate(f):
            words = line.strip().split(maxsplit=1)
            transcript_id = words[0]
            file_path = transcript_mappings[transcript_id]
            word_count = len(words[1].split(" "))
            if word_count >= minimum_word_count:
                t: Transcript = {
                    "id": transcript_id,
                    "filepath": file_path,
                    "text": words[1],
                    "word_count": len(words[1].split(" ")),
                }
                transcripts.append(t)

    # select suitable transcripts
    relevant_transcripts: list[Transcript] = []
    sample_counter: int = 0
    
    for transcript in transcripts:
        sample_counter += 1
        check: bool = check_context(transcript["text"])
        if check:
            print(f"--> sample {sample_counter} added ({len(relevant_transcripts) + 1}/{limit})")
            relevant_transcripts.append(transcript)
            if len(relevant_transcripts) == limit:
                save_relevant_transcripts(store_path, relevant_transcripts)
                print(f"Found enough transcripts: {store_path}")
                return relevant_transcripts
        else:
            print(f"--> sample {sample_counter} ignored ({len(relevant_transcripts)}/{limit})")
            print(transcript["text"])
    print(f"Only {len(relevant_transcripts)} / {limit} transcripts found!")
    return None
