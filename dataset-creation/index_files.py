from sqa_types import Transcript

def load_transcript(base_path: str, minimum_word_count = 70, limit = 5) -> list[Transcript]:
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

    # cut transcripts to have at most debug_transcript_limit transcripts
    return transcripts[:limit] if limit else transcripts