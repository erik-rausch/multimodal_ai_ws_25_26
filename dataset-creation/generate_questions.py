import json
from typing import cast
from sqa_types import QuestionAnswerPair

from ask_lisa import ask_lisa, LisaLevelResponse

# load system prompt
with open("../system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

def generate_qa(transcript: str) -> list[QuestionAnswerPair]:
    qa_res = ask_lisa(system_prompt, transcript)

    try:
        qa_all = json.loads(qa_res)
    except json.JSONDecodeError:
        print("Antwort war kein gültiges JSON:", qa_res)
        return []

    level1 = cast(LisaLevelResponse, qa_all.get("level_1", {}))
    level2 = cast(LisaLevelResponse, qa_all.get("level_2", {}))
    level3 = cast(LisaLevelResponse, qa_all.get("level_3", {}))

    return [
        {**level1, "difficulty": 1},
        {**level2, "difficulty": 2},
        {**level3, "difficulty": 3}
    ]