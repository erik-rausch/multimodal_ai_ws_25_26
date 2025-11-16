import json
from enum import Enum
from typing import TypedDict, cast

from ask_lisa import ask_lisa, LisaLevelResponse


class QuestionDifficulty(Enum):
    EASY = 1
    MEDIUM = 2
    HARD = 3

class QuestionAnswerPair(TypedDict):
    question: str
    answer: str
    difficulty: int

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
        {**level1, "difficulty": QuestionDifficulty.EASY},
        {**level2, "difficulty": QuestionDifficulty.MEDIUM},
        {**level3, "difficulty": QuestionDifficulty.HARD}
    ]