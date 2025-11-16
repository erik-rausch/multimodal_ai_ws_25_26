from typing import TypedDict

class Transcript(TypedDict):
    id: str
    filepath: str
    text: str
    word_count: int