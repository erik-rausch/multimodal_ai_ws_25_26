from io import BytesIO
from typing import TypedDict

class Transcript(TypedDict):
    id: str
    filepath: str
    text: str
    word_count: int

class QuestionAnswerPair(TypedDict):
    question: str
    answer: str
    difficulty: int

class SpokenQuestionAnswerPair(QuestionAnswerPair):
    audio_buffer: BytesIO
