from ask_lisa import ask_lisa

with open("../../system_prompts/eval_answer_prompt.txt", "r") as f:
    eval_prompt = f.read()


def evaluate(context: str, question: str, expected_answer: str, generated_answer: str) -> int:
    res = ask_lisa(eval_prompt, f"""
        Kontext: {context}
        Frage: {question}
        Ground Truth Antwort: {expected_answer}
        Generierte Antwort: {generated_answer}
    """)
    if res is None or len(res) > 1:
        return -1
    return int(res)


