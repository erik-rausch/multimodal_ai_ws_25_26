import json
from typing import cast
from sqa_types import QuestionAnswerPair
from ask_lisa import ask_lisa, LisaLevelResponse

prompts = "../system_prompts/"
improvement_file = "out/improvements.jsonl"

with open(prompts + "check_context_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt_check_context = f.read()

with open(prompts + "generate_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt_generate = f.read()

with open(prompts + "verify_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt_verify = f.read() + system_prompt_generate


def log_improvement(old_answer: str, new_answer: str):
    """Loggt Verbesserungen in eine JSONL-Datei, falls improvement_file gesetzt ist."""
    if not improvement_file:
        return
    
    entry = {
        "old_answer": old_answer,
        "new_answer": new_answer
    }

    try:
        with open(improvement_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print("Fehler beim Schreiben der Improvement-Datei:", e)


def check_context(context):
    res = ask_lisa(system_prompt_check_context, context)
    try:
        return json.loads(res).get("context") == "ok"
    except json.JSONDecodeError:
        print("Antwort war kein gültiges JSON:", res)
        return False


def generate_qa(context: str) -> list[QuestionAnswerPair]:
    qa_res = ask_lisa(system_prompt_generate, context)

    try:
        qa_all = json.loads(qa_res)
    except json.JSONDecodeError:
        print("Antwort war kein gültiges JSON:", qa_res)
        return None

    verify = ask_lisa(
        system_prompt_verify,
        f"""
        Kontext: {context}
        Antwort: {qa_res}
        """
    )

    final_json = qa_all

    try:
        verified = json.loads(verify)

        if verified.get("quality") == "ok":
            print("No improvement needed")

        else:
            print("Questions improved!")

            # Alte und neue Antwort loggen
            new_data = verified.get("data")
            log_improvement(
                old_answer=qa_res,
                new_answer=json.dumps(new_data, ensure_ascii=False)
            )

            final_json = new_data

    except json.JSONDecodeError:
        print("Antwort war kein gültiges JSON:", qa_res)
        return None

    level1 = cast(LisaLevelResponse, final_json.get("level_1", {}))
    level2 = cast(LisaLevelResponse, final_json.get("level_2", {}))
    level3 = cast(LisaLevelResponse, final_json.get("level_3", {}))

    return [
        {**level1, "difficulty": 1},
        {**level2, "difficulty": 2},
        {**level3, "difficulty": 3},
    ]
