import pandas as pd
from pathlib import Path

def export_parquet(transcripts, question_audios, out_path: str = "dataset-creation/output/dataset.parquet"):
    rows = []
    for t in transcripts:
        tid = t["id"]
        context_text = t["text"]
        context_audio = None
        try:
            if Path(t["filepath"]).exists():
                with open(t["filepath"], "rb") as rf:
                    context_audio = rf.read()
        except Exception:
            context_audio = None

        qlist = question_audios.get(tid, [])

        def get_q(qlist, idx):
            if idx < len(qlist):
                q = qlist[idx]
                # audio_buffer was stored as raw bytes in main.py
                return q.get("audio_buffer"), q.get("question"), q.get("answer")
            return None, None, None

        q1_audio, q1_text, a1_text = get_q(qlist, 0)
        q2_audio, q2_text, a2_text = get_q(qlist, 1)
        q3_audio, q3_text, a3_text = get_q(qlist, 2)

        rows.append({
            "transcript_id": tid,
            "context_text": context_text,
            "context_audio": context_audio,
            "q1_audio": q1_audio, "q1_text": q1_text, "a1_text": a1_text,
            "q2_audio": q2_audio, "q2_text": q2_text, "a2_text": a2_text,
            "q3_audio": q3_audio, "q3_text": q3_text, "a3_text": a3_text,
        })

    df = pd.DataFrame(rows, columns=[
        "transcript_id", "context_text", "context_audio",
        "q1_audio", "q1_text", "a1_text",
        "q2_audio", "q2_text", "a2_text",
        "q3_audio", "q3_text", "a3_text",
    ])

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    # write parquet with pyarrow engine (binary columns preserved)
    df.to_parquet(out_file, engine="pyarrow", index=False)