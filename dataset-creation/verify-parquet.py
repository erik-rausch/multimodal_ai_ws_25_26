# load parquet with pandas
import pandas as pd
from pathlib import Path

def verify_parquet(parquet_path: str):
    if not Path(parquet_path).exists():
        print(f"Parquet file does not exist: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path, engine="pyarrow")
    print(f"Loaded parquet file: {parquet_path}")
    print(f"Number of rows: {len(df)}")

    # get binary wav from column q1_audio of first row and store to directory validation-out/q1_audio.wav
    output_dir = Path("validation-out")
    output_dir.mkdir(parents=True, exist_ok=True)

    first_row = df.iloc[42]
    audio_column_names = ["q1_audio", "context_audio", "q2_audio", "q3_audio"]

    for col_name in audio_column_names:
        q1_audio = first_row[col_name]
        if q1_audio is not None:
            output_path = output_dir / f"{col_name}.wav"
            with open(output_path, "wb") as f:
                f.write(q1_audio)
            print(f"Extracted {col_name} of first row to: {output_path}")
        else:
            print(f"No {col_name} found in the first row.")

    print("Verification complete.")

verify_parquet("out/validate.parquet")