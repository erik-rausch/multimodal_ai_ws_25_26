import os
import pandas as pd
from pydub import AudioSegment
from pydub.utils import mediainfo

def get_total_audio_duration(filenames):
    total_duration = 0
    
    for filename in filenames:
        wav_file = f"/training-1/asr/raw/asr_bundestag_clean/wavs/{filename}.wav"
        
        try:
            audio = AudioSegment.from_wav(wav_file)
            duration_seconds = len(audio) / 1000
            total_duration += duration_seconds
            
            print(f"{wav_file}: {duration_seconds:.2f} Sekunden")
            
        except FileNotFoundError:
            print(f"Warnung: {wav_file} nicht gefunden")
        except Exception as e:
            print(f"Fehler bei {wav_file}: {e}")
    
    return total_duration


if __name__ == "__main__":
    
    # parquet holen
    parquet_file = "../dataset-creation/out/validate.parquet"
    df = pd.read_parquet(parquet_file)
    
    # nur transcript holen
    filenames = df['transcript_id'].tolist()
    
    total = get_total_audio_duration(filenames)
    
    print(f"\n{'='*50}")
    print(f"Gesamtdauer: {total:.2f} Sekunden")
    print(f"Gesamtdauer: {total/60:.2f} Minuten")
    print(f"{'='*50}")
