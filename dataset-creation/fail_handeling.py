import shutil
from pathlib import Path

log_path = Path("out/failed.txt")

def add_failed_log(sample_id: str):
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        f.write(sample_id + "\n")

def remove_failed_samples():
    if not log_path.exists():
        print("Found no fails to retry")
        return

    with open(log_path, "r") as f:
        failed_samples = f.read().splitlines()

    for sample in failed_samples:
        target = Path(f"tts-audios/{sample}")
        if target.exists() and target.is_dir():
            print(f"Removed for retry: {target}")
            shutil.rmtree(target)
        else:
            print(f"Failed sample not found: {target}")

    print(f"Deleting log file: {log_path}")
    log_path.unlink()
