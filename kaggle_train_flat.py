import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORK_DATA = ROOT / "data" / "reaction_meme_dataset"
IMAGE_DIR = WORK_DATA / "images"
MODEL_DIR = ROOT / "model"


def run(command):
    print("+", " ".join(command))
    subprocess.check_call(command, cwd=ROOT)


def prepare_files():
    WORK_DATA.mkdir(parents=True, exist_ok=True)
    if not (WORK_DATA / "labels_reviewed.csv").exists():
        shutil.copy2(ROOT / "labels_reviewed.csv", WORK_DATA / "labels_reviewed.csv")
    if not IMAGE_DIR.exists():
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ROOT / "images.zip") as archive:
            archive.extractall(IMAGE_DIR)


def main():
    prepare_files()
    run(
        [
            sys.executable,
            "src/train.py",
            "--csv",
            str(WORK_DATA / "labels_reviewed.csv"),
            "--image_root",
            str(WORK_DATA),
            "--out_dir",
            str(MODEL_DIR),
        ]
    )

    metrics_path = MODEL_DIR / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        print("\nBest classifier:", metrics.get("best_classifier"))
        for result in metrics.get("results", []):
            print(
                f"{result['name']}: "
                f"accuracy={result['accuracy']:.3f}, "
                f"macro_f1={result['macro_f1']:.3f}"
            )

    output_dir = Path("/kaggle/working")
    if output_dir.exists():
        shutil.make_archive(str(output_dir / "reaction_meme_model"), "zip", root_dir=MODEL_DIR)
        print("\nSaved output:", output_dir / "reaction_meme_model.zip")


if __name__ == "__main__":
    main()
