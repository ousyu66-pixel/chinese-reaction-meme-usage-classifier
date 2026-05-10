import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_CSV = PROJECT_ROOT / "data" / "reaction_meme_dataset" / "labels_reviewed.csv"
IMAGE_ROOT = PROJECT_ROOT / "data" / "reaction_meme_dataset"
MODEL_DIR = PROJECT_ROOT / "model"


def run(command):
    print("+", " ".join(command))
    subprocess.check_call(command, cwd=PROJECT_ROOT)


def main():
    if not DATA_CSV.exists():
        raise FileNotFoundError(f"Missing reviewed labels file: {DATA_CSV}")

    run(
        [
            sys.executable,
            "src/train.py",
            "--csv",
            str(DATA_CSV),
            "--image_root",
            str(IMAGE_ROOT),
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
        shutil.make_archive(
            str(output_dir / "reaction_meme_model"),
            "zip",
            root_dir=MODEL_DIR,
        )
        print("\nSaved Kaggle output:", output_dir / "reaction_meme_model.zip")


if __name__ == "__main__":
    main()

