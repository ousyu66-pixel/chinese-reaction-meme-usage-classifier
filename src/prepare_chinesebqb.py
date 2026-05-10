import argparse
import csv
import io
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from label_schema import weak_label


TEXT_HINT_COLUMNS = [
    "text",
    "caption",
    "description",
    "content",
    "prompt",
    "query",
    "answer",
    "label",
]


def find_image_column(example):
    for key, value in example.items():
        if isinstance(value, Image.Image):
            return key
        if isinstance(value, dict) and ("bytes" in value or "path" in value):
            return key
    return None


def image_from_value(value):
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes"):
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    raise ValueError("Could not decode image value.")


def extract_text(example):
    parts = []
    for key in TEXT_HINT_COLUMNS:
        value = example.get(key)
        if isinstance(value, str):
            parts.append(value)
    if not parts:
        for value in example.values():
            if isinstance(value, str) and len(value) <= 500:
                parts.append(value)
    return " ".join(parts).strip()


def choose_split(index):
    bucket = index % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "validation"
    return "test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_id", default="mrzjy/ChineseBQB")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--out_dir", default="data/reaction_meme_dataset")
    parser.add_argument("--streaming", action="store_true", default=True)
    args = parser.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset_id, split=args.split, streaming=args.streaming)

    rows = []
    iterator = iter(dataset)
    first = next(iterator)
    image_column = find_image_column(first)
    if image_column is None:
        raise ValueError("No image column found in the dataset.")

    examples = [first]
    for example in iterator:
        examples.append(example)
        if len(examples) >= args.limit:
            break

    for index, example in enumerate(tqdm(examples, desc="Exporting images")):
        try:
            image = image_from_value(example[image_column])
        except Exception as exc:
            print(f"Skipping item {index}: {exc}")
            continue

        filename = f"{index:05d}.jpg"
        image_path = image_dir / filename
        image.save(image_path, quality=95)

        text = extract_text(example)
        rows.append(
            {
                "filename": filename,
                "image": f"images/{filename}",
                "text": text,
                "weak_label": weak_label(text),
                "label": weak_label(text),
                "split": choose_split(len(rows)),
                "source_dataset": args.dataset_id,
                "notes": "",
            }
        )

    for csv_name in ["metadata.csv", "labels_to_review.csv"]:
        with (out_dir / csv_name).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(rows)} examples to {out_dir}")
    print("Review labels_to_review.csv, then save it as labels.csv before training.")


if __name__ == "__main__":
    main()

