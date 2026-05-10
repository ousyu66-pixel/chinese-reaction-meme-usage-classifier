import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/reaction_meme_dataset/labels.csv")
    parser.add_argument("--image_root", default="data/reaction_meme_dataset")
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    image_root = Path(args.image_root)
    df = pd.read_csv(args.csv)
    df["image"] = df["image"].apply(lambda path: str((image_root / path).resolve()))

    splits = {}
    for split, split_df in df.groupby("split"):
        dataset = Dataset.from_pandas(split_df.reset_index(drop=True))
        splits[split] = dataset.cast_column("image", Image())

    DatasetDict(splits).push_to_hub(args.dataset_id, private=args.private)
    print(f"Pushed dataset to https://huggingface.co/datasets/{args.dataset_id}")


if __name__ == "__main__":
    main()

