import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm
from transformers import AutoProcessor, CLIPModel

try:
    from transformers import ChineseCLIPModel
except ImportError:
    ChineseCLIPModel = None


DEFAULT_EMBEDDING_MODEL = "OFA-Sys/chinese-clip-vit-base-patch16"


def load_embedding_model(model_name, device):
    processor = AutoProcessor.from_pretrained(model_name)
    if "chinese-clip" in model_name.lower() and ChineseCLIPModel is not None:
        model = ChineseCLIPModel.from_pretrained(model_name)
    else:
        model = CLIPModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return processor, model


def extract_embeddings(df, image_root, processor, model, device, batch_size):
    vectors = []
    paths = [image_root / image_path for image_path in df["image"].tolist()]
    for start in tqdm(range(0, len(paths), batch_size), desc="Embedding images"):
        batch_paths = paths[start : start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            features = model.get_image_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            elif hasattr(features, "image_embeds"):
                features = features.image_embeds
            features = features / features.norm(dim=-1, keepdim=True)
        vectors.append(features.cpu().numpy())
    return np.vstack(vectors)


def build_classifiers():
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        ),
        "linear_svm": make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", random_state=42)),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/reaction_meme_dataset/labels.csv")
    parser.add_argument("--image_root", default="data/reaction_meme_dataset")
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--out_dir", default="model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--model_id")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=["image", "label", "split"]).reset_index(drop=True)
    df = df[df["label"].astype(str).str.len() > 0].reset_index(drop=True)

    if df["label"].nunique() < 2:
        raise ValueError("Need at least two labels to train a classifier.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_embedding_model(args.embedding_model, device)
    x = extract_embeddings(df, Path(args.image_root), processor, model, device, args.batch_size)

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    train_mask = df["split"].eq("train").to_numpy()
    eval_mask = df["split"].isin(["validation", "test"]).to_numpy()

    if not train_mask.any() or not eval_mask.any():
        raise ValueError("Need train and validation/test rows.")

    results = []
    fitted = {}
    for name, classifier in build_classifiers().items():
        classifier.fit(x[train_mask], y[train_mask])
        pred = classifier.predict(x[eval_mask])
        result = {
            "name": name,
            "accuracy": accuracy_score(y[eval_mask], pred),
            "macro_f1": f1_score(y[eval_mask], pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y[eval_mask], pred, average="weighted", zero_division=0),
            "classification_report": classification_report(
                y[eval_mask],
                pred,
                labels=list(range(len(encoder.classes_))),
                target_names=encoder.classes_,
                zero_division=0,
                output_dict=True,
            ),
        }
        results.append(result)
        fitted[name] = classifier
        print(f"{name}: macro_f1={result['macro_f1']:.3f}, accuracy={result['accuracy']:.3f}")

    best = max(results, key=lambda item: item["macro_f1"])
    bundle = {
        "classifier": fitted[best["name"]],
        "label_encoder": encoder,
        "embedding_model": args.embedding_model,
        "best_classifier": best["name"],
        "class_names": encoder.classes_.tolist(),
    }
    joblib.dump(bundle, out_dir / "classifier.joblib")

    metrics = {
        "embedding_model": args.embedding_model,
        "best_classifier": best["name"],
        "results": results,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        f"# Chinese Reaction Meme Usage Classifier\n\nBest classifier: {best['name']}\n\nMacro F1: {best['macro_f1']:.3f}\n",
        encoding="utf-8",
    )

    if args.push_to_hub:
        if not args.model_id:
            raise ValueError("--model_id is required with --push_to_hub")
        from huggingface_hub import HfApi

        HfApi().upload_folder(folder_path=str(out_dir), repo_id=args.model_id, repo_type="model")

    print(f"Saved model to {out_dir / 'classifier.joblib'}")


if __name__ == "__main__":
    main()
