import os
from pathlib import Path

import gradio as gr
import joblib
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, CLIPModel

try:
    from transformers import ChineseCLIPModel
except ImportError:
    ChineseCLIPModel = None


LABEL_TEXT = {
    "agreement": "Agreement or support.",
    "refusal": "Refusal or unwillingness.",
    "mocking": "Sarcasm, teasing, or ridicule.",
    "comfort": "Comfort, care, or encouragement.",
    "confusion": "Confusion, shock, or speechlessness.",
    "celebration": "Happiness, success, or celebration.",
    "neutral": "General meme usage.",
}


def resolve_model_dir():
    local_dir = Path("model")
    if (local_dir / "classifier.joblib").exists():
        return local_dir
    repo_id = os.getenv("MODEL_REPO_ID", "")
    if repo_id:
        return Path(snapshot_download(repo_id))
    raise FileNotFoundError("Train a model into ./model or set MODEL_REPO_ID.")


BUNDLE = joblib.load(resolve_model_dir() / "classifier.joblib")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSOR = AutoProcessor.from_pretrained(BUNDLE["embedding_model"])

if "chinese-clip" in BUNDLE["embedding_model"].lower() and ChineseCLIPModel is not None:
    EMBEDDING_MODEL = ChineseCLIPModel.from_pretrained(BUNDLE["embedding_model"]).to(DEVICE)
else:
    EMBEDDING_MODEL = CLIPModel.from_pretrained(BUNDLE["embedding_model"]).to(DEVICE)
EMBEDDING_MODEL.eval()


def embed_image(image):
    inputs = PROCESSOR(images=[image.convert("RGB")], return_tensors="pt")
    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
    with torch.no_grad():
        features = EMBEDDING_MODEL.get_image_features(**inputs)
        if hasattr(features, "pooler_output"):
            features = features.pooler_output
        elif hasattr(features, "image_embeds"):
            features = features.image_embeds
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy()


def predict(image):
    if image is None:
        return "Please upload a meme image.", {}

    vector = embed_image(image)
    classifier = BUNDLE["classifier"]
    encoder = BUNDLE["label_encoder"]
    pred = classifier.predict(vector)[0]
    label = encoder.inverse_transform([pred])[0]

    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(vector)[0]
        scores = {encoder.inverse_transform([i])[0]: float(score) for i, score in enumerate(probabilities)}
    else:
        scores = {name: 0.0 for name in encoder.classes_}
        scores[label] = 1.0

    return f"Prediction: {label}\n\n{LABEL_TEXT.get(label, '')}\n\nBest classifier: {BUNDLE['best_classifier']}", scores


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload a Chinese reaction meme"),
    outputs=[gr.Textbox(label="Prediction"), gr.Label(label="Class scores")],
    title="Chinese Reaction Meme Usage Classifier",
    description="Classifies Chinese reaction memes by likely chat usage using image embeddings.",
)


if __name__ == "__main__":
    demo.launch()
