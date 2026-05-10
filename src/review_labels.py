import argparse
from pathlib import Path

import gradio as gr
import pandas as pd
from PIL import Image

from label_schema import LABELS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/reaction_meme_dataset")
    parser.add_argument("--server_port", type=int, default=7860)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    review_path = dataset_dir / "labels_to_review.csv"
    output_path = dataset_dir / "labels.csv"
    csv_path = output_path if output_path.exists() else review_path
    df = pd.read_csv(csv_path)
    current = {"index": 0, "df": df}

    def load_item(index):
        index = max(0, min(int(index), len(current["df"]) - 1))
        current["index"] = index
        row = current["df"].iloc[index]
        image = Image.open(dataset_dir / row["image"]).convert("RGB")
        note_value = row.get("notes", "")
        if pd.isna(note_value):
            note_value = ""
        return image, row.get("text", ""), row.get("label", "neutral"), note_value, index, f"{index + 1}/{len(current['df'])}"

    def save_label(label, notes):
        index = current["index"]
        current["df"].loc[index, "label"] = label
        current["df"].loc[index, "notes"] = notes
        current["df"].to_csv(output_path, index=False, encoding="utf-8")
        next_index = min(index + 1, len(current["df"]) - 1)
        return load_item(next_index)

    with gr.Blocks(title="Meme Label Reviewer") as demo:
        gr.Markdown("# Meme Label Reviewer")
        progress = gr.Textbox(label="Progress", interactive=False)
        image = gr.Image(type="pil", label="Image")
        text = gr.Textbox(label="Dataset text/caption", interactive=False)
        label = gr.Dropdown(choices=LABELS, label="Reviewed label")
        notes = gr.Textbox(label="Notes")
        index = gr.Number(value=0, label="Index", precision=0)
        with gr.Row():
            go = gr.Button("Go to index")
            save_next = gr.Button("Save and next", variant="primary")

        go.click(load_item, inputs=index, outputs=[image, text, label, notes, index, progress])
        save_next.click(save_label, inputs=[label, notes], outputs=[image, text, label, notes, index, progress])
        demo.load(load_item, inputs=index, outputs=[image, text, label, notes, index, progress])

    demo.launch(server_name="127.0.0.1", server_port=args.server_port)


if __name__ == "__main__":
    main()



