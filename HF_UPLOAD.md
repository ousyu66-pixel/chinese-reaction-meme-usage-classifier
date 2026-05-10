# Hugging Face Upload Guide

Recommended repo names:

- Dataset: `ousyu66-pixel/chinese-reaction-meme-usage`
- Model: `ousyu66-pixel/chinese-reaction-meme-usage-classifier`
- Space: `ousyu66-pixel/chinese-reaction-meme-usage-demo`

Log in:

```bash
hf auth login
```

Upload dataset:

```bash
python src/publish_dataset.py ^
  --csv data/reaction_meme_dataset/labels_reviewed.csv ^
  --image_root data/reaction_meme_dataset ^
  --dataset_id ousyu66-pixel/chinese-reaction-meme-usage
```

Upload model:

```bash
hf upload ousyu66-pixel/chinese-reaction-meme-usage-classifier model . --repo-type model
```

Create and upload Space:

```bash
hf repo create ousyu66-pixel/chinese-reaction-meme-usage-demo --type space --space-sdk gradio
hf upload ousyu66-pixel/chinese-reaction-meme-usage-demo hf_space . --repo-type space
```

Final links:

- GitHub: https://github.com/ousyu66-pixel/chinese-reaction-meme-usage-classifier
- Dataset: https://huggingface.co/datasets/ousyu66-pixel/chinese-reaction-meme-usage
- Model: https://huggingface.co/ousyu66-pixel/chinese-reaction-meme-usage-classifier
- Demo: https://huggingface.co/spaces/ousyu66-pixel/chinese-reaction-meme-usage-demo
