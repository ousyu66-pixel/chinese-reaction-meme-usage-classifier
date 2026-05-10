# Chinese Reaction Meme Usage Classifier

This project builds a custom image dataset from public Chinese meme resources, then trains classifiers on CLIP-style image embeddings.

Recommended assignment topic:

> Chinese reaction meme usage classification using image embeddings.

Instead of detecting sexism, this simpler version classifies how a Chinese meme is likely used in chat.

## Classes

| Label | Meaning |
| --- | --- |
| `agreement` | agree, support, approve |
| `refusal` | reject, unwilling, no |
| `mocking` | sarcasm, teasing, ridicule |
| `comfort` | encouragement, care, consolation |
| `confusion` | confused, speechless, shocked |
| `celebration` | happy, excited, success |
| `neutral` | general meme that does not fit above |

## Workflow

1. Download/sample public Chinese meme data.
2. Export images into a local dataset folder.
3. Create a weakly labeled CSV for fast human review.
4. Review labels manually.
5. Train multiple classifiers on image embeddings.
6. Deploy the Gradio demo to Hugging Face Spaces.

## Install

```bash
pip install -r requirements.txt
```

## Step 1: Prepare A Review Dataset

Default source is `mrzjy/ChineseBQB` on Hugging Face.

```bash
python src/prepare_chinesebqb.py --limit 400 --out_dir data/reaction_meme_dataset
```

This creates:

```text
data/reaction_meme_dataset/
├── images/
├── metadata.csv
└── labels_to_review.csv
```

## Step 2: Review Labels

Open the labeling helper:

```bash
python src/review_labels.py --dataset_dir data/reaction_meme_dataset
```

When finished, the reviewed file should be saved as:

```text
data/reaction_meme_dataset/labels.csv
```

You can also edit `labels_to_review.csv` manually in Excel and save it as `labels.csv`.

## Step 3: Train

```bash
python src/train.py --csv data/reaction_meme_dataset/labels_reviewed.csv --image_root data/reaction_meme_dataset --out_dir model
```

The training script uses image embeddings and trains:

- logistic regression
- linear SVM
- random forest
- k-nearest neighbors

The best model is selected by macro F1.

## Step 4: Run Demo

```bash
python app.py
```

## Hugging Face

Publish dataset:

```bash
python src/publish_dataset.py \
  --csv data/reaction_meme_dataset/labels.csv \
  --image_root data/reaction_meme_dataset \
  --dataset_id YOUR_USERNAME/chinese-reaction-meme-usage
```

Train and push model:

```bash
python src/train.py \
  --csv data/reaction_meme_dataset/labels.csv \
  --image_root data/reaction_meme_dataset \
  --out_dir model \
  --push_to_hub \
  --model_id YOUR_USERNAME/chinese-reaction-meme-usage-classifier
```

Create a Gradio Space and set:

```text
MODEL_REPO_ID=YOUR_USERNAME/chinese-reaction-meme-usage-classifier
```

## VG Notes

For pass with distinction, use at least 180-300 reviewed examples. The report should clearly explain that the dataset was created by filtering public Chinese meme data and manually reviewing labels into a new usage taxonomy.

## Current Dataset And Results

The current reviewed dataset contains 299 usable images after removing one static-frame sample marked for removal.

| Label | Count |
| --- | ---: |
| `agreement` | 22 |
| `celebration` | 16 |
| `comfort` | 21 |
| `confusion` | 23 |
| `mocking` | 105 |
| `neutral` | 84 |
| `refusal` | 28 |

Evaluation on the validation/test rows:

| Classifier | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Logistic regression | 0.295 | 0.230 |
| Linear SVM | 0.318 | 0.241 |
| Random forest | 0.386 | 0.131 |
| KNN | 0.284 | 0.154 |

The selected classifier is `linear_svm`, because it achieved the highest macro F1.
