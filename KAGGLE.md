# Kaggle Training Guide

This folder is ready to upload to Kaggle.

## Option A: Upload As A Kaggle Dataset

1. Compress this whole folder as a zip.
2. Go to Kaggle -> Datasets -> New Dataset.
3. Upload the zip.
4. Create a new Kaggle Notebook.
5. Add your uploaded dataset as input.
6. In the notebook, run:

```python
import os
import shutil
from pathlib import Path

input_root = Path("/kaggle/input")
project = next(input_root.glob("*/chinese-reaction-meme-usage-project"), None)

if project is None:
    # If Kaggle extracts the files directly under the dataset folder:
    for candidate in input_root.glob("*"):
        if (candidate / "kaggle_train.py").exists():
            project = candidate
            break

work = Path("/kaggle/working/chinese-reaction-meme-usage-project")
if work.exists():
    shutil.rmtree(work)
shutil.copytree(project, work)
os.chdir(work)

!pip install -q -r requirements.txt
!python kaggle_train.py
```

The trained model zip will be created at:

```text
/kaggle/working/reaction_meme_model.zip
```

## Option B: Upload As Notebook Files

You can also upload the project files into a Kaggle Notebook directly, but Dataset upload is cleaner because it keeps the images and CSV together.

## Expected Runtime

On Kaggle with internet enabled:

- dependency/model download: usually 3-10 minutes
- embedding 299 images: usually 1-5 minutes with GPU, longer on CPU
- classifier training: under 1 minute

If `OFA-Sys/chinese-clip-vit-base-patch16` fails to download, edit the training command and use:

```bash
--embedding_model openai/clip-vit-base-patch32
```

