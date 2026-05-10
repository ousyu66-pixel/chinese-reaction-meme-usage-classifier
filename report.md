# Chinese Reaction Meme Usage Classification Using Image Embeddings

**GitHub repository:** TODO  
**Hugging Face dataset:** TODO  
**Hugging Face model:** TODO  
**Hugging Face demo Space:** TODO  

## 1. Problem Definition

This project addresses a classification problem in the domain of Chinese online reaction memes. In chat contexts, reaction memes are used as communicative signals rather than only as images: a meme can express agreement, refusal, teasing, comfort, confusion, celebration, or a neutral/abstract reaction. The challenge is that many Chinese memes rely on visual style, facial expression, embedded text, internet slang, and shared cultural context. A text-only classifier would miss visual cues, while a generic image classifier would not directly model communicative intent. Therefore, this project uses image embeddings as compact visual representations for meme usage classification.

## 2. Dataset

A custom dataset was created by processing a public Chinese meme source and manually reviewing the labels for this assignment. The starting point was a sampled Chinese reaction meme collection. GIF-like or animated sources were converted into static image files for image-classification training. I manually reviewed the meme images and assigned one of seven usage labels: `agreement`, `refusal`, `mocking`, `comfort`, `confusion`, `celebration`, and `neutral`. Low-quality or context-lost samples were marked for removal. The final dataset contains 299 usable images. The dataset is imbalanced because many reviewed memes were abstract humor or teasing, which is common in Chinese reaction meme culture.

## 3. Method

The embedding model is `OFA-Sys/chinese-clip-vit-base-patch16`, a Chinese CLIP-style vision-language model. For each meme image, the system extracts a fixed image embedding and trains lightweight classifiers on top of those frozen embeddings. The classifiers output one of seven communicative usage labels: `agreement`, `refusal`, `mocking`, `comfort`, `confusion`, `celebration`, or `neutral`. This means the model does not describe the full content of the image; it predicts the most likely chat function of the meme.

Four classifiers were trained and compared: logistic regression, linear support vector machine, random forest, and k-nearest neighbors. The data was divided into train, validation, and test rows, and performance was measured with accuracy and macro F1. Macro F1 was treated as the main metric because the label distribution is imbalanced. The demo returns the predicted label for an uploaded image. Because the selected model is a linear SVM, the demo score display should be interpreted as a class decision rather than a calibrated probability.

## 4. Results

| Classifier | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Logistic regression | 0.295 | 0.230 |
| Linear SVM | 0.318 | 0.241 |
| Random forest | 0.386 | 0.131 |
| KNN | 0.284 | 0.154 |

The best model by macro F1 was the linear SVM. The scores are modest, but this is expected for a small, context-dependent meme dataset. Many images are ambiguous without the surrounding chat context, and some original GIFs lose meaning when represented as static images. The model was more reliable for broad categories such as `mocking` and `neutral`, while smaller classes such as `celebration` and `comfort` were more difficult. A limitation is that each image has only one label, even though a meme can express mixed meanings depending on context.

## 5. Reflection on AI Use

AI tools were used to build the data-processing pipeline, embedding extraction code, classifier training script, and Gradio demo. The most useful part was quickly turning the project idea into a reproducible workflow. However, the dataset design required human judgment. At first, I tried to force AI to generate synthetic meme images, but the memes were not realistic and I felt they could not really be called "memes", so I switched to processing existing Chinese meme images and manually choosing labels. I also found that automatic labels were unreliable because Chinese meme meanings often depend on slang, irony, and context. The final dataset therefore reflects my own judgment, but I think it is still not enough, because memes may have mixed emotions while I only typed a single result. If the result were multi-label, the work would become harder but also more interesting. This experience showed that AI tools are helpful for implementation, but we remain responsible for data quality, ethical use of sources, evaluation, and interpretation. Finally, I think this is a good way to understand code. If we do not apply code to real use, the memory can disappear quickly, but if we learn code from the project itself, even when some parts are from AI tools, we can still learn useful skills.

I have to say it is too easy for us to get an answer from AI, so the question is how to make sure the result is high quality and how to keep thinking to generate our own ideas.

## References

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning*.

OFA-Sys. (n.d.). `chinese-clip-vit-base-patch16`. Hugging Face. https://huggingface.co/OFA-Sys/chinese-clip-vit-base-patch16

ChineseBQB dataset source. Hugging Face. https://huggingface.co/datasets/mrzjy/ChineseBQB
