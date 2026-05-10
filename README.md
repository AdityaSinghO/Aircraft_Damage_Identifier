# Aircraft Damage Classification and Captioning

A deep learning project that automates aircraft damage detection using a fine-tuned VGG16 model for binary classification (dent vs. crack), and generates natural language captions and summaries of damage images using the BLIP transformer model.

---

## Project Overview

Manual aircraft inspection is time-consuming and error-prone. This project addresses that by:

- **Classifying** aircraft damage images into two categories: `dent` or `crack`, using transfer learning with VGG16.
- **Captioning and summarizing** damage images using the BLIP (Bootstrapping Language-Image Pretraining) pretrained model from Hugging Face, wrapped in a custom Keras layer.

---

## Dataset

The project uses the [Aircraft Damage Dataset](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZjXM4RKxlBK9__ZjHBLl5A/aircraft-damage-dataset-v1.tar), originally sourced from [Roboflow](https://universe.roboflow.com/youssef-donia-fhktl/aircraft-damage-detection-1j9qk) (License: CC BY 4.0).

The dataset is downloaded and extracted automatically by the notebook. Its structure:

```
aircraft_damage_dataset_v1/
├── train/
│   ├── dent/
│   └── crack/
├── valid/
│   ├── dent/
│   └── crack/
└── test/
    ├── dent/
    └── crack/
```

---

## Requirements

Install all dependencies before running the notebook:

```bash
pip install pandas==2.2.3
pip install tensorflow_cpu==2.17.1
pip install pillow==11.1.0
pip install matplotlib==3.9.2
pip install transformers==4.38.2
pip install torch==2.2.0+cpu torchvision==0.17.0+cpu torchaudio==2.2.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
```

> If using a GPU, remove the `_cpu` suffix from `tensorflow_cpu` and adjust the PyTorch install URL accordingly. You may also want to comment out the TensorFlow warning suppression lines at the top of the notebook.

---

## Notebook Structure

### Part 1 — Classification (VGG16)

| Section | Description |
|---------|-------------|
| 1.1 Dataset Preparation | Downloads and extracts the dataset; defines train/valid/test directories |
| 1.2 Data Preprocessing | Creates `ImageDataGenerator` instances and `flow_from_directory` generators |
| 1.3 Model Definition | Loads pretrained VGG16 (frozen), adds custom Dense + Dropout classifier head |
| 1.4 Model Training | Trains for 5 epochs using Adam (lr=0.0001) and binary crossentropy loss |
| 1.5 Visualizing Training Results | Plots training and validation loss and accuracy curves |
| 1.6 Model Evaluation | Evaluates the model on the test set, reports loss and accuracy |
| 1.7 Visualizing Predictions | Displays test images with true and predicted labels side by side |

**Model architecture:**
- Base: VGG16 pretrained on ImageNet (all layers frozen)
- Custom head: `Flatten → Dense(512, ReLU) → Dropout(0.3) → Dense(512, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)`

**Training config:**
- Input size: 224×224×3
- Batch size: 32
- Epochs: 5
- Optimizer: Adam (lr=0.0001)
- Loss: Binary Crossentropy

---

### Part 2 — Image Captioning & Summarization (BLIP)

| Section | Description |
|---------|-------------|
| 2.1 Loading BLIP Model | Loads `Salesforce/blip-image-captioning-base` processor and model from Hugging Face |
| 2.2 Generating Captions and Summaries | Runs caption and summary generation on example damage images |

**Custom Keras Layer — `BlipCaptionSummaryLayer`:**

A `tf.keras.layers.Layer` subclass that wraps the BLIP model. Accepts an image path and a task string (`"caption"` or `"summary"`), preprocesses the image with PIL, runs it through BLIP, and returns the generated text via `tf.py_function`.

**Helper function:**
```python
generate_text(image_path, task)
# image_path: tf.constant string path to image
# task: tf.constant, either "caption" or "summary"
```

---

## Tasks Summary

The notebook includes 10 graded tasks:

| Task | Description |
|------|-------------|
| 1 | Create `valid_generator` from `valid_datagen` |
| 2 | Create `test_generator` from `test_datagen` |
| 3 | Load VGG16 with ImageNet weights, `include_top=False` |
| 4 | Compile model with Adam, binary crossentropy, accuracy metric |
| 5 | Train the model for `n_epochs` epochs |
| 6 | Plot training and validation accuracy curves |
| 7 | Visualize a test image prediction at `index_to_plot=1` |
| 8 | Implement `generate_text` helper using `BlipCaptionSummaryLayer` |
| 9 | Generate a caption for a test dent image using BLIP |
| 10 | Generate a summary for the same test image using BLIP |

---

## Notes

- Predictions from neural networks are probabilistic — a misclassification between `dent` and `crack` is expected behavior, especially with a small number of training epochs.
- BLIP captions and summaries may not be perfectly accurate for domain-specific damage imagery, as the model was not fine-tuned on aviation data.
- A random seed of `42` is set across Python, NumPy, and TensorFlow for reproducibility.
