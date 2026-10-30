# Skin Lesion Image Classification with ResNet50 — HAM10000

A TensorFlow/Keras transfer-learning project that compares **partial fine-tuning** and **full fine-tuning** of an ImageNet-pretrained **ResNet50** on the [HAM10000 skin lesion dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).

> **Educational project, not a medical diagnostic tool.** The binary labels below are a teaching simplification and **do not represent a clinically valid definition of benign versus malignant lesions**. Do not use predictions for diagnosis or treatment decisions.

## Dataset

The HAM10000 dataset contains dermatoscopic images and metadata, including `image_id`, `lesion_id`, and `dx` (diagnostic category). Its seven diagnostic categories are:

| Code | Category |
| --- | --- |
| `akiec` | Actinic keratoses and intraepithelial carcinoma |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `df` | Dermatofibroma |
| `mel` | Melanoma |
| `nv` | Melanocytic nevi |
| `vasc` | Vascular lesions |

**Labeling used in this notebook:** The `akiec` category is excluded. `mel` and `bcc` receive label `1`; all remaining categories receive label `0`. For convenience, the notebook prints these as **“Malignant”** and **“Benign”**, but these names are only shorthand for this custom grouping and must not be interpreted as medically validated diagnoses.

## Notebook and workflow

The implementation is in [`Skin Model.ipynb`](Skin%20Model.ipynb), designed to run in **Google Colab**.

1. Install the Kaggle CLI and upload your `kaggle.json` API credentials in Colab.
2. Download and extract the dataset into `/content/skin_data`.
3. Load `HAM10000_metadata.csv`, remove `akiec`, and create the custom binary `label` column.
4. Match `image_id` values to JPEGs in `ham10000_images_part_1` and `ham10000_images_part_2`.
5. Split into training (80%) and held-out (20%) sets using `GroupShuffleSplit` grouped by `lesion_id`. This helps prevent images of the same lesion from appearing in both sets.
6. Resize images to **224 × 224**, apply ResNet50 `preprocess_input`, and build `tf.data` pipelines with batch size **16**.
7. Train a new sigmoid classifier head while the ResNet50 backbone is frozen.
8. Start two separate experiments from the **same head-only checkpoint**: partial fine-tuning and full fine-tuning.
9. Print classification reports, save each model, and optionally upload an image for inference.

### Architecture

```text
Input image (224 × 224 × 3)
        ↓
ResNet50 (ImageNet pretrained, include_top=False)
        ↓
Global average pooling (2048 features)
        ↓
Dense(1, activation='sigmoid')
        ↓
Binary score for the custom label 1
```

The model is trained end-to-end with a Keras classification head; **no SVM is used** in this notebook.

### Training configurations

| Stage | Trainable layers | Optimizer | Epochs | Saved artifact |
| --- | --- | --- | ---: | --- |
| Head-only | New Dense classifier; ResNet50 frozen | Adam (`1e-3`) | 2 | `head_start.weights.h5` |
| Partial fine-tuning | Classifier + trainable non-BatchNorm layers among the final 20 ResNet50 layers | Adam (`1e-5`) | 3 | `partial_finetuned.keras` |
| Full fine-tuning | Classifier + all non-BatchNorm ResNet50 layers | Adam (`1e-5`) | 3 | `full_finetuned.keras` |

Both fine-tuning runs begin from the same head-only weights, and BatchNormalization layers remain frozen during fine-tuning. The loss is `binary_crossentropy`, with `accuracy` tracked during training.

## How to run in Google Colab

1. Open `Skin Model.ipynb` in Colab and select a suitable runtime (GPU recommended).
2. Obtain a Kaggle API token (`kaggle.json`) from your Kaggle account. **Never commit this file to GitHub.**
3. Run notebook cells in order. The notebook downloads the dataset using:

   ```bash
   kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
   ```

4. Allow the head-only, partial, and full training stages to finish.
5. Download the resulting `.keras` files from Colab if you want to reuse them outside that session.

**Dependencies:** Python, TensorFlow/Keras, NumPy, pandas, scikit-learn, and the Kaggle CLI. Google Colab is also used for file uploads.

## Predict on a new image

After training, the notebook's last cell prompts you to upload a JPEG or another supported image and loads `/content/partial_finetuned.keras`. To run inference separately, provide a path to your saved model and image:

```python
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input

model = load_model('partial_finetuned.keras')
# Alternatively: load_model('full_finetuned.keras')

image = load_img('example.jpg', target_size=(224, 224))
image = img_to_array(image)
image = preprocess_input(np.expand_dims(image, axis=0))

score = float(model.predict(image, verbose=0)[0][0])
label = 1 if score >= 0.5 else 0

print('Custom binary label:', label)
print('Model score for label 1:', round(score, 4))
```

A score of `0.5` or above maps to **custom label 1** (`mel` or `bcc` in the training data); a lower score maps to **custom label 0** (the other retained categories). The score is a model output, **not a calibrated clinical risk estimate**. The `.keras` files are created by running the notebook; they are not included here by default.

## Evaluation and limitations

The notebook uses `classification_report` to display precision, recall, and F1-score for the two custom classes, alongside accuracy tracked during fitting. **No numerical results are claimed in this README** because actual run outputs are not supplied here.

Important considerations:

- The notebook passes the **test split as `validation_data` during training**, then reports performance on that same split. For a trustworthy final evaluation, create a separate lesion-grouped validation split and reserve the test split until the end.
- The notebook does not explicitly stratify the lesion-grouped split; inspect class balance after splitting.
- HAM10000 labels and the notebook's binary mapping should not be treated as a clinical malignancy classifier.
- External validation, dataset-bias checks, calibration, and clinical review would be necessary before any medical application.

## Project files

```text
project/
├── Skin Model.ipynb         # Dataset download, training, evaluation, inference
├── README.md                # This documentation
├── partial_finetuned.keras  # Generated after training (optional)
└── full_finetuned.keras     # Generated after training (optional)
```

## Dataset credit

Dataset: [Skin Cancer MNIST: HAM10000 on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000), curated by the HAM10000 dataset authors. Check the dataset page for original attribution and applicable usage terms.
