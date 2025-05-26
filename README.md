# 🎓 Emotional Speech Classification using CNN and Spectrograms

This project demonstrates how to classify human emotions from voice recordings using a Convolutional Neural Network (CNN) trained on spectrogram images generated from the **RAVDESS** dataset. The pipeline is structured across two Jupyter notebooks and has been developed and tested using **Python 3.11**.

## 📁 Project Structure

```
.
├── 01_audio_to_spectrogram.ipynb   # Notebook to generate spectrograms from audio
├── 02_cnn_model_training.ipynb     # CNN training and evaluation notebook
├── dataset/                        # Contains actor folders with RAVDESS audio files
├── spectrograms/                   # Generated image dataset of spectrograms
└── emotion_cnn_model.keras         # Saved trained model for reuse
```

## 🔧 Requirements

Ensure that you are using **Python 3.11** as this version is compatible with TensorFlow.

### 1. Set up a virtual environment

We recommend using `venv`:

```bash
python3.11 -m venv venv
source venv/bin/activate    # On Windows use: venv\Scripts\activate
```

### 2. Install dependencies

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## ▶️ How to Run the Project

### Step 1: Generate Spectrograms

Open and run **`01_audio_to_spectrogram.ipynb`**.

This notebook:

- Loads all `.wav` files from the `dataset/` folder
- Converts them into mel spectrogram images using Librosa
- Saves them in a new `spectrograms/` folder
- Splits the dataset into training, validation, and test sets
- Prepares `.npy` arrays for model training

Ensure you run **all cells sequentially**

### Step 2: Train and Evaluate the CNN

Open **`02_cnn_model_training.ipynb`**.

This notebook:

- Loads the spectrogram images and labels
- Builds a CNN model using TensorFlow/Keras
- Trains the model using the training and validation sets
- Evaluates model performance on the test set
- Outputs:
  - Accuracy/loss graphs
  - Confusion matrix
  - Classification report
  - Final model saved to `emotion_cnn_model.keras`

## 📈 Expected Performance

After successful training, the model achieves around:

- **Test Accuracy:** ~88%
- **Macro F1-Score:** ~0.88

This result is competitive with recent benchmarks on the RAVDESS dataset.

## 💾 Exporting the Model

The final trained model is saved automatically in Keras format:

```bash
emotion_cnn_model.keras
```

You can reuse this model in Streamlit apps or external deployments using:

```python
from tensorflow.keras.models import load_model
model = load_model("emotion_cnn_model.keras")
```

## 📌 Notes

- Ensure your dataset structure follows the RAVDESS folder format (`Actor_01/`, `Actor_02/`, ...).
- The audio files should be in `.wav` format and named according to the RAVDESS convention.

## 👨‍🎓 Author

RG Swart – North-West University (2025)  
Student Number: `42320755`
