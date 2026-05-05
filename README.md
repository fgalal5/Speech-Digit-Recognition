# Spoken Digit Recognition with PCEN + MFCC Features and BiLSTM

This project trains a spoken digit recognition model in **Google Colab** using a **feature-fusion pipeline** and a **bidirectional LSTM classifier**.

The notebook loads audio files from **Google Drive** (audio recordings too large to upload, if interested in reproducing our results please contact me for our google drive that includes the data folder), extracts two complementary feature streams:

- **PCEN + delta + delta-delta**
- **MFCC + delta + delta-delta**

These are concatenated into a single frame-level representation and passed to a **2-layer bidirectional LSTM** for classification into **10 spoken digit classes (0–9)**.

---

## Overview

The system is designed to classify spoken digits under both clean and noisy conditions. It evaluates performance on:

- **Clean test data**
- **Noisy test data at 5 dB SNR (babble noise)**
- **Noisy test data at 10 dB SNR (babble noise)**

The feature pipeline combines:

### 1. PCEN stream
Per-Channel Energy Normalization (PCEN) is applied to mel spectrograms. This helps improve robustness to loudness variation and background noise.

PCEN features used:
- 64 PCEN coefficients
- 64 first-order deltas
- 64 second-order deltas

**Total PCEN dimensions: 192**

### 2. MFCC stream
MFCCs capture compact spectral envelope information useful for speech discrimination.

MFCC features used:
- 13 MFCC coefficients
- 13 first-order deltas
- 13 second-order deltas

**Total MFCC dimensions: 39**

### Final fused feature
The two streams are stacked together:

- **192 (PCEN stream) + 39 (MFCC stream) = 231 dimensions per frame**

---

## Model

The classifier is a **BiLSTM-based sequence model**:

- **Input size:** 231
- **LSTM hidden size:** 128
- **Number of LSTM layers:** 2
- **Bidirectional:** Yes
- **LSTM dropout:** 0.2
- **Classifier head:**
  - Linear(256 -> 32)
  - ReLU
  - Dropout(0.3)
  - Linear(32 -> 10)

After the LSTM, the notebook performs **mean pooling over valid time frames only**, ignoring padded frames.

---

## Notebook workflow

The notebook performs the following steps:

1. Mounts Google Drive in Colab
2. Locates the dataset folders
3. Loads `.wav` files using `torchaudio`
4. Extracts fused PCEN + MFCC features
5. Builds padded mini-batches for variable-length utterances
6. Trains the BiLSTM model
7. Evaluates on:
   - clean test set
   - noisy 5 dB test set
   - noisy 10 dB test set
8. Saves the checkpoint with the **best 10 dB accuracy**
9. Reloads the best checkpoint
10. Plots confusion matrices for all test conditions

---

## Expected dataset structure

The notebook expects the following Google Drive directory structure:

```text
MyDrive/
└── ECE_M214A/
    └── project_214A/
        └── M214_project_data/
            ├── train_clean/
            ├── test_clean/
            ├── test_snr_5db_babble/
            └── test_snr_10db_babble/
```

The notebook uses these paths:

```python
TRAIN_DIR = "/content/drive/MyDrive/ECE_M214A/project_214A/M214_project_data/train_clean"
TEST_CLEAN_DIR = "/content/drive/MyDrive/ECE_M214A/project_214A/M214_project_data/test_clean"
TEST_NOISY_5DB_DIR = "/content/drive/MyDrive/ECE_M214A/project_214A/M214_project_data/test_snr_5db_babble"
TEST_NOISY_10DB_DIR = "/content/drive/MyDrive/ECE_M214A/project_214A/M214_project_data/test_snr_10db_babble"
```

### File naming convention
Labels are extracted from the filename using:

```python
int(base.split("_")[0])
```

So filenames should begin with the digit label, for example:

```text
0_example.wav
7_speaker3.wav
9_utterance12.wav
```

---

## Running in Google Colab

### 1. Open the notebook in Colab
Upload the notebook to Google Drive or open it directly in Colab.

### 2. Mount Google Drive
The notebook already includes:

```python
from google.colab import drive

drive.mount('/content/drive')
```

When prompted, authorize access to your Google Drive.

### 3. Confirm dataset paths
Make sure your dataset folders match the expected structure above.

### 4. Install dependencies if needed
Depending on your Colab runtime, you may need:

```python
!pip install librosa torchaudio scikit-learn matplotlib
```

PyTorch is usually already available in Colab.

### 5. Enable GPU
In Colab:

- Go to **Runtime -> Change runtime type**
- Set **Hardware accelerator** to **GPU**

The notebook automatically uses CUDA if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

### 6. Run all cells
Run the notebook from top to bottom.

---

## Feature extraction details

### Audio loading
Audio is loaded with `torchaudio.load()` and converted to a 1D NumPy array.

### Silence trimming
Before feature extraction, the notebook removes low-energy leading and trailing regions using:

```python
librosa.effects.trim(audio, top_db=40)
```

### Spectral settings
The notebook uses:

- `WIN_LENGTH = 200`
- `HOP_LENGTH = 80`
- `N_FFT = 256`
- `N_MELS = 64`
- `N_MFCC = 13`

### Normalization
- **PCEN stream:** mean normalization
- **MFCC stream:** cepstral mean/variance normalization (CMVN)

### Delta handling
A `_safe_delta()` helper avoids invalid delta computation for very short utterances by returning zeros when needed.

---

## Training settings

The notebook uses the following training hyperparameters:

```python
BATCH_SIZE = 32
NUM_EPOCHS = 40
LR = 3e-4
```

Additional training details:

- Loss: **CrossEntropyLoss**
- Optimizer: **Adam**
- Gradient clipping: **max norm = 1.0**
- Reproducibility: fixed random seed

The model tracks the best accuracies on:

- clean set
- 5 dB noisy set
- 10 dB noisy set

The saved checkpoint is chosen based on **best 10 dB performance**.

---

## Evaluation outputs

At the end of training, the notebook reports:

- Load/features time
- Training time
- Evaluation time
- Total runtime
- Final accuracies on:
  - clean test set
  - noisy 5 dB test set
  - noisy 10 dB test set

It also plots confusion matrices for:

- Clean
- Noisy 5 dB
- Noisy 10 dB

---

## Main code components

### Utility functions
- `set_seed()` — sets seeds for reproducibility
- `load_audio()` — loads a waveform and sample rate
- `_cmvn()` — mean/variance normalization for MFCCs
- `_safe_delta()` — safe delta/delta-delta computation

### Feature extraction
- `extract_feature(audio, fs)` — builds the full 231-dim feature matrix
- `extract_feature_from_file(audio_file)` — wrapper for file-based extraction

### Dataset pipeline
- `FeatureDataset` — stores feature matrices and labels
- `collate_pad()` — pads variable-length sequences into batched tensors
- `load_dir()` — loads all `.wav` files from a directory and extracts features

### Model
- `SimpleLSTM` — bidirectional LSTM classifier with masked mean pooling

### Evaluation
- `evaluate()` — computes accuracy and optionally plots confusion matrices

---

## Tensor shapes

The padded batch shape before entering the model is:

```text
(B, 1, F, T)
```

where:

- `B` = batch size
- `F` = feature dimension = 231
- `T` = number of frames

Inside the model, this is rearranged to:

```text
(B, T, F)
```

for LSTM processing.

---

## Notes and assumptions

- The notebook assumes all audio files are `.wav`
- It expects labels to be encoded in the filename prefix
- It is written for **Google Colab**, not a local Jupyter environment by default
- Paths are hardcoded to a specific Google Drive layout and may need editing for another setup
- The best checkpoint is selected using **10 dB noisy accuracy**, which biases model selection toward robustness under noise

