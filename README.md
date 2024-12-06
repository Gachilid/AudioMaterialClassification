# Audio-Based Material Classification

This project implements a CNN-based classifier for audio material classification using Mel-spectrograms. It includes scripts for data preprocessing, model training, and inference.

## Files in the Repository

1. **`Highest_Accuracy_CNN_Audio_Based_Material_Classification.ipynb`**:
   - Contains the full pipeline for the project, including:
     - Data preprocessing (Mel-spectrogram generation).
     - CNN model definition and training.
     - K-Fold cross-validation results.
   - Provides an end-to-end demonstration of model performance.

2. **`run_trained_model.py`**:
   - Core inference script.
   - Loads the trained model and processes WAV file inputs.
   - Outputs the predicted class IDs for each input audio file.

3. **`README.md`**:
   - This document, describing the repository structure and usage.

---

## Installation and Setup

Follow these steps to set up the environment and run the scripts.

### Prerequisites
Make sure the following dependencies are installed:
- Python 3.8+
- Required libraries:
  ```bash
  pip install numpy torch librosa matplotlib pillow soundfile sklearn
Clone the Repository
Clone this repository to your local machine:

bash
Copy code
git clone https://github.com/Gachilid/AudioMaterialClassification.git
cd AudioMaterialClassification
Running the Project
1. Training the Model
Use the provided Jupyter Notebook to train the CNN classifier:

bash
Copy code
Highest_Accuracy_CNN_Audio_Based_Material_Classification.ipynb
Follow the steps in the notebook to preprocess the data, define the model, and train it using K-Fold cross-validation.
Save the trained model weights using:
python
Copy code
torch.save(model.state_dict(), "best_model.pth")
2. Inference
Run run_trained_model.py to make predictions on a list of WAV files:

python
Copy code
from run_trained_model import run_trained_model

# Example WAV file inputs
wav_files = ["path/to/audio1.wav", "path/to/audio2.wav"]

# Get predictions
predictions = run_trained_model(wav_files)
print("Predictions:", predictions)
Directory Structure
arduino
Copy code
AudioMaterialClassification/
├── Highest_Accuracy_CNN_Audio_Based_Material_Classification.ipynb
├── run_trained_model.py
├── README.md
├── best_model.pth (optional, saved after training)
├── Data/
    ├── water/
    ├── table/
    ├── sofa/
    ├── railing/
    ├── glass/
    ├── blackboard/
    ├── ben/
Model Performance
K-Fold Cross-Validation Results:

Train Accuracy: 96.43%
Validation Accuracy: 97.26%
Notes
Ensure that WAV files used for testing are preprocessed in the same way as the training data.
If the model weights (best_model.pth) are not provided, you'll need to train the model using the provided notebook.
Acknowledgements
This project was developed as part of the CIS 5190/4190 Fall 2024 coursework. Contributors include Alexis Powell, Zitong Ren, and Jiaming Li.
