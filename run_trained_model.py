def run_trained_model(X):
    """
    Run the trained model on the given input WAV file paths.
    
    Args:
    X: array of shape (N,), where each element is a WAV file path.

    Returns:
    predictions: array of shape (N,), where each element is the predicted class ID.
    """
    # Define the CNN classifier model to match the training architecture
    class CNNClassifier(nn.Module):
        def __init__(self, num_classes=7):
            super(CNNClassifier, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)  # Convolution layer 1
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)  # Convolution layer 2
            self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
            self.fc1 = nn.Linear(64 * 32 * 32, num_classes)  # Fully connected layer

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))  # Apply first convolution, ReLU, and pooling
            x = self.pool(torch.relu(self.conv2(x)))  # Apply second convolution, ReLU, and pooling
            x = x.view(x.size(0), -1)  # Flatten feature maps
            x = self.fc1(x)  # Apply the fully connected layer
            return x

    # Step 1: Featurize WAV files into Mel spectrograms
    def featurize_wav_files(X, target_size=(128, 128)):
        """
        Featurize WAV files into consistent Mel spectrograms of the target size.
        Args:
        X: List of WAV file paths.
        target_size: Target size (n_mels, time_frames) for spectrograms.

        Returns:
        features: Numpy array of shape (N, 3, target_size[0], target_size[1]).
        """
        features = []
        for file_path in X:
            y, sr = librosa.load(file_path, sr=22050)  # Load WAV file
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=target_size[0])  # Generate Mel spectrogram
            log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to log scale

            # Resize time dimension to match target size
            if log_mel_spec.shape[1] > target_size[1]:
                log_mel_spec = log_mel_spec[:, :target_size[1]]  # Truncate
            else:
                pad_width = target_size[1] - log_mel_spec.shape[1]
                log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='constant')  # Pad

            # Expand to 3 channels to match the model's input requirements
            log_mel_spec = np.stack([log_mel_spec] * 3, axis=0)
            features.append(log_mel_spec)
        return np.array(features)

    # Featurize input WAV files
    features = featurize_wav_files(X)
    features = torch.tensor(features, dtype=torch.float32)  # Convert features to PyTorch tensor

    # Step 2: Download model weights
    def download_model_weights():
        """
        Download pre-trained model weights from Google Drive.
        Returns:
        Path to the downloaded weights file.
        """
        url = "https://drive.google.com/uc?id=1Z79uSqiK079hGhXPZKXYsXBqhhHUH18p"
        output = "model_weights.pth"
        gdown.download(url, output, fuzzy=True)  # Download the weights file
        return output

    # Load model weights
    weight_path = download_model_weights()

    # Step 3: Setup the classifier and load the pre-trained weights
    model = CNNClassifier(num_classes=7)
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))  # Load weights into model
    model.eval()  # Set the model to evaluation mode

    # Step 4: Perform inference
    with torch.no_grad():  # Disable gradient calculations for inference
        predictions = model(features).argmax(dim=1).numpy()  # Get the predicted class for each input

    return predictions
