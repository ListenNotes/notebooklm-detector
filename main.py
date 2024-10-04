import os
import numpy as np
import librosa
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

# Define function to extract features from an audio file
def extract_features(file_path, n_mfcc=13):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        # Taking the mean of each feature across time
        mfcc_mean = np.mean(mfcc, axis=1)
        chroma_mean = np.mean(chroma, axis=1)
        spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

        return np.hstack([mfcc_mean, chroma_mean, spectral_contrast_mean])
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        return None


# Load dataset and extract features
def load_dataset(dataset_path):
    features = []
    labels = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav') or file.endswith('.mp3') or file.endswith('.mp4'):
                file_path = os.path.join(root, file)
                label = 'ai' if 'ai' in file_path.lower() else 'human'
                print(f'file: {file} / file_path: {file_path} => {label}')
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    labels.append(1 if label == 'ai' else 0)
    return np.array(features), np.array(labels)


# Train and save the model
def train_model(dataset_path, model_save_path='ai_speech_detector.pkl'):
    X, y = load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = SVC(kernel='rbf', probability=True)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model accuracy: {accuracy * 100:.2f}%')

    with open(model_save_path, 'wb') as f:
        pickle.dump((scaler, model), f)
    print(f'Model saved to {model_save_path}')


# Load the model and make predictions
def predict(file_path, model_path='ai_speech_detector.pkl'):
    with open(model_path, 'rb') as f:
        scaler, model = pickle.load(f)

    features = extract_features(file_path)
    if features is None:
        return "Could not extract features from the given audio file."

    features = scaler.transform([features])
    prediction = model.predict(features)
    return 'AI Generated' if prediction[0] == 1 else 'Human'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI Speech Detector')
    parser.add_argument('--action', type=str, choices=['train', 'predict'], required=True,
                        help='Action to perform: train or predict')
    parser.add_argument('--dataset_path', type=str, help='Path to the dataset for training')
    parser.add_argument('--file_path', type=str, help='Path to the audio file for prediction')
    parser.add_argument('--model_path', type=str, default='ai_speech_detector.pkl', help='Path to the model file')

    args = parser.parse_args()

    if args.action == 'train':
        if not args.dataset_path:
            print('Please provide the dataset path for training using --dataset_path')
        else:
            train_model(args.dataset_path, model_save_path=args.model_path)
    elif args.action == 'predict':
        if not args.file_path:
            print('Please provide the file path for prediction using --file_path')
        else:
            result = predict(args.file_path, model_path=args.model_path)
            print(f'The audio is: {result}')

    # Train the model on a dataset of AI vs human audio samples
    # dataset_path = './datasets'  # Replace with the actual path to the dataset
    # train_model(dataset_path)

    # Example prediction on a new audio file
    # file_path = './test/a.mp3'  # Replace with the actual path to the audio file
    # result = predict(file_path)
    # print(f'The audio is: {result}')