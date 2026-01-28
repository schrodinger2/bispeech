import librosa
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib


def extract_features(path, sr=16000, n_mfcc=13, duration=3.0):
    # duration in seconds
    audio, _ = librosa.load(path, sr=sr)
    
    # trim or pad to desired length
    n_samples = int(sr * duration)
    if len(audio) > n_samples:
        audio = audio[:n_samples]  # truncate
    else:
        audio = np.pad(audio, (0, max(0, n_samples - len(audio))), mode='constant')

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    features = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])
    return features

# def extract_features(path, sr=16000, n_mfcc=13):
    
#     audio, _ = librosa.load(path, sr=sr)

#     mfcc = librosa.feature.mfcc(
#         y=audio,
#         sr=sr,
#         n_mfcc=n_mfcc
#     )

#     features = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])

#     return features


X = []  # features
y = []  # labels
ns = 2200 # number of samples
for i in range(1,ns):
    if i > 100:
        feats = extract_features(f"data/AR/{i}.flac")
    else:
        feats = extract_features(f"data/AR/{i}.wav")
    X.append(feats)
    y.append(0)  
    feats = extract_features(f"data/EN/{i}.flac")
    X.append(feats)
    y.append(1)

X = np.array(X)
y = np.array(y)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=1000))
])

clf.fit(X, y)


joblib.dump(clf, "lid_model.joblib")


