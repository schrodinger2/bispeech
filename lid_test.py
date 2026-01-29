import joblib
from lid import extract_features
import numpy as np

clf = joblib.load("lid_model.joblib")

# ns = 2600 # number of samples
# rightcountar = 0
# rightcounten = 0
# for i in range(2201,ns):
#     feats = extract_features(f"data/ARtest/{i}.flac").reshape(1, -1)  
#     prediction = clf.predict(feats)[0] # this should return 0 if ar and 1 if en
#     if not prediction:
#         rightcountar += 1
#     feats = extract_features(f"data/ENtest/{i}.flac").reshape(1, -1)
#     prediction = clf.predict(feats)[0] # this should return 0 if ar and 1 if en
#     if prediction:
#         rightcounten += 1

# print("arabic acc = " ,rightcountar/4,"%" )
# print("english acc = " ,rightcounten/4,"%" )


X_test = []
y_test = []

for i in range(2201, 2600):
    X_test.append(extract_features(f"data/ARtest/{i}.flac"))
    y_test.append(0)

    X_test.append(extract_features(f"data/ENtest/{i}.flac"))
    y_test.append(1)

X_test = np.array(X_test)
y_test = np.array(y_test)

preds = clf.predict(X_test)
acc = (preds == y_test).mean()
print("acc =", acc)