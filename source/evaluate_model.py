import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from model_architecture import SeizureModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = r"D:\project"

print("Loading dataset...")

X = np.load(DATA_PATH + r"\X_balanced.npy")
y = np.load(DATA_PATH + r"\y_balanced.npy")

print("Dataset shape:", X.shape)

# Same split used in training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)

print("Test shape:", X_test.shape)

print("Loading trained model...")

model = SeizureModel().to(DEVICE)

model.load_state_dict(
    torch.load(r"D:\project\seizure_model.pth", map_location=DEVICE)
)

model.eval()

print("Running inference...")

with torch.no_grad():

    outputs = torch.sigmoid(model(X_test)).cpu().numpy()

preds = (outputs > 0.5).astype(int)

# --------------------------
# Metrics
# --------------------------

acc = accuracy_score(y_test, preds)
prec = precision_score(y_test, preds)
rec = recall_score(y_test, preds)
f1 = f1_score(y_test, preds)

print("\nEvaluation Results")

print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1 Score :", f1)

# --------------------------
# Confusion Matrix
# --------------------------

cm = confusion_matrix(y_test, preds)

print("\nConfusion Matrix")
print(cm)

plt.figure(figsize=(5,5))

plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks([0,1],["Normal","Seizure"])
plt.yticks([0,1],["Normal","Seizure"])

for i in range(2):
    for j in range(2):
        plt.text(j,i,cm[i,j],ha="center",va="center")

plt.colorbar()

plt.show()

# --------------------------
# ROC Curve
# --------------------------

fpr, tpr, _ = roc_curve(y_test, outputs)

roc_auc = auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1],"--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()

print("\nAUC Score:", roc_auc)