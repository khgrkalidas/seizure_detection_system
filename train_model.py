import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from model_architecture import SeizureModel


# ---------------------------------------------------
# DEVICE
# ---------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)


# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

X = np.load(r"D:\project\X_balanced.npy")
y = np.load(r"D:\project\y_balanced.npy")

print("Dataset shape:", X.shape)


# ---------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42

)


# ---------------------------------------------------
# DATASET CLASS
# ---------------------------------------------------

class EEGDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


# ---------------------------------------------------
# DATALOADERS
# ---------------------------------------------------

train_loader = DataLoader(

    EEGDataset(X_train, y_train),
    batch_size=32,
    shuffle=True

)

test_loader = DataLoader(

    EEGDataset(X_test, y_test),
    batch_size=32

)


# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

model = SeizureModel().to(DEVICE)


# ---------------------------------------------------
# WEIGHTED LOSS
# ---------------------------------------------------

pos_weight = torch.tensor([3.0]).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ---------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------

EPOCHS = 25

best_f1 = 0

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")


    # ---------------------------------------------------
    # EVALUATION
    # ---------------------------------------------------

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(DEVICE)

            outputs = torch.sigmoid(model(X_batch))

            preds = (outputs > 0.5).float().cpu().numpy()

            all_preds.extend(preds)

            all_labels.extend(y_batch.numpy())


    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("Precision:", round(precision,4))
    print("Recall   :", round(recall,4))
    print("F1 Score :", round(f1,4))


    # ---------------------------------------------------
    # SAVE BEST MODEL
    # ---------------------------------------------------

    if f1 > best_f1:

        best_f1 = f1

        torch.save(
            model.state_dict(),
            r"D:\project\seizure_model.pth"
        )

        print("Best model saved")


# ---------------------------------------------------
# FINAL CONFUSION MATRIX
# ---------------------------------------------------

cm = confusion_matrix(all_labels, all_preds)

print("\nConfusion Matrix")

print(cm)


print("\nTraining Finished")