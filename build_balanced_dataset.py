import numpy as np

X_PATH = r"D:\project\dataset_X.dat"
Y_PATH = r"D:\project\dataset_y.dat"

MEAN_PATH = r"D:\project\eeg_mean.npy"
STD_PATH = r"D:\project\eeg_std.npy"

OUTPUT_X = r"D:\project\X_balanced.npy"
OUTPUT_Y = r"D:\project\y_balanced.npy"

WINDOWS = 1712173
CHANNELS = 23
SAMPLES = 1024

print("Opening dataset (memmap)...")

X = np.memmap(X_PATH, dtype=np.float32, mode="r")
y = np.memmap(Y_PATH, dtype=np.int8, mode="r")

X = X.reshape(WINDOWS, CHANNELS, SAMPLES)

print("Loading normalization stats")

mean = np.load(MEAN_PATH)
std = np.load(STD_PATH)

print("Finding seizure and normal indices")

seizure_idx = np.where(y == 1)[0]
normal_idx = np.where(y == 0)[0]

print("Seizures:", len(seizure_idx))
print("Normals:", len(normal_idx))

np.random.seed(42)

normal_sample = np.random.choice(
    normal_idx,
    size=len(seizure_idx) * 3,
    replace=False
)

balanced_idx = np.concatenate([seizure_idx, normal_sample])

np.random.shuffle(balanced_idx)

print("Extracting balanced dataset")

X_bal = X[balanced_idx]
y_bal = y[balanced_idx]

print("Applying normalization")

X_bal = (X_bal - mean) / std

print("Saving dataset")

np.save(OUTPUT_X, X_bal)
np.save(OUTPUT_Y, y_bal)

print("\nDataset created")

print("Shape:", X_bal.shape)
print("Seizures:", np.sum(y_bal))
print("Normals:", len(y_bal) - np.sum(y_bal))