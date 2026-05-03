import numpy as np

X_path = r"D:\project\dataset_X.dat"

WINDOWS = 1712173
CHANNELS = 23
SAMPLES = 1024

CHUNK = 2000   # windows per chunk

print("Opening memmap...")

X = np.memmap(X_path, dtype=np.float32, mode="r")
X = X.reshape(WINDOWS, CHANNELS, SAMPLES)

total_sum = 0.0
total_sq = 0.0
total_count = 0

print("Streaming dataset...")

for i in range(0, WINDOWS, CHUNK):

    end = min(i + CHUNK, WINDOWS)

    batch = X[i:end]

    total_sum += np.sum(batch)
    total_sq += np.sum(batch ** 2)
    total_count += batch.size

    if i % 50000 == 0:
        print(f"Processed {i}/{WINDOWS}")

mean = total_sum / total_count

var = (total_sq / total_count) - (mean ** 2)

std = np.sqrt(var)

print("\nMean:", mean)
print("Std:", std)

np.save(r"D:\project\eeg_mean.npy", mean)
np.save(r"D:\project\eeg_std.npy", std)

print("\nSaved normalization stats")