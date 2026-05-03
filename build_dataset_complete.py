import os
import re
import numpy as np
import mne
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

DATASET_PATH = r"D:\project\data set edf"

X_PATH = r"D:\project\dataset_X.dat"
Y_PATH = r"D:\project\dataset_y.dat"

WINDOW_SIZE = 1024
STEP_SIZE = 512
SAMPLING_RATE = 256

THREAD_WORKERS = 4

TARGET_CHANNELS = [
"FP1-F7","F7-T7","T7-P7","P7-O1",
"FP1-F3","F3-C3","C3-P3","P3-O1",
"FP2-F4","F4-C4","C4-P4","P4-O2",
"FP2-F8","F8-T8","T8-P8","P8-O2",
"FZ-CZ","CZ-PZ",
"P7-T7","T7-FT9","FT9-FT10","FT10-T8","T8-P8"
]

write_lock = Lock()


def parse_summary(summary_file):

    seizure_dict = {}
    current_file = None

    with open(summary_file, "r") as f:
        lines = f.readlines()

    for line in lines:

        if "File Name:" in line:
            current_file = line.split(":")[1].strip()
            seizure_dict[current_file] = []

        if "Seizure Start Time" in line:
            start = int(re.findall(r"\d+", line)[0])

        if "Seizure End Time" in line:
            end = int(re.findall(r"\d+", line)[0])
            seizure_dict[current_file].append((start, end))

    return seizure_dict


def clean_name(name):

    name = name.upper()
    name = name.replace("-REF","")

    parts = name.split("-")

    if parts[-1].isdigit():
        name = "-".join(parts[:-1])

    return name


def select_channels(raw):

    cleaned = [clean_name(c) for c in raw.ch_names]

    idx = []

    for target in TARGET_CHANNELS:

        found = False

        for i,ch in enumerate(cleaned):

            if ch == target:
                idx.append(i)
                found = True
                break

        if not found:
            return None

    return raw.get_data(picks=idx)


def process_edf(args):

    edf_path, seizures, X_file, y_file = args

    try:

        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        raw.filter(0.5,40,verbose=False)

        signal = select_channels(raw)

        if signal is None:
            print("Skipping missing channels:", edf_path)
            return 0,0

        total_samples = signal.shape[1]

        seizures_samples = [
            (s*SAMPLING_RATE, e*SAMPLING_RATE)
            for s,e in seizures
        ]

        local_total = 0
        local_seizure = 0

        for start in range(0,total_samples-WINDOW_SIZE,STEP_SIZE):

            end = start + WINDOW_SIZE

            window = signal[:,start:end]

            if window.shape != (23,WINDOW_SIZE):
                continue

            label = 0

            for s,e in seizures_samples:
                if start < e and end > s:
                    label = 1
                    local_seizure += 1
                    break

            with write_lock:

                window.astype(np.float32).tofile(X_file)
                np.array([label],dtype=np.int8).tofile(y_file)

            local_total += 1

        print("Processed:", os.path.basename(edf_path),
              "| windows:", local_total)

        return local_total, local_seizure

    except Exception as e:

        print("Skipping error:", edf_path)
        return 0,0


def collect_tasks():

    tasks = []

    patients = sorted(os.listdir(DATASET_PATH))

    for patient in patients:

        patient_path = os.path.join(DATASET_PATH,patient)

        if not os.path.isdir(patient_path):
            continue

        summary = os.path.join(patient_path,patient+"-summary.txt")

        if not os.path.exists(summary):
            continue

        seizure_info = parse_summary(summary)

        edfs = [f for f in os.listdir(patient_path) if f.endswith(".edf")]

        for edf in edfs:

            path = os.path.join(patient_path,edf)

            seizures = seizure_info.get(edf,[])

            tasks.append((path,seizures))

    return tasks


def main():

    tasks = collect_tasks()

    print("Total EDF files:",len(tasks))
    print("Threads used:",THREAD_WORKERS)

    X_file = open(X_PATH,"wb")
    y_file = open(Y_PATH,"wb")

    total_windows = 0
    seizure_windows = 0

    executor_tasks = []

    with ThreadPoolExecutor(max_workers=THREAD_WORKERS) as executor:

        for edf_path,seizures in tasks:

            executor_tasks.append(
                executor.submit(
                    process_edf,
                    (edf_path,seizures,X_file,y_file)
                )
            )

        for future in as_completed(executor_tasks):

            total,sz = future.result()

            total_windows += total
            seizure_windows += sz

    X_file.close()
    y_file.close()

    print("\nDataset creation finished")
    print("Total windows:",total_windows)
    print("Seizure windows:",seizure_windows)
    print("Normal windows:",total_windows-seizure_windows)


if __name__ == "__main__":
    main()