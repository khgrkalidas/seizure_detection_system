import numpy as np
import torch
import mne
import os

from model_architecture import SeizureModel

DEVICE = "cpu"
WINDOW = 1024
STEP = 256

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MEAN = np.load(os.path.join(BASE_DIR, "eeg_mean.npy"))
STD  = np.load(os.path.join(BASE_DIR, "eeg_std.npy"))


# =============================
# CHANNEL HANDLING
# =============================
def clean(name):
    return name.upper().replace("-REF", "")


def extract_channels(raw, target_channels):
    cleaned = [clean(c) for c in raw.ch_names]
    total = raw.n_times

    signals = []

    for target in target_channels:
        idx = None
        for i, ch in enumerate(cleaned):
            if ch == target:
                idx = i
                break

        if idx is not None:
            sig = raw.get_data(picks=[idx])[0]
        else:
            sig = np.zeros(total)

        signals.append(sig)

    return np.vstack(signals)


# =============================
# ARTIFACT HANDLING
# =============================
def artifact_score(seg):
    amp = np.max(np.abs(seg))
    var = np.var(seg)

    score = 0
    if amp > 500:
        score += 1
    if var < 1e-2:
        score += 1

    return score


# =============================
# MODEL LOADING
# =============================
def load_model(path):
    model = SeizureModel()

    state = torch.load(path, map_location=DEVICE)

    fixed = {}
    for k, v in state.items():
        k = k.replace("attention.", "attn.")
        k = k.replace("module.", "")
        fixed[k] = v

    model.load_state_dict(fixed, strict=False)
    model.eval()

    return model


# =============================
# MERGE EVENTS
# =============================
def merge_events(events, gap=20):
    if not events:
        return []

    events = sorted(events)
    merged = [events[0]]

    for s, e in events[1:]:
        ls, le = merged[-1]

        if s - le <= gap:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))

    return merged


# =============================
# MAIN DETECTION
# =============================
def run_detection_stream(edf_path, model_path, target_channels):

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    raw.filter(0.5, 40, verbose=False)

    data = extract_channels(raw, target_channels)
    sfreq = int(raw.info["sfreq"])

    model = load_model(model_path)

    ptr = 0
    total = data.shape[1]

    prob_hist = []
    time_hist = []

    in_event = False
    onset = None

    events = []

    while ptr + WINDOW < total:

        raw_seg = data[:, ptr:ptr+WINDOW]

        seg = raw_seg * 1e6
        seg = (seg - MEAN[:, None]) / (STD[:, None] + 1e-6)

        x = torch.tensor(seg).float().unsqueeze(0)

        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()

        # artifact penalty
        prob *= (1 - 0.25 * artifact_score(seg))

        t = ptr / sfreq

        prob_hist.append(prob)
        time_hist.append(t)

        if len(prob_hist) > 200:
            prob_hist.pop(0)
            time_hist.pop(0)

        smooth = np.mean(prob_hist[-10:]) if len(prob_hist) >= 10 else prob

        event = None
        confidence = 0

        # =============================
        # HIGH SENSITIVITY START
        # =============================
        if not in_event and (smooth > 0.5 or prob > 0.8):

            in_event = True

            # TRACK BACK
            for i in range(len(prob_hist)-1, -1, -1):
                if prob_hist[i] < 0.3:
                    onset = time_hist[i]
                    break
            else:
                onset = t

        # =============================
        # END
        # =============================
        elif in_event and smooth < 0.3:

            in_event = False
            offset = t

            duration = offset - onset

            # =============================
            # ✅ MIN DURATION = 12 sec
            # =============================
            if duration >= 12:

                seg_probs = [
                    p for p, tt in zip(prob_hist, time_hist)
                    if onset <= tt <= offset
                ]

                confidence = float(np.mean(seg_probs))

                # =============================
                # ✅ CONFIDENCE FILTER (CRITICAL)
                # =============================
                if confidence >= 0.6:
                    events.append((onset, offset))
                    event = (onset, offset)

        yield {
            "signal": seg,
            "raw": raw_seg,
            "time": t,
            "prob": prob,
            "event": event,
            "confidence": confidence
        }

        ptr += STEP

    # =============================
    # FINAL CLEANUP
    # =============================
    merged = merge_events(events)

    final_events = [
        (s, e) for (s, e) in merged
        if (e - s) >= 12
    ]

    print("\n===== FINAL EVENTS =====")
    for s, e in final_events:
        print(f"{s:.2f}s → {e:.2f}s")