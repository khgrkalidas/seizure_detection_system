import sys
import numpy as np
import mne
import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog,
    QVBoxLayout, QWidget, QHBoxLayout, QTableWidget,
    QTableWidgetItem
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from detection_engine import run_detection_stream


CHANNELS = [
    "FP1-F7","F7-T7","T7-P7","P7-O1",
    "FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2",
    "FP2-F8","F8-T8","T8-P8","P8-O2",
    "FZ-CZ","CZ-PZ",
    "P7-T7","T7-FT9","FT9-FT10","FT10-T8","T8-P8"
]


class Dashboard(QMainWindow):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("EEG Seizure Detection Dashboard")
        self.setGeometry(100, 100, 1200, 800)

        self.edf_path = None
        self.model_path = "seizure_model.pth"

        self.running = False
        self.paused = False

        self.stream = None
        self.events = []
        self.last_event = None

        # ✅ NEW (minimal addition)
        self.current_onset = None

        self.sfreq = 256

        self.buffer = None
        self.time_buffer = None

        self.max_points = 256 * 120

        self.init_ui()

    # ================= UI =================
    def init_ui(self):

        main_layout = QHBoxLayout()
        left_layout = QVBoxLayout()

        self.load_btn = QPushButton("Load EDF")
        self.start_btn = QPushButton("Start Detection")

        self.load_btn.clicked.connect(self.load_edf)
        self.start_btn.clicked.connect(self.start_detection)

        left_layout.addWidget(self.load_btn)
        left_layout.addWidget(self.start_btn)

        left_layout.addStretch()

        self.pause_btn = QPushButton("Pause")
        self.stop_btn = QPushButton("Stop")
        self.export_btn = QPushButton("Export CSV")
        self.export_img_btn = QPushButton("Export Seizure Images")

        self.pause_btn.clicked.connect(self.pause)
        self.stop_btn.clicked.connect(self.stop)
        self.export_btn.clicked.connect(self.export_csv)
        self.export_img_btn.clicked.connect(self.export_images)

        left_layout.addWidget(self.pause_btn)
        left_layout.addWidget(self.stop_btn)
        left_layout.addWidget(self.export_btn)
        left_layout.addWidget(self.export_img_btn)

        right_layout = QVBoxLayout()

        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(self.canvas)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["No", "Onset", "Offset"])
        right_layout.addWidget(self.table)

        main_layout.addLayout(left_layout, 1)
        main_layout.addLayout(right_layout, 5)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # ================= LOAD EDF =================
    def load_edf(self):
        file, _ = QFileDialog.getOpenFileName(self, "Open EDF", "", "EDF Files (*.edf)")
        if file:
            self.edf_path = file
            self.load_btn.setText(f"Loaded: {file.split('/')[-1]}")

            raw = mne.io.read_raw_edf(file, preload=False, verbose=False)
            self.sfreq = int(raw.info["sfreq"])

    # ================= START =================
    def start_detection(self):

        if not self.edf_path:
            return

        self.running = True
        self.paused = False

        self.buffer = None
        self.time_buffer = None
        self.events = []
        self.last_event = None
        self.current_onset = None

        self.stream = run_detection_stream(
            self.edf_path,
            self.model_path,
            CHANNELS
        )

        self.run_loop()

    # ================= LOOP =================
    def run_loop(self):

        while self.running:

            QApplication.processEvents()

            if self.paused:
                continue

            loop_start = time.time()

            try:
                data = next(self.stream)
            except StopIteration:
                print(">>> FILE FINISHED <<<")
                self.running = False
                break

            # ✅ FIX: RAW signal
            signal = data["raw"]

            # ✅ FIX: correct time axis
            t = data["time"]
            window_len = signal.shape[1] / self.sfreq
            t_arr = np.linspace(t - window_len, t, signal.shape[1])

            if self.buffer is None:
                self.buffer = signal
                self.time_buffer = t_arr
            else:
                self.buffer = np.concatenate([self.buffer, signal], axis=1)
                self.time_buffer = np.concatenate([self.time_buffer, t_arr])

                if self.buffer.shape[1] > self.max_points:
                    self.buffer = self.buffer[:, -self.max_points:]
                    self.time_buffer = self.time_buffer[-self.max_points:]

            self.update_plot(self.buffer, self.time_buffer)

            # ================= EVENT TRACKING =================
            if data["event"] is not None:
                if self.last_event != data["event"]:
                    self.last_event = data["event"]

                    onset, offset = data["event"]
                    self.events.append((onset, offset))
                    self.add_table(onset, offset)

                    self.current_onset = None

            # ongoing event tracking
            if data["event"] is not None:
                if self.current_onset is None:
                    self.current_onset = data["event"][0]

            elapsed = time.time() - loop_start
            time.sleep(max(0.12 - elapsed, 0))

    # ================= PLOT =================
    def update_plot(self, signal, time_axis):

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        n, T = signal.shape

        window_sec = 10
        t_end = time_axis[-1]
        t_start = max(time_axis[0], t_end - window_sec)

        mask = time_axis >= t_start
        signal = signal[:, mask]
        time_axis = time_axis[mask]

        norm_signal = np.zeros_like(signal)

        for i in range(n):
            ch = signal[i]
            ch = ch - np.mean(ch)
            ch = ch / (np.std(ch) * 2 + 1e-6)
            norm_signal[i] = ch

        spacing = 3

        for i in range(n):
            y = norm_signal[i] + i * spacing
            ax.plot(time_axis, y, color='blue', linewidth=0.5)

        ax.set_yticks(np.arange(n) * spacing)
        ax.set_yticklabels(CHANNELS[:n])

        ax.set_xlabel("Time (seconds)")
        ax.set_ylabel("Channels")

        ax.set_ylim(-spacing, n * spacing)
        ax.set_xlim(t_start, t_end)

        self.figure.subplots_adjust(
            left=0.06,
            right=0.99,
            top=0.97,
            bottom=0.08
        )

        ax.grid(False)
        self.canvas.draw()

    # ================= TABLE =================
    def add_table(self, onset, offset):

        row = self.table.rowCount()
        self.table.insertRow(row)

        self.table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
        self.table.setItem(row, 1, QTableWidgetItem(f"{onset:.2f}"))
        self.table.setItem(row, 2, QTableWidgetItem(f"{offset:.2f}"))

    # ================= EXPORT CSV =================
    def export_csv(self):

        if self.time_buffer is None:
            return

        export_events = self.events.copy()

        if self.current_onset is not None:
            current_time = self.time_buffer[-1]
            export_events.append((self.current_onset, current_time))

        with open("seizures.csv", "w") as f:
            f.write("No,Onset,Offset\n")
            for i, (on, off) in enumerate(export_events):
                f.write(f"{i+1},{on:.2f},{off:.2f}\n")

    # ================= EXPORT IMAGES =================
    def export_images(self):

        if self.time_buffer is None:
            return

        raw = mne.io.read_raw_edf(self.edf_path, preload=True, verbose=False)
        sfreq = int(raw.info["sfreq"])

        export_events = self.events.copy()

        if self.current_onset is not None:
            current_time = self.time_buffer[-1]
            export_events.append((self.current_onset, current_time))

        for i, (onset, offset) in enumerate(export_events):

            start = int(onset * sfreq)
            end = int(offset * sfreq)

            data = raw.get_data(start=start, stop=end) * 1e6
            t = np.linspace(onset, offset, data.shape[1])

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(12, 6))

            for ch in range(data.shape[0]):
                sig = data[ch]
                sig = sig - np.mean(sig)
                sig = sig / (np.std(sig) * 2 + 1e-6)
                data[ch] = sig

            spacing = 3

            for ch in range(min(len(CHANNELS), data.shape[0])):
                y = data[ch] + ch * spacing
                ax.plot(t, y, color='blue', linewidth=0.5)

            ax.axvspan(onset, offset, color='red', alpha=0.2)

            ax.set_title(f"Seizure {i+1} | {onset:.2f}s - {offset:.2f}s")

            ax.set_yticks(np.arange(len(CHANNELS)) * spacing)
            ax.set_yticklabels(CHANNELS)

            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Channels")

            plt.tight_layout()
            plt.savefig(f"seizure_{i+1}.png")
            plt.close()

    # ================= CONTROLS =================
    def pause(self):
        self.paused = not self.paused

    def stop(self):
        self.running = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Dashboard()
    win.show()
    sys.exit(app.exec_())