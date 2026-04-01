import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
import joblib
import os
import time
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- SETTINGS & THEME ---
APP_BG = "#1a1a2e"
PANEL_BG = "#16213e"
TEXT_COLOR = "#ffffff"
GREEN = "#00b894"
RED = "#d63031"
YELLOW = "#fbc531"
ACCENT = "#0984e3"

def extract_features(audio, sr):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    return np.hstack([mfcc_mean, centroid, rolloff, zcr])

class MachineHealthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Health Monitor")
        self.root.geometry("1000x900")
        self.root.configure(bg=APP_BG)
        
        # State variables
        self.models_loaded = False
        self.binary_model = None
        self.multi_model = None
        self.label_encoder = None
        self.is_recording = False
        self.data_path = "data" # Default local data folder
        
        # Ensure directories exist
        for cat in ['normal', 'mechanical', 'electrical', 'wear']:
            os.makedirs(os.path.join(self.data_path, cat), exist_ok=True)
        
        self.setup_styles()
        self.create_widgets()
        self.load_existing_models()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background=APP_BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=PANEL_BG, foreground="white", padding=[20, 8], font=("Arial", 11, "bold"))
        style.map("TNotebook.Tab", background=[("selected", ACCENT)], foreground=[("selected", "white")])
        style.configure("TProgressbar", thickness=25, troughcolor=PANEL_BG, background=GREEN)

    def create_widgets(self):
        # Header
        header = tk.Frame(self.root, bg=APP_BG, pady=15)
        header.pack(fill="x")
        tk.Label(header, text="MACHINE HEALTH MONITOR PRO", font=("Arial", 28, "bold"), fg=TEXT_COLOR, bg=APP_BG).pack()
        
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=15, pady=10)

        # TAB 1: REAL-TIME INSPECTION
        self.inspect_tab = tk.Frame(self.notebook, bg=APP_BG)
        self.notebook.add(self.inspect_tab, text="  🔍  REAL-TIME INSPECTION  ")
        self.setup_inspection_ui()

        # TAB 2: DATA COLLECTION & TRAINING
        self.train_tab = tk.Frame(self.notebook, bg=APP_BG)
        self.notebook.add(self.train_tab, text="  ⚙️  COLLECT & TRAIN  ")
        self.setup_training_ui()

    def setup_inspection_ui(self):
        # Top Controls
        top_ctrl = tk.Frame(self.inspect_tab, bg=APP_BG, pady=10)
        top_ctrl.pack(fill="x")
        
        tk.Button(top_ctrl, text="📁 Load Audio File", font=("Arial", 11), bg=ACCENT, fg="white", width=18, command=self.load_file).pack(side="left", padx=10)
        self.live_btn = tk.Button(top_ctrl, text="🔴 Start Real-time Check", font=("Arial", 11, "bold"), bg=RED, fg="white", width=22, command=self.start_recording)
        self.live_btn.pack(side="left", padx=10)
        
        self.status_dot = tk.Canvas(top_ctrl, width=15, height=15, bg=APP_BG, highlightthickness=0)
        self.status_dot.pack(side="right", padx=20)
        self.update_dot("gray")

        # Live Waveform
        self.fig, self.ax = plt.subplots(figsize=(8, 2))
        self.fig.patch.set_facecolor("#0a0a1a")
        self.ax.set_facecolor("#0a0a1a")
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.inspect_tab)
        self.canvas.get_tk_widget().pack(fill="x", padx=15, pady=10)

        # Dashboard
        self.dash = tk.Frame(self.inspect_tab, bg=PANEL_BG, padx=30, pady=25, bd=1, relief="ridge")
        self.dash.pack(fill="x", padx=15, pady=5)
        
        self.res_title = tk.Label(self.dash, text="SYSTEM READY", font=("Arial", 24, "bold"), fg="white", bg=PANEL_BG)
        self.res_title.pack(anchor="w")
        
        self.res_sub = tk.Label(self.dash, text="Awaiting input for diagnostic analysis...", font=("Arial", 12), fg="#a2a2d0", bg=PANEL_BG)
        self.res_sub.pack(anchor="w", pady=5)
        
        self.health_canvas = tk.Canvas(self.dash, height=30, bg="#050510", highlightthickness=0)
        self.health_canvas.pack(fill="x", pady=15)

        # Logs
        tk.Label(self.inspect_tab, text="LOG HISTORY", font=("Arial", 9, "bold"), fg="#57606f", bg=APP_BG).pack(anchor="w", padx=20)
        self.log_box = tk.Text(self.inspect_tab, height=12, bg="#050510", fg="#ecf0f1", font=("Consolas", 10), state="disabled", borderwidth=0, padx=10, pady=10)
        self.log_box.pack(fill="both", expand=True, padx=15, pady=(5, 15))

    def setup_training_ui(self):
        container = tk.Frame(self.train_tab, bg=APP_BG, padx=20, pady=20)
        container.pack(fill="both", expand=True)

        # LEFT SIDE: COLLECTION
        coll_frame = tk.LabelFrame(container, text=" STEP 1: Record Training Data ", font=("Arial", 11, "bold"), bg=APP_BG, fg=ACCENT, padx=15, pady=15)
        coll_frame.pack(side="left", fill="both", expand=True, padx=10)

        tk.Label(coll_frame, text="Target Label:", bg=APP_BG, fg="white", font=("Arial", 10)).pack(anchor="w")
        self.cat_var = tk.StringVar(value="normal")
        cat_dropdown = ttk.Combobox(coll_frame, textvariable=self.cat_var, values=['normal', 'mechanical', 'electrical', 'wear'], state="readonly", font=("Arial", 12))
        cat_dropdown.pack(fill="x", pady=5)

        self.rec_train_btn = tk.Button(coll_frame, text="🎤 Record 3s for Training", font=("Arial", 12, "bold"), bg=ACCENT, fg="white", height=2, command=self.record_for_training)
        self.rec_train_btn.pack(fill="x", pady=15)

        self.coll_stats = tk.Label(coll_frame, text="Samples Collected: Checking...", bg=APP_BG, fg="#a2a2d0", justify="left")
        self.coll_stats.pack(fill="x")
        self.refresh_stats()

        # RIGHT SIDE: TRAINING
        train_ctrl = tk.LabelFrame(container, text=" STEP 2: Train Model ", font=("Arial", 11, "bold"), bg=APP_BG, fg=GREEN, padx=15, pady=15)
        train_ctrl.pack(side="left", fill="both", expand=True, padx=10)

        tk.Label(train_ctrl, text="This will rebuild the model using all\nrecorded data in the subfolders.", bg=APP_BG, fg="#bdc3c7", justify="center").pack(pady=10)
        
        self.train_now_btn = tk.Button(train_ctrl, text="🚀 Train Now", font=("Arial", 14, "bold"), bg=GREEN, fg="white", height=2, command=self.manual_train)
        self.train_now_btn.pack(fill="x", pady=20)
        
        self.train_prog = ttk.Progressbar(train_ctrl, orient="horizontal", mode="determinate")
        self.train_prog.pack(fill="x", pady=10)
        
        self.train_status = tk.Label(train_ctrl, text="Ready", bg=APP_BG, fg="white")
        self.train_status.pack()

    # --- RECORDING FOR TRAINING ---
    def record_for_training(self):
        category = self.cat_var.get()
        def task():
            self.rec_train_btn.config(state="disabled", text="Recording...")
            fs = 22050
            duration = 3
            try:
                recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
                sd.wait()
                audio = recording.flatten()
                
                # Save file
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rec_{ts}.wav"
                save_path = os.path.join(self.data_path, category, filename)
                sf.write(save_path, audio, fs)
                
                self.root.after(0, lambda: messagebox.showinfo("Success", f"Recorded and labeled as: {category.upper()}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed to record: {e}"))
            finally:
                self.root.after(0, lambda: self.rec_train_btn.config(state="normal", text="🎤 Record 3s for Training"))
                self.root.after(0, self.refresh_stats)

        threading.Thread(target=task, daemon=True).start()

    def refresh_stats(self):
        text = "Current Dataset:\n"
        for cat in ['normal', 'mechanical', 'electrical', 'wear']:
            count = len([f for f in os.listdir(os.path.join(self.data_path, cat)) if f.endswith('.wav')])
            text += f"• {cat.capitalize()}: {count} samples\n"
        self.coll_stats.config(text=text)

    # --- CORE LOGIC ---
    def update_dot(self, color):
        self.status_dot.delete("all")
        self.status_dot.create_oval(2, 2, 13, 13, fill=color, outline="")

    def load_existing_models(self):
        if os.path.exists('model_binary.pkl'):
            try:
                self.binary_model = joblib.load('model_binary.pkl')
                self.multi_model = joblib.load('model_multiclass.pkl')
                self.label_encoder = joblib.load('label_encoder.pkl')
                self.models_loaded = True
                self.update_dot(GREEN)
            except: pass

    def load_file(self):
        if not self.models_loaded:
            messagebox.showwarning("Warning", "Please train the model first.")
            return
        f = filedialog.askopenfilename(filetypes=[("Audio", "*.wav")])
        if f:
            audio, sr = librosa.load(f, sr=22050)
            self.process_audio(audio, sr, os.path.basename(f))

    def start_recording(self):
        if not self.models_loaded:
            messagebox.showwarning("Warning", "Please train the model first.")
            return
        if self.is_recording: return
        
        def run():
            self.is_recording = True
            self.update_dot(RED)
            fs = 22050
            duration = 3
            try:
                self.root.after(0, lambda: self.res_title.config(text="RECORDING LIVE..."))
                rec = sd.rec(int(duration * fs), samplerate=fs, channels=1)
                sd.wait()
                audio = rec.flatten()
                self.root.after(0, lambda: self.process_audio(audio, fs, "Live Mic"))
            finally:
                self.is_recording = False
                self.root.after(0, lambda: self.update_dot(GREEN))

        threading.Thread(target=run, daemon=True).start()

    def process_audio(self, audio, sr, source):
        # Viz
        self.ax.clear()
        self.ax.plot(audio[:10000], color=ACCENT, linewidth=0.7)
        self.ax.axis('off')
        self.canvas.draw()

        # Predict
        feat = extract_features(audio, sr).reshape(1, -1)
        pred_bin = self.binary_model.predict(feat)[0]
        prob = np.max(self.binary_model.predict_proba(feat)) * 100
        
        is_healthy = (pred_bin == 0)
        color = GREEN if is_healthy else RED
        label = "HEALTHY" if is_healthy else "FAULTY"
        
        # Diagnosis
        diag_idx = self.multi_model.predict(feat)[0]
        diag_name = self.label_encoder.inverse_transform([diag_idx])[0]
        
        # Update Dashboard
        self.dash.config(bg=color)
        self.res_title.config(text=f"STATUS: {label}", bg=color, fg="white")
        self.res_sub.config(text=f"Detection: {diag_name.upper()} | Confidence: {prob:.1f}%", bg=color, fg="white")
        
        # Health Bar
        self.health_canvas.delete("all")
        self.root.update_idletasks()
        score = prob if is_healthy else (100 - prob)
        w = self.health_canvas.winfo_width()
        self.health_canvas.create_rectangle(0, 0, (score/100)*w, 30, fill="white", outline="")

        # Log
        self.log_box.config(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("1.0", f"[{ts}] {source} -> {label} ({diag_name}) | Health: {int(score)}%\n")
        self.log_box.config(state="disabled")

    def manual_train(self):
        def proc():
            self.train_now_btn.config(state="disabled")
            self.train_status.config(text="Extracting features...")
            X, y = [], []
            cats = ['normal', 'mechanical', 'electrical', 'wear']
            
            try:
                for idx, c in enumerate(cats):
                    p = os.path.join(self.data_path, c)
                    files = [f for f in os.listdir(p) if f.endswith('.wav')]
                    for f in files:
                        audio, sr = librosa.load(os.path.join(p, f), sr=22050)
                        X.append(extract_features(audio, sr))
                        y.append(c)
                    self.train_prog['value'] = (idx + 1) * 20

                if not X:
                    messagebox.showerror("Error", "No data to train on!")
                    return

                self.train_status.config(text="Fitting models...")
                le = LabelEncoder()
                y_enc = le.fit_transform(y)
                y_bin = np.array([0 if i == 'normal' else 1 for i in y])

                m_bin = RandomForestClassifier(n_estimators=100).fit(X, y_bin)
                m_multi = RandomForestClassifier(n_estimators=100).fit(X, y_enc)
                
                joblib.dump(m_bin, 'model_binary.pkl')
                joblib.dump(m_multi, 'model_multiclass.pkl')
                joblib.dump(le, 'label_encoder.pkl')
                
                self.binary_model, self.multi_model, self.label_encoder = m_bin, m_multi, le
                self.models_loaded = True
                self.root.after(0, lambda: self.update_dot(GREEN))
                self.root.after(0, lambda: messagebox.showinfo("Done", "Training Complete! Models updated."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.train_now_btn.config(state="normal")
                self.train_status.config(text="Ready")
                self.train_prog['value'] = 0

        threading.Thread(target=proc, daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = MachineHealthApp(root)
    root.mainloop()