import tkinter as tk
import winsound
from tkinter import ttk, filedialog, messagebox
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
import joblib
import os
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# --- THEME ---
BG      = "#f0f2f5"
CARD    = "#ffffff"
TEXT    = "#1a1a1a"
MUTED   = "#6b7280"
BLUE    = "#2563eb"
GREEN   = "#16a34a"
RED     = "#dc2626"
BORDER  = "#e5e7eb"


def extract_features(audio, sr):
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak          # normalize before any feature extraction
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    rolloff  = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    zcr      = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    return np.hstack([np.mean(mfcc, axis=1), centroid, rolloff, zcr])


class MachineHealthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Health Monitor")
        self.root.geometry("860x720")
        self.root.configure(bg=BG)

        self.models_loaded = False
        self.binary_model = self.multi_model = self.label_encoder = None
        self.is_recording = False
        self.is_realtime  = False
        self._last_audio  = None
        self._last_sr     = None
        self.data_path = "data"

        for cat in ['normal', 'mechanical', 'electrical', 'wear']:
            os.makedirs(os.path.join(self.data_path, cat), exist_ok=True)

        self._setup_styles()
        self._build_ui()
        self._load_models()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.is_realtime = False
        sd.stop()
        plt.close("all")
        self.root.destroy()
        os._exit(0)

    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure("TNotebook",      background=BG,   borderwidth=0)
        s.configure("TNotebook.Tab",  background=CARD, foreground=MUTED,
                    padding=[18, 7],  font=("Segoe UI", 10, "bold"))
        s.map("TNotebook.Tab",
              background=[("selected", BLUE)],
              foreground=[("selected", "white")])
        s.configure("TProgressbar", thickness=8, troughcolor=BORDER, background=BLUE)
        s.configure("TCombobox",    fieldbackground=CARD, background=CARD, foreground=TEXT)

    # ── helpers ──────────────────────────────────────────────────────────────

    def _card(self, parent, **kw):
        return tk.Frame(parent, bg=CARD, bd=1, relief="flat",
                        highlightbackground=BORDER, highlightthickness=1, **kw)

    def _btn(self, parent, text, color, cmd, **kw):
        return tk.Button(parent, text=text, bg=color, fg="white",
                         font=("Segoe UI", 10, "bold"), relief="flat",
                         activebackground=color, cursor="hand2",
                         padx=14, pady=6, command=cmd, **kw)

    def _label(self, parent, text, size=10, bold=False, color=TEXT, **kw):
        weight = "bold" if bold else "normal"
        return tk.Label(parent, text=text, bg=parent["bg"], fg=color,
                        font=("Segoe UI", size, weight), **kw)

    # ── main layout ──────────────────────────────────────────────────────────

    def _build_ui(self):
        # title bar
        bar = tk.Frame(self.root, bg=CARD, pady=12,
                       highlightbackground=BORDER, highlightthickness=1)
        bar.pack(fill="x")
        self._label(bar, "Machine Health Monitor", size=14, bold=True).pack(side="left", padx=20)
        self.dot = tk.Canvas(bar, width=12, height=12, bg=CARD, highlightthickness=0)
        self.dot.pack(side="right", padx=20)
        self._set_dot("gray")

        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=14, pady=12)

        self.inspect_tab = tk.Frame(nb, bg=BG)
        nb.add(self.inspect_tab, text="  Inspect  ")

        self.train_tab = tk.Frame(nb, bg=BG)
        nb.add(self.train_tab, text="  Train  ")

        self._build_inspect()
        self._build_train()

    # ── inspect tab ──────────────────────────────────────────────────────────

    def _build_inspect(self):
        # controls row
        ctrl = tk.Frame(self.inspect_tab, bg=BG, pady=8)
        ctrl.pack(fill="x", padx=14)
        self._btn(ctrl, "Load Audio File", BLUE, self.load_file).pack(side="left", padx=(0, 8))
        self.live_btn = self._btn(ctrl, "Start Live Check", RED, self.start_recording)
        self.live_btn.pack(side="left", padx=(0, 8))
        self.rt_btn = self._btn(ctrl, "Real-time Monitor", "#7c3aed", self._toggle_realtime)
        self.rt_btn.pack(side="left", padx=(0, 8))
        self.replay_btn = self._btn(ctrl, "Replay", MUTED, self._replay)
        self.replay_btn.config(state="disabled")
        self.replay_btn.pack(side="left")

        # waveform card
        wf_card = self._card(self.inspect_tab, pady=6)
        wf_card.pack(fill="x", padx=14, pady=(4, 8))
        self.fig, (self.ax_wave, self.ax_spec) = plt.subplots(
            2, 1, figsize=(7, 3.6), facecolor=CARD,
            constrained_layout=True,
            gridspec_kw={"hspace": 0.55}
        )
        for ax in (self.ax_wave, self.ax_spec):
            ax.set_facecolor("#f8fafc")
            for spine in ax.spines.values():
                spine.set_color(BORDER)
            ax.tick_params(colors=MUTED, labelsize=7.5)
        self._draw_empty_plots()
        self.wf_canvas = FigureCanvasTkAgg(self.fig, master=wf_card)
        self.wf_canvas.get_tk_widget().pack(fill="x")

        # result card
        self.result_card = self._card(self.inspect_tab, padx=20, pady=16)
        self.result_card.pack(fill="x", padx=14, pady=(0, 8))

        self.res_status = self._label(self.result_card, "—", size=20, bold=True)
        self.res_status.pack(anchor="w")
        self.res_detail = self._label(self.result_card, "Load a file or start a live check.", color=MUTED)
        self.res_detail.pack(anchor="w", pady=(4, 10))

        self.health_bar = tk.Canvas(self.result_card, height=6, bg=BORDER, highlightthickness=0)
        self.health_bar.pack(fill="x")

        # log
        log_frame = tk.Frame(self.inspect_tab, bg=BG)
        log_frame.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self._label(log_frame, "Log", size=9, color=MUTED).pack(anchor="w", pady=(0, 4))
        self.log_box = tk.Text(log_frame, height=9, bg=CARD, fg=TEXT,
                               font=("Consolas", 9), state="disabled",
                               relief="flat", padx=10, pady=8,
                               highlightbackground=BORDER, highlightthickness=1)
        self.log_box.pack(fill="both", expand=True)

    # ── train tab ────────────────────────────────────────────────────────────

    def _build_train(self):
        outer = tk.Frame(self.train_tab, bg=BG, padx=14, pady=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        # ── collect card ──
        cc = self._card(outer, padx=16, pady=16)
        cc.grid(row=0, column=0, sticky="nsew", padx=(0, 7))

        self._label(cc, "1  Collect", size=11, bold=True).pack(anchor="w", pady=(0, 12))
        self._label(cc, "Label", color=MUTED, size=9).pack(anchor="w")
        self.cat_var = tk.StringVar(value="normal")
        ttk.Combobox(cc, textvariable=self.cat_var,
                     values=['normal', 'mechanical', 'electrical', 'wear'],
                     state="readonly", font=("Segoe UI", 10)).pack(fill="x", pady=(2, 12))

        self.rec_btn = self._btn(cc, "Record 3s", BLUE, self.record_for_training)
        self.rec_btn.pack(fill="x", pady=(0, 14))

        self.coll_stats = self._label(cc, "", color=MUTED, size=9, justify="left")
        self.coll_stats.pack(anchor="w")
        self._refresh_stats()

        # ── train card ──
        tc = self._card(outer, padx=16, pady=16)
        tc.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        self._label(tc, "2  Train", size=11, bold=True).pack(anchor="w", pady=(0, 12))
        self._label(tc, "Trains a new model from all recorded data.",
                    color=MUTED, size=9, wraplength=260, justify="left").pack(anchor="w", pady=(0, 14))

        self.train_btn = self._btn(tc, "Train Model", GREEN, self.manual_train)
        self.train_btn.pack(fill="x", pady=(0, 12))

        self.train_prog = ttk.Progressbar(tc, orient="horizontal", mode="determinate")
        self.train_prog.pack(fill="x", pady=(0, 8))

        self.train_status = self._label(tc, "Ready", color=MUTED, size=9)
        self.train_status.pack(anchor="w")

    # ── recording / training logic ────────────────────────────────────────────

    def record_for_training(self):
        category = self.cat_var.get()
        def task():
            self.rec_btn.config(state="disabled", text="Recording…")
            fs, dur = 22050, 3
            try:
                rec = sd.rec(int(dur * fs), samplerate=fs, channels=1)
                sd.wait()
                audio = rec.flatten()
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                sf.write(os.path.join(self.data_path, category, f"rec_{ts}.wav"), audio, fs)
                self.root.after(0, lambda: messagebox.showinfo(
                    "Saved", f"Saved as: {category}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, lambda: self.rec_btn.config(state="normal", text="Record 3s"))
                self.root.after(0, self._refresh_stats)
        threading.Thread(target=task, daemon=True).start()

    def _refresh_stats(self):
        lines = []
        for cat in ['normal', 'mechanical', 'electrical', 'wear']:
            n = len([f for f in os.listdir(os.path.join(self.data_path, cat)) if f.endswith('.wav')])
            lines.append(f"{cat.capitalize()}: {n}")
        self.coll_stats.config(text="\n".join(lines))

    def _draw_empty_plots(self):
        for ax, title in ((self.ax_wave, "Waveform"), (self.ax_spec, "Frequency Spectrum")):
            ax.clear()
            ax.set_facecolor("#f8fafc")
            ax.set_title(title, fontsize=8, color=MUTED, loc="left", pad=4)
            ax.set_xlabel("" , fontsize=7.5, color=MUTED)
            ax.tick_params(colors=MUTED, labelsize=7.5)
            for spine in ax.spines.values():
                spine.set_color(BORDER)
        self.ax_wave.set_ylabel("Amplitude", fontsize=7.5, color=MUTED)
        self.ax_spec.set_ylabel("Power (dB)", fontsize=7.5, color=MUTED)
        self.fig.canvas.draw_idle()

    def _plot_audio(self, audio, sr, accent):
        # ── waveform ──────────────────────────────────────────────────────
        self.ax_wave.clear()
        self.ax_wave.set_facecolor("#f8fafc")
        t = np.linspace(0, len(audio) / sr, num=len(audio))
        # downsample for drawing speed
        step = max(1, len(audio) // 4000)
        self.ax_wave.plot(t[::step], audio[::step], color=accent, linewidth=0.7)
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-9)
        peak   = np.max(np.abs(audio))
        self.ax_wave.set_title(
            f"Waveform   RMS {rms_db:.1f} dB   Peak {peak:.3f}",
            fontsize=8, color=MUTED, loc="left", pad=4
        )
        self.ax_wave.set_xlabel("Time (s)", fontsize=7.5, color=MUTED)
        self.ax_wave.set_ylabel("Amplitude", fontsize=7.5, color=MUTED)
        self.ax_wave.set_xlim(0, len(audio) / sr)
        self.ax_wave.axhline(0, color=BORDER, linewidth=0.5)
        self.ax_wave.grid(True, linestyle="--", linewidth=0.4, color=BORDER, alpha=0.8)
        self.ax_wave.tick_params(colors=MUTED, labelsize=7.5)
        for sp in self.ax_wave.spines.values():
            sp.set_color(BORDER)

        # ── FFT power spectrum ────────────────────────────────────────────
        self.ax_spec.clear()
        self.ax_spec.set_facecolor("#f8fafc")
        n   = len(audio)
        fft = np.abs(np.fft.rfft(audio * np.hanning(n)))
        fft_db  = 20 * np.log10(fft / (n / 2) + 1e-9)
        freqs   = np.fft.rfftfreq(n, d=1 / sr)
        # limit to 0-8 kHz (most machine fault content lives here)
        mask    = freqs <= 8000
        dom_hz  = freqs[mask][np.argmax(fft_db[mask])]
        self.ax_spec.plot(freqs[mask] / 1000, fft_db[mask], color=accent, linewidth=0.7)
        self.ax_spec.axvline(dom_hz / 1000, color=RED, linewidth=0.8, linestyle="--", alpha=0.7)
        self.ax_spec.text(dom_hz / 1000 + 0.05, self.ax_spec.get_ylim()[1] * 0.95 if self.ax_spec.get_ylim()[1] else -10,
                          f"{dom_hz:.0f} Hz", color=RED, fontsize=7)
        self.ax_spec.set_title(
            f"Frequency Spectrum   Dominant {dom_hz:.0f} Hz",
            fontsize=8, color=MUTED, loc="left", pad=4
        )
        self.ax_spec.set_xlabel("Frequency (kHz)", fontsize=7.5, color=MUTED)
        self.ax_spec.set_ylabel("Power (dB)",       fontsize=7.5, color=MUTED)
        self.ax_spec.set_xlim(0, 8)
        self.ax_spec.grid(True, linestyle="--", linewidth=0.4, color=BORDER, alpha=0.8)
        self.ax_spec.tick_params(colors=MUTED, labelsize=7.5)
        for sp in self.ax_spec.spines.values():
            sp.set_color(BORDER)

        self.wf_canvas.draw()

    def _set_dot(self, color):
        self.dot.delete("all")
        self.dot.create_oval(1, 1, 11, 11, fill=color, outline="")

    def _load_models(self):
        if os.path.exists('model_binary.pkl'):
            try:
                self.binary_model  = joblib.load('model_binary.pkl')
                self.multi_model   = joblib.load('model_multiclass.pkl')
                self.label_encoder = joblib.load('label_encoder.pkl')
                self.models_loaded = True
                self._set_dot(GREEN)
            except Exception:
                pass

    def load_file(self):
        if not self.models_loaded:
            messagebox.showwarning("No Model", "Train the model first.")
            return
        path = filedialog.askopenfilename(filetypes=[("Audio", "*.wav")])
        if path:
            audio, sr = librosa.load(path, sr=22050)
            self._analyse(audio, sr, os.path.basename(path))

    def start_recording(self):
        if not self.models_loaded:
            messagebox.showwarning("No Model", "Train the model first.")
            return
        if self.is_recording:
            return
        def run():
            self.is_recording = True
            self._set_dot(RED)
            self.root.after(0, lambda: self.res_status.config(text="Recording…", fg=MUTED))
            fs, dur = 22050, 3
            try:
                rec = sd.rec(int(dur * fs), samplerate=fs, channels=1)
                sd.wait()
                self.root.after(0, lambda: self._analyse(rec.flatten(), fs, "Live Mic"))
            finally:
                self.is_recording = False
                self.root.after(0, lambda: self._set_dot(GREEN))
        threading.Thread(target=run, daemon=True).start()

    def _toggle_realtime(self):
        if not self.models_loaded:
            messagebox.showwarning("No Model", "Train the model first.")
            return
        if self.is_realtime:
            self.is_realtime = False
            self.rt_btn.config(text="Real-time Monitor", bg="#7c3aed")
        else:
            self.is_realtime = True
            self.rt_btn.config(text="Stop Monitoring", bg=RED)
            threading.Thread(target=self._realtime_loop, daemon=True).start()

    def _realtime_loop(self):
        fs, chunk = 22050, 22050  # 1-second chunks
        while self.is_realtime:
            rec = sd.rec(chunk, samplerate=fs, channels=1)
            sd.wait()
            if not self.is_realtime:
                break
            audio = rec.flatten()
            self.root.after(0, lambda a=audio: self._realtime_update(a, fs))

    def _realtime_update(self, audio, sr):
        # ── plots ────────────────────────────────────────────────────────
        self.ax_wave.clear()
        self.ax_wave.set_facecolor("#f8fafc")
        t = np.linspace(0, len(audio) / sr, num=len(audio))
        self.ax_wave.plot(t, audio, color="#7c3aed", linewidth=0.7)
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-9)
        peak   = np.max(np.abs(audio))
        self.ax_wave.set_title(f"Waveform   RMS {rms_db:.1f} dB   Peak {peak:.3f}",
                               fontsize=8, color=MUTED, loc="left", pad=4)
        self.ax_wave.set_xlabel("Time (s)", fontsize=7.5, color=MUTED)
        self.ax_wave.set_ylabel("Amplitude", fontsize=7.5, color=MUTED)
        self.ax_wave.set_xlim(0, len(audio) / sr)
        self.ax_wave.axhline(0, color=BORDER, linewidth=0.5)
        self.ax_wave.grid(True, linestyle="--", linewidth=0.4, color=BORDER, alpha=0.8)
        self.ax_wave.tick_params(colors=MUTED, labelsize=7.5)
        for sp in self.ax_wave.spines.values():
            sp.set_color(BORDER)

        self.ax_spec.clear()
        self.ax_spec.set_facecolor("#f8fafc")
        n      = len(audio)
        fft    = np.abs(np.fft.rfft(audio * np.hanning(n)))
        fft_db = 20 * np.log10(fft / (n / 2) + 1e-9)
        freqs  = np.fft.rfftfreq(n, d=1 / sr)
        mask   = freqs <= 8000
        dom_hz = freqs[mask][np.argmax(fft_db[mask])]
        self.ax_spec.plot(freqs[mask] / 1000, fft_db[mask], color="#7c3aed", linewidth=0.7)
        self.ax_spec.axvline(dom_hz / 1000, color=RED, linewidth=0.8, linestyle="--", alpha=0.7)
        self.ax_spec.set_title(f"Frequency Spectrum   Dominant {dom_hz:.0f} Hz",
                               fontsize=8, color=MUTED, loc="left", pad=4)
        self.ax_spec.set_xlabel("Frequency (kHz)", fontsize=7.5, color=MUTED)
        self.ax_spec.set_ylabel("Power (dB)",       fontsize=7.5, color=MUTED)
        self.ax_spec.set_xlim(0, 8)
        self.ax_spec.grid(True, linestyle="--", linewidth=0.4, color=BORDER, alpha=0.8)
        self.ax_spec.tick_params(colors=MUTED, labelsize=7.5)
        for sp in self.ax_spec.spines.values():
            sp.set_color(BORDER)
        self.wf_canvas.draw()

        # ── prediction ───────────────────────────────────────────────────
        feat    = extract_features(audio, sr).reshape(1, -1)
        pred    = self.binary_model.predict(feat)[0]
        prob    = np.max(self.binary_model.predict_proba(feat)) * 100
        diag    = self.label_encoder.inverse_transform([self.multi_model.predict(feat)[0]])[0]
        healthy = pred == 0
        color   = GREEN if healthy else RED
        label   = "Healthy" if healthy else "Fault Detected"
        score   = prob if healthy else 100 - prob

        self.res_status.config(text=label, fg=color)
        self.res_detail.config(text=f"{diag.capitalize()}  ·  {prob:.0f}% confidence")

        self.health_bar.delete("all")
        self.root.update_idletasks()
        w = self.health_bar.winfo_width()
        self.health_bar.create_rectangle(0, 0, (score / 100) * w, 6, fill=color, outline="")

        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("1.0", f"[{ts}]  Live  {label:<18} {diag}  {int(score)}%\n")
        self.log_box.config(state="disabled")

    def _replay(self):
        if self._last_audio is None:
            return
        def play():
            self.root.after(0, lambda: self.replay_btn.config(state="disabled", text="Playing…"))
            tmp_path = os.path.join(self.data_path, "_replay_tmp.wav")
            try:
                audio = self._last_audio
                peak = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak * 0.95
                sf.write(tmp_path, audio, self._last_sr)
                winsound.PlaySound(tmp_path, winsound.SND_FILENAME)
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Playback Error", str(e)))
            finally:
                self.root.after(0, lambda: self.replay_btn.config(state="normal", text="Replay"))
        threading.Thread(target=play, daemon=True).start()

    def _analyse(self, audio, sr, source):
        self._last_audio = audio
        self._last_sr    = sr
        self.replay_btn.config(state="normal")
        # prediction
        feat    = extract_features(audio, sr).reshape(1, -1)
        pred    = self.binary_model.predict(feat)[0]
        prob    = np.max(self.binary_model.predict_proba(feat)) * 100
        diag    = self.label_encoder.inverse_transform([self.multi_model.predict(feat)[0]])[0]
        healthy = pred == 0
        color   = GREEN if healthy else RED
        label   = "Healthy" if healthy else "Fault Detected"
        score   = prob if healthy else 100 - prob

        self._plot_audio(audio, sr, color)

        self.res_status.config(text=label, fg=color)
        self.res_detail.config(text=f"{diag.capitalize()}  ·  {prob:.0f}% confidence")

        # health bar
        self.health_bar.delete("all")
        self.root.update_idletasks()
        w = self.health_bar.winfo_width()
        self.health_bar.create_rectangle(0, 0, (score / 100) * w, 6, fill=color, outline="")

        # log
        self.log_box.config(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("1.0", f"[{ts}]  {source:<20} {label:<18} {diag}  {int(score)}%\n")
        self.log_box.config(state="disabled")

    def manual_train(self):
        def proc():
            self.train_btn.config(state="disabled")
            cats = ['normal', 'mechanical', 'electrical', 'wear']
            X, y = [], []
            try:
                for idx, c in enumerate(cats):
                    self.root.after(0, lambda c=c: self.train_status.config(text=f"Loading {c}…"))
                    for f in [f for f in os.listdir(os.path.join(self.data_path, c)) if f.endswith('.wav')]:
                        audio, sr = librosa.load(os.path.join(self.data_path, c, f), sr=22050)
                        X.append(extract_features(audio, sr))
                        y.append(c)
                    self.root.after(0, lambda v=(idx+1)*20: self.train_prog.config(value=v))

                if not X:
                    messagebox.showerror("Error", "No audio files found in data folders.")
                    return

                self.root.after(0, lambda: self.train_status.config(text="Fitting…"))
                le   = LabelEncoder()
                yenc = le.fit_transform(y)
                ybin = np.array([0 if i == 'normal' else 1 for i in y])

                m_bin   = RandomForestClassifier(n_estimators=100).fit(X, ybin)
                m_multi = RandomForestClassifier(n_estimators=100).fit(X, yenc)

                joblib.dump(m_bin,   'model_binary.pkl')
                joblib.dump(m_multi, 'model_multiclass.pkl')
                joblib.dump(le,      'label_encoder.pkl')

                self.binary_model, self.multi_model, self.label_encoder = m_bin, m_multi, le
                self.models_loaded = True
                self.root.after(0, lambda: self._set_dot(GREEN))
                self.root.after(0, lambda: messagebox.showinfo("Done", "Training complete."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, lambda: self.train_btn.config(state="normal"))
                self.root.after(0, lambda: self.train_status.config(text="Ready"))
                self.root.after(0, lambda: self.train_prog.config(value=0))

        threading.Thread(target=proc, daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = MachineHealthApp(root)
    root.mainloop()
