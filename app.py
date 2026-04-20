import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
import noisereduce as nr
import joblib
import os
import threading
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ── Theme ─────────────────────────────────────────────────────────────────────
BG     = "#f0f2f5"
CARD   = "#ffffff"
TEXT   = "#1a1a1a"
MUTED  = "#6b7280"
BLUE   = "#2563eb"
GREEN  = "#16a34a"
RED    = "#dc2626"
BORDER = "#e5e7eb"
PURPLE = "#7c3aed"

# ── Audio constants ────────────────────────────────────────────────────────────
SAMPLE_RATE       = 22050
N_MFCC            = 40
MAX_FREQ_HZ       = 8000
N_ESTIMATORS      = 100
DRAW_POINTS       = 4000   # max points when downsampling waveform for drawing
REPLAY_NORM_PEAK  = 0.95   # ceiling for replay normalization
LOG_EPS           = 1e-9   # prevent log(0)
RT_CHUNK          = SAMPLE_RATE  # real-time chunk size = 1 second

# ── Plot constants ─────────────────────────────────────────────────────────────
PLOT_LW           = 0.7
PLOT_FS           = 7.5
DOM_FREQ_LW       = 0.8
DOM_FREQ_X_OFF    = 0.05
GRID_LW           = 0.4
ZERO_LINE_LW      = 0.5
PLOT_TITLE_FS     = 8
PLOT_BG           = "#f8fafc"


def extract_features(audio, sr):
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    mfcc     = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    rolloff  = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    zcr      = np.mean(librosa.feature.zero_crossing_rate(y=audio))
    return np.hstack([np.mean(mfcc, axis=1), centroid, rolloff, zcr])


class MachineHealthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Health Monitor")
        self.root.geometry("860x820")
        self.root.configure(bg=BG)

        self.models_loaded  = False
        self.binary_model   = self.multi_model = self.label_encoder = None
        self.is_realtime    = False
        self._last_audio    = None
        self._last_sr       = None
        self.data_path      = "data"

        # open-ended recording streams
        self._stream        = None   # inspect tab
        self._rec_buffer    = []
        self._train_stream  = None   # train tab
        self._train_buffer  = []

        for cat in ['normal', 'mechanical', 'electrical', 'wear']:
            os.makedirs(os.path.join(self.data_path, cat), exist_ok=True)

        self._setup_styles()
        self._build_ui()
        self._load_models()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.is_realtime = False
        self._stop_inspect_stream()
        self._stop_train_stream()
        sd.stop()
        plt.close("all")
        self.root.destroy()
        os._exit(0)

    # ── styles ────────────────────────────────────────────────────────────────

    def _setup_styles(self):
        s = ttk.Style()
        s.theme_use('clam')
        s.configure("TNotebook",     background=BG,   borderwidth=0)
        s.configure("TNotebook.Tab", background=CARD, foreground=MUTED,
                    padding=[18, 7], font=("Segoe UI", 10, "bold"))
        s.map("TNotebook.Tab",
              background=[("selected", BLUE)],
              foreground=[("selected", "white")])
        s.configure("TProgressbar", thickness=8, troughcolor=BORDER, background=BLUE)
        s.configure("TCombobox",    fieldbackground=CARD, background=CARD, foreground=TEXT)

    # ── widget helpers ────────────────────────────────────────────────────────

    def _card(self, parent, **kw):
        return tk.Frame(parent, bg=CARD, bd=1, relief="flat",
                        highlightbackground=BORDER, highlightthickness=1, **kw)

    def _btn(self, parent, text, color, cmd, **kw):
        return tk.Button(parent, text=text, bg=color, fg="white",
                         font=("Segoe UI", 10, "bold"), relief="flat",
                         activebackground=color, cursor="hand2",
                         padx=14, pady=6, command=cmd, **kw)

    def _label(self, parent, text, size=10, bold=False, color=TEXT, **kw):
        return tk.Label(parent, text=text, bg=parent["bg"], fg=color,
                        font=("Segoe UI", size, "bold" if bold else "normal"), **kw)

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
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

    # ── inspect tab ───────────────────────────────────────────────────────────

    def _build_inspect(self):
        ctrl = tk.Frame(self.inspect_tab, bg=BG, pady=8)
        ctrl.pack(fill="x", padx=14)
        self._btn(ctrl, "Load Audio File", BLUE, self.load_file).pack(side="left", padx=(0, 8))
        self.live_btn = self._btn(ctrl, "Start Live Check", RED, self._toggle_live)
        self.live_btn.pack(side="left", padx=(0, 8))
        self.rt_btn = self._btn(ctrl, "Real-time Monitor", PURPLE, self._toggle_realtime)
        self.rt_btn.pack(side="left", padx=(0, 8))
        self.replay_btn = self._btn(ctrl, "Replay", MUTED, self._replay)
        self.replay_btn.config(state="disabled")
        self.replay_btn.pack(side="left")

        # Confidence threshold slider
        slider_frame = tk.Frame(self.inspect_tab, bg=BG)
        slider_frame.pack(fill="x", padx=14, pady=(0, 8))
        self._label(slider_frame, "Confidence Threshold", size=9, color=MUTED).pack(anchor="w", pady=(0, 4))
        slider_container = tk.Frame(slider_frame, bg=BG)
        slider_container.pack(fill="x", pady=(0, 4))
        self.conf_slider = ttk.Scale(slider_container, from_=0, to=100, orient="horizontal")
        self.conf_slider.set(50)
        self.conf_slider.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.conf_value = self._label(slider_container, "50%", size=9, color=MUTED, width=4)
        self.conf_value.pack(side="left")
        self.conf_slider.config(command=self._update_slider_label)

        wf_card = self._card(self.inspect_tab, pady=6)
        wf_card.pack(fill="x", padx=14, pady=(4, 8))
        self.fig, (self.ax_wave, self.ax_spec, self.ax_denoise) = plt.subplots(
            3, 1, figsize=(7, 5.2), facecolor=CARD, constrained_layout=True
        )
        for ax in (self.ax_wave, self.ax_spec, self.ax_denoise):
            ax.set_facecolor(PLOT_BG)
            for spine in ax.spines.values():
                spine.set_color(BORDER)
            ax.tick_params(colors=MUTED, labelsize=PLOT_FS)
        self._draw_empty_plots()
        self.wf_canvas = FigureCanvasTkAgg(self.fig, master=wf_card)
        self.wf_canvas.get_tk_widget().pack(fill="x")

        result_card = self._card(self.inspect_tab, padx=20, pady=16)
        result_card.pack(fill="x", padx=14, pady=(0, 8))
        self.res_status = self._label(result_card, "—", size=20, bold=True)
        self.res_status.pack(anchor="w")
        self.res_detail = self._label(result_card, "Load a file or start a live check.", color=MUTED)
        self.res_detail.pack(anchor="w", pady=(4, 10))
        self.health_bar = tk.Canvas(result_card, height=6, bg=BORDER, highlightthickness=0)
        self.health_bar.pack(fill="x")

        log_frame = tk.Frame(self.inspect_tab, bg=BG)
        log_frame.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self._label(log_frame, "Log", size=9, color=MUTED).pack(anchor="w", pady=(0, 4))
        self.log_box = tk.Text(log_frame, height=6, bg=CARD, fg=TEXT,
                               font=("Consolas", 9), state="disabled",
                               relief="flat", padx=10, pady=8,
                               highlightbackground=BORDER, highlightthickness=1)
        self.log_box.pack(fill="both", expand=True)

    # ── train tab ─────────────────────────────────────────────────────────────

    def _build_train(self):
        outer = tk.Frame(self.train_tab, bg=BG, padx=14, pady=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)

        cc = self._card(outer, padx=16, pady=16)
        cc.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        self._label(cc, "1  Collect", size=11, bold=True).pack(anchor="w", pady=(0, 12))
        self._label(cc, "Label", color=MUTED, size=9).pack(anchor="w")
        self.cat_var = tk.StringVar(value="normal")
        ttk.Combobox(cc, textvariable=self.cat_var,
                     values=['normal', 'mechanical', 'electrical', 'wear'],
                     state="readonly", font=("Segoe UI", 10)).pack(fill="x", pady=(2, 12))
        self.rec_btn = self._btn(cc, "Start Recording", BLUE, self._toggle_train_record)
        self.rec_btn.pack(fill="x", pady=(0, 14))
        self.coll_stats = self._label(cc, "", color=MUTED, size=9, justify="left")
        self.coll_stats.pack(anchor="w")
        self._refresh_stats()

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

    # ── open-ended recording — inspect ────────────────────────────────────────

    def _toggle_live(self):
        if not self.models_loaded:
            messagebox.showwarning("No Model", "Train the model first.")
            return
        if self._stream is None:
            self._rec_buffer = []
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1,
                callback=self._inspect_callback
            )
            self._stream.start()
            self._set_dot(RED)
            self.live_btn.config(text="Stop & Analyse", bg=GREEN)
            self.res_status.config(text="Recording…", fg=MUTED)
        else:
            self._stop_inspect_stream()

    def _inspect_callback(self, indata, frames, time_info, status):
        self._rec_buffer.append(indata.copy())

    def _stop_inspect_stream(self):
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        self.live_btn.config(text="Start Live Check", bg=RED)
        self._set_dot(GREEN)
        if self._rec_buffer:
            audio = np.concatenate(self._rec_buffer).flatten()
            self._rec_buffer = []
            self._analyse(audio, SAMPLE_RATE, "Live Mic")

    # ── open-ended recording — train ──────────────────────────────────────────

    def _toggle_train_record(self):
        if self._train_stream is None:
            self._train_buffer = []
            self._train_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1,
                callback=self._train_callback
            )
            self._train_stream.start()
            self.rec_btn.config(text="Stop & Save", bg=RED)
        else:
            self._stop_train_stream()

    def _train_callback(self, indata, frames, time_info, status):
        self._train_buffer.append(indata.copy())

    def _stop_train_stream(self):
        if self._train_stream is None:
            return
        self._train_stream.stop()
        self._train_stream.close()
        self._train_stream = None
        self.rec_btn.config(text="Start Recording", bg=BLUE)
        if not self._train_buffer:
            return
        audio = np.concatenate(self._train_buffer).flatten()
        self._train_buffer = []
        self._save_training_audio(audio)

    def _save_training_audio(self, audio):
        def task():
            self.rec_btn.config(state="disabled")
            try:
                peak = np.max(np.abs(audio))
                normalized = audio / peak * 0.95 if peak > 0 else audio
                category = self.cat_var.get()
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(self.data_path, category, f"rec_{ts}.wav")
                sf.write(path, normalized, SAMPLE_RATE, subtype='PCM_16')
                self.root.after(0, lambda: messagebox.showinfo("Saved", f"Saved as: {category}"))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.root.after(0, lambda: self.rec_btn.config(state="normal"))
                self.root.after(0, self._refresh_stats)
        threading.Thread(target=task, daemon=True).start()

    def _refresh_stats(self):
        lines = []
        for cat in ['normal', 'mechanical', 'electrical', 'wear']:
            n = len([f for f in os.listdir(os.path.join(self.data_path, cat))
                     if f.endswith('.wav')])
            lines.append(f"{cat.capitalize()}: {n}")
        self.coll_stats.config(text="\n".join(lines))

    # ── plots ─────────────────────────────────────────────────────────────────

    def _style_ax(self, ax):
        ax.set_facecolor(PLOT_BG)
        ax.tick_params(colors=MUTED, labelsize=PLOT_FS)
        ax.grid(True, linestyle="--", linewidth=GRID_LW, color=BORDER, alpha=0.8)
        for sp in ax.spines.values():
            sp.set_color(BORDER)

    def _draw_empty_plots(self):
        for ax, title, ylabel in (
            (self.ax_wave,   "Waveform",          "Amplitude"),
            (self.ax_spec,   "Frequency Spectrum", "Power (dB)"),
            (self.ax_denoise,"Denoised",           "Amplitude"),
        ):
            ax.clear()
            ax.set_facecolor(PLOT_BG)
            ax.set_title(title, fontsize=PLOT_TITLE_FS, color=MUTED, loc="left", pad=4)
            ax.set_ylabel(ylabel, fontsize=PLOT_FS, color=MUTED)
            ax.tick_params(colors=MUTED, labelsize=PLOT_FS)
            for sp in ax.spines.values():
                sp.set_color(BORDER)
        self.fig.canvas.draw_idle()

    def _draw_plots(self, audio, sr, wave_color):
        """Render all three subplots for the given audio."""
        t    = np.linspace(0, len(audio) / sr, num=len(audio))
        step = max(1, len(audio) // DRAW_POINTS)

        # ── waveform ─────────────────────────────────────────────────────
        self.ax_wave.clear()
        self._style_ax(self.ax_wave)
        self.ax_wave.plot(t[::step], audio[::step], color=wave_color, linewidth=PLOT_LW)
        rms_db = 20 * np.log10(np.sqrt(np.mean(audio ** 2)) + LOG_EPS)
        peak   = np.max(np.abs(audio))
        self.ax_wave.set_title(
            f"Waveform   RMS {rms_db:.1f} dB   Peak {peak:.3f}",
            fontsize=PLOT_TITLE_FS, color=MUTED, loc="left", pad=4
        )
        self.ax_wave.set_xlabel("Time (s)",  fontsize=PLOT_FS, color=MUTED)
        self.ax_wave.set_ylabel("Amplitude", fontsize=PLOT_FS, color=MUTED)
        self.ax_wave.set_xlim(0, len(audio) / sr)
        self.ax_wave.axhline(0, color=BORDER, linewidth=ZERO_LINE_LW)

        # ── FFT spectrum ──────────────────────────────────────────────────
        self.ax_spec.clear()
        self._style_ax(self.ax_spec)
        n       = len(audio)
        fft     = np.abs(np.fft.rfft(audio * np.hanning(n)))
        fft_db  = 20 * np.log10(fft / (n / 2) + LOG_EPS)
        freqs   = np.fft.rfftfreq(n, d=1 / sr)
        mask    = freqs <= MAX_FREQ_HZ
        dom_hz  = freqs[mask][np.argmax(fft_db[mask])]
        self.ax_spec.plot(freqs[mask] / 1000, fft_db[mask], color=wave_color, linewidth=PLOT_LW)
        self.ax_spec.axvline(dom_hz / 1000, color=RED, linewidth=DOM_FREQ_LW,
                             linestyle="--", alpha=0.7)
        ylim = self.ax_spec.get_ylim()
        self.ax_spec.text(dom_hz / 1000 + DOM_FREQ_X_OFF, ylim[1] - (ylim[1] - ylim[0]) * 0.1,
                          f"{dom_hz:.0f} Hz", color=RED, fontsize=7)
        self.ax_spec.set_title(
            f"Frequency Spectrum   Dominant {dom_hz:.0f} Hz",
            fontsize=PLOT_TITLE_FS, color=MUTED, loc="left", pad=4
        )
        self.ax_spec.set_xlabel("Frequency (kHz)", fontsize=PLOT_FS, color=MUTED)
        self.ax_spec.set_ylabel("Power (dB)",       fontsize=PLOT_FS, color=MUTED)
        self.ax_spec.set_xlim(0, MAX_FREQ_HZ / 1000)

        # ── denoised waveform ─────────────────────────────────────────────
        self.ax_denoise.clear()
        self._style_ax(self.ax_denoise)
        denoised = nr.reduce_noise(y=audio, sr=sr)
        dn_rms   = 20 * np.log10(np.sqrt(np.mean(denoised ** 2)) + LOG_EPS)
        self.ax_denoise.plot(t[::step], denoised[::step], color=BLUE, linewidth=PLOT_LW)
        self.ax_denoise.set_title(
            f"Denoised   RMS {dn_rms:.1f} dB",
            fontsize=PLOT_TITLE_FS, color=MUTED, loc="left", pad=4
        )
        self.ax_denoise.set_xlabel("Time (s)",  fontsize=PLOT_FS, color=MUTED)
        self.ax_denoise.set_ylabel("Amplitude", fontsize=PLOT_FS, color=MUTED)
        self.ax_denoise.set_xlim(0, len(audio) / sr)
        self.ax_denoise.axhline(0, color=BORDER, linewidth=ZERO_LINE_LW)

        self.wf_canvas.draw()

    # ── real-time monitor ─────────────────────────────────────────────────────

    def _toggle_realtime(self):
        if not self.models_loaded:
            messagebox.showwarning("No Model", "Train the model first.")
            return
        if self.is_realtime:
            self.is_realtime = False
            self.rt_btn.config(text="Real-time Monitor", bg=PURPLE)
        else:
            self.is_realtime = True
            self.rt_btn.config(text="Stop Monitoring", bg=RED)
            threading.Thread(target=self._realtime_loop, daemon=True).start()

    def _realtime_loop(self):
        while self.is_realtime:
            rec = sd.rec(RT_CHUNK, samplerate=SAMPLE_RATE, channels=1)
            sd.wait()
            if not self.is_realtime:
                break
            audio = rec.flatten()
            self.root.after(0, lambda a=audio: self._realtime_update(a))

    def _realtime_update(self, audio):
        self._draw_plots(audio, SAMPLE_RATE, PURPLE)

        # Check audio energy (RMS) to detect if sound is actually playing
        rms = np.sqrt(np.mean(audio ** 2))
        energy_threshold = 0.01  # Threshold for detecting if sound is present

        feat       = extract_features(audio, SAMPLE_RATE).reshape(1, -1)
        pred       = self.binary_model.predict(feat)[0]
        prob       = np.max(self.binary_model.predict_proba(feat)) * 100
        diag       = self.label_encoder.inverse_transform([self.multi_model.predict(feat)[0]])[0]
        threshold  = self.conf_slider.get()

        # If no meaningful sound detected, show "No Motor Sound"
        if rms < energy_threshold:
            display_label = "No Motor Sound"
            color = MUTED
            score = 0
        elif prob < threshold:
            display_label = "No Motor Sound"
            color = MUTED
            score = 0
        else:
            if diag == 'normal':
                display_label = 'Normal'
                color = GREEN
                score = prob
            else:
                display_label = f"{diag.capitalize()} Fault"
                color = RED
                score = 100 - prob

        self.res_status.config(text=display_label, fg=color)
        self.res_detail.config(text=f"{prob:.0f}% confidence")

        self.health_bar.delete("all")
        self.root.update_idletasks()
        w = self.health_bar.winfo_width()
        self.health_bar.create_rectangle(0, 0, (score / 100) * w, 6, fill=color, outline="")

        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("1.0", f"[{ts}]  Live  {display_label:<18} {prob:.0f}%\n")
        self.log_box.config(state="disabled")

    # ── replay ────────────────────────────────────────────────────────────────

    def _replay(self):
        if self._last_audio is None:
            return
        def play():
            self.root.after(0, lambda: self.replay_btn.config(state="disabled", text="Playing…"))
            try:
                audio = self._last_audio
                peak  = np.max(np.abs(audio))
                if peak > 0:
                    audio = audio / peak * REPLAY_NORM_PEAK
                sd.play(audio, self._last_sr)
                sd.wait()
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Playback Error", str(e)))
            finally:
                self.root.after(0, lambda: self.replay_btn.config(state="normal", text="Replay"))
        threading.Thread(target=play, daemon=True).start()

    # ── core analysis ─────────────────────────────────────────────────────────

    def _update_slider_label(self, value):
        """Update confidence threshold slider display value."""
        self.conf_value.config(text=f"{int(float(value))}%")

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
            audio, sr = librosa.load(path, sr=SAMPLE_RATE)
            self._analyse(audio, sr, os.path.basename(path))

    def _analyse(self, audio, sr, source):
        self._last_audio = audio
        self._last_sr    = sr
        self.replay_btn.config(state="normal")

        # Check audio energy (RMS) to detect if sound is actually playing
        rms = np.sqrt(np.mean(audio ** 2))
        energy_threshold = 0.01  # Threshold for detecting if sound is present

        feat       = extract_features(audio, sr).reshape(1, -1)
        pred       = self.binary_model.predict(feat)[0]
        prob       = np.max(self.binary_model.predict_proba(feat)) * 100
        multi_probs = self.multi_model.predict_proba(feat)[0]  # Get all class probabilities
        max_class_prob = np.max(multi_probs) * 100  # Max probability among all classes
        diag       = self.label_encoder.inverse_transform([self.multi_model.predict(feat)[0]])[0]
        threshold  = self.conf_slider.get()

        # ── DETECTION GATES ────────────────────────────────────────────────────────────
        # If no meaningful sound detected, show "No Sound"
        if rms < energy_threshold:
            display_label = "No Sound"
            color = MUTED
            score = 0
        # If model confidence is low across all classes, likely unknown audio
        # Threshold: max class probability < 60% = not confident → "No Machine Sound"
        elif max_class_prob < 60:
            display_label = "No Machine Sound"
            color = MUTED
            score = 0
        elif prob < threshold:
            display_label = "No Machine Sound"
            color = MUTED
            score = 0
        else:
            if diag == 'normal':
                display_label = 'Normal'
                color = GREEN
                score = prob
            else:
                display_label = f"{diag.capitalize()} Fault"
                color = RED
                score = 100 - prob

        self._draw_plots(audio, sr, color)

        self.res_status.config(text=display_label, fg=color)
        self.res_detail.config(text=f"{prob:.0f}% confidence")

        self.health_bar.delete("all")
        self.root.update_idletasks()
        w = self.health_bar.winfo_width()
        self.health_bar.create_rectangle(0, 0, (score / 100) * w, 6, fill=color, outline="")

        self.log_box.config(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("1.0", f"[{ts}]  {source:<20} {display_label:<18} {prob:.0f}%\n")
        self.log_box.config(state="disabled")

    def manual_train(self):
        def proc():
            self.train_btn.config(state="disabled")
            cats = ['normal', 'mechanical', 'electrical', 'wear']
            X, y = [], []
            try:
                for idx, c in enumerate(cats):
                    self.root.after(0, lambda c=c: self.train_status.config(text=f"Loading {c}…"))
                    for f in [f for f in os.listdir(os.path.join(self.data_path, c))
                              if f.endswith('.wav')]:
                        audio, sr = librosa.load(os.path.join(self.data_path, c, f),
                                                 sr=SAMPLE_RATE)
                        X.append(extract_features(audio, sr))
                        y.append(c)
                    self.root.after(0, lambda v=(idx + 1) * 20: self.train_prog.config(value=v))

                if not X:
                    messagebox.showerror("Error", "No audio files found in data folders.")
                    return

                self.root.after(0, lambda: self.train_status.config(text="Fitting…"))
                le   = LabelEncoder()
                yenc = le.fit_transform(y)
                ybin = np.array([0 if i == 'normal' else 1 for i in y])

                m_bin   = RandomForestClassifier(n_estimators=N_ESTIMATORS).fit(X, ybin)
                m_multi = RandomForestClassifier(n_estimators=N_ESTIMATORS).fit(X, yenc)

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
