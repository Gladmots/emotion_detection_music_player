# main.py 

import os, time, threading, random
from pathlib import Path

import numpy as np
import cv2
import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageTk
import onnxruntime as ort
import yt_dlp, requests

# ---- Make VLC discoverable on Windows before importing python-vlc ----
if os.name == "nt":
    for p in (r"C:\Program Files\VideoLAN\VLC", r"C:\Program Files (x86)\VideoLAN\VLC"):
        if Path(p).exists():
            try: os.add_dll_directory(p)
            except Exception: pass
            os.environ.setdefault("VLC_PLUGIN_PATH", str(Path(p, "plugins")))
            break
import vlc

# ---------------- Config ----------------
APP_TITLE = "🎧 Emotion-Based Music Player 3000"
WINDOW_SIZE = "1100x800"
FRAME_W, FRAME_H = 960, 540

DETECT_INTERVAL = 1.5            # seconds between emotion inferences
SWITCH_COOLDOWN = 15             # min seconds between track changes
CONF_THRESHOLD = 0.55            # min probability to accept a new emotion

BARS = 48
BAR_MS = 80

# FER+ class order from ONNX model zoo:
FERPLUS_CLASSES = [
    "Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"
]

# Map FER+ -> your music moods
EMOTION_TO_MOOD = {
    "Happy": "Happy",
    "Surprise": "Surprise",
    "Sad": "Sad",
    "Angry": "Angry",
    "Neutral": "Neutral",
    "Fear": "Fear",
    "Disgust": "Angry",     # collapse
    "Contempt": "Neutral",  # collapse
}

# youTube search query
MOOD_TO_QUERY = {
    "Happy":    "happy upbeat pop hits",
    "Sad":      "soothing acoustic sad songs",
    "Angry":    "motivational energetic rock",
    "Neutral":  "lofi chill beats",
    "Surprise": "trending edm party mix",
    "Fear":     "calming piano music",
}

MOOD_PLAYLISTS = {
    # Example:
    # "Happy": {"playlist": "https://www.youtube.com/playlist?list=YOUR_PLAYLIST_ID"},
    # "Sad":   {"videos": [
    #     "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    #     "https://www.youtube.com/watch?v=oHg5SJYRHA0"
    # ]},
}

# ---------------- FER+ model setup ----------------
MODELS_DIR = Path("models"); MODELS_DIR.mkdir(exist_ok=True)
FERPLUS_PATH = MODELS_DIR / "emotion-ferplus-8.onnx"

FERPLUS_URLS = [
    "https://raw.githubusercontent.com/onnx/models/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
    "https://huggingface.co/webml/models/resolve/main/emotion-ferplus-8.onnx",
]

# ---------------- Globals ----------------
running = True
latest_frame = None
last_mood = ""
last_switch_time = 0.0
last_box = None
last_prob = 0.0

player = None
vlc_instance = None
cap = None
ort_sess = None
input_name = None

# ---------------- Utils ----------------
def ensure_ferplus():
    if FERPLUS_PATH.exists():
        return
    last_err = None
    for url in FERPLUS_URLS:
        try:
            print(f"[setup] Downloading FER+ ONNX model from: {url}")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(FERPLUS_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            # sanity check (~35MB). Small files are likely HTML/LFS pointers.
            if FERPLUS_PATH.stat().st_size < 5_000_000:
                raise ValueError("Downloaded file too small; likely an HTML page or LFS pointer.")
            print(f"[setup] Saved {FERPLUS_PATH}")
            return
        except Exception as e:
            last_err = e
            print(f"[setup] Failed from {url}: {e}")
            if FERPLUS_PATH.exists():
                try: FERPLUS_PATH.unlink()
                except: pass
    raise RuntimeError(f"Could not download FER+ model from known mirrors. Last error: {last_err}")

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-8)

def crop_largest_face(frame_bgr):
    """Detect faces and return the largest crop (x,y,w,h) and image ROI. None if not found."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None, None
    # pick largest
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    x0, y0 = max(0, x), max(0, y)
    roi = frame_bgr[y0:y0+h, x0:x0+w]
    return (x0, y0, w, h), roi

def preprocess_ferplus(face_bgr):
    """FER+ expects grayscale 64x64, NCHW float32 (0..1)."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    roi = roi / 255.0
    # NCHW
    return roi[np.newaxis, np.newaxis, :, :]

def load_onnx():
    global ort_sess, input_name
    ensure_ferplus()
    providers = ["CPUExecutionProvider"]
    ort_sess = ort.InferenceSession(str(FERPLUS_PATH), providers=providers)
    input_name = ort_sess.get_inputs()[0].name

def infer_emotion(frame_bgr):
    """Return (mood, prob, box) or (None, 0.0, None)."""
    box, face = crop_largest_face(frame_bgr)
    if face is None:
        return None, 0.0, None
    inp = preprocess_ferplus(face)
    out = ort_sess.run(None, {input_name: inp})[0]  # shape (1,8)
    probs = softmax(out[0].astype(np.float32))
    idx = int(np.argmax(probs))
    fer_label = FERPLUS_CLASSES[idx]
    mood = EMOTION_TO_MOOD.get(fer_label, "Neutral")
    return mood, float(probs[idx]), box

def _yt_search(query):
    ydl_opts = {"quiet": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch1:{query}", download=False)
        entry = (info.get("entries") or [info])[0]
        url = entry.get("webpage_url") or entry.get("url")
        title = entry.get("title", "Unknown Title")
        return url, title

def get_track_for_mood(mood: str):
    """Pick from curated sources if present; otherwise fallback to search."""
    src = MOOD_PLAYLISTS.get(mood)
    ydl_opts = {"quiet": True, "noplaylist": False}
    if src:
        # 1) explicit video list
        vids = src.get("videos") if isinstance(src, dict) else None
        if vids:
            pick = random.choice(vids)
            return pick, "Curated pick"
        # 2) playlist
        playlist = src.get("playlist") if isinstance(src, dict) else None
        if playlist:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(playlist, download=False)
                entries = info.get("entries") or []
                if entries:
                    pick = random.choice(entries)
                    url = pick.get("webpage_url") or pick.get("url")
                    title = pick.get("title", "Unknown Title")
                    return url, title
    # 3) fallback: search
    query = MOOD_TO_QUERY.get(mood, "relaxing ambient music")
    return _yt_search(query)

def stream_audio(url: str, set_status):
    global player, vlc_instance
    try:
        if player:
            player.stop(); player.release(); player = None
        ydl_opts = {"format": "bestaudio/best", "quiet": True, "noplaylist": True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            audio_url = info["url"]
        if vlc_instance is None:
            vlc_instance = vlc.Instance("--no-video")
        player = vlc_instance.media_player_new()
        media = vlc_instance.media_new(audio_url)
        player.set_media(media)
        player.audio_set_volume(85)
        player.play()
    except Exception as e:
        set_status(f"Playback error: {e}")

# ---------------- UI ----------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")
app = ctk.CTk()
app.title(APP_TITLE)
app.geometry(WINDOW_SIZE)
app.minsize(1000, 700)

emotion_var = ctk.StringVar(value="Detecting Emotion…")
song_var = ctk.StringVar(value="Initializing Music…")

bg = ctk.CTkFrame(app, fg_color="#0b0e1a", corner_radius=0)
bg.place(relwidth=1, relheight=1)

# (FIXED) width/height set in constructors, not in .place()
card = ctk.CTkFrame(
    bg, width=1000, height=600,
    fg_color="#131729", corner_radius=24, border_width=3, border_color="#9b59b6"
)
card.place(relx=0.5, rely=0.08, anchor="n")

video_frame = ctk.CTkFrame(
    card, width=960, height=560,
    fg_color="#0f1426", corner_radius=18, border_width=2, border_color="#ff00ff"
)
video_frame.place(relx=0.5, rely=0.0, anchor="n")

video_label = ctk.CTkLabel(video_frame, text="")
video_label.place(relx=0.5, rely=0.5, anchor="center")

# Big emotion label
emotion_label = ctk.CTkLabel(
    bg, textvariable=emotion_var,
    font=ctk.CTkFont(size=40, weight="bold"),
    text_color="#00eaff"
)
emotion_label.place(relx=0.5, rely=0.82, anchor="center")

# Confidence bar
conf_bar = ctk.CTkProgressBar(bg, width=420, height=14, corner_radius=10, progress_color="#00ffd5")
conf_bar.place(relx=0.5, rely=0.87, anchor="center")
conf_bar.set(0.0)  # 0..1

song_label = ctk.CTkLabel(
    bg, textvariable=song_var,
    font=ctk.CTkFont(size=22, weight="bold"),
    text_color="#ff59ff"
)
song_label.place(relx=0.5, rely=0.92, anchor="center")

viz_frame = ctk.CTkFrame(
    bg, width=900, height=120,
    fg_color="#0f0f1f", corner_radius=18, border_color="#8e44ad", border_width=2
)
viz_frame.place(relx=0.5, rely=0.97, anchor="s")

viz_canvas = tk.Canvas(viz_frame, width=860, height=88, bg="#0f0f1f", highlightthickness=0)
viz_canvas.place(relx=0.5, rely=0.5, anchor="center")

bar_width = 12
gap = (860 - BARS * bar_width) / (BARS - 1)
bars = [
    viz_canvas.create_rectangle(i * (bar_width + gap), 88, i * (bar_width + gap) + bar_width, 88,
                                fill="#00eaff", outline="")
    for i in range(BARS)
]

def animate_visualizer():
    if not running:
        return
    heights = np.random.randint(18, 82, size=BARS)
    for r, h in zip(bars, heights):
        x0, _, x1, _ = viz_canvas.coords(r)
        viz_canvas.coords(r, x0, 88 - int(h), x1, 88)
    viz_canvas.after(BAR_MS, animate_visualizer)

def pulse_emotion_color():
    colors = ["#00eaff", "#ff59ff", "#8e44ad", "#00ffd5"]; idx = 0
    def tick():
        nonlocal idx
        if running:
            emotion_label.configure(text_color=colors[idx % len(colors)])
            idx += 1
            app.after(400, tick)
    tick()

# ---------------- Loops ----------------
def detection_loop():
    global latest_frame, last_mood, last_switch_time, last_box, last_prob
    last_infer = 0.0
    while running:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05); continue
        frame = cv2.flip(frame, 1)
        latest_frame = frame
        now = time.time()
        if now - last_infer >= DETECT_INTERVAL:
            last_infer = now
            try:
                mood, prob, box = infer_emotion(frame)
                last_box = box
                last_prob = prob
                if mood is None:
                    emotion_var.set("No face")
                    conf_bar.set(0.0)
                else:
                    emotion_var.set(f"{mood}")
                    conf_bar.set(max(0.0, min(1.0, prob)))
                    if (mood != last_mood) and (prob >= CONF_THRESHOLD) and (now - last_switch_time >= SWITCH_COOLDOWN):
                        last_mood, last_switch_time = mood, now
                        url, title = get_track_for_mood(mood)
                        song_var.set(title)
                        stream_audio(url, set_status=lambda s: song_var.set(s))
            except Exception:
                # keep UI running even if inference fails
                pass
        time.sleep(0.01)

def update_video():
    if not running:
        return
    if latest_frame is not None:
        # draw overlays on a copy
        disp = cv2.resize(latest_frame, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA).copy()
        if last_box is not None:
            x,y,w,h = last_box
            # scale box if camera capture size differs from display size

            cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 234, 255), 2)
            label = f"{last_mood or '—'} {int(last_prob*100)}%"
            cv2.putText(disp, label, (x, max(0, y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 89, 255), 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    video_label.after(30, update_video)

# ---------------- Lifecycle ----------------
def on_start():
    global cap
    load_onnx()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    threading.Thread(target=detection_loop, daemon=True).start()
    update_video(); animate_visualizer(); pulse_emotion_color()

def on_close():
    global running, cap, player, vlc_instance
    running = False
    try:
        if player: player.stop(); player.release()
        if vlc_instance: vlc_instance.release()
        if cap: cap.release()
    finally:
        app.destroy()

app.protocol("WM_DELETE_WINDOW", on_close)

if __name__ == "__main__":

    emotion_var.set("Happy")
    song_var.set("Shape of You")
    on_start()
    app.mainloop()
