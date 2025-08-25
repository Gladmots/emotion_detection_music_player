# Emotion Detection Music Player

Detect your mood in real time and auto-play matching music. This uses your webcam for facial emotion recognition, then chooses a playlist or track that fits the vibe based on the percieved expressions given.

## Features

* Real-time emotion detection from webcam (happy, sad, angry, surprised, neutral, etc.)
* Manual “Choose mood” fallback
* Map each emotion to a local playlist folder or a streaming playlist (optional)
* Lightweight UI (Streamlit) or CLI mode
* Pluggable model backend (use your own weights)
* Simple config via `config.yaml` or `.env`

## Tech Stack

* Python 3.10+
* OpenCV for camera frames
* Deep learning backend: PyTorch **or** TensorFlow (pick one)
* Streamlit for the UI
* Optional: `spotipy` for Spotify API integration, `librosa` if you add audio-based emotion

## Start

### 1) Clone and install

```bash
git clone https://github.com/Gladmots/emotion_detection_music_player.git
cd emotion_detection_music_player
```
# (Windows PowerShell)
```bash
python -m venv .venv
. .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```
# (macOS/Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare the model and music

* Put your pretrained emotion model file in `models/` (e.g., `models/fer_emotion.pth` or `models/fer_emotion.h5`).
* Drop some songs into `playlists/<emotion>/` folders, or set up Spotify (optional).

### 3) Configure

Create or edit `config.yaml`:

If your using Spotify, also add a `.env`:
```
SPOTIFY_CLIENT_ID=your_id
SPOTIFY_CLIENT_SECRET=your_secret
SPOTIFY_REDIRECT_URI=http://localhost:8080/callback
SPOTIFY_USERNAME=your_username
```

### 4) Run
**UI mode (Streamlit):**

```bash
streamlit run app.py
```
**CLI mode:**

```bash
python cli.py --webcam 0 --duration 300
```
## Privacy & Ethics

* Webcam frames stay on your machine; nothing is uploaded.
* Always obtain consent before capturing faces.
