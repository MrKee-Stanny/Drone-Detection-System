"""
Revised drone-detection script:
- Uses atomic JSON writes
- Validates arecord availability
- Uses logging
- Maps model classes robustly
- Respects sample rate consistency (set to model/training rate via SR)
- Adds graceful shutdown via threading.Event and signal handlers
- Uses subprocess.run with timeout and captured stderr/stdout
"""

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, HTTPServer

import joblib
import librosa
import numpy as np
import soundfile as sf
import math

# --- CONFIG - defaults (overridable via CLI) ---
DEFAULT_MODEL_PATH = "/home/drone2025/Desktop/Drone Detection System/ML-Model-main/ml_model.pkl"
DEFAULT_RESULTS_FILE = "results.json"
DEFAULT_AUDIO_OUT = "sample.wav"
WAVEFORM_PATH = "waveform.json"
DEFAULT_AUDIO_DURATION = 3     # seconds
DEFAULT_FS = 22050             # sample rate used for feature extraction / model (match training)
DEFAULT_CONFIDENCE_THRESHOLD = 80  # percent
DEFAULT_CARD = 2
DEFAULT_DEV = 0
DEFAULT_CHANNELS = 1
HTTP_BIND = "0.0.0.0"
HTTP_PORT = 8000
WEB_DIR = "web"


# --- Helpers / Globals ---
stop_event = threading.Event()
lock = threading.Lock()
logger = logging.getLogger("drone_detector")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

def atomic_write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

def check_command(cmd_name):
    return shutil.which(cmd_name) is not None

def save_waveform(out_path, target_len=1024, sr=DEFAULT_FS):
    # Load mono PCM (soundfile preserves original dtype)
    data, sr_loaded = sf.read(out_path, dtype="float32", always_2d=False)
    # If stereo, mix to mono
    if data.ndim > 1:
        data = data.mean(axis=1)
    # Optionally resample to model sr to keep consistent length
    if sr_loaded != sr:
        data = librosa.resample(data, sr_loaded, sr)
    # Downsample to a fixed number of points for efficient transfer
    total = len(data)
    if total <= target_len:
        # pad/trim to target_len
        if total < target_len:
            pad = target_len - total
            data = np.pad(data, (0, pad), mode="constant")
        samples = data
    else:
        # block-aggregate (RMS or mean) to reduce points
        hop = total / target_len
        samples = np.zeros(target_len, dtype="float32")
        for i in range(target_len):
            start = int(math.floor(i * hop))
            end = int(min(total, math.floor((i + 1) * hop)))
            if end <= start:
                samples[i] = data[start]
            else:
                samples[i] = float(np.mean(data[start:end]))
    # Normalize to -1..1 then quantize to int16 for smaller JSON (optional)
    maxv = max(1e-9, float(np.max(np.abs(samples))))
    samples = (samples / maxv).astype("float32")
    # Convert to a list of floats (or ints) and atomic write to web dir
    payload = {"timestamp": int(time.time()), "sr": sr, "points": samples.tolist()}
    atomic_write_json(WAVEFORM_PATH, payload)


def extract_features(file_path, sr=DEFAULT_FS, n_mfcc=13):
    """
    Load audio and return mean MFCC vector (shape: n_mfcc,).
    Maintains mono mix by default; adjust if model expects different channels.
    """
    y, sr_loaded = librosa.load(file_path, sr=sr, mono=True)
    mfccs = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)


def run_arecord(out_path, duration, fs, channels, card, dev, timeout_buffer=10):
    cmd = [
        "arecord",
        "-D", f"plughw:{card},{dev}",
        "-f", "S16_LE",
        "-r", str(fs),
        "-c", str(channels),
        "-d", str(duration),
        out_path,
    ]
    logger.debug("Running arecord: %s", " ".join(cmd))
    # timeout slightly larger than duration to avoid hangs
    return subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=duration + timeout_buffer)


def detect_loop(model, results_file, out_file, audio_duration, fs, channels, card, dev):
    """
    Main loop: record, extract, predict, write JSON.
    """
    # Determine index corresponding to the positive/drone class.
    classes = list(model.classes_)
    # Heuristic: choose the class with label == 1, or the class with name 'drone' (case-insensitive),
    # otherwise assume the second column of predict_proba corresponds to the positive class.
    drone_index = None
    if 1 in classes:
        drone_index = classes.index(1)
    else:
        for idx, c in enumerate(classes):
            if isinstance(c, str) and "drone" in c.lower():
                drone_index = idx
                break
    if drone_index is None:
        drone_index = 1 if len(classes) > 1 else 0
    logger.info("Model classes: %s; using proba index %d for 'drone'", classes, drone_index)

    if not check_command("arecord"):
        logger.warning("'arecord' not found on PATH. Recording will fail.")

    while not stop_event.is_set():
        try:
            logger.info("Recording %ds -> %s", audio_duration, out_file)
            run_arecord(out_file, audio_duration, fs, channels, card, dev)

            features = extract_features(out_file, sr=fs)
            features = features.reshape(1, -1)

            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            proba_for_drone = float(proba[drone_index])

            confidence_pct = proba_for_drone * 100.0
            
            # Map label name if possible
            try:
                predicted_label = classes[list(model.predict_proba(features)[0]).index(max(proba))]
            except Exception:
                predicted_label = pred

            status = "ALERT" if predicted_label == 1 else "NO ALERT"
            data = {
                "timestamp": int(time.time()),
                "predicted_label": str(predicted_label),
                "confidence_percent": round(confidence_pct, 4),
                "status": status,
            }

            # Atomic write
            with lock:
                atomic_write_json(results_file, data)
                save_waveform(out_file, target_len=1024, sr=fs)

            logger.info("Result: %s", data)

            # Simple debounce / pacing
            time.sleep(0.1)
        except subprocess.TimeoutExpired:
            logger.error("arecord timed out.")
        except FileNotFoundError as e:
            logger.exception("Recording command not found: %s", e)
            time.sleep(5)
        except Exception:
            logger.exception("Detection loop error; retrying in 2s")
            time.sleep(2)


def start_http_server(bind, port, web_dir):
    """
    Serve files from web_dir. This blocks until stop_event is set; uses HTTPServer.
    """
    if not os.path.isdir(web_dir):
        logger.warning("Web directory '%s' not found; creating.", web_dir)
        os.makedirs(web_dir, exist_ok=True)

    os.chdir(web_dir)
    handler = SimpleHTTPRequestHandler
    httpd = HTTPServer((bind, port), handler)

    # Run server in a separate thread so we can shutdown cleanly
    def serve():
        logger.info("Serving at http://%s:%d", bind, port)
        try:
            while not stop_event.is_set():
                httpd.handle_request()
        except Exception:
            logger.exception("HTTP server error")
        finally:
            httpd.server_close()
            logger.info("HTTP server stopped")

    t = threading.Thread(target=serve, name="http-server", daemon=True)
    t.start()
    return t


def install_signal_handlers():
    def _handle(signum, frame):
        logger.info("Signal %d received, shutting down...", signum)
        stop_event.set()

    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _handle)


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Drone detection service")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--results", default=DEFAULT_RESULTS_FILE)
    parser.add_argument("--out", default=DEFAULT_AUDIO_OUT)
    parser.add_argument("--duration", type=int, default=DEFAULT_AUDIO_DURATION)
    parser.add_argument("--fs", type=int, default=DEFAULT_FS)
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS)
    parser.add_argument("--card", type=int, default=DEFAULT_CARD)
    parser.add_argument("--dev", type=int, default=DEFAULT_DEV)
    parser.add_argument("--threshold", type=float, default=DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument("--bind", default=HTTP_BIND)
    parser.add_argument("--port", type=int, default=HTTP_PORT)
    parser.add_argument("--webdir", default=WEB_DIR)
    args = parser.parse_args()

    # Validate paths and load model
    if not os.path.isfile(args.model):
        logger.error("Model file not found: %s", args.model)
        return

    logger.info("Loading model from %s", args.model)
    model = joblib.load(args.model)
    logger.info("Model loaded. Classes: %s", model.classes_)

    # Ensure results file exists initially
    initial = {"timestamp": int(time.time()), "status": "INIT", "confidence_percent": 0.0}
    atomic_write_json(args.results, initial)

    install_signal_handlers()

    # Start HTTP server
    start_http_server(args.bind, args.port, args.webdir)

    # Start detection thread
    detect_thread = threading.Thread(
        target=detect_loop,
        args=(model, args.results, args.out, args.duration, args.fs, args.channels, args.card, args.dev),
        name="detector",
        daemon=True,
    )
    detect_thread.start()

    # Wait for shutdown
    try:
        while not stop_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        stop_event.set()

    logger.info("Waiting for threads to finish...")
    detect_thread.join(timeout=2)
    logger.info("Exiting.")


if __name__ == "__main__":
    main()