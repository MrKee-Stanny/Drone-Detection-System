#!/usr/bin/env python3
import os
import time
import subprocess
import argparse

OUTDIR = "training_data"
DRONE_DIR = os.path.join(OUTDIR, "drone")
NONDRONE_DIR = os.path.join(OUTDIR, "non-drone")
CARD = 2
DEV = 0
DURATION = 3
FS = 22050
CHANNELS = 1

def ensure_dirs():
    os.makedirs(DRONE_DIR, exist_ok=True)
    os.makedirs(NONDRONE_DIR, exist_ok=True)

def safe_filename(prefix="sample"):
    ts = int(time.time() * 1000)
    return f"{prefix}_{ts}.wav"

def record_arecord(path, duration=DURATION, fs=FS, channels=CHANNELS, card=CARD, dev=DEV):
    dev_spec = f"plughw:{card},{dev}"
    cmd = [
        "arecord",
        "-D", dev_spec,
        "-f", "S16_LE",
        "-r", str(fs),
        "-c", str(channels),
        "-d", str(duration),
        path,
    ]
    subprocess.run(cmd, check=True)

def main():
    global OUTDIR, DRONE_DIR, NONDRONE_DIR, DURATION, FS
    parser = argparse.ArgumentParser(description="Record 3s labeled WAVs using arecord (plughw:2,0)")
    parser.add_argument("--outdir", default=OUTDIR)
    parser.add_argument("--duration", type=int, default=DURATION)
    parser.add_argument("--fs", type=int, default=FS)
    args = parser.parse_args()

    
    OUTDIR = args.outdir
    DRONE_DIR = os.path.join(OUTDIR, "drone")
    NONDRONE_DIR = os.path.join(OUTDIR, "non-drone")
    DURATION = args.duration
    FS = args.fs

    ensure_dirs()

    print("Using arecord device: plughw:2,0")
    print("Press 'd' + Enter to record a 'drone' sample, 'n' + Enter for 'non-drone', 'q' to quit.")

    choice = input("\nEnter label (d/n/q): ").strip().lower()

    i = 0

    while i != 60 :
        if choice == "q":
            print("Exiting.")
            break
        if choice not in ("d", "n"):
            print("Invalid choice.")
            continue

        i = i + 1

        label_dir = DRONE_DIR if choice == "d" else NONDRONE_DIR
        prefix = "drone" if choice == "d" else "nondrone"
        filename = safe_filename(prefix=prefix)
        out_path = os.path.join(label_dir, filename)

        print(f"Recording {DURATION}s -> {out_path}")
        try:
            record_arecord(out_path, duration=DURATION, fs=FS)
            print("Saved:", out_path)
        except subprocess.CalledProcessError as e:
            print("Recording failed:", e)
        except FileNotFoundError:
            print("arecord not found. Install alsa-utils and ensure arecord is on PATH.")
            break

if __name__ == "__main__":
    main()