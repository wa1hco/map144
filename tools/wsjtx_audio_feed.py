#!/usr/bin/env python3
"""
wsjtx_audio_feed.py -- Phase 0 of the MAP144 -> WSJT-X audio bridge.

Streams a 12 kHz MSK144 WAV to a PipeWire/Pulse null sink IN REAL TIME, aligned
to UTC 15 s period boundaries, so WSJT-X (Radio=None, input = "Monitor of
map144_out", mode MSK144) can decode it.  This validates the entire audio
ingestion path -- virtual device, format, level, UTC framing, decode -- with
ZERO changes to MAP144 or the radio.

Band dead / no live signals?  That's the point: this replays a known-good ping
(default: the KB8OTK +9 dB period) every 15 s, so WSJT-X gets a fresh decodable
period each cycle and you can confirm it prints "KB8OTK N4JP FM14".

Chain: 12 kHz mono int16  --resample x4-->  48 kHz mono  --paplay--> null sink
       --monitor--> WSJT-X.  Levels preserved (no renormalisation) so WSJT-X
       sees the same SNR the recording had.

Usage:
    python3 tools/wsjtx_audio_feed.py [wav]      # default: KB8OTK +9 dB
    python3 tools/wsjtx_audio_feed.py --remove   # delete the null sink
Ctrl-C to stop (sink is left in place for the next run).
"""
import sys
import os
import time
import wave
import shutil
import argparse
import subprocess
import numpy as np
from scipy.signal import resample_poly

SINK = "map144_out"
OUT48 = "scratch/wsjtx_feed_48k.wav"
PERIOD = 15.0
DEFAULT_WAV = "/home/jeff/.local/share/WSJT-X - flex/save/260511_104230.wav"  # KB8OTK +9 dB


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def sink_exists():
    return any(SINK in ln for ln in sh("pactl list short sinks").stdout.splitlines())


def ensure_sink():
    if sink_exists():
        print(f"[feed] sink '{SINK}' already present")
        return
    r = sh(f"pactl load-module module-null-sink sink_name={SINK} "
           f"object.linger=1 media.class=Audio/Sink "
           f"sink_properties=device.description={SINK}")
    if r.returncode != 0:
        sys.exit(f"[feed] failed to create sink: {r.stderr.strip()}")
    print(f"[feed] created null sink '{SINK}' (record from 'Monitor of {SINK}')")


def remove_sink():
    for ln in sh("pactl list short modules").stdout.splitlines():
        if "module-null-sink" in ln and SINK in ln:
            mid = ln.split()[0]
            sh(f"pactl unload-module {mid}")
            print(f"[feed] unloaded module {mid} (sink {SINK})")
            return
    print(f"[feed] no '{SINK}' null-sink module found")


def pick_player():
    """Prefer paplay (pulse-compat), fall back to pw-play (native PipeWire)."""
    if shutil.which("paplay"):
        return lambda path: ["paplay", f"--device={SINK}", path]
    if shutil.which("pw-play"):
        return lambda path: ["pw-play", "--target", SINK, path]
    sys.exit("[feed] neither 'paplay' nor 'pw-play' found "
             "(install pulseaudio-utils or pipewire-bin)")


def prep_48k(wav_in):
    w = wave.open(wav_in)
    sr, n, ch = w.getframerate(), w.getnframes(), w.getnchannels()
    a = np.frombuffer(w.readframes(n), np.int16)
    w.close()
    if ch > 1:
        a = a[::ch]                          # take first channel
    if sr != 48000:
        a = resample_poly(a.astype(np.float64), 48000, sr)
    a = np.clip(np.round(a), -32768, 32767).astype(np.int16)   # level preserved, clip-guard
    os.makedirs("scratch", exist_ok=True)
    w2 = wave.open(OUT48, "w")
    w2.setnchannels(1)
    w2.setsampwidth(2)
    w2.setframerate(48000)
    w2.writeframes(a.tobytes())
    w2.close()
    return len(a) / 48000.0


def main():
    ap = argparse.ArgumentParser(description="Phase 0 WSJT-X audio feeder")
    ap.add_argument("wav", nargs="?", default=DEFAULT_WAV, help="12 kHz MSK144 WAV to replay")
    ap.add_argument("--remove", action="store_true", help="delete the null sink and exit")
    args = ap.parse_args()

    if args.remove:
        remove_sink()
        return

    if not os.path.isfile(args.wav):
        sys.exit(f"[feed] WAV not found: {args.wav}")

    ensure_sink()
    play = pick_player()
    dur = prep_48k(args.wav)
    print(f"[feed] prepared {OUT48}: {dur:.1f}s @ 48 kHz from {os.path.basename(args.wav)}")
    print(f"""
[feed] ---- WSJT-X setup (one time) ----
   Settings -> Audio -> Input  =  Monitor of {SINK}
   Mode = MSK144      Radio = None      dial 50.260 (cosmetic only)
   Decodes should appear once per 15 s period.
[feed] Replaying every {PERIOD:.0f}s, aligned to the UTC period boundary.  Ctrl-C to stop.
""")
    proc = None
    try:
        while True:
            wait = PERIOD - (time.time() % PERIOD)
            time.sleep(wait)
            t0 = time.strftime("%H:%M:%S", time.gmtime())
            # Fire AT the boundary and DON'T block.  paplay takes ~15.1 s (the
            # 15.0 s file + startup latency) -- just over one period -- so a
            # blocking call overshot the next boundary and replayed every 30 s.
            # Non-blocking Popen holds a strict 15 s cadence; the ~0.1 s tail
            # overlap at the boundary (mostly inter-ping noise) is harmless.
            proc = subprocess.Popen(play(OUT48))
            print(f"[feed] {t0} UTC  ->  streamed one {dur:.1f}s period")
            time.sleep(1.0)   # step clear of the boundary before recomputing wait
    except KeyboardInterrupt:
        print(f"\n[feed] stopped.  Sink '{SINK}' left in place "
              f"(run with --remove to delete it).")


if __name__ == "__main__":
    main()
