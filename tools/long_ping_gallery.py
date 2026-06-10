#!/usr/bin/env python3
"""
long_ping_gallery.py -- Find MSK144 decodes that needed multi-frame coherent
integration and render spectrograms.

Detection method:
  For each WSJT-X decode in ALL.TXT, run _freq_search_avg for each of the
  six SPD navmasks (1-frame, 2-frame, 3-frame combos).  Find the minimum
  navg that achieves xmax_norm >= DECODE_THRESH (1.3).

  Decodes where:
    - single-frame (navg=1) fails, AND
    - two- or three-frame (navg=2-3) succeeds
  are confirmed coherent-integration cases.

  Decodes where even three-frame averaging fails are likely WSJT-X deep-
  decode cases (5-7 frames), flagged separately.

Output: scratch/long_ping_gallery/  (PNGs + summary.txt)
"""

import argparse
import collections
import sys
import wave
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from map144_app.msk144_spd import (
    _freq_search_avg, _NAV_PATTERNS, NSPM, FS, FC_JT9, SYNC_THRESHOLD,
)
from tools.qrz_lookup import QrzClient, load_credentials, QrzError

# ── Constants ─────────────────────────────────────────────────────────────────
DECODE_THRESH = SYNC_THRESHOLD   # 1.3 normalised
FRAME_DUR     = NSPM / FS        # 72 ms

# freq search params (match SPD ntol0/delf_inner)
NTOL_HZ   = 8.0
DELF_HZ   = 2.0

# spectrogram display
NFFT            = 512
HOP             = 48
PLOT_BW         = 700.0   # Hz half-width for spectrogram freq axis
PLOT_PAD        = 0.3     # s time padding around marked burst
NOISE_HEADROOM  = 8.0     # dB below median = cmap bottom
NOISE_RANGE     = 22.0    # dB above median = cmap top

_CMAP_CACHE = None

def _cmap():
    """MAP144 GUI colormap: black → navy → blue → cyan → pale-yellow → white."""
    global _CMAP_CACHE
    if _CMAP_CACHE is None:
        stops_rgb = [
            (0, 0, 0), (0, 0, 64), (0, 0, 128), (0, 64, 192),
            (0, 128, 255), (64, 192, 255), (128, 255, 255),
            (255, 255, 128), (255, 255, 255),
        ]
        colors = [(r/255.0, g/255.0, b/255.0) for r, g, b in stops_rgb]
        _CMAP_CACHE = LinearSegmentedColormap.from_list('map144_gui', colors, N=256)
    return _CMAP_CACHE

SAVE_DIR  = Path("/home/jeff/.local/share/WSJT-X - flex/save")
ALL_TXT   = Path("/home/jeff/.local/share/WSJT-X - flex/ALL.TXT")

MY_GRID   = "FN42"   # WA1HCO default; override with --my-grid
MY_CALL   = "WA1HCO"


# ── Distance helpers ──────────────────────────────────────────────────────────

_NON_GRID = {"RR73", "RR74"}   # look like grids but aren't
_MSG_KEYWORDS = {"CQ", "DE", "TEST", "73", "RR73", "RR74"}

def grid_from_message(message: str) -> str | None:
    """Extract the first 4-char Maidenhead grid square from a decode message."""
    for token in message.split():
        if token.upper() in _NON_GRID:
            continue
        if (len(token) == 4
                and token[0].upper() in "ABCDEFGHIJKLMNOPQR"
                and token[1].upper() in "ABCDEFGHIJKLMNOPQR"
                and token[2].isdigit()
                and token[3].isdigit()):
            return token.upper()
    return None


def calls_from_message(message: str) -> list[str]:
    """Extract callsign-like tokens from a message (no grids, no keywords)."""
    calls = []
    for token in message.split():
        t = token.upper()
        if t in _MSG_KEYWORDS:
            continue
        if grid_from_message(t):   # skip grid squares
            continue
        if t.startswith(("+", "-")) and t[1:].isdigit():  # signal report
            continue
        if t.isdigit():
            continue
        if len(t) >= 3 and t.replace("/", "").isalnum():
            calls.append(t)
    return calls


def maidenhead_to_latlon(grid: str) -> tuple[float, float]:
    """Center lat/lon of a 4-char Maidenhead locator."""
    g = grid.upper()
    lon = (ord(g[0]) - ord("A")) * 20.0 - 180.0 + (ord(g[2]) - ord("0")) * 2.0 + 1.0
    lat = (ord(g[1]) - ord("A")) * 10.0 -  90.0 + (ord(g[3]) - ord("0")) * 1.0 + 0.5
    return lat, lon


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine great-circle distance in km."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(phi1) * np.cos(phi2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


# ── Audio helpers ─────────────────────────────────────────────────────────────

def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        raw  = w.readframes(w.getnframes())
        rate = w.getframerate()
    if rate != int(FS):
        raise ValueError(f"Expected {int(FS)} Hz, got {rate}")
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0


def to_baseband(audio: np.ndarray, fc: float) -> np.ndarray:
    """Real audio → complex baseband at fc, RMS-normalised (SPD convention)."""
    analytic = hilbert(audio.astype(np.float64)).astype(np.complex64)
    t  = np.arange(len(analytic), dtype=np.float64) / FS
    bb = (analytic * np.exp(-2j * np.pi * fc * t)).astype(np.complex64)
    rms = float(np.sqrt((np.abs(bb.astype(np.complex128)) ** 2).mean()))
    if rms > 1e-12:
        bb = (bb * (np.sqrt(2.0) / rms)).astype(np.complex64)
    return bb


def period_wav(ts: str) -> Path:
    date, time = ts[:6], ts[7:]
    hh, mm, ss = int(time[:2]), int(time[2:4]), int(time[4:6])
    return SAVE_DIR / f"{date}_{hh:02d}{mm:02d}{(ss//15)*15:02d}.wav"


# ── Navmask scan ──────────────────────────────────────────────────────────────

def find_min_navg(bb: np.ndarray, dt: float,
                  fc_off: float) -> tuple[int | None, float]:
    """
    Try each SPD navmask on a 3-NSPM window centered on dt.
    Return (min_navg_that_decodes, xmax_norm_at_that_navg).
    min_navg=None if all patterns fail.
    """
    n_total = len(bb)
    # centre of the 3-frame window: place frame 1 (middle) starting at DT
    i_start = int(dt * FS) - NSPM // 2
    i_start = max(0, min(i_start, n_total - 3 * NSPM))
    window  = bb[i_start: i_start + 4 * NSPM]   # 4*NSPM for non-circular extract
    if len(window) < 3 * NSPM:
        return None, 0.0

    best_by_navg: dict[int, float] = {}
    for navmask in _NAV_PATTERNS:
        navg = int(sum(navmask))
        if navg in best_by_navg:
            continue   # already found better or equal
        c_avg, xmax_norm, *_ = _freq_search_avg(
            window, navmask, ntol=NTOL_HZ, delf=DELF_HZ,
        )
        if xmax_norm >= DECODE_THRESH:
            if navg not in best_by_navg or xmax_norm > best_by_navg[navg]:
                best_by_navg[navg] = xmax_norm

    if not best_by_navg:
        return None, 0.0
    min_navg = min(best_by_navg)
    return min_navg, best_by_navg[min_navg]


# ── Spectrogram helpers ───────────────────────────────────────────────────────

def stft_power(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    win    = np.hanning(NFFT).astype(np.float32)
    n_bins = NFFT // 2 + 1
    n_fr   = 1 + (len(audio) - NFFT) // HOP
    S = np.empty((n_fr, n_bins), np.float32)
    for t in range(n_fr):
        seg  = audio[t * HOP: t * HOP + NFFT] * win
        S[t] = np.abs(np.fft.rfft(seg, NFFT)) ** 2
    times = (np.arange(n_fr) * HOP + NFFT / 2.0) / FS
    freqs = np.fft.rfftfreq(NFFT, 1.0 / FS)
    return times, freqs, S


def plot_ping(wav_path: Path, dec: dict, min_navg: int,
              xmax_norm: float, out_path: Path) -> None:
    try:
        audio = load_wav(wav_path)
    except Exception as e:
        print(f"  skip {wav_path.name}: {e}")
        return

    times, freqs, S = stft_power(audio)
    S_db = 10.0 * np.log10(S + 1e-12)

    audio_hz = dec["audio_hz"]
    dt       = dec["dt"]

    # zoom window: DT ± min_navg frames + padding
    t_lo = max(0.0,        dt - PLOT_PAD - FRAME_DUR)
    t_hi = min(times[-1],  dt + PLOT_PAD + min_navg * FRAME_DUR)
    f_lo = max(0.0,        audio_hz - PLOT_BW)
    f_hi = min(freqs[-1],  audio_hz + PLOT_BW)

    t_mask = (times >= t_lo) & (times <= t_hi)
    f_mask = (freqs >= f_lo) & (freqs <= f_hi)
    S_crop = S_db[np.ix_(t_mask, f_mask)]
    t_crop = times[t_mask]
    f_crop = freqs[f_mask]

    # median-anchored dynamic range (matches gallery_corpus / MAP144 GUI)
    med  = float(np.median(S_crop))
    vmin = med - NOISE_HEADROOM
    vmax = med + NOISE_RANGE

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.pcolormesh(t_crop, f_crop, S_crop.T,
                  cmap=_cmap(), vmin=vmin, vmax=vmax,
                  shading="auto", rasterized=True)

    # mark the frames used in the best navmask
    for fi in range(min_navg):
        ft = dt + fi * FRAME_DUR
        ax.axvspan(ft, ft + FRAME_DUR, alpha=0.12, color="cyan")
    ax.axvline(dt, color="white", lw=0.8, ls=":", alpha=0.8, label="DT")
    ax.axhline(audio_hz, color="lime", lw=0.6, ls="--", alpha=0.5)

    label = "coherent integration" if min_navg >= 2 else "single-frame"
    if min_navg is None:
        label = "SPD fail (likely 5-7 frame)"

    ax.set_ylabel("Audio Hz")
    ax.set_xlabel("Time in period (s)")
    ax.set_xlim(t_lo, t_hi)
    dist_str = f"  {dec['dist_km']:.0f} km" if dec.get("dist_km") else ""
    grid_str = f" {dec['grid']}" if dec.get("grid") else ""
    ax.set_title(
        f"{wav_path.stem}  |  {dec['message']}{grid_str}{dist_str}\n"
        f"SNR {dec['snr']:+d} dB   DT {dt:.2f} s   "
        f"min_navg={min_navg}   xmax={xmax_norm:.2f}   "
        f"audio {audio_hz:.0f} Hz   [{label}]",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── ALL.TXT parsing ───────────────────────────────────────────────────────────

def _resolve_grid(message: str, qrz: QrzClient | None,
                  my_call: str) -> tuple[str | None, str]:
    """Return (grid4, source_label) for a message.

    Tries in order: (1) grid token in message text, (2) QRZ cache lookup of
    the non-self callsigns in the message.  Returns (None, '') if unresolvable.
    """
    grid = grid_from_message(message)
    if grid:
        return grid, "msg"
    if qrz is None:
        return None, ""
    for call in calls_from_message(message):
        if call == my_call.upper():
            continue
        try:
            rec = qrz.lookup(call)
        except (QrzError, Exception):
            continue
        if rec and rec.get("grid"):
            return rec["grid"][:4].upper(), f"qrz:{call}"
    return None, ""


def parse_all_txt(max_snr: int, min_dist_km: float,
                  my_grid: str, my_call: str,
                  qrz: QrzClient | None) -> dict[Path, list[dict]]:
    my_lat, my_lon = maidenhead_to_latlon(my_grid)
    by_wav: dict[Path, list[dict]] = collections.defaultdict(list)
    n_no_grid = 0
    n_too_close = 0
    with open(ALL_TXT) as fh:
        for line in fh:
            if "MSK144" not in line or " Rx " not in line:
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                snr  = int(parts[4])
                dt   = float(parts[5])
                freq = float(parts[6])
            except ValueError:
                continue
            if snr > max_snr:
                continue
            wav = period_wav(parts[0])
            if not wav.exists():
                continue
            message = " ".join(parts[7:]).strip()
            grid, grid_src = _resolve_grid(message, qrz, my_call)
            if min_dist_km > 0:
                if grid is None:
                    n_no_grid += 1
                    continue
                r_lat, r_lon = maidenhead_to_latlon(grid)
                dist_km = great_circle_km(my_lat, my_lon, r_lat, r_lon)
                if dist_km < min_dist_km:
                    n_too_close += 1
                    continue
            else:
                dist_km = 0.0
                if grid:
                    r_lat, r_lon = maidenhead_to_latlon(grid)
                    dist_km = great_circle_km(my_lat, my_lon, r_lat, r_lon)
            by_wav[wav].append(dict(
                ts=parts[0], snr=snr, dt=dt,
                audio_hz=freq,
                message=message,
                grid=grid or "",
                grid_src=grid_src,
                dist_km=dist_km,
            ))
    if min_dist_km > 0:
        print(f"  Distance filter: skipped {n_no_grid} (no grid) + "
              f"{n_too_close} (< {min_dist_km:.0f} km)")
    return dict(by_wav)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Gallery of MSK144 pings that needed multi-frame coherent integration")
    ap.add_argument("--min-navg", type=int, default=2,
                    help="Min navg to include (2=any coherent integration, default 2)")
    ap.add_argument("--max-snr",  type=int, default=5,
                    help="Skip decodes with SNR above this dB (default 5)")
    ap.add_argument("--min-dist-km", type=float, default=800.0,
                    help="Min great-circle distance in km (default 800); 0=no filter")
    ap.add_argument("--my-grid", type=str, default=MY_GRID,
                    help=f"Your Maidenhead grid square (default {MY_GRID})")
    ap.add_argument("--my-call", type=str, default=MY_CALL,
                    help=f"Your callsign (default {MY_CALL})")
    ap.add_argument("--qrz-lookup", action="store_true",
                    help="Look up missing grids via QRZ.com (requires credentials)")
    ap.add_argument("--include-fails", action="store_true",
                    help="Also include decodes where even 3-frame SPD fails")
    ap.add_argument("--max-plots", type=int, default=200)
    ap.add_argument("--out", type=Path, default=Path("scratch/long_ping_gallery"))
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    creds = load_credentials() if args.qrz_lookup else None
    qrz = QrzClient(
        username=creds[0] if creds else None,
        password=creds[1] if creds else None,
        allow_network=args.qrz_lookup,
    )

    print(f"Parsing ALL.TXT (SNR <= {args.max_snr} dB, "
          f"dist >= {args.min_dist_km:.0f} km from {args.my_grid}) …", flush=True)
    if args.qrz_lookup:
        print("  QRZ network lookups enabled for messages without a grid", flush=True)
    else:
        print(f"  QRZ cache-only fallback ({len(qrz.cache)} cached callsigns)", flush=True)
    by_wav = parse_all_txt(args.max_snr, args.min_dist_km,
                           args.my_grid, args.my_call, qrz)
    qrz.flush()
    n_wav = len(by_wav)
    n_dec = sum(len(v) for v in by_wav.values())
    print(f"  {n_wav} WAVs, {n_dec} decode candidates")

    candidates: list[dict] = []
    n_single = 0
    n_fail   = 0

    for i, (wav_path, decodes) in enumerate(sorted(by_wav.items())):
        if i % 250 == 0:
            print(f"  WAV {i}/{n_wav} …", flush=True)
        try:
            audio = load_wav(wav_path)
        except Exception:
            continue

        for dec in decodes:
            fc_off = dec["audio_hz"] - FC_JT9
            try:
                bb = to_baseband(audio, FC_JT9)
            except Exception:
                continue

            min_navg, xmax_norm = find_min_navg(bb, dec["dt"], fc_off)

            if min_navg is None:
                n_fail += 1
                if args.include_fails:
                    candidates.append(dict(
                        wav_path=wav_path, dec=dec,
                        min_navg=None, xmax_norm=0.0,
                    ))
            elif min_navg == 1:
                n_single += 1
            else:
                if min_navg >= args.min_navg:
                    candidates.append(dict(
                        wav_path=wav_path, dec=dec,
                        min_navg=min_navg, xmax_norm=xmax_norm,
                    ))

    print(f"\nResults (SNR <= {args.max_snr} dB):")
    print(f"  single-frame decodable : {n_single}")
    print(f"  coherent integration   : {len([c for c in candidates if c['min_navg']])} "
          f"(navg >= {args.min_navg})")
    print(f"  SPD fail (deep decode?): {n_fail}")
    print(f"  Total candidates       : {len(candidates)}")

    if not candidates:
        print("Nothing to plot.")
        return

    candidates.sort(key=lambda c: (-(c["min_navg"] or 0), c["dec"]["snr"]))

    rendered = 0
    for c in candidates:
        if rendered >= args.max_plots:
            break
        dec      = c["dec"]
        wav_path = c["wav_path"]
        navg_str = str(c["min_navg"]) if c["min_navg"] else "fail"
        out_png  = out_dir / f"{wav_path.stem}_{dec['audio_hz']:.0f}Hz_navg{navg_str}.png"
        print(f"  {wav_path.stem}  navg={navg_str}  SNR{dec['snr']:+d}  {dec['message']}")
        plot_ping(wav_path, dec, c["min_navg"] or 3, c["xmax_norm"], out_png)
        rendered += 1

    summary = out_dir / "summary.txt"
    with open(summary, "w") as sf:
        sf.write(f"MSK144 coherent-integration gallery\n")
        sf.write(f"decode_thresh={DECODE_THRESH}  min_navg={args.min_navg}  "
                 f"max_snr={args.max_snr} dB\n\n")
        sf.write(f"{'timestamp':<18} {'snr':>4} {'dt':>6} {'navg':>6} {'xmax':>6} {'km':>6}  message\n")
        sf.write("-" * 80 + "\n")
        for c in candidates:
            d = c["dec"]
            navg_s = str(c["min_navg"]) if c["min_navg"] else "fail"
            km_s = f"{d['dist_km']:.0f}" if d.get("dist_km") else "?"
            sf.write(f"{d['ts']:<18} {d['snr']:>4} {d['dt']:>6.2f} "
                     f"{navg_s:>6} {c['xmax_norm']:>6.2f} {km_s:>6}  {d['message']}\n")

    print(f"\nRendered {rendered} plot(s) → {out_dir}")
    print(f"Summary  → {summary}")


if __name__ == "__main__":
    main()
