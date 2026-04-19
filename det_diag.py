#!/home/jeff/ham/map144/.venv/bin/python3
"""Offline detection metric diagnostic for short50.wav.

For each manifest ping, compute the peak pair_metric and hop-above-threshold
count in the correct channel, bypassing the coincidence gates. Shows whether
the issue is (a) metric never reaches threshold or (b) gate suppression.

Usage:
    python det_diag.py [--manifest ...] [--wav ...]
"""
import json
import sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile as sio

from map144_app.channelizer import (apply_channelizer, N_CHANNELS, CH_SAMPLE_RATE,
                                     make_channelizer_state, design_channelizer_filter)
from map144_app.processing import (
    CH_DETECT_SIZE, _CH_DETECT_HOP,
    _DETECT_WINDOW, _SQ_FREQ, _LO_MASK, _HI_MASK,
    _METRIC_HIST_DEPTH, _METRIC_HIST_STRIDE,
    DETECT_THRESH_DB,
)

WAV  = 'MSK144/sim/short50.wav'
JSON = 'MSK144/sim/short50.json'

# ── Load WAV ──────────────────────────────────────────────────────────────────
fs_wav, data = sio.read(WAV)
if data.ndim == 2:
    iq_full = (data[:, 0] + 1j * data[:, 1]).astype(np.complex64)
else:
    iq_full = data.astype(np.complex64)

print(f"WAV: {len(iq_full)/fs_wav:.1f}s @ {fs_wav} Hz  complex IQ")

with open(JSON) as f:
    manifest = json.load(f)
placements = sorted(manifest['placements'], key=lambda p: p['delay_s'])

# ── Channel state and accumulation buffer ─────────────────────────────────────
CHUNK_SIZE_IQ = 2048
# channel_offset_hz = 1500 Hz for simulation file (pan_center == calling_freq)
_lp_taps = design_channelizer_filter(fs_wav)
ch_state  = make_channelizer_state(N_CHANNELS, _lp_taps, fs_wav, channel_offset_hz=1500.0)
ch_buf     = np.zeros((N_CHANNELS, CH_DETECT_SIZE * 4), dtype=np.complex64)
ch_buf_end = 0

# Rolling history for 25th-percentile baseline
metric_hist_buf  = np.zeros((_METRIC_HIST_DEPTH, N_CHANNELS), dtype=np.float32)
metric_hist_idx  = 0
metric_hist_cnt  = 0
pct25 = np.ones(N_CHANNELS, dtype=np.float32) * 1e-30
pct25_ctr = 0

# Accumulate (hop_time, pair_metric[ch]) for all hops
all_hop_times   = []
all_hop_metrics = []

sample_abs = 0

pos = 0
while pos < len(iq_full):
    chunk = iq_full[pos:pos + CHUNK_SIZE_IQ].astype(np.complex64)
    pos  += len(chunk)

    ch_out = apply_channelizer(chunk, ch_state)
    new_n  = ch_out.shape[1]

    cap = ch_buf.shape[1]
    if ch_buf_end + new_n > cap:
        keep = cap - new_n
        if keep > 0:
            ch_buf[:, :keep] = ch_buf[:, cap - keep:cap]
        ch_buf_end = max(keep, 0)
    ch_buf[:, ch_buf_end:ch_buf_end + new_n] = ch_out
    ch_buf_end += new_n

    while ch_buf_end >= CH_DETECT_SIZE:
        block = ch_buf[:, :CH_DETECT_SIZE]
        sq    = block ** 2
        X_sq  = np.fft.fft(sq * _DETECT_WINDOW[np.newaxis, :], axis=1)
        plin  = np.abs(X_sq) / CH_DETECT_SIZE
        plin_sh = np.fft.fftshift(plin, axes=1)
        lo_peak = np.max(plin_sh[:, _LO_MASK], axis=1)
        hi_peak = np.max(plin_sh[:, _HI_MASK], axis=1)
        raw_lin = (lo_peak + hi_peak) / 2.0

        metric_hist_buf[metric_hist_idx] = raw_lin
        metric_hist_idx = (metric_hist_idx + 1) % _METRIC_HIST_DEPTH
        if metric_hist_cnt < _METRIC_HIST_DEPTH:
            metric_hist_cnt += 1

        pct25_ctr += 1
        if pct25_ctr % 10 == 0 or metric_hist_cnt < 10:
            hist   = (metric_hist_buf[:metric_hist_cnt]
                      if metric_hist_cnt < _METRIC_HIST_DEPTH
                      else metric_hist_buf)
            hist_s = hist[::_METRIC_HIST_STRIDE]
            k      = max(0, int(len(hist_s) * 0.25))
            pct25  = np.maximum(np.partition(hist_s, k, axis=0)[k], 1e-30)

        pair_metric = (10.0 * np.log10(np.maximum(raw_lin / pct25, 1e-30))).astype(np.float32)
        t_hop       = (sample_abs + _CH_DETECT_HOP // 2) / CH_SAMPLE_RATE

        all_hop_times.append(t_hop)
        all_hop_metrics.append(pair_metric.copy())

        ch_buf[:, :ch_buf_end - _CH_DETECT_HOP] = ch_buf[:, _CH_DETECT_HOP:ch_buf_end]
        ch_buf_end  -= _CH_DETECT_HOP
        sample_abs  += _CH_DETECT_HOP

hop_times   = np.array(all_hop_times)
hop_metrics = np.column_stack(all_hop_metrics).T  # (N_HOPS, N_CHANNELS)

print(f"Processed {len(hop_times)} hops  ({len(hop_times)*_CH_DETECT_HOP/CH_SAMPLE_RATE:.1f}s equivalent)\n")

# ── Per-ping analysis ─────────────────────────────────────────────────────────
# channel index: ch_signed % N_CHANNELS  (signed channel → array index)
print(f"{'#':>3}  {'msg':<14}  {'t':>5}  {'ch':>4}  {'snr':>4}  {'w':>5}  "
      f"{'pk_met':>7}  {'n>thr':>5}  note")
print("─" * 78)

CHANNEL_OFFSET_HZ = 1500.0   # (calling_freq - pan_center) + 1500; = 1500 for sim file

for i, p in enumerate(placements):
    ch_s   = round((p['center_hz'] - CHANNEL_OFFSET_HZ) / 1000)
    ch_idx = ch_s % N_CHANNELS
    t0     = p['delay_s']
    tau_s  = p['width_ms'] / 1000.0
    t1     = t0 + 5 * tau_s + 0.1   # 5τ + 100 ms guard

    mask = (hop_times >= t0 - 0.05) & (hop_times <= t1)
    if not np.any(mask):
        note = "no hops"
        print(f"{i+1:3d}  {p['msg']:<14}  {t0:5.2f}  {ch_s:+4d}  "
              f"{p['snr_db']:+3d}dB  {p['width_ms']:4d}ms  {'---':>7}  {'---':>5}  {note}")
        continue

    m_win = hop_metrics[mask, ch_idx]
    pk    = float(np.max(m_win))
    n_ab  = int(np.sum(m_win > DETECT_THRESH_DB))

    # Check simultaneous per-hop channel count (accurate gate model)
    # Gate fires if ≥ _COINCIDENCE_MAX_CH channels exceed threshold in the same hop.
    max_simultaneous = int(np.max(np.sum(hop_metrics[mask, :] > DETECT_THRESH_DB, axis=1)))

    note = ""
    if pk < DETECT_THRESH_DB:
        note = "BELOW_THRESH"
    elif max_simultaneous >= 8:
        note = f"GATE_RISK: {max_simultaneous} ch in 1 hop"

    print(f"{i+1:3d}  {p['msg']:<14}  {t0:5.2f}  {ch_s:+4d}  "
          f"{p['snr_db']:+3d}dB  {p['width_ms']:4d}ms  "
          f"{pk:>+7.1f}dB  {n_ab:>5}  {note}")

print(f"\nThreshold = {DETECT_THRESH_DB} dB above 25th-pct baseline")
print(f"Gate fires if ≥ 8 fresh channels exceed threshold simultaneously")
