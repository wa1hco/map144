#!/usr/bin/env python3
"""MSK144 decode sensitivity sweep — find MAP144's SNR decode threshold.

Generates realistic MSK144 pings at precisely known SNR using
generate_msk144.py's msk144code-based generator (exact symbol sequences),
applies the same IQ→audio chain that detection.py uses, runs jt9 on each,
and measures decode probability vs SNR.

This answers: "At what SNR does jt9 start failing to decode pings
that MAP144 has already detected?"

The synthetic signal uses the WSJT-X 2500 Hz bandwidth SNR convention, so
the resulting threshold is directly comparable to WSJT-X's own decode
sensitivity (accounting for the expected ~3–4 dB coherent-vs-incoherent
offset — see map144.py SNR reporting section).

Signal model
------------
One MSK144 ping per trial, Gamma-enveloped (τ = ``--width-ms``),
pure-H polarisation (worst-case for H-only receiver; add ``--random-pol``
to simulate random Faraday rotation).  White Gaussian noise at 48 kHz.
Signal placed at the MSK144 audio centre (1500 Hz) so no frequency
mixing is needed before jt9.

The WAV delivered to jt9 matches detection.py's layout exactly:
    [100 ms zeros] [500 ms AWGN noise] [1200 ms signal + noise] [100 ms zeros]

Modes
-----
Single sweep (default)::

    python decode_sensitivity.py [--snr-min -10] [--snr-max 4]
                                  [--snr-step 1] [--trials 50]
                                  [--msg MESSAGE] [--width-ms 150]
                                  [--random-pol] [--workers 4]
                                  [--results-out FILE.json] [--plot FILE.png]

Comparison of four standard scenarios (τ=150 ms and τ=50 ms × pure-H and
random-pol) in one run, produces a single overlay plot::

    python decode_sensitivity.py --compare [--snr-min -10] [--snr-max 6]
                                  [--trials 100] [--workers 4]
                                  [--plot sensitivity_compare.png]
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import decimate

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from generate_msk144 import generate_ping_msk144code
from map144_app.detection import JT9_BASE_ARGS
from map144_app.msk144_spd import msk144_spd_decode, FC_JT9

# ── Audio layout constants — must match detection.py ─────────────────────────
_IQ_RATE    = 48_000
_AUDIO_RATE = 12_000
_DECIMATE   = _IQ_RATE // _AUDIO_RATE      # 4
_FC_HZ      = 1500.0                       # jt9 audio centre

_PAD_S  = 0.100   # 100 ms zero pad each end
_PRE_S  = 0.500   # 500 ms pre-burst noise baseline
_POST_S = 1.200   # 1200 ms post-detection capture

_NOISE_SIGMA = 0.01   # AWGN amplitude; SNR is calibrated relative to this


# ── Single-trial worker ───────────────────────────────────────────────────────

def run_trial(msg: str, snr_db: float, seed: int,
              width_ms: int = 150, random_pol: bool = False,
              flat_frame: bool = False) -> dict:
    """Run one decode trial at the given SNR.  Returns a result dict."""
    rng = np.random.default_rng(seed)
    pol = 'random' if random_pol else 'pure-H'

    # Generate the ping signal at 48 kHz IQ (no noise) with _PRE_S of lead
    # silence so that after AWGN is added there is a clean 500 ms noise baseline.
    delay_s = _PRE_S
    ping_h, _ping_v, pol_meta = generate_ping_msk144code(
        msg,
        center_hz=_FC_HZ,
        delay_s=delay_s,
        snr_db=snr_db,
        noise_sigma=_NOISE_SIGMA,
        width_ms=width_ms,
        pol_scenario=pol,
        rng=rng,
        sample_rate=_IQ_RATE,
        flat_frame=flat_frame,
    )

    # Trim / extend to [pre + post] IQ samples
    total_iq = int((_PRE_S + _POST_S) * _IQ_RATE)
    if len(ping_h) < total_iq:
        ping_h = np.pad(ping_h, (0, total_iq - len(ping_h)))
    else:
        ping_h = ping_h[:total_iq]

    # Add complex AWGN
    noise_iq = (rng.standard_normal(total_iq) +
                1j * rng.standard_normal(total_iq)).astype(np.complex64) * _NOISE_SIGMA
    iq_noisy = ping_h + noise_iq

    # IQ → audio: Re(IQ) → decimate 48k → 12k (signal already centred at 1500 Hz)
    audio = decimate(np.real(iq_noisy).astype(np.float64), _DECIMATE,
                     zero_phase=False).astype(np.float32)

    # 100 ms zero pads (matches detection.py's iq_padded layout)
    pad_n = int(_PAD_S * _AUDIO_RATE)
    audio = np.concatenate([np.zeros(pad_n, dtype=np.float32),
                            audio,
                            np.zeros(pad_n, dtype=np.float32)])

    # Peak-normalise to 0.9 — same as detection.py
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.9

    # Write 16-bit WAV and run jt9
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(_AUDIO_RATE)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        result = subprocess.run(
            JT9_BASE_ARGS + [tmp_path],
            capture_output=True, text=True, timeout=15,
        )
        decoded_msg = None
        jt9_snr     = None
        for line in result.stdout.splitlines():
            tokens = line.split()
            if len(tokens) >= 6:
                decoded_msg = ' '.join(tokens[5:]).strip()
                try:
                    jt9_snr = int(tokens[1])
                except (ValueError, IndexError):
                    pass
                break
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        'snr_db':      snr_db,
        'seed':        seed,
        'success':     decoded_msg is not None,
        'decoded_msg': decoded_msg,
        'jt9_snr':     jt9_snr,
        'theta_deg':   pol_meta.get('theta0_deg'),
    }


def run_trial_spd(msg: str, snr_db: float, seed: int,
                  width_ms: int = 150, random_pol: bool = False,
                  flat_frame: bool = False,
                  use_complex_iq: bool = False) -> dict:
    """Like run_trial but uses msk144_spd_decode (coherent frame averaging).

    use_complex_iq=True: mix 48 kHz IQ to baseband and decimate both I and Q
    channels before calling msk144_spd_decode.  This bypasses the Hilbert
    transform inside SPD, recovering the ~3 dB noise-doubling penalty of the
    real-audio path.
    """
    rng = np.random.default_rng(seed)
    pol = 'random' if random_pol else 'pure-H'

    delay_s = _PRE_S
    ping_h, _ping_v, pol_meta = generate_ping_msk144code(
        msg,
        center_hz=_FC_HZ,
        delay_s=delay_s,
        snr_db=snr_db,
        noise_sigma=_NOISE_SIGMA,
        width_ms=width_ms,
        pol_scenario=pol,
        rng=rng,
        sample_rate=_IQ_RATE,
        flat_frame=flat_frame,
    )

    total_iq = int((_PRE_S + _POST_S) * _IQ_RATE)
    if len(ping_h) < total_iq:
        ping_h = np.pad(ping_h, (0, total_iq - len(ping_h)))
    else:
        ping_h = ping_h[:total_iq]

    noise_iq = (rng.standard_normal(total_iq) +
                1j * rng.standard_normal(total_iq)).astype(np.complex64) * _NOISE_SIGMA
    iq_noisy = ping_h + noise_iq

    pad_n = int(_PAD_S * _AUDIO_RATE)

    if use_complex_iq:
        # Mix 48 kHz IQ to 0 Hz (baseband) then decimate both I and Q.
        # This avoids the Hilbert noise-doubling penalty inside msk144_spd_decode.
        t_48k = np.arange(total_iq, dtype=np.float64) / _IQ_RATE
        iq_bb = (iq_noisy * np.exp(-2j * np.pi * _FC_HZ * t_48k)).astype(np.complex64)
        i_dec = decimate(np.real(iq_bb).astype(np.float64), _DECIMATE, zero_phase=False)
        q_dec = decimate(np.imag(iq_bb).astype(np.float64), _DECIMATE, zero_phase=False)
        iq_12k = (i_dec + 1j * q_dec).astype(np.complex64)
        iq_padded = np.concatenate([
            np.zeros(pad_n, dtype=np.complex64),
            iq_12k,
            np.zeros(pad_n, dtype=np.complex64),
        ])
        try:
            decoded_msg, jt9_snr, navg, _fest, xmax_norm = msk144_spd_decode(
                complex_baseband=iq_padded,
            )
        except Exception:
            decoded_msg = None
            jt9_snr     = None
            navg        = 0
            xmax_norm   = 0.0
    else:
        audio = decimate(np.real(iq_noisy).astype(np.float64), _DECIMATE,
                         zero_phase=False).astype(np.float32)
        audio = np.concatenate([np.zeros(pad_n, dtype=np.float32),
                                audio,
                                np.zeros(pad_n, dtype=np.float32)])
        peak = float(np.max(np.abs(audio)))
        if peak > 0:
            audio = audio / peak * 0.9
        try:
            decoded_msg, jt9_snr, navg, _fest, xmax_norm = msk144_spd_decode(
                audio, fc=FC_JT9, fs=_AUDIO_RATE,
            )
        except Exception:
            decoded_msg = None
            jt9_snr     = None
            navg        = 0
            xmax_norm   = 0.0

    return {
        'snr_db':      snr_db,
        'seed':        seed,
        'success':     decoded_msg is not None,
        'decoded_msg': decoded_msg,
        'jt9_snr':     jt9_snr,
        'navg':        navg,
        'xmax_norm':   xmax_norm,
        'theta_deg':   pol_meta.get('theta0_deg'),
    }


def run_sweep_spd(msg: str, snr_range, n_trials: int,
                  width_ms: int = 150, random_pol: bool = False,
                  workers: int = 4, flat_frame: bool = False,
                  use_complex_iq: bool = False) -> list[dict]:
    """SPD-path sweep — mirrors run_sweep but calls run_trial_spd."""
    tasks = [
        (msg, snr, seed, width_ms, random_pol, flat_frame, use_complex_iq)
        for snr in snr_range
        for seed in range(n_trials)
    ]

    print(f"  {'SNR':>6}  {'trials':>6}  {'decoded':>7}  {'prob':>6}  {'navg':>6}  {'sync pk':>8}  {'jt9 SNR':>8}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}")

    done_by_snr: dict[float, list] = {snr: [] for snr in snr_range}
    results: list[dict] = []

    if workers == 1:
        for args in tasks:
            try:
                r = run_trial_spd(*args)
            except Exception:
                r = {'snr_db': args[1], 'seed': args[2], 'success': False,
                     'decoded_msg': None, 'jt9_snr': None, 'navg': 0, 'xmax_norm': 0.0, 'theta_deg': None}
            done_by_snr[r['snr_db']].append(r)
            results.append(r)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_trial_spd, *args): args for args in tasks}
            for f in concurrent.futures.as_completed(futures):
                try:
                    r = f.result()
                except Exception:
                    args = futures[f]
                    r = {'snr_db': args[1], 'seed': args[2], 'success': False,
                         'decoded_msg': None, 'jt9_snr': None, 'navg': 0, 'xmax_norm': 0.0, 'theta_deg': None}
                done_by_snr[r['snr_db']].append(r)
                results.append(r)

    for snr in snr_range:
        batch     = done_by_snr[snr]
        n_ok      = sum(1 for r in batch if r['success'])
        prob      = n_ok / len(batch) if batch else 0.0
        navgs     = [r['navg']     for r in batch if r.get('navg')]
        xmaxs     = [r['xmax_norm'] for r in batch if r.get('xmax_norm')]
        jt9snrs   = [r['jt9_snr']  for r in batch if r.get('jt9_snr') is not None]
        mean_navg = f"{np.mean(navgs):.1f}"   if navgs   else "  —"
        mean_xmax = f"{np.mean(xmaxs):.2f}"  if xmaxs   else "  —"
        mean_jt9  = f"{np.mean(jt9snrs):+.1f}" if jt9snrs else "  —"
        print(f"  {snr:+6.1f}  {len(batch):6d}  {n_ok:7d}  {prob:6.2f}  {mean_navg:>6}  {mean_xmax:>8}  {mean_jt9:>8}")

    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def results_to_probs(results: list[dict]) -> dict[float, float]:
    by_snr: dict[float, list] = {}
    for r in results:
        by_snr.setdefault(r['snr_db'], []).append(r)
    return {snr: sum(r['success'] for r in rs) / len(rs)
            for snr, rs in by_snr.items()}


def find_snr50(probs: dict[float, float]) -> float | None:
    snrs = sorted(probs)
    ps   = [probs[s] for s in snrs]
    for i in range(len(snrs) - 1):
        if ps[i] <= 0.5 <= ps[i + 1] or ps[i] >= 0.5 >= ps[i + 1]:
            if ps[i + 1] == ps[i]:
                return snrs[i]
            frac = (0.5 - ps[i]) / (ps[i + 1] - ps[i])
            return snrs[i] + frac * (snrs[i + 1] - snrs[i])
    return None


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(msg: str, snr_range, n_trials: int,
              width_ms: int = 150, random_pol: bool = False,
              workers: int = 4, flat_frame: bool = False) -> list[dict]:
    """Run the full SNR sweep; return list of per-trial result dicts."""
    tasks = [
        (msg, snr, seed, width_ms, random_pol, flat_frame)
        for snr in snr_range
        for seed in range(n_trials)
    ]

    print(f"  {'SNR':>6}  {'trials':>6}  {'decoded':>7}  {'prob':>6}  {'jt9 SNR (mean)':>15}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*15}")

    done_by_snr: dict[float, list] = {snr: [] for snr in snr_range}
    results: list[dict] = []

    if workers == 1:
        for args in tasks:
            try:
                r = run_trial(*args)
            except Exception:
                r = {'snr_db': args[1], 'seed': args[2], 'success': False,
                     'decoded_msg': None, 'jt9_snr': None, 'theta_deg': None}
            done_by_snr[r['snr_db']].append(r)
            results.append(r)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_trial, *args): args for args in tasks}
            for f in concurrent.futures.as_completed(futures):
                try:
                    r = f.result()
                except Exception:
                    args = futures[f]
                    r = {'snr_db': args[1], 'seed': args[2], 'success': False,
                         'decoded_msg': None, 'jt9_snr': None, 'theta_deg': None}
                done_by_snr[r['snr_db']].append(r)
                results.append(r)

    for snr in snr_range:
        batch = done_by_snr[snr]
        n_ok  = sum(1 for r in batch if r['success'])
        prob  = n_ok / len(batch) if batch else 0.0
        jt9s  = [r['jt9_snr'] for r in batch if r['jt9_snr'] is not None]
        mean_jt9 = f"{np.mean(jt9s):+.1f} dB" if jt9s else "  —"
        print(f"  {snr:+6.1f}  {len(batch):6d}  {n_ok:7d}  {prob:6.2f}  {mean_jt9:>15}")

    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def _single_plot(probs: dict, out_path: str, msg: str, width_ms: int,
                 random_pol: bool, snr50: float | None, title_extra: str = ''):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    snrs = sorted(probs)
    ps   = [probs[s] for s in snrs]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#f8f8f8')

    ax.plot(snrs, ps, 'o-', color='steelblue', linewidth=2, markersize=6,
            label='MAP144 / jt9 decode probability')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='50% threshold')

    if snr50 is not None:
        ax.axvline(snr50, color='steelblue', linestyle=':', linewidth=1.5,
                   label=f'50% point: {snr50:+.1f} dB')

    pol_label = 'random pol.' if random_pol else 'pure-H'
    extra = f'  {title_extra}' if title_extra else f'  τ={width_ms} ms  {pol_label}'
    ax.set_title(f'MAP144 MSK144 decode sensitivity\nmsg={msg!r}{extra}', fontsize=11)
    ax.set_xlabel('True SNR  (dB, WSJT-X 2500 Hz reference)', fontsize=11)
    ax.set_ylabel('Decode probability', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(linestyle=':', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved → {out_path}")
    plt.close(fig)


def _compare_plot(scenarios: list[dict], out_path: str, msg: str):
    """Multi-curve presentation plot.

    scenarios — list of dicts with keys:
        label, probs (snr→prob dict), snr50, color, marker, linestyle
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#f8f8f8')

    ax.axhline(0.5, color='#888', linestyle='--', linewidth=1, zorder=1)

    for sc in scenarios:
        snrs = sorted(sc['probs'])
        ps   = [sc['probs'][s] for s in snrs]
        ax.plot(snrs, ps,
                marker=sc['marker'], linestyle=sc['linestyle'],
                color=sc['color'], linewidth=2, markersize=6,
                label=sc['label'], zorder=3)
        if sc['snr50'] is not None:
            ax.axvline(sc['snr50'], color=sc['color'],
                       linestyle=':', linewidth=1, alpha=0.6, zorder=2)
            ax.text(sc['snr50'] + 0.1, 0.07,
                    f"{sc['snr50']:+.1f} dB",
                    color=sc['color'], fontsize=8, va='bottom')

    ax.set_title(
        f'MAP144 MSK144 decode sensitivity — scenario comparison\n'
        f'msg={msg!r}   SNR ref: WSJT-X 2500 Hz bandwidth',
        fontsize=11,
    )
    ax.set_xlabel('True SNR  (dB)', fontsize=11)
    ax.set_ylabel('Decode probability', fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(linestyle=':', alpha=0.4)

    # Second x-axis showing WSJT-X-reported SNR (offset +3.7 dB)
    ax2 = ax.twiny()
    x1lo, x1hi = ax.get_xlim()
    ax2.set_xlim(x1lo + 3.7, x1hi + 3.7)
    ax2.set_xlabel('Approx. WSJT-X reported SNR  (dB)', fontsize=10, color='#555')
    ax2.tick_params(colors='#555')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Comparison plot saved → {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

_COMPARE_SCENARIOS = [
    # (width_ms, random_pol, label, color, marker, linestyle)
    (150, False, 'τ=150 ms, pure-H  (best case)',       'steelblue',  'o', '-'),
    (150, True,  'τ=150 ms, random pol.',               '#d62728',    's', '--'),
    ( 50, False, 'τ=50 ms,  pure-H  (short ping)',      '#2ca02c',    '^', '-'),
    ( 50, True,  'τ=50 ms,  random pol.  (worst case)', '#ff7f0e',    'D', '--'),
]


def _run_spd(args, snr_range, use_complex_iq: bool = False):
    """SPD sweep with Gamma envelope — measures multi-frame averaging benefit."""
    pol_label = 'random polarisation' if args.random_pol else 'pure-H (best case)'
    iq_label  = 'complex IQ (no Hilbert)' if use_complex_iq else 'real audio (Hilbert)'
    print('MSK144 decode sensitivity — SPD coherent averaging decoder')
    print(f'  message  : {args.msg!r}')
    print(f'  τ        : {args.width_ms} ms')
    print(f'  pol      : {pol_label}')
    print(f'  input    : {iq_label}')
    print(f'  SNR range: {args.snr_min:+.0f} to {args.snr_max:+.0f} dB  '
          f'step {args.snr_step:.1f} dB  ({len(snr_range)} levels × {args.trials} trials)')
    print(f'  workers  : {args.workers}\n')

    results = run_sweep_spd(args.msg, snr_range, args.trials,
                            width_ms=args.width_ms,
                            random_pol=args.random_pol,
                            workers=args.workers,
                            use_complex_iq=use_complex_iq)

    if args.results_out:
        Path(args.results_out).write_text(json.dumps(results, indent=2))
        print(f'Results saved → {args.results_out}')

    probs = results_to_probs(results)
    snr50 = find_snr50(probs)

    if snr50 is not None:
        print(f'\n50% decode probability at SNR ≈ {snr50:+.1f} dB')
        print(f'(WSJT-X reported SNR at threshold ≈ {snr50 + 3.7:+.1f} dB)')
    elif sorted(probs.values())[0] > 0.5:
        print('\nDecode probability > 50% across full range — try lower --snr-min')
    else:
        print('\nDecode probability < 50% across full range — try higher --snr-max')

    if args.plot:
        _single_plot(probs, args.plot, args.msg, args.width_ms, args.random_pol, snr50,
                     title_extra='SPD coherent averaging')


def _run_spd_flat(args, snr_range, use_complex_iq: bool = False):
    """SPD single flat-frame sweep — direct comparison with QEX Figure 4."""
    iq_label = 'complex IQ (no Hilbert)' if use_complex_iq else 'real audio (Hilbert)'
    print('MSK144 decode sensitivity — SPD decoder, flat frame (QEX reference)')
    print(f'  message  : {args.msg!r}')
    print(f'  envelope : flat, single 72 ms frame, constant amplitude')
    print(f'  decoder  : msk144_spd coherent averaging')
    print(f'  input    : {iq_label}')
    print(f'  SNR range: {args.snr_min:+.0f} to {args.snr_max:+.0f} dB  '
          f'step {args.snr_step:.1f} dB  ({len(snr_range)} levels × {args.trials} trials)')
    print(f'  workers  : {args.workers}\n')

    results = run_sweep_spd(args.msg, snr_range, args.trials,
                            workers=args.workers, flat_frame=True,
                            use_complex_iq=use_complex_iq)

    if args.results_out:
        Path(args.results_out).write_text(json.dumps(results, indent=2))
        print(f'Results saved → {args.results_out}')

    probs = results_to_probs(results)
    snr50 = find_snr50(probs)

    if snr50 is not None:
        print(f'\n50% decode probability at SNR ≈ {snr50:+.1f} dB')
        print(f'(WSJT-X reported SNR at threshold ≈ {snr50 + 3.7:+.1f} dB)')
        print(f'QEX reference: 50% at ≈ −1.4 dB  →  gap = {snr50 - (-1.4):+.1f} dB')
    else:
        print('\nDid not cross 50% — adjust --snr-min / --snr-max')

    if args.plot:
        _single_plot(probs, args.plot, args.msg,
                     width_ms=0, random_pol=False, snr50=snr50,
                     title_extra='SPD decoder, flat frame (QEX reference)')


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument('--compare', action='store_true',
                    help='Run all four standard scenarios and produce a comparison plot')
    ap.add_argument('--snr-min',    type=float, default=-10.0)
    ap.add_argument('--snr-max',    type=float, default=+6.0)
    ap.add_argument('--snr-step',   type=float, default=1.0)
    ap.add_argument('--trials',     type=int,   default=50,
                    help='AWGN trials per SNR level (default 50)')
    ap.add_argument('--msg',        default='CQ K2GV FN30',
                    help='MSK144 message to encode (default: CQ K2GV FN30)')
    ap.add_argument('--width-ms',   type=int, default=150,
                    help='Single-sweep only: Gamma τ in ms (default 150)')
    ap.add_argument('--random-pol', action='store_true',
                    help='Single-sweep only: random Faraday rotation each trial')
    ap.add_argument('--flat', action='store_true',
                    help='Use flat-amplitude single frame (no Gamma envelope) — '
                         'matches QEX/WSJT-X sensitivity test conditions')
    ap.add_argument('--spd', action='store_true',
                    help='Use msk144_spd coherent frame-averaging decoder instead of '
                         'plain jt9; can be combined with --flat for apples-to-apples '
                         'single-frame comparison or with Gamma envelope for multi-frame '
                         'averaging benefit')
    ap.add_argument('--spd-iq', action='store_true',
                    help='Like --spd but feeds complex baseband IQ directly to SPD, '
                         'bypassing the Hilbert transform.  Recovers the ~3 dB noise-'
                         'doubling penalty of the real-audio path.  Requires --spd or '
                         '--flat --spd.')
    ap.add_argument('--workers',    type=int, default=4,
                    help='Parallel jt9 processes (default 4)')
    ap.add_argument('--results-out', default=None, metavar='FILE',
                    help='Save raw trial results to JSON (single sweep)')
    ap.add_argument('--plot',       default=None, metavar='FILE',
                    help='Save plot to FILE')
    args = ap.parse_args()

    snr_range = list(np.arange(args.snr_min, args.snr_max + 0.01, args.snr_step))

    use_iq = getattr(args, 'spd_iq', False)
    if args.compare:
        _run_compare(args, snr_range)
    elif args.flat and (args.spd or use_iq):
        _run_spd_flat(args, snr_range, use_complex_iq=use_iq)
    elif args.flat:
        _run_flat(args, snr_range)
    elif args.spd or use_iq:
        _run_spd(args, snr_range, use_complex_iq=use_iq)
    else:
        _run_single(args, snr_range)


def _run_single(args, snr_range):
    pol_label = 'random polarisation' if args.random_pol else 'pure-H (best case)'
    print('MSK144 decode sensitivity sweep')
    print(f'  message  : {args.msg!r}')
    print(f'  τ        : {args.width_ms} ms')
    print(f'  pol      : {pol_label}')
    print(f'  SNR range: {args.snr_min:+.0f} to {args.snr_max:+.0f} dB  '
          f'step {args.snr_step:.1f} dB  ({len(snr_range)} levels × {args.trials} trials)')
    print(f'  workers  : {args.workers}\n')

    results = run_sweep(args.msg, snr_range, args.trials,
                        width_ms=args.width_ms,
                        random_pol=args.random_pol,
                        workers=args.workers)

    if args.results_out:
        Path(args.results_out).write_text(json.dumps(results, indent=2))
        print(f'Results saved → {args.results_out}')

    probs = results_to_probs(results)
    snr50 = find_snr50(probs)

    if snr50 is not None:
        print(f'\n50% decode probability at SNR ≈ {snr50:+.1f} dB')
        print(f'(WSJT-X reported SNR at threshold ≈ {snr50 + 3.7:+.1f} dB)')
    elif sorted(probs.values())[0] > 0.5:
        print('\nDecode probability > 50% across full range — try lower --snr-min')
    else:
        print('\nDecode probability < 50% across full range — try higher --snr-max')

    if args.plot:
        _single_plot(probs, args.plot, args.msg, args.width_ms, args.random_pol, snr50)


def _run_flat(args, snr_range):
    """Single sweep with flat-amplitude single frame — QEX apples-to-apples test."""
    print('MSK144 decode sensitivity — flat frame (QEX reference)')
    print(f'  message  : {args.msg!r}')
    print(f'  envelope : flat, single 72 ms frame, constant amplitude')
    print(f'  SNR range: {args.snr_min:+.0f} to {args.snr_max:+.0f} dB  '
          f'step {args.snr_step:.1f} dB  ({len(snr_range)} levels × {args.trials} trials)')
    print(f'  workers  : {args.workers}\n')

    results = run_sweep(args.msg, snr_range, args.trials,
                        workers=args.workers, flat_frame=True)

    if args.results_out:
        Path(args.results_out).write_text(json.dumps(results, indent=2))
        print(f'Results saved → {args.results_out}')

    probs = results_to_probs(results)
    snr50 = find_snr50(probs)

    if snr50 is not None:
        print(f'\n50% decode probability at SNR ≈ {snr50:+.1f} dB')
        print(f'(WSJT-X reported SNR at threshold ≈ {snr50 + 3.7:+.1f} dB)')
        print(f'QEX reference: 50% at ≈ −1.4 dB  →  gap = {snr50 - (-1.4):+.1f} dB')
    else:
        print('\nDid not cross 50% — adjust --snr-min / --snr-max')

    if args.plot:
        _single_plot(probs, args.plot, args.msg,
                     width_ms=0, random_pol=False, snr50=snr50,
                     title_extra='flat frame (QEX reference)')


def _run_compare(args, snr_range):
    print('MSK144 decode sensitivity — comparison sweep')
    print(f'  message  : {args.msg!r}')
    print(f'  SNR range: {args.snr_min:+.0f} to {args.snr_max:+.0f} dB  '
          f'step {args.snr_step:.1f} dB  ({len(snr_range)} levels × {args.trials} trials)')
    print(f'  workers  : {args.workers}')
    print(f'  scenarios: {len(_COMPARE_SCENARIOS)}\n')

    scenario_results = []
    for width_ms, random_pol, label, color, marker, linestyle in _COMPARE_SCENARIOS:
        pol_str = 'random-pol' if random_pol else 'pure-H'
        print(f'── {label} ──')
        results = run_sweep(args.msg, snr_range, args.trials,
                            width_ms=width_ms,
                            random_pol=random_pol,
                            workers=args.workers)
        probs = results_to_probs(results)
        snr50 = find_snr50(probs)
        if snr50 is not None:
            print(f'   50% at {snr50:+.1f} dB  (WSJT-X ≈ {snr50 + 3.7:+.1f} dB)\n')
        else:
            print()
        scenario_results.append({
            'label': label, 'probs': probs, 'snr50': snr50,
            'color': color, 'marker': marker, 'linestyle': linestyle,
        })

    print('─' * 50)
    print('Summary:')
    for sc in scenario_results:
        s50 = f"{sc['snr50']:+.1f} dB" if sc['snr50'] is not None else '  —'
        print(f"  {sc['label']:<40}  50% at {s50}")

    if args.plot:
        _compare_plot(scenario_results, args.plot, args.msg)


if __name__ == '__main__':
    main()
