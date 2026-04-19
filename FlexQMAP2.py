#!/usr/bin/env python3
# Copyright (c) 2025 Thomas S. Knutsen, LA3PNA
# Copyright (C) 2026 Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# FlexQMAP2.py is a derived work from qmap_flex.py by Thomas S. Knutsen, LA3PNA.
# The TIMF2 packet format, preprocessing pipeline (DC block, gain, IQ swap/invert,
# complex frequency mix), and CLI interface are based on qmap_flex.py.
# The radio interface is rewritten to use the map144 flexclient library
# (VITA-49 UDP, no audio driver) in place of sounddevice/WASAPI.
#
# Private use only. Not licensed for redistribution.
"""
FlexQMAP2.py — Flex DAXIQ -> QMAP (Linrad TIMF2 UDP) bridge

Uses the map144 flexclient library to receive IQ samples directly from the
FlexRadio VITA-49 UDP stream (no sounddevice / audio driver needed).
Packs samples into Linrad TIMF2 packets and sends them to QMAP via UDP.

Packet format: 24-byte TIMF2 header + 174 * 8 bytes payload = 1416 bytes.
Center frequency is tracked live from SmartSDR pan/slice status messages.

Usage:
    python FlexQMAP2.py --radio 192.168.1.100 [options]

Other useful flags:
    --daxiq N           DAX IQ channel (default 1)
    --host H            QMAP host (default 127.0.0.1)
    --port P            QMAP UDP port (default 50004)
    --gain G            Linear gain applied before packing (default 1.0)
    --mix HZ            Complex frequency shift in Hz (default 0, disabled)
    --dcblock           Remove DC per block
    --swapiq / --noswapiq  Swap I and Q (default: no swap)
    --invertq           Negate Q channel
    --nrx {-1,2}        TIMF2 nrx field for float32 mode (default -1)
    --int16             Use int16 payload (default, required by QMAP)
    --int16_scale N     Full-scale integer for int16 mode (default 8192)
    --iptrinc           Increment iptr by 174 per packet
    --iblk_zero         Force iblk=0 in header
    --rms               Print RMS/max once per second
    --debug             Verbose logging
"""

import argparse
import math
import socket
import struct
import sys
import time
from datetime import datetime, timezone

import numpy as np

# map144 flex client
from flexclient.client import FlexDAXIQ

# ── TIMF2 packet constants ────────────────────────────────────────────────────
PKT_SAMPLES     = 174                           # samples per packet
BYTES_PER_PKT   = 24 + PKT_SAMPLES * 8         # 1416 bytes
HEADER_FMT      = "<d i f i H b c"             # little-endian
# cfreq(double,8) msec(int32,4) userfreq(float,4) iptr(int32,4) iblk(uint16,2) nrx(int8,1) iusb(char,1)

# FlexRadio DAXIQ samples arrive in ±32768 ADC scale — normalise to ±1.0
FLEX_FULL_SCALE = 32768.0

# QMAP output sample rate — must match QMAP configuration
QMAP_RATE = 96000


# ── Linear resampler ──────────────────────────────────────────────────────────
def _resample(samples: np.ndarray, in_rate: int, out_rate: int) -> np.ndarray:
    """Resample complex64 array from in_rate to out_rate using linear interpolation."""
    if in_rate == out_rate:
        return samples
    n_in  = len(samples)
    n_out = int(round(n_in * out_rate / in_rate))
    t_in  = np.arange(n_in,  dtype=np.float64)
    t_out = np.linspace(0, n_in - 1, n_out, dtype=np.float64)
    real  = np.interp(t_out, t_in, samples.real)
    imag  = np.interp(t_out, t_in, samples.imag)
    return (real + 1j * imag).astype(np.complex64)


# ── TIMF2 header assembly ─────────────────────────────────────────────────────
def _ms_utc_midnight() -> int:
    now     = datetime.now(timezone.utc)
    midnite = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return int((now - midnite).total_seconds() * 1000)


def _pack_packet(iq: np.ndarray, cfreq_mhz: float, iblk: int, iptr: int,
                 nrx: int, use_int16: bool, int16_scale: int,
                 iblk_zero: bool) -> bytes:
    """Build one 1416-byte TIMF2 packet from `iq` (174 complex64 samples, ±1.0)."""
    header = struct.pack(
        HEADER_FMT,
        float(cfreq_mhz),
        _ms_utc_midnight(),
        0.0,                        # userfreq
        int(iptr),
        0 if iblk_zero else int(iblk & 0xFFFF),
        int(nrx),
        bytes([0]),                 # iusb
    )

    if use_int16:
        # i*2 format: int16 I, int16 Q, 4 zero bytes per slot
        s = np.clip(iq, -1.0 + 1e-7, 1.0 - 1e-7) * float(int16_scale)
        interleaved = np.empty(PKT_SAMPLES * 2, dtype=np.int16)
        interleaved[0::2] = s.real.astype(np.int16)
        interleaved[1::2] = s.imag.astype(np.int16)
        # Expand to 8-byte slots: int16 I, int16 Q, 4 zero bytes
        payload = np.zeros(PKT_SAMPLES * 8, dtype=np.uint8)
        payload.view(np.int16)[0::4] = interleaved[0::2]  # I
        payload.view(np.int16)[1::4] = interleaved[1::2]  # Q
    else:
        # r*4 format: float32 I, float32 Q per slot — interleaved, little-endian
        interleaved = np.empty(PKT_SAMPLES * 2, dtype=np.float32)
        interleaved[0::2] = iq.real
        interleaved[1::2] = iq.imag
        payload = interleaved.astype('<f4').tobytes()

    return header + bytes(payload)


# ── Signal preprocessing ──────────────────────────────────────────────────────
def _preprocess(block: np.ndarray, args,
                mix_state: dict) -> np.ndarray:
    """Apply dc-block, gain, swap/invert, and complex mix to a complex64 block."""
    b = block.astype(np.complex64, copy=True)

    if args.dcblock:
        b -= np.mean(b)

    if args.gain != 1.0:
        b *= np.float32(args.gain)

    if args.swapiq:
        b = b.imag + 1j * b.real      # swap I and Q

    if args.invertq:
        b = b.real - 1j * b.imag      # negate Q

    # Complex frequency shift: multiply by exp(j*2π*f*t)
    if args.mix != 0.0:
        n   = len(b)
        fs  = float(args.sample_rate)
        f   = float(args.mix)
        inc = 2.0 * math.pi * f / fs

        if abs(abs(f) - fs / 2.0) < 1.0:
            # Exact Fs/2 shift: alternate ×(-1) — avoids float phase drift
            signs = np.ones(n, dtype=np.float32)
            start = mix_state["parity"]
            signs[start::2] = -1.0
            b *= signs
            mix_state["parity"] ^= (n & 1)
        else:
            t = mix_state["phase"] + inc * np.arange(n, dtype=np.float64)
            rotator = (np.cos(t) + 1j * np.sin(t)).astype(np.complex64)
            b *= rotator
            mix_state["phase"] = float((mix_state["phase"] + inc * n) % (2.0 * math.pi))

    return b


# ── Main bridge loop ──────────────────────────────────────────────────────────
def run_bridge(args):
    if args.radio:
        print(f"Connecting to FlexRadio at {args.radio}, DAX IQ ch {args.daxiq}...")
    else:
        print(f"Auto-discovering FlexRadio on LAN, DAX IQ ch {args.daxiq}...")
    client = FlexDAXIQ(
        radio_ip     = args.radio,
        sample_rate  = args.sample_rate,
        dax_channel  = args.daxiq,
    )
    try:
        client.start()
    except Exception as e:
        print(f"Failed to start FlexDAXIQ: {e}", file=sys.stderr)
        sys.exit(1)

    radio_ip = client._tcp.radio.ip
    print(f"Connected to FlexRadio at {radio_ip}, DAX IQ ch {args.daxiq}")
    print(f"Streaming -> {args.host}:{args.port}")

    sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)

    use_int16 = args.int16
    nrx       = 1 if use_int16 else args.nrx

    iblk = 0
    iptr = 0
    buf  = np.zeros(0, dtype=np.complex64)  # accumulation buffer

    mix_state      = {"phase": 0.0, "parity": 0}
    pkts_sent      = 0
    last_stats     = time.monotonic()
    last_rms       = time.monotonic()
    first_pkt      = True
    rate_samples   = 0
    rate_t0        = None
    rate_done      = False
    actual_rate    = args.sample_rate  # updated after measurement

    try:
        while True:
            pkt = client.get_samples(timeout=2.0)
            if pkt is None:
                if args.debug:
                    print("[bridge] waiting for samples...", flush=True)
                continue

            # Measure actual sample rate over first 5 seconds
            now = time.monotonic()
            if not rate_done:
                if rate_t0 is None:
                    rate_t0 = now
                rate_samples += len(pkt.samples)
                elapsed = now - rate_t0
                if elapsed >= 10.0:
                    measured_rate = rate_samples / elapsed
                    # Round to nearest standard rate
                    for std in (24000, 48000, 96000, 192000):
                        if abs(measured_rate - std) < std * 0.05:
                            actual_rate = std
                            break
                    print(f"[diag] measured sample rate: {measured_rate:.0f} Hz "
                          f"-> using {actual_rate} Hz", flush=True)
                    if actual_rate != QMAP_RATE:
                        if args.resample:
                            print(f"[diag] resampling {actual_rate} -> {QMAP_RATE} Hz", flush=True)
                        else:
                            print(f"[diag] native {actual_rate} Hz passthrough (use --resample to upsample)", flush=True)
                    rate_done = True

            # Normalise from ±32768 to ±1.0
            samples = pkt.samples / np.float32(FLEX_FULL_SCALE)

            # Resample to QMAP rate only if explicitly requested
            if args.resample and actual_rate != QMAP_RATE:
                samples = _resample(samples, actual_rate, QMAP_RATE)

            # Preprocess whole VITA packet block, then accumulate
            samples = _preprocess(samples, args, mix_state)
            buf = np.concatenate((buf, samples))

            # Emit as many complete 174-sample TIMF2 packets as we can
            while len(buf) >= PKT_SAMPLES:
                block = buf[:PKT_SAMPLES]
                buf   = buf[PKT_SAMPLES:]

                # Get current center frequency from SmartSDR status
                setup = client._dax_setup
                cfreq_mhz = (setup.slice_frequency_mhz
                             or setup.pan_frequency_mhz
                             or 0.0)

                if cfreq_mhz == 0.0:
                    print("[bridge] center freq unknown, skipping packet "
                          f"(slice={setup.slice_frequency_mhz} pan={setup.pan_frequency_mhz})",
                          flush=True)
                    continue

                data = _pack_packet(block, cfreq_mhz, iblk, iptr,
                                    nrx, use_int16, args.int16_scale, args.iblk_zero)

                if first_pkt:
                    rms = float(np.sqrt(np.mean(np.abs(block) ** 2)))
                    mx  = float(np.max(np.abs(block)))
                    dbfs = 20 * math.log10(max(rms, 1e-12))
                    print(f"[diag] first packet: cfreq={cfreq_mhz:.6f} MHz  "
                          f"RMS={rms:.5f} ({dbfs:.1f} dBFS)  max={mx:.5f}  "
                          f"nrx={nrx}  swapiq={args.swapiq}  mix={args.mix}  "
                          f"int16={use_int16}", flush=True)
                    first_pkt = False
                    hdr = data[:24]
                    cfreq_check = struct.unpack_from('<d', hdr, 0)[0]
                    msec_check  = struct.unpack_from('<i', hdr, 8)[0]
                    iptr_check  = struct.unpack_from('<i', hdr, 16)[0]
                    iblk_check  = struct.unpack_from('<H', hdr, 20)[0]
                    nrx_check   = struct.unpack_from('<b', hdr, 22)[0]
                    iusb_check  = struct.unpack_from('<B', hdr, 23)[0]
                    i0 = struct.unpack_from('<f', data, 24)[0]
                    q0 = struct.unpack_from('<f', data, 28)[0]
                    print(f"[diag] header: cfreq={cfreq_check:.6f} msec={msec_check} "
                          f"iptr={iptr_check} iblk={iblk_check} nrx={nrx_check} iusb={iusb_check}", flush=True)
                    print(f"[diag] payload[0]: I={i0:.6f} Q={q0:.6f}", flush=True)
                    print(f"[diag] packet size={len(data)} bytes", flush=True)

                if len(data) == BYTES_PER_PKT:
                    try:
                        sock.sendto(data, target)
                        pkts_sent += 1
                    except Exception as e:
                        print(f"UDP send error: {e}", file=sys.stderr)

                iblk = (iblk + 1) & 0xFFFF
                if args.iptrinc:
                    iptr = (iptr + PKT_SAMPLES) & 0x7FFFFFFF

            now = time.monotonic()

            if args.rms and now - last_rms >= 1.0:
                if len(samples):
                    rms = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
                    mx  = float(np.max(np.abs(samples)))
                    print(f"RMS={rms:.4f}  max|x|={mx:.4f}", flush=True)
                last_rms = now

            if args.debug and now - last_stats >= 1.0:
                setup = client._dax_setup
                cfreq = setup.slice_frequency_mhz or setup.pan_frequency_mhz or 0.0
                print(f"pkts/s={pkts_sent}  cfreq={cfreq:.6f} MHz  "
                      f"nrx={nrx}  gain={args.gain}  int16={'ON' if use_int16 else 'OFF'}",
                      flush=True)
                pkts_sent  = 0
                last_stats = now

    except KeyboardInterrupt:
        print("\nExiting.")
    finally:
        client.stop()
        sock.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Flex DAXIQ (VITA-49) → QMAP TIMF2 UDP bridge (no audio driver)"
    )
    ap.add_argument("--radio",       default=None,             help="FlexRadio IP address (omit to auto-discover)")
    ap.add_argument("--daxiq",       type=int,   default=1,    help="DAX IQ channel (default 1)")
    ap.add_argument("--host",        default="127.0.0.1",      help="QMAP host (default 127.0.0.1)")
    ap.add_argument("--port",        type=int,   default=50004, help="QMAP UDP port (default 50004)")
    ap.add_argument("--sample_rate", type=int,   default=96000, help="DAXIQ sample rate Hz (default 96000)")
    ap.add_argument("--gain",        type=float, default=1.0,   help="Linear gain (default 1.0)")
    ap.add_argument("--mix",         type=float, default=0.0,
                    help="Complex frequency shift Hz (default 0, disabled)")
    ap.add_argument("--dcblock",     action="store_true",       help="Remove DC per VITA packet")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--swapiq",   dest="swapiq", action="store_true",  help="Swap I and Q (default)")
    g.add_argument("--noswapiq", dest="swapiq", action="store_false", help="Do not swap I and Q")
    ap.set_defaults(swapiq=False)
    ap.add_argument("--invertq",     action="store_true",       help="Negate Q channel")
    ap.add_argument("--nrx",         type=int,   default=-1,   choices=[-1, 2],
                    help="TIMF2 nrx field for float32 mode: -1 (default) or 2")
    ap.add_argument("--int16",       action="store_true", default=True, help="Use int16 payload (i*2 format, default)")
    ap.add_argument("--int16_scale", type=int,   default=8192,  help="Full-scale for int16 (default 8192)")
    ap.add_argument("--iptrinc",     action="store_true",       help="Increment iptr by 174 per packet")
    ap.add_argument("--iblk_zero",   action="store_true",       help="Force iblk=0 in header")
    ap.add_argument("--resample",    action="store_true",       help="Resample to 96000 Hz if radio delivers a lower rate")
    ap.add_argument("--rms",         action="store_true",       help="Print RMS/max once per second")
    ap.add_argument("--debug",       action="store_true",       help="Verbose output")
    args = ap.parse_args()
    run_bridge(args)


if __name__ == "__main__":
    main()
