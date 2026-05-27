# Copyright (C) 2026  Jeff Millar, WA1HCO <wa1hco@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Radio client setup, background thread runtime loop, and shutdown handling.

This module owns the boundary between raw IQ sources (FlexRadio DAXIQ or WAV
file) and the DSP pipeline in ``processing.py``.  Its functions are mixed into
``MAP144Visualizer`` and cover three concerns: source lifecycle management,
sample ingress normalisation, and clean shutdown.

Signal-level conventions
------------------------
The internal dBFS reference is ±1.0 (0 dBFS = ±1.0 float).  Every source is
normalised to this convention at ingress, close to the hardware driver:

    Flex DAXIQ
        FlexRadio VITA-49 float32 payloads use an ADC full scale of
        approximately ±32768 (16-bit-style values expressed as float32).
        ``FLEX_DAXIQ_FULL_SCALE = 32768.0`` is divided out at ingress in
        ``run_radio_source`` before samples reach the pipeline.

    WAV files
        ``_load_wav_complex`` already normalises to [-1, 1] (int16 ÷ 32768,
        float32 clipped to ±1, etc.).  No further scaling is needed.

    Other SDR sources (AirSpy HF+, RTL-SDR, etc.)
        These deliver float32 in [-1, 1] natively.  Add a new ingress branch
        in ``run_radio_source`` without any additional scaling.

Functions
---------
setup_radio_client(self)
    Imports the ``flexclient`` module at runtime (avoids a hard dependency when
    running in WAV-only mode), instantiates ``FlexDAXIQ`` with the configured
    centre frequency, sample rate, and DAX channel, then starts a
    ``QtCore.QThread`` running ``run_radio_source``.

run_radio_source(self)
    Background thread main loop.  Polls ``source_mode`` on every iteration:

    * **"wav"** mode — stops the radio client if running, then calls
      ``_process_wav_source_step`` which loads (or reloads) the WAV file,
      feeds one chunk per real-time sleep, and wraps the file when exhausted.
    * **"radio"** mode — starts the radio client (once), drains packets from
      ``radio_client.sample_queue``, divides by ``FLEX_DAXIQ_FULL_SCALE``
      to normalise FlexRadio's ±32768 ADC range to the ±1.0 internal
      convention, then calls ``process_iq_data``.

    The loop is tolerant of transient errors (prints traceback, sleeps briefly,
    retries) and exits cleanly when ``self.running`` is set to False by
    ``closeEvent``.

_load_wav_complex(path, target_rate)
    Reads a WAV file of any common format (8-bit unsigned, 16-bit signed,
    32-bit int or 32-bit float) and returns complex64 IQ:

    * **Mono** files — the real (mono) channel is converted to analytic signal
      via ``scipy.signal.hilbert`` to create a single-sideband complex
      representation.  Without this, setting Q=0 produces a symmetric spectrum
      whose squared-domain version yields mirror-image tone pairs at negative
      frequencies, causing the MSK144 detector to recover a negative fc.
    * **Stereo** files — channel 0 is I, channel 1 is Q.

    Sample rate conversion to ``target_rate`` is done by
    ``_resample_linear`` (linear interpolation; adequate for display and
    detection but not for high-fidelity decode — the decimation in
    ``detection.extract_and_decode`` handles that path separately).

_resample_linear(samples, src_rate, dst_rate)
    Resample a complex array by linear interpolation of I and Q independently.
    Fast and allocation-light for moderate rate ratios; passband ripple is
    acceptable for the 48 kHz display pipeline.

_process_wav_source_step(self)
    Called repeatedly by ``run_radio_source`` in WAV mode.  Loads or reloads
    the file when ``selected_wav_path`` changes, feeds ``fft_size × 4``-sample
    chunks (enough for two full FFT blocks with 50 % overlap), wraps the index
    at end-of-file for continuous replay, and sleeps for the real-time
    equivalent of the chunk duration to avoid spinning the CPU.

_reset_wav_timeline(self)
    Clears all spectrogram staging buffers, noise-floor estimates, energy
    buffers, ring buffer, LP-filter state, and boundary counters back to their
    initial sentinel values (−130 dBFS / NaN / 0).  Called whenever a new WAV
    file is loaded so stale data from a previous file does not bleed into the
    new display.

_start_radio_source(self) / _stop_radio_source(self)
    Thin wrappers around ``radio_client.start()`` / ``radio_client.stop()`` with
    error handling and the ``_radio_started`` guard to prevent double-starts.

_get_tuned_frequency_mhz(self)
    Reads the current centre frequency from the radio client's ``_dax_setup``
    object (prefers Slice frequency over Pan centre) and returns a 3-tuple
    ``(freq_mhz, source_label, bandwidth_hz)``.  In WAV mode returns
    ``(center_freq_mhz, "WAV", None)``.  Called by ``displays.py`` on every
    refresh to keep frequency axis labels and the window title current.

closeEvent(self, event)
    Qt close handler: persists window geometry and dB-scale slider values to
    ``QSettings``, sets ``self.running = False`` to stop the background thread,
    stops the display timer, stops the radio client, waits up to 2 s for the
    thread to exit (terminates forcibly if it does not), then accepts the event.
"""

import logging
import queue
import time
import importlib

logger = logging.getLogger(__name__)

import numpy as np
from scipy.signal import hilbert

from PyQt5 import QtCore

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

def _fg_resize_for_plot_count(self):
    """Resize the Fast Graph window so each pane keeps the same pixel height
    when the set of visible plots changes (e.g. WAV→B210 adds the previous-15s
    pair, doubling the plot count).  Called once at source/dual-pol transitions,
    not on the 100 ms render timer.
    """
    fg_win = getattr(self, '_fast_graph_win', None)
    if fg_win is None:
        return
    dual    = getattr(self, 'dual_pol', False)
    is_live = getattr(self, 'source_mode', 'idle') not in ('idle', 'wav')
    # Count how many stretch=1 plots will be visible after this transition.
    n = 1                          # realtime_plot (H) always visible
    if dual:    n += 1             # realtime_plot_v
    if is_live: n += 1             # spectrogram_plot
    if is_live and dual: n += 1    # spectrogram_plot_v
    # Resize the window so each pane gets a constant target height.
    _pane_h = max(fg_win.height() // max(1, getattr(self, '_fg_last_plot_count', n)), 150)
    self._fg_last_plot_count = n
    from PyQt5.QtCore import QTimer
    QTimer.singleShot(0, lambda: fg_win.resize(fg_win.width(), max(_pane_h * n, 300)))


# FlexRadio DAXIQ float32 payloads use ADC full scale ≈ ±32768.
# Dividing by this constant at ingress normalises samples to the ±1.0 internal standard.
FLEX_DAXIQ_FULL_SCALE = 32768.0


def setup_radio_client(self):
    """Start the runtime thread.  The DAXIQ connection is deferred until
    the user selects 'Flex Radio' from the File menu."""
    # Do not overwrite radio_client if setup_ui() already created it
    # (e.g. when restoring a saved radio source mode at startup).
    if not hasattr(self, 'radio_client'):
        self.radio_client = None

    self.client_thread = QtCore.QThread()
    self.client_thread.run = self.run_radio_source
    self.client_thread.setTerminationEnabled(True)
    self.client_thread.start()


def _connect_airspy_client(self):
    """Instantiate AirspyHFSource on first use (called from on_select_source_airspy)."""
    if getattr(self, 'airspy_client', None) is not None:
        self.airspy_client.center_freq_mhz = self.calling_freq_mhz
        return
    try:
        from .airspy_source import AirspyHFSource
        self.airspy_client = AirspyHFSource(
            center_freq_mhz=self.calling_freq_mhz,
            target_rate=self.sample_rate,
        )
        print("[airspy] AirspyHFSource created", flush=True)
    except Exception as exc:
        print(f"[airspy] AirspyHFSource creation failed: {exc}", flush=True)
        self.airspy_client = None


def _set_dual_pol(self, enabled: bool):
    """Switch dual_pol mode and reset all pol-dependent buffers.

    Call before the first IQ chunk of a new source arrives so there is no
    stale H/V data in the channeliser or waterfall buffers.
    """
    if getattr(self, 'dual_pol', False) == enabled:
        return
    self.dual_pol = enabled
    # Resize Fast Graph so each pane keeps its pixel height when the V panes
    # appear/disappear.  Do this here (not in the render loop) so it fires once.
    _fg_resize_for_plot_count(self)
    # Reset channeliser buffers for both polarizations.
    self._rebuild_channelizer_state()
    # Reset V-channel waterfall buffer.
    self._sbuf_v_end = 0
    # Reset V-channel spectrograms to blank so stale H data doesn't show.
    if not enabled:
        self.realtime_data_v   = np.full((self.max_history, self.fft_size), -130.0)
        self.spectrogram_data_v = np.full((self.max_history, self.fft_size), -130.0)
        self.spec_staging_v     = np.full((self.max_history, self.fft_size), -130.0)
    print(f"[source] dual_pol={'ON' if enabled else 'OFF'}", flush=True)


def _start_airspy_source(self) -> bool:
    if getattr(self, '_airspy_started', False):
        return True
    if getattr(self, 'airspy_client', None) is None:
        return False
    try:
        _set_dual_pol(self, False)
        self.airspy_client.start()
        self._airspy_started = True
        if hasattr(self, '_jt9_markers'):
            self._jt9_markers.clear()
        if hasattr(self, 'decode_panel'):
            self.decode_panel.clear()
        return True
    except Exception as exc:
        print(f"[airspy] start error: {exc}", flush=True)
        import traceback; traceback.print_exc()
        return False


def _stop_airspy_source(self):
    if not getattr(self, '_airspy_started', False):
        return
    try:
        self.airspy_client.stop()
    except Exception:
        pass
    self._airspy_started = False


def _connect_rtlsdr_client(self):
    """Instantiate NesdrSmartSource on first use (called from on_select_source_rtlsdr)."""
    if getattr(self, 'rtlsdr_client', None) is not None:
        self.rtlsdr_client.center_freq_mhz = self.calling_freq_mhz
        return
    try:
        from .rtlsdr_source import NesdrSmartSource
        self.rtlsdr_client = NesdrSmartSource(
            center_freq_mhz=self.calling_freq_mhz,
            target_rate=self.sample_rate,
        )
        print("[rtlsdr] NesdrSmartSource created", flush=True)
    except Exception as exc:
        print(f"[rtlsdr] NesdrSmartSource creation failed: {exc}", flush=True)
        self.rtlsdr_client = None


def _start_rtlsdr_source(self) -> bool:
    if getattr(self, '_rtlsdr_started', False):
        return True
    if getattr(self, 'rtlsdr_client', None) is None:
        return False
    try:
        _set_dual_pol(self, False)
        self.rtlsdr_client.start()
        self._rtlsdr_started = True
        if hasattr(self, '_jt9_markers'):
            self._jt9_markers.clear()
        if hasattr(self, 'decode_panel'):
            self.decode_panel.clear()
        return True
    except Exception as exc:
        print(f"[rtlsdr] start error: {exc}", flush=True)
        import traceback; traceback.print_exc()
        self.rtlsdr_client = None   # prevent retry spam — user must re-select source
        return False


def _stop_rtlsdr_source(self):
    if not getattr(self, '_rtlsdr_started', False):
        return
    try:
        self.rtlsdr_client.stop()
    except Exception:
        pass
    self._rtlsdr_started = False


def _connect_sdrangel_client(self):
    """Instantiate SDRAngelSource(s) on first use (called from on_select_source_sdrangel).

    Reads connection settings from QSettings.  Always creates both H (device set N)
    and V (device set N+1) clients — MAP144 always uses SDRangel with a USRP B210
    which has two RX streams.  auto_configure_b210() creates the device sets if
    they don't yet exist.  V-channel UDP port is primary port - 1.
    """
    if getattr(self, 'sdrangel_client', None) is not None:
        self.sdrangel_client.calling_freq_mhz = self.calling_freq_mhz
        sc_v = getattr(self, 'sdrangel_client_v', None)
        if sc_v is not None:
            sc_v.calling_freq_mhz = self.calling_freq_mhz
        return
    try:
        from .sdrangel_source import SDRAngelSource
        from .visualizer import _SETTINGS as _S
        _host = str(_S.value('sdrangel_host', 'localhost'))
        try:
            _rest_port = int(_S.value('sdrangel_rest_port', 8091))
        except (ValueError, TypeError):
            _rest_port = 8091
        try:
            _udp_port = int(_S.value('sdrangel_udp_port', 9999))
        except (ValueError, TypeError):
            _udp_port = 9999
        try:
            _dev_set = int(_S.value('sdrangel_device_set', 0))
        except (ValueError, TypeError):
            _dev_set = 0
        try:
            _lo_offset = float(_S.value('sdrangel_lo_offset_khz', 50.0))
        except (ValueError, TypeError):
            _lo_offset = 50.0
        try:
            _dev_rate = int(_S.value('sdrangel_dev_sample_rate', 192000))
        except (ValueError, TypeError):
            _dev_rate = 192000
        try:
            _gain = float(_S.value('sdrangel_gain_db', 40.0))
        except (ValueError, TypeError):
            _gain = 40.0
        try:
            _rx_scale_raw = float(_S.value('sdrangel_rx_scale', 0.0))
            _rx_scale = _rx_scale_raw if _rx_scale_raw > 0 else None
        except (ValueError, TypeError):
            _rx_scale = None
        _exe_path = str(_S.value('sdrangel_exe', '')).strip() or None

        # H channel (device set _dev_set, stream 0).
        self.sdrangel_client = SDRAngelSource(
            host             = _host,
            rest_port        = _rest_port,
            udp_port         = _udp_port,
            device_set       = _dev_set,
            calling_freq_mhz = self.calling_freq_mhz,
            lo_offset_khz    = _lo_offset,
            target_rate      = self.sample_rate,
            exe_path         = _exe_path,
            dev_sample_rate  = _dev_rate,
            gain_db          = _gain,
            stream_index     = 0,
            rx_scale         = _rx_scale,
        )
        print("[sdrangel] SDRAngelSource (H/stream0) created", flush=True)

        # V channel (device set _dev_set+1, stream 1) — always created for B210.
        _udp_port_v = _udp_port - 1
        _dev_set_v  = _dev_set + 1
        self.sdrangel_client_v = SDRAngelSource(
            host             = _host,
            rest_port        = _rest_port,
            udp_port         = _udp_port_v,
            device_set       = _dev_set_v,
            calling_freq_mhz = self.calling_freq_mhz,
            lo_offset_khz    = _lo_offset,
            target_rate      = self.sample_rate,
            exe_path         = _exe_path,
            dev_sample_rate  = _dev_rate,
            gain_db          = _gain,
            stream_index     = 1,
            rx_scale         = _rx_scale,
        )
        print(f"[sdrangel] SDRAngelSource (V/stream1) created: device_set={_dev_set_v}"
              f"  UDP port={_udp_port_v}", flush=True)

    except Exception as exc:
        print(f"[sdrangel] SDRAngelSource creation failed: {exc}", flush=True)
        self.sdrangel_client   = None
        self.sdrangel_client_v = None


def _start_sdrangel_source(self) -> bool:
    if getattr(self, '_sdrangel_started', False):
        return True
    if getattr(self, 'sdrangel_client', None) is None:
        return False
    try:
        sc_v  = getattr(self, 'sdrangel_client_v', None)
        is_dual = sc_v is not None
        _set_dual_pol(self, is_dual)
        self.sdrangel_client.start()
        if is_dual:
            try:
                sc_v.start()
            except Exception as exc:
                print(f"[sdrangel] V-channel start failed: {exc}", flush=True)
                self.sdrangel_client_v = None
                _set_dual_pol(self, False)
        self._sdrangel_started = True
        if hasattr(self, '_jt9_markers'):
            self._jt9_markers.clear()
        if hasattr(self, 'decode_panel'):
            self.decode_panel.clear()
        return True
    except Exception as exc:
        print(f"[sdrangel] start error: {exc}", flush=True)
        import traceback; traceback.print_exc()
        self.sdrangel_client = None
        return False


def _stop_sdrangel_source(self):
    if not getattr(self, '_sdrangel_started', False):
        return
    try:
        self.sdrangel_client.stop()
    except Exception:
        pass
    sc_v = getattr(self, 'sdrangel_client_v', None)
    if sc_v is not None:
        try:
            sc_v.stop()
        except Exception:
            pass
    self._sdrangel_started = False
    # Close the SDRangel process so hardware is released for other sources.
    sc = getattr(self, 'sdrangel_client', None)
    if sc is not None:
        from .sdrangel_source import _terminate_sdrangel, _save_sdrangel_layout
        _terminate_sdrangel(sc.host, sc.rest_port)
        # SDRangel has now exited and flushed its Qt settings file.
        # Auto-save the layout so the next launch restores it.
        _save_sdrangel_layout()


def _sdrangel_ensure_ssb_service(self):
    """Find or create SSBDemod + virtual audio device for WSJT-X use.

    Creates a PipeWire/PulseAudio null sink named 'MAP144-WSJT-X' (and
    'MAP144-WSJT-X-V' for the V channel when dual-pol is active).
    Configures SSBDemod in each device set to output to the respective sink.
    Returns a dict with keys 'audio_h', 'audio_v' (or None if not created).
    """
    from .sdrangel_source import create_virtual_audio_sink
    sc   = getattr(self, 'sdrangel_client',   None)
    sc_v = getattr(self, 'sdrangel_client_v', None)
    result = {'audio_h': None, 'audio_v': None}

    if sc is None:
        return result

    # H channel.
    name_h = 'MAP144-WSJT-X'
    dev_h  = create_virtual_audio_sink(name_h)
    if dev_h:
        ch_h = sc.ensure_ssb_receiver(audio_device=dev_h)
        result['audio_h'] = dev_h
        print(f"[sdrangel] SSB service H: sink='{dev_h}'  ch={ch_h}", flush=True)
    else:
        ch_h = sc.ensure_ssb_receiver()
        print(f"[sdrangel] SSB service H: ch={ch_h} (no virtual audio created)", flush=True)

    # V channel (dual-pol only).
    if sc_v is not None:
        name_v = 'MAP144-WSJT-X-V'
        dev_v  = create_virtual_audio_sink(name_v)
        if dev_v:
            ch_v = sc_v.ensure_ssb_receiver(audio_device=dev_v)
            result['audio_v'] = dev_v
            print(f"[sdrangel] SSB service V: sink='{dev_v}'  ch={ch_v}", flush=True)
        else:
            ch_v = sc_v.ensure_ssb_receiver()
            print(f"[sdrangel] SSB service V: ch={ch_v} (no virtual audio created)",
                  flush=True)

    return result


def _connect_usrp_client(self):
    """Instantiate USRPSource on first use (called from on_select_source_usrp)."""
    if getattr(self, 'usrp_client', None) is not None:
        self.usrp_client.center_freq_mhz = self.calling_freq_mhz
        return
    try:
        from .usrp_source import USRPSource
        from .visualizer import _SETTINGS as _S
        try:
            _gain = float(_S.value('usrp_gain_db', 50.0))
        except (ValueError, TypeError):
            _gain = 50.0
        _ant  = str(_S.value('usrp_antenna',      'RX2'))
        _ant1 = str(_S.value('usrp_antenna_ch1', 'RX2'))
        try:
            _gain1 = float(_S.value('usrp_gain_db_ch1', _gain))
        except (ValueError, TypeError):
            _gain1 = _gain
        self.usrp_client = USRPSource(
            center_freq_mhz=self.calling_freq_mhz,
            target_rate=self.sample_rate,
            gain_db=_gain,
            antenna=_ant,
            dual_channel=True,        # RX 1 always enabled for polarization diversity
        )
        self.usrp_client._gain_ch1    = _gain1
        self.usrp_client._antenna_ch1 = _ant1
        logger.debug("[usrp] USRPSource created")
        _fg_resize_for_plot_count(self)
    except Exception as exc:
        logger.error("[usrp] USRPSource creation failed: %s", exc)
        self.usrp_client = None


def _start_usrp_source(self) -> bool:
    if getattr(self, '_usrp_started', False):
        return True
    if getattr(self, 'usrp_client', None) is None:
        return False
    try:
        _set_dual_pol(self, getattr(self.usrp_client, 'dual_channel', False))
        self.usrp_client.start()
        self._usrp_started = True
        if hasattr(self, '_jt9_markers'):
            self._jt9_markers.clear()
        if hasattr(self, 'decode_panel'):
            self.decode_panel.clear()
        return True
    except Exception as exc:
        logger.error("[usrp] start error: %s", exc)
        import traceback; traceback.print_exc()
        self.usrp_client = None   # prevent retry spam
        return False


def _stop_usrp_source(self):
    if not getattr(self, '_usrp_started', False):
        return
    try:
        self.usrp_client.stop()
    except Exception:
        pass
    self._usrp_started = False


def _connect_radio_client(self):
    """Instantiate FlexDAXIQ on first use (called from on_select_source_radio)."""
    if self.radio_client is not None:
        return   # already connected
    import time as _time
    t0 = _time.monotonic()
    try:
        print(f"[radio] importing flexclient...", flush=True)
        flex_client_module = importlib.import_module('flexclient')
        t_import = _time.monotonic()
        print(f"[radio] import done ({t_import - t0:.2f}s); calling FlexDAXIQ(...)", flush=True)
        flex_client_class = flex_client_module.FlexDAXIQ
        _radio_ip_edit = getattr(self, '_flex_radio_ip_edit', None)
        _radio_ip = _radio_ip_edit.text().strip() if _radio_ip_edit is not None else ''
        _radio_ip = _radio_ip or None
        print(f"[radio] calling FlexDAXIQ(radio_ip={_radio_ip!r})", flush=True)
        self.radio_client = flex_client_class(
            radio_ip=_radio_ip,
            center_freq_mhz=self.center_freq_mhz,
            sample_rate=self.sample_rate,
            dax_channel=1,
            bind_client_id=getattr(self, 'bind_client_id', None),
        )
        t_connect = _time.monotonic()
        print(f"[radio] FlexDAXIQ connected ({t_connect - t_import:.2f}s in ctor; {t_connect - t0:.2f}s total)", flush=True)
    except Exception as exc:
        t_fail = _time.monotonic()
        print(f"[radio] FlexDAXIQ connection failed after {t_fail - t0:.2f}s: {exc}", flush=True)
        self.radio_client = None


def _resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples.astype(np.complex64)
    if samples.size == 0:
        return np.array([], dtype=np.complex64)

    out_len = int(round(samples.size * (dst_rate / src_rate)))
    old_x = np.arange(samples.size, dtype=np.float64)
    new_x = np.linspace(0, samples.size - 1, out_len, dtype=np.float64)

    re = np.interp(new_x, old_x, np.real(samples).astype(np.float64))
    im = np.interp(new_x, old_x, np.imag(samples).astype(np.float64))
    return (re + 1j * im).astype(np.complex64)


def _load_wav_complex(path: str, target_rate: int) -> tuple[np.ndarray, int]:
    """Load a WAV file and return complex IQ samples resampled to target_rate.

    Supports both PCM (fmt_code=1) and IEEE float32 (fmt_code=3) WAV files,
    including capture files written by capture.save_capture().
    """
    import struct as _struct

    with open(path, 'rb') as f:
        file_data = f.read()

    if file_data[:4] != b'RIFF' or file_data[8:12] != b'WAVE':
        raise ValueError(f"Not a RIFF/WAVE file: {path}")

    pos = 12
    fmt_code = channels = src_rate = sample_width = None
    audio_bytes = None
    while pos < len(file_data) - 8:
        chunk_id   = file_data[pos:pos+4]
        chunk_size = _struct.unpack_from('<I', file_data, pos+4)[0]
        pos += 8
        if chunk_id == b'fmt ':
            fmt_code    = _struct.unpack_from('<H', file_data, pos)[0]
            channels    = _struct.unpack_from('<H', file_data, pos+2)[0]
            src_rate    = _struct.unpack_from('<I', file_data, pos+4)[0]
            sample_width = _struct.unpack_from('<H', file_data, pos+14)[0] // 8
        elif chunk_id == b'data':
            audio_bytes = file_data[pos:pos+chunk_size]
        pos += chunk_size

    if audio_bytes is None or fmt_code is None:
        raise ValueError(f"Malformed WAV file: {path}")

    if fmt_code == 3:
        # IEEE float32 — capture files written by capture.save_capture()
        raw = np.frombuffer(audio_bytes, dtype=np.float32)
        data = raw.astype(np.float32)
    elif fmt_code == 1:
        # PCM integer
        if sample_width == 1:
            data = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32)
            data = (data - 128.0) / 128.0
        elif sample_width == 2:
            data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            data = np.frombuffer(audio_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported PCM sample width {sample_width}: {path}")
    else:
        raise ValueError(f"Unsupported WAV format code {fmt_code}: {path}")

    if channels == 1:
        # Convert real mono to analytic (single-sideband) IQ.
        # Setting Q=0 leaves a symmetric spectrum whose squared version produces
        # mirror-image tone pairs at negative frequencies, causing the detector to
        # recover a negative fc and mix the signal to the wrong frequency.
        iq0 = hilbert(data.astype(np.float64)).astype(np.complex64)
        iq0 = _resample_linear(iq0, src_rate, target_rate)
        return iq0.reshape(-1, 1).astype(np.complex64), target_rate
    elif channels == 2:
        # Standard single-channel IQ WAV: left=I, right=Q.
        frames = data.reshape(-1, 2)
        iq0 = (frames[:, 0] + 1j * frames[:, 1]).astype(np.complex64)
        iq0 = _resample_linear(iq0, src_rate, target_rate)
        return iq0.reshape(-1, 1).astype(np.complex64), target_rate
    elif channels == 4:
        # Dual-channel IQ WAV: ch0_I, ch0_Q, ch1_I, ch1_Q interleaved per frame.
        frames = data.reshape(-1, 4)
        iq0 = (frames[:, 0] + 1j * frames[:, 1]).astype(np.complex64)
        iq1 = (frames[:, 2] + 1j * frames[:, 3]).astype(np.complex64)
        iq0 = _resample_linear(iq0, src_rate, target_rate)
        iq1 = _resample_linear(iq1, src_rate, target_rate)
        return np.column_stack((iq0, iq1)).astype(np.complex64), target_rate
    else:
        raise ValueError(f"Unsupported WAV channel count {channels} (expected 1, 2, or 4): {path}")


def _polarization_combine(samples: np.ndarray) -> np.ndarray:
    """Pass IQ samples to process_iq_data.

    For dual-pol sources, ``samples`` is shape ``(N, 2)`` complex64 (col 0 = H,
    col 1 = V).  process_iq_data detects the shape and handles both channels.
    For single-pol sources, ``samples`` is ``(N, 1)`` or ``(N,)``.
    """
    return samples


def _start_radio_source(self) -> bool:
    if self._radio_started:
        return True
    if self.radio_client is None:
        return False
    try:
        _set_dual_pol(self, False)
        self.radio_client.start()
        self._radio_started = True
        if hasattr(self, '_jt9_markers'):
            self._jt9_markers.clear()
        if hasattr(self, 'decode_panel'):
            self.decode_panel.clear()
        return True
    except Exception as exc:
        print(f"Radio client start error: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            self.radio_client.stop()
        except Exception:
            pass
        self.radio_client = None  # prevent thread from retrying with broken client
        return False


def _stop_radio_source(self):
    if not self._radio_started:
        return
    try:
        self.radio_client.stop()
    except Exception:
        pass
    self._radio_started = False


def _reset_wav_timeline(self):
    self._wav_time_cursor = 0.0
    self._sbuf_end   = 0
    self._sbuf_v_end = 0
    self._ch_buf_end = 0
    self._metric_hist_idx = 0
    self._metric_hist_cnt = 0
    self.raw_buffer    = np.array([], dtype=np.complex64)

    # Fill in-place rather than replacing array objects.  pyqtgraph ImageItem
    # may hold a zero-copy raw C pointer into the numpy buffer for deferred
    # QImage rendering; replacing the Python object frees the old buffer while
    # Qt may still be reading it (dangling pointer → segfault on paint event).
    self.spectrogram_data[:]   = -130.0
    self.spec_staging[:]       = -130.0
    self.realtime_data[:]      = -130.0
    # Always clear V spectrograms too — on single-pol WAVs this ensures
    # stale V data from a previous dual-pol run doesn't linger in the panes.
    self.spectrogram_data_v[:] = -130.0
    self.spec_staging_v[:]     = -130.0
    self.realtime_data_v[:]    = -130.0

    self.spec_staging_filled = False
    self.realtime_filled = False
    self.accumulated_noise_floor[:] = -125.0
    self.realtime_noise_floor[:]    = -125.0

    self.spec_boundary = 0
    self._realtime_boundary = 0
    self._sbuf_t0 = 0.0
    self._prev_staging_idx = None
    self._prev_rt_idx      = None

    self.time_in_window = 0.0
    self.next_boundary = self.history_secs

    # Reset ring buffer, detection state, and SNR heatmap history
    self._ch_snr_history_h[:] = 0.0
    self._ch_snr_history_v[:] = 0.0
    self._ch_snr_write_idx  = 0
    self._ch_snr_boundary   = 0
    self._ch_snr_hop_count  = 0
    self._iq_ring[:] = 0
    self._iq_ring_pos = 0
    self._iq_abs_sample = 0
    self._iq_t0_wall = None
    if hasattr(self, '_sync_buf'):
        self._sync_buf[:]   = 0
        self._sync_buf_v[:] = 0
    self._detect_cooldowns = {}
    self._iq_ring_gen = getattr(self, '_iq_ring_gen', 0) + 1
    if hasattr(self, '_jt9_markers'):
        self._jt9_markers.clear()
    if hasattr(self, 'decode_panel'):
        self.decode_panel.clear()
    # Drain any queued decode results from the previous WAV so stale marker
    # updates don't arrive after _jt9_markers has been cleared.
    dq = getattr(self, '_decode_queue', None)
    if dq is not None:
        import queue as _q
        while True:
            try:
                dq.get_nowait()
            except _q.Empty:
                break

    # Reset percentile baseline so the first frames of a new WAV are not
    # compared against a stale noise floor from the previous run.
    # _pct25_ctr drives the gated recompute; resetting to 0 forces an
    # immediate recompute on the very first detection frame.
    self._pct25_ctr = 0
    if hasattr(self, '_pct25'):
        del self._pct25
    self._nb_spec_avg = None   # reset blanker per-bin averages too


def _process_wav_source_step(self):
    wav_path = self.selected_wav_path
    if not wav_path:
        time.sleep(0.1)
        return

    nonce = getattr(self, '_wav_load_nonce', 0)
    nonce_loaded = getattr(self, '_wav_nonce_loaded', -1)

    if self._wav_samples is None or self._wav_path_loaded != wav_path or nonce_loaded != nonce:
        try:
            samples, _ = _load_wav_complex(wav_path, self.sample_rate)
            self._wav_samples = samples
            self._wav_path_loaded = wav_path
            self._wav_index = 0
            self._wav_done = False
            self._wav_nonce_loaded = nonce
            self._wav_run_start_time = datetime.now(timezone.utc)
            _set_dual_pol(self, samples.ndim == 2 and samples.shape[1] == 2)
            # _set_dual_pol already calls _rebuild_channelizer_state which
            # resets _ch_buf_end, _ch_buf_v_end, _sbuf_end, _sbuf_v_end.
            _reset_wav_timeline(self)
            print(f"Loaded WAV source: {wav_path} ({len(samples)} samples @ {self.sample_rate} Hz)"
                  f"{' [dual-pol]' if self.dual_pol else ''}", flush=True)
        except Exception as exc:
            print(f"WAV load error: {exc}", flush=True)
            time.sleep(0.5)
            return

    if self._wav_samples is None or len(self._wav_samples) == 0:
        time.sleep(0.1)
        return

    # Pause when playback has finished (play once, no wrap)
    if getattr(self, '_wav_done', False):
        time.sleep(0.1)
        return

    chunk_size = self.fft_size * 4
    start = self._wav_index
    end = start + chunk_size

    if end > len(self._wav_samples):
        # Final partial chunk
        chunk = self._wav_samples[start:]
        self._wav_index = len(self._wav_samples)
        if chunk.size > 0:
            chunk = _polarization_combine(chunk.astype(np.complex64))
            wav_seconds = float(self._wav_time_cursor)
            ts_int = int(wav_seconds)
            ts_frac = int((wav_seconds - ts_int) * 1e12)
            self.process_iq_data(chunk, ts_int, ts_frac)
        # Mark done and trigger comparison
        self._wav_done = True
        print(f"WAV playback complete: {wav_path}", flush=True)
        threading.Thread(
            target=_run_wav_comparison,
            args=(wav_path, getattr(self, '_wav_run_start_time', None),
                  list(getattr(self, '_jt9_threads', []))),
            daemon=True,
        ).start()
        return

    chunk = _polarization_combine(self._wav_samples[start:end].astype(np.complex64))
    self._wav_index = end

    # WAV samples are already in [-1, 1] from _load_wav_complex — no ingress scaling needed.
    wav_seconds = float(self._wav_time_cursor)
    ts_int = int(wav_seconds)
    ts_frac = int((wav_seconds - ts_int) * 1e12)
    self.process_iq_data(chunk, ts_int, ts_frac)


def _run_wav_comparison(wav_path: str, run_start: datetime | None,
                        jt9_threads: list | None = None) -> None:
    """Read manifest alongside wav_path, compare against decode log, write report.

    Runs in a daemon thread after WAV playback completes.  Writes a timestamped
    report to MSK144/detections/ with a per-simulation-file sequence number.
    """
    # Wait for all jt9 decode threads that were running when WAV finished.
    # jt9 has a 20-second timeout; cap our wait at 25 s to avoid hanging forever.
    if jt9_threads:
        print(f"[compare] Waiting for {len(jt9_threads)} pending jt9 thread(s)...", flush=True)
        for t in jt9_threads:
            t.join(timeout=25.0)

    wav_p = Path(wav_path)
    manifest_p = wav_p.with_suffix('.json')
    if not manifest_p.exists():
        print(f"[compare] No manifest found at {manifest_p}, skipping comparison", flush=True)
        return

    try:
        manifest = json.loads(manifest_p.read_text())
    except Exception as exc:
        print(f"[compare] Failed to read manifest: {exc}", flush=True)
        return

    placements = manifest.get("placements", [])

    # Read decode log and launch log, filtered to entries since run_start
    detections_dir = Path(__file__).parent.parent / 'MSK144' / 'detections'

    def _read_jsonl(path):
        entries = []
        if not path.exists():
            return entries
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if run_start is not None:
                    ts_str = entry.get("timestamp", "")
                    try:
                        entry_dt = datetime.fromisoformat(ts_str)
                        if entry_dt.tzinfo is None:
                            entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                        if entry_dt < run_start:
                            continue
                    except ValueError:
                        pass
                entries.append(entry)
        return entries

    log_path     = detections_dir / 'decodes.jsonl'
    launch_path  = detections_dir / 'launches.jsonl'
    decodes      = _read_jsonl(log_path)
    launches     = _read_jsonl(launch_path)

    # Build message lookup — index by full line and by each whitespace token
    # so that 13-char free-text messages match regardless of jt9 field prefix.
    decode_by_msg: dict[str, list[dict]] = {}
    for dec in decodes:
        raw = dec.get("message", "").strip()
        decode_by_msg.setdefault(raw, []).append(dec)
        for token in raw.split():
            if token != raw:
                decode_by_msg.setdefault(token, []).append(dec)

    freq_tol_khz = 2.0
    time_tol_s   = 1.5

    def _pos_match(pl):
        centre_khz = pl.get("center_hz", 0.0) / 1000.0
        delay_s    = pl.get("delay_s", 0.0)
        best, best_dist = None, float("inf")
        for dec in decodes:
            df = abs(dec.get("fc_khz", float("inf")) - centre_khz)
            dt = abs(dec.get("t_sec",  float("inf")) - delay_s)
            if df <= freq_tol_khz and dt <= time_tol_s:
                dist = (df / freq_tol_khz) ** 2 + (dt / time_tol_s) ** 2
                if dist < best_dist:
                    best_dist, best = dist, dec
        return best

    def _launch_theta(pl):
        """Return theta_deg from the launch log entry nearest to placement pl."""
        centre_khz = pl.get("center_hz", 0.0) / 1000.0
        delay_s    = pl.get("delay_s", 0.0)
        best, best_dist = None, float("inf")
        for lch in launches:
            df = abs(lch.get("radio_khz", float("inf")) - centre_khz)
            dt = abs(lch.get("t_sec",     float("inf")) - delay_s)
            if df <= freq_tol_khz and dt <= time_tol_s:
                dist = (df / freq_tol_khz) ** 2 + (dt / time_tol_s) ** 2
                if dist < best_dist:
                    best_dist, best = dist, lch
        return best.get("theta_deg") if best else None

    lines = []
    lines.append(f"MSK144 Comparison Report")
    lines.append(f"Simulation : {wav_p.name}")
    lines.append(f"Manifest   : {manifest_p.name}")
    lines.append(f"Generated  : {manifest.get('generated_at', 'unknown')}")
    lines.append(f"Run start  : {run_start.isoformat() if run_start else 'unknown'}")
    if 'noise_floor_dbfs' in manifest:
        lines.append(f"Noise floor: {manifest['noise_floor_dbfs']} dBFS   "
                     f"Atten: {manifest.get('atten_db', '?')} dB")
    else:
        lines.append(f"AWGN SNR   : {manifest.get('awgn_snr_db', '?')} dB   "
                     f"Atten: {manifest.get('atten_db', '?')} dB")
    n_launched   = len(launches)
    n_no_decode  = sum(1 for l in launches if l.get("outcome") == "no_decode")
    n_timeout    = sum(1 for l in launches if l.get("outcome") == "timeout")
    lines.append(f"Decode log : {log_path}  ({len(decodes)} entries in this run)")
    lines.append(f"jt9 launches: {n_launched} total  "
                 f"({len(decodes)} decoded  {n_no_decode} no_decode  {n_timeout} timeout)")
    lines.append("")
    lines.append("Message: A [P/N] ff ttt [P/N] dd www  — freq±kHz  time-ds  SNR±dB  width-ms")
    lines.append("")

    # Detect whether manifest has polarization data (dual-pol WAV)
    _has_pol = any('theta0_deg' in pl for pl in placements)

    col_w = 13
    _pol_hdr = "  θ_gen  θ_meas" if _has_pol else ""
    lines.append(f"{'#':>3}  {'Message':<{col_w}}  {'t (s)':>6}  {'fc (kHz)':>8}  "
                 f"{'gen SNR':>7}  {'wid ms':>7}{_pol_hdr}  Status")
    lines.append("-" * (70 + (16 if _has_pol else 0)))

    n_decoded = n_no_decode = n_missed = 0
    matched_ids: set[int] = set()

    def _snr_str(v):
        if v is None:
            return "      ?"
        # Accept either int or float (older ramp manifests stored snr_db as float).
        return f"{int(round(v)):>+4d} dB"

    def _pol_str(theta_gen, theta_meas):
        if not _has_pol:
            return ""
        g = f"{theta_gen:>5.0f}°" if theta_gen is not None else "    ?"
        m = f"{theta_meas:>5.0f}°" if theta_meas is not None else "    ?"
        return f"  {g}  {m}"

    placements = sorted(placements, key=lambda p: p.get("delay_s", 0.0))

    for i, pl in enumerate(placements):
        msg       = pl.get("msg", "")
        c_khz     = pl.get("center_hz", 0.0) / 1000.0
        delay_s   = pl.get("delay_s", 0.0)
        snr_db    = pl.get("snr_db")
        width_ms  = pl.get("width_ms")
        theta_gen = pl.get("theta0_deg")
        gen_snr   = _snr_str(snr_db)
        width_str = f"{width_ms:>4d} ms" if width_ms is not None else "     ?"
        prefix    = f"{i+1:>3}  {msg:<{col_w}}  {delay_s:>6.2f}  {c_khz:>8.2f}  {gen_snr}  {width_str}"

        hits = decode_by_msg.get(msg, [])
        if hits:
            dec = hits[0]
            # Mark every decode of this message as matched so repeated detections
            # of the same signal don't appear as false alarms.
            for h in hits:
                matched_ids.add(id(h))
            det_t      = dec.get('t_sec')
            theta_meas = dec.get('theta_deg')
            extra      = f"  ×{len(hits)}" if len(hits) > 1 else ""
            lines.append(f"{prefix}{_pol_str(theta_gen, theta_meas)}  DECODED{extra}   "
                         f"(t={det_t:.2f}s  fc={dec.get('radio_khz', dec.get('fc_khz', float('nan'))):.2f} kHz)")
            n_decoded += 1
        else:
            theta_meas_lch = _launch_theta(pl)
            if theta_meas_lch is not None:
                # Detected (jt9 launched) but jt9 did not return the correct message.
                # Check for a wrong-message decode near this placement (position match).
                near = _pos_match(pl)
                if near and id(near) not in matched_ids:
                    matched_ids.add(id(near))
                    det_t = near.get('t_sec')
                    note  = f"  (wrong msg: '{near.get('message','')}'  t={det_t:.2f}s)"
                else:
                    note  = ""
                lines.append(f"{prefix}{_pol_str(theta_gen, theta_meas_lch)}  NO DECODE{note}")
                n_no_decode += 1
            else:
                # No launch found near this placement — signal never reached the detector.
                lines.append(f"{prefix}{_pol_str(theta_gen, None)}  MISSED")
                n_missed += 1

    false_alarms = [d for d in decodes if id(d) not in matched_ids]

    lines.append("")
    lines.append("─" * 70)
    total = len(placements)
    pct   = 100 * n_decoded / total if total else 0
    lines.append(f"Total signals  : {total}")
    lines.append(f"Decoded        : {n_decoded:>3}  ({pct:.0f}%)")
    lines.append(f"No decode      : {n_no_decode:>3}  (detected, jt9 launched, no correct message)")
    lines.append(f"Missed         : {n_missed:>3}  (never detected — below threshold)")
    lines.append(f"jt9 launches   : {n_launched:>3}  "
                 f"({len(decodes)} decoded  {n_no_decode} no_decode  {n_timeout} timeout)")
    if false_alarms:
        lines.append(f"False alarms   : {len(false_alarms):>3}  (decoded with no matching signal)")
        for dec in false_alarms:
            jsnr  = dec.get('jt9_snr_db')
            snr_s = f"  snr={jsnr:+d} dB" if jsnr is not None else ""
            lines.append(f"    t={dec.get('t_sec',0.0):.3f}s  "
                         f"fc={dec.get('fc_khz',0.0):.3f} kHz"
                         f"{snr_s}  '{dec.get('message','')}'")
    lines.append("")

    report_text = "\n".join(lines)
    print(report_text, flush=True)

    # Write report file: {wav_stem}_run{seq:03d}.txt in detections dir
    detections_dir.mkdir(parents=True, exist_ok=True)
    stem = wav_p.stem
    existing = sorted(detections_dir.glob(f"{stem}_run*.txt"))
    seq = len(existing) + 1
    report_path = detections_dir / f"{stem}_run{seq:03d}.txt"
    report_path.write_text(report_text)
    print(f"[compare] Report written: {report_path}", flush=True)


def run_radio_source(self):
    """Run selected source (radio or WAV file) and feed processing pipeline."""
    while self.running:
        try:
            if self.source_mode == "idle":
                if self._radio_started:
                    _stop_radio_source(self)
                time.sleep(0.1)
                continue

            if self.source_mode == "wav":
                if self._radio_started:
                    _stop_radio_source(self)
                if getattr(self, '_airspy_started', False):
                    _stop_airspy_source(self)
                if getattr(self, '_rtlsdr_started', False):
                    _stop_rtlsdr_source(self)
                if getattr(self, '_usrp_started', False):
                    _stop_usrp_source(self)
                if getattr(self, '_sdrangel_started', False):
                    _stop_sdrangel_source(self)
                _process_wav_source_step(self)
                continue

            if self.source_mode == "rtlsdr":
                if self._radio_started:
                    _stop_radio_source(self)
                if getattr(self, '_airspy_started', False):
                    _stop_airspy_source(self)
                if getattr(self, '_usrp_started', False):
                    _stop_usrp_source(self)
                if getattr(self, '_sdrangel_started', False):
                    _stop_sdrangel_source(self)
                if not _start_rtlsdr_source(self):
                    time.sleep(1.0)
                    continue
                try:
                    drained = 0
                    while drained < 64:
                        try:
                            packet = self.rtlsdr_client.sample_queue.get_nowait()
                        except queue.Empty:
                            if drained == 0:
                                try:
                                    packet = self.rtlsdr_client.sample_queue.get(timeout=1.0)
                                except queue.Empty:
                                    break
                            else:
                                break
                        # RTL-SDR outputs ±1.0 float32 after uint8 conversion — no scaling.
                        chunk = _polarization_combine(np.asarray(packet.samples, dtype=np.complex64))
                        self.process_iq_data(chunk, packet.timestamp_int, packet.timestamp_frac)
                        drained += 1
                except Exception as exc:
                    print(f"[rtlsdr] queue/process error: {exc}", flush=True)
                    import traceback; traceback.print_exc()
                continue

            if self.source_mode == "airspy":
                if self._radio_started:
                    _stop_radio_source(self)
                if getattr(self, '_usrp_started', False):
                    _stop_usrp_source(self)
                if getattr(self, '_sdrangel_started', False):
                    _stop_sdrangel_source(self)
                if not _start_airspy_source(self):
                    time.sleep(1.0)
                    continue
                try:
                    drained = 0
                    while drained < 64:
                        try:
                            packet = self.airspy_client.sample_queue.get_nowait()
                        except queue.Empty:
                            if drained == 0:
                                try:
                                    packet = self.airspy_client.sample_queue.get(timeout=1.0)
                                except queue.Empty:
                                    break
                            else:
                                break
                        # Airspy HF+ outputs ±1.0 float32 — no scaling needed.
                        chunk = _polarization_combine(np.asarray(packet.samples, dtype=np.complex64))
                        self.process_iq_data(chunk, packet.timestamp_int, packet.timestamp_frac)
                        drained += 1
                except Exception as exc:
                    print(f"[airspy] queue/process error: {exc}", flush=True)
                    import traceback; traceback.print_exc()
                continue

            if self.source_mode == "usrp":
                if self._radio_started:
                    _stop_radio_source(self)
                if getattr(self, '_airspy_started', False):
                    _stop_airspy_source(self)
                if getattr(self, '_rtlsdr_started', False):
                    _stop_rtlsdr_source(self)
                if getattr(self, '_sdrangel_started', False):
                    _stop_sdrangel_source(self)
                if not _start_usrp_source(self):
                    time.sleep(1.0)
                    continue
                try:
                    drained = 0
                    while drained < 64:
                        try:
                            packet = self.usrp_client.sample_queue.get_nowait()
                        except queue.Empty:
                            if drained == 0:
                                try:
                                    packet = self.usrp_client.sample_queue.get(timeout=1.0)
                                except queue.Empty:
                                    break
                            else:
                                break
                        # UHD fc32 stream outputs ±1.0 float32 — no scaling needed.
                        chunk = _polarization_combine(np.asarray(packet.samples, dtype=np.complex64))
                        self.process_iq_data(chunk, packet.timestamp_int, packet.timestamp_frac)
                        drained += 1
                except Exception as exc:
                    logger.exception("[usrp] queue/process error: %s", exc)
                continue

            if self.source_mode == "sdrangel":
                if self._radio_started:
                    _stop_radio_source(self)
                if getattr(self, '_airspy_started', False):
                    _stop_airspy_source(self)
                if getattr(self, '_rtlsdr_started', False):
                    _stop_rtlsdr_source(self)
                if getattr(self, '_usrp_started', False):
                    _stop_usrp_source(self)
                if not _start_sdrangel_source(self):
                    time.sleep(1.0)
                    continue
                try:
                    sc_v    = getattr(self, 'sdrangel_client_v', None)
                    is_dual = sc_v is not None and getattr(self, 'dual_pol', False)
                    drained = 0
                    while drained < 64:
                        try:
                            packet = self.sdrangel_client.sample_queue.get_nowait()
                        except queue.Empty:
                            if drained == 0:
                                try:
                                    packet = self.sdrangel_client.sample_queue.get(timeout=1.0)
                                except queue.Empty:
                                    # Check whether the SDRangel process is still alive.
                                    import map144_app.sdrangel_source as _ss
                                    _proc = _ss._sdrangel_proc
                                    if _proc is not None and _proc.poll() is not None:
                                        print(f"[sdrangel] process died (exit {_proc.poll()})"
                                              " — stopping clients and relaunching", flush=True)
                                        _ss._sdrangel_proc = None
                                        # Stop recv threads so sockets are freed before restart.
                                        for _sc in (getattr(self, 'sdrangel_client', None),
                                                    getattr(self, 'sdrangel_client_v', None)):
                                            if _sc is not None:
                                                try: _sc.stop()
                                                except Exception: pass
                                        self._sdrangel_started = False
                                    break
                            else:
                                break
                        # SDRangel S16LE → ±1.0 done in recv_loop — no ingress scaling needed.
                        if is_dual:
                            # Pair H packet with a V packet from the second device set.
                            # Both streams run from the same hardware clock (B210);
                            # timestamps are wall-clock anchored and should match
                            # within a packet duration.  Take one V packet per H packet;
                            # if V queue is empty, fall back to single-pol for this chunk.
                            try:
                                pkt_v = sc_v.sample_queue.get_nowait()
                                n     = min(len(packet.samples), len(pkt_v.samples))
                                h_iq  = np.asarray(packet.samples[:n], dtype=np.complex64)
                                v_iq  = np.asarray(pkt_v.samples[:n],  dtype=np.complex64)
                                chunk = np.column_stack((h_iq.ravel(), v_iq.ravel()))
                            except queue.Empty:
                                chunk = _polarization_combine(
                                    np.asarray(packet.samples, dtype=np.complex64))
                        else:
                            chunk = _polarization_combine(
                                np.asarray(packet.samples, dtype=np.complex64))
                        self.process_iq_data(chunk, packet.timestamp_int, packet.timestamp_frac)
                        drained += 1
                except Exception as exc:
                    print(f"[sdrangel] queue/process error: {exc}", flush=True)
                    import traceback; traceback.print_exc()
                continue

            if not _start_radio_source(self):
                time.sleep(1.0)
                continue


            # Drain packets and batch them before calling process_iq_data.
            #
            # FlexRadio sends 128-sample VITA-49 UDP packets at 375 pkt/s.
            # Calling process_iq_data once per packet has two problems:
            #   1. The noise blanker (NB_FFT_SIZE=256) never fires because
            #      128 < 256 → n_blocks=0 every call → all Flex IQ passes
            #      through unblanked, causing excess false detections.
            #   2. Python/numpy call overhead at 375/s is significant.
            # Accumulate packets until the batch reaches _FLEX_BATCH_SAMPLES
            # (1024 = 4 × NB_FFT_SIZE = ~21 ms), then call once with the
            # combined chunk and the timestamp of the first packet in the batch.

            _FLEX_BATCH_SAMPLES = 1024

            try:
                # Anchor wall clock once per drain cycle; compute per-batch
                # wall time from cumulative sample count within the cycle.
                # One time.time() call per ~170 ms drain instead of per 21 ms
                # batch eliminates OS scheduling jitter in the heatmap/spectrogram
                # write index while staying crystal-accurate within the drain.
                _drain_t0      = time.time()
                _drain_samples = 0   # samples dispatched to process_iq_data so far
                drained = 0
                _batch_chunks   = []
                _batch_ts_int   = None
                _batch_ts_frac  = None
                _batch_n        = 0

                while drained < 64:
                    try:
                        packet = self.radio_client.sample_queue.get_nowait()
                    except queue.Empty:
                        if drained == 0 and _batch_n == 0:
                            # Nothing available — block briefly rather than spinning.
                            try:
                                packet = self.radio_client.sample_queue.get(timeout=1.0)
                            except queue.Empty:
                                break
                        else:
                            break

                    # Normalise FlexRadio DAXIQ samples from ±32768 ADC scale to ±1.0.
                    # Flex is always single-channel; apply polarization combine (ch0 passthrough).
                    pkt_chunk = (_polarization_combine(np.asarray(packet.samples, dtype=np.complex64))
                                 / FLEX_DAXIQ_FULL_SCALE)
                    drained += 1

                    if _batch_ts_int is None:
                        # Timestamp of the first packet anchors the batch.
                        _batch_ts_int  = packet.timestamp_int
                        # Convert timestamp_frac to picoseconds based on the TSF
                        # field carried in each packet.  The Flex may send either:
                        #   TSF=1  sample count within current second
                        #   TSF=2  real-time picoseconds (already correct)
                        # Applying the TSF=1 formula to TSF=2 values multiplies by
                        # ~20 million and makes staging_idx jump across the whole
                        # spectrogram, producing hopping vertical stripes.
                        _tsf = getattr(packet, 'tsf', 1)
                        if _tsf == 2:
                            _batch_ts_frac = int(packet.timestamp_frac)
                        else:
                            _batch_ts_frac = int(
                                packet.timestamp_frac * (1_000_000_000_000 / self.sample_rate)
                            )


                    # Gate processing during transmit — TX leakback floods the
                    # NB and channelizer with false detections.  Discard packets
                    # while TX is active; reset NB state on the TX→RX transition
                    # so it re-learns the noise floor from a clean signal.
                    now_tx = getattr(self.radio_client, 'transmitting', False)
                    was_tx = getattr(self, '_flex_was_tx', False)
                    self._flex_was_tx = now_tx

                    if now_tx:
                        # Drain and discard — don't let the queue fill up during TX.
                        _batch_chunks  = []
                        _batch_ts_int  = None
                        _batch_ts_frac = None
                        _batch_n       = 0
                        # Reset channelizer buffer on TX entry so there's nothing
                        # stale to process when RX resumes.
                        if not was_tx:
                            self._ch_buf_end = 0
                            self._ch_buf[:] = 0
                        # Empty remaining queued packets without processing them.
                        while True:
                            try:
                                self.radio_client.sample_queue.get_nowait()
                            except queue.Empty:
                                break
                        break   # exit inner drain loop; outer loop will re-enter

                    if was_tx and not now_tx:
                        # TX just ended — flush queue, purge NB state so it
                        # re-initialises from the first clean RX block.
                        # Also set a settling gate so AGC transients in the first
                        # ~500 ms don't produce broadband false detections.
                        pass  # TX→RX transition: NB state reset (debug)
                        while True:
                            try:
                                self.radio_client.sample_queue.get_nowait()
                            except queue.Empty:
                                break
                        self._nb_spec_avg = None
                        if hasattr(self, '_pct25'):
                            del self._pct25
                        self._pct25_ctr = 0
                        # Reset channelizer buffer and state so stale TX samples
                        # don't overflow the accumulation buffer on re-entry.
                        self._ch_buf_end = 0
                        self._ch_buf[:] = 0
                        from .channelizer import make_channelizer_state, N_CHANNELS
                        self._ch_state = make_channelizer_state(N_CHANNELS, self._ch_taps, self.sample_rate)
                        # 500 ms settling gate at current sample rate
                        self._tx_settle_remaining = int(self.sample_rate * 0.5)
                        _batch_chunks  = []
                        _batch_ts_int  = None
                        _batch_ts_frac = None
                        _batch_n       = 0
                        break   # re-enter outer loop cleanly

                    _batch_chunks.append(pkt_chunk)
                    _batch_n += len(pkt_chunk)

                    if _batch_n >= _FLEX_BATCH_SAMPLES:
                        # Pass timestamp_int=0 to signal "no valid radio clock" —
                        # process_iq_data then falls back to its session-level
                        # sample-clock anchor (_iq_t0_wall + _iq_abs_sample/sr),
                        # which is monotonic across drain cycles and immune to
                        # the wall-clock jitter that the older _drain_t0 +
                        # _drain_samples scheme introduced (a fresh time.time()
                        # per drain meant ~5–20 ms of jitter per drain boundary
                        # during CPU bursts, which manifested as spec_staging
                        # gap-fill and visible 200 ms repeat-banding in the
                        # fast graph during noise bursts).
                        self.process_iq_data(np.concatenate(_batch_chunks), 0, 0)
                        _drain_samples += _batch_n
                        _batch_chunks  = []
                        _batch_ts_int  = None
                        _batch_ts_frac = None
                        _batch_n       = 0

                # Flush any remaining samples shorter than a full batch.
                if _batch_chunks and not getattr(self, '_flex_was_tx', False):
                    self.process_iq_data(np.concatenate(_batch_chunks), 0, 0)
            except Exception as exc:
                print(f"Queue get/process error: {exc}", flush=True)
                import traceback
                traceback.print_exc()
                continue

        except Exception as exc:
            print(f"Source loop error: {exc}", flush=True)
            import traceback
            traceback.print_exc()
            time.sleep(0.25)

    _stop_radio_source(self)
    _stop_usrp_source(self)
    _stop_sdrangel_source(self)


def _get_tuned_frequency_mhz(self):
    """Return current tuned frequency and source label from radio client status."""
    if self.source_mode == "wav":
        return self.center_freq_mhz, "WAV", None
    if self.source_mode == "airspy":
        ac = getattr(self, 'airspy_client', None)
        freq = ac.center_freq_mhz_actual if ac is not None else self.calling_freq_mhz
        return freq, "Airspy HF+", None
    if self.source_mode == "rtlsdr":
        rc = getattr(self, 'rtlsdr_client', None)
        freq = rc.center_freq_mhz_actual if rc is not None else self.calling_freq_mhz
        return freq, "NESDR Smart", None
    if self.source_mode == "usrp":
        uc = getattr(self, 'usrp_client', None)
        freq = uc.center_freq_mhz_actual if uc is not None else self.calling_freq_mhz
        return freq, "USRP B210", None
    if self.source_mode == "sdrangel":
        # UDP Sink output is centered on calling_freq (lo_offset applied inside SDRangel).
        return self.calling_freq_mhz, "SDRangel", None

    tuned_freq_mhz = None
    tuned_source = None
    tuned_bandwidth_hz = None
    if hasattr(self, 'radio_client') and self.radio_client:
        dax_setup = getattr(self.radio_client, '_dax_setup', None)
        if dax_setup:
            slice_freq = getattr(dax_setup, 'slice_frequency_mhz', None)
            pan_freq = getattr(dax_setup, 'pan_frequency_mhz', None)
            pan_bw = getattr(dax_setup, 'pan_bandwidth_hz', None)
            # DAXIQ always delivers IQ centred at the panadapter centre, regardless
            # of where the associated slice is tuned.  The slice dial frequency is a
            # cursor in the pan and does NOT determine the IQ stream centre.
            # Prefer pan_freq; fall back to slice_freq only when pan is unknown.
            if pan_freq is not None:
                tuned_freq_mhz = pan_freq
                tuned_source = 'Pan'
                tuned_bandwidth_hz = pan_bw
            elif slice_freq is not None:
                tuned_freq_mhz = slice_freq
                tuned_source = 'Slice'
    return tuned_freq_mhz, tuned_source, tuned_bandwidth_hz


def closeEvent(self, event):
    """Clean up on window close."""
    from .visualizer import _SETTINGS

    # Signal panel windows that this is a real shutdown so their closeEvent
    # does not ignore the event and block teardown.
    self._app_closing = True

    _SETTINGS.setValue('source_mode',          self.source_mode)
    _SETTINGS.setValue('selected_wav_path',    self.selected_wav_path or '')
    _SETTINGS.setValue('window_geometry',      self.saveGeometry())
    _SETTINGS.setValue('min_level',            self.min_level)
    _SETTINGS.setValue('max_level',            self.max_level)
    _SETTINGS.setValue('detect_min_level_f',   self.detect_min_level)
    _SETTINGS.setValue('detect_max_level_f',   self.detect_max_level)
    _SETTINGS.setValue('sync_detect_min_level_f', getattr(self, 'sync_detect_min_level', -5.0))
    _SETTINGS.setValue('sync_detect_max_level_f', getattr(self, 'sync_detect_max_level', 20.0))
    _SETTINGS.setValue('nb_factor',            self.nb_factor)
    _SETTINGS.setValue('nb_factor_v',          getattr(self, 'nb_factor_v', self.nb_factor))
    _td_sl = getattr(self, 'td_scale_slider', None)
    _SETTINGS.setValue('td_scale',             _td_sl.value() if _td_sl else 10)
    _SETTINGS.setValue('td_span',              int(getattr(self, 'td_span_ms', 200)))

    for win, geo_key, vis_key in [
        (getattr(self, '_fast_graph_win', None), 'fast_graph_geometry', 'fast_graph_visible'),
        (getattr(self, '_detect_win',     None), 'detect_geometry',     'detect_visible'),
        (getattr(self, '_sync_detect_win',None), 'sync_detect_geometry','sync_detect_visible'),
        (getattr(self, '_iq_nb_win',      None), 'iq_nb_geometry',      'iq_nb_visible'),
        (getattr(self, '_reporting_win',  None), 'reporting_geometry',  'reporting_visible'),
        (getattr(self, '_flex_win',       None), 'flex_geometry',       None),
        (getattr(self, '_usrp_win',       None), 'usrp_geometry',       None),
        (getattr(self, '_airspy_win',     None), 'airspy_geometry',     None),
        (getattr(self, '_rtlsdr_win',     None), 'rtlsdr_geometry',     None),
        (getattr(self, '_sdrangel_win',   None), 'sdrangel_geometry',   None),
        (getattr(self, '_screenshot_win', None), 'screenshot_geometry', None),
    ]:
        if win is not None:
            _SETTINGS.setValue(geo_key, win.saveGeometry())
            if vis_key is not None:
                _SETTINGS.setValue(vis_key, win.isVisible())

    print("Shutting down...")
    self.running = False

    if hasattr(self, 'update_timer'):
        self.update_timer.stop()

    if hasattr(self, 'radio_client'):
        _stop_radio_source(self)
    if getattr(self, '_airspy_started', False):
        _stop_airspy_source(self)
    if getattr(self, '_rtlsdr_started', False):
        _stop_rtlsdr_source(self)
    if getattr(self, '_usrp_started', False):
        _stop_usrp_source(self)
    if getattr(self, '_sdrangel_started', False):
        _stop_sdrangel_source(self)

    if hasattr(self, 'client_thread') and self.client_thread.isRunning():
        self.client_thread.quit()
        if not self.client_thread.wait(2000):
            print("Thread did not exit cleanly, terminating...")
            self.client_thread.terminate()
            self.client_thread.wait()

    # Stop reporter (flushes pending PSKReporter spots, closes UDP socket)
    rpt = getattr(self, 'reporter', None)
    if rpt is not None:
        rpt.stop()

    # Close all panel windows explicitly.  Since _PanelWindow is now a true
    # top-level window (parent=None), quitOnLastWindowClosed will not fire
    # until every top-level window is closed.  _app_closing=True above ensures
    # their closeEvent accepts rather than ignoring the event.
    for _attr in ('_fast_graph_win', '_detect_win', '_sync_detect_win', '_iq_nb_win',
                  '_reporting_win', '_flex_win', '_usrp_win', '_airspy_win', '_rtlsdr_win',
                  '_sdrangel_win', '_screenshot_win'):
        _w = getattr(self, _attr, None)
        if _w is not None:
            _w.close()

    # Close any open analysis windows (saves their settings automatically via closeEvent).
    for _w in list(getattr(self, '_analysis_windows', [])):
        try:
            _w.close()
        except Exception:
            pass

    print("Shutdown complete")
    event.accept()
