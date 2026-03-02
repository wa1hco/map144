"""Flex client setup, background thread runtime loop, and shutdown handling."""

import queue
import time
import wave
import importlib

import numpy as np

from PyQt5 import QtCore

# Internal signal level standard used throughout the FFT/display pipeline:
# complex float samples where |sample| ~= 32768 is near full-scale (0 dBFS reference).
INTERNAL_FULL_SCALE = 32768.0

# Source-specific ingress scaling into the internal standard above.
# - WAV loader yields roughly [-1, 1] normalized floats, so scale up to internal full-scale.
# - Flex DAXIQ payload varies by source/version; determine scale at runtime once.
WAV_INPUT_SCALE = INTERNAL_FULL_SCALE
FLEX_INPUT_SCALE = 1.0


def setup_flex_client(self):
    """Initialize source backend and start runtime thread."""
    flex_client_module = importlib.import_module('flexclient')
    flex_client_class = flex_client_module.FlexDAXIQ
    self.flex_client = flex_client_class(
        center_freq_mhz=self.center_freq_mhz,
        sample_rate=self.sample_rate,
        dax_channel=1,
        bind_client_id=getattr(self, 'bind_client_id', None),
    )
    self._flex_ingress_scale = None

    self.client_thread = QtCore.QThread()
    self.client_thread.run = self.run_flex_client
    self.client_thread.setTerminationEnabled(True)
    self.client_thread.start()


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
    with wave.open(path, 'rb') as wf:
        src_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif sample_width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        float_try = np.frombuffer(raw, dtype=np.float32)
        if np.all(np.isfinite(float_try)) and np.max(np.abs(float_try)) <= 10:
            data = float_try.astype(np.float32)
            max_abs = float(np.max(np.abs(data))) if data.size else 0.0
            if max_abs > 1.0:
                data /= max_abs
        else:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width {sample_width}")

    if channels == 1:
        iq = data.astype(np.complex64)
    else:
        frames = data.reshape(-1, channels)
        i = frames[:, 0]
        q = frames[:, 1]
        iq = (i + 1j * q).astype(np.complex64)

    iq = _resample_linear(iq, src_rate, target_rate)
    return iq.astype(np.complex64), target_rate


def _start_flex_source(self) -> bool:
    if self._flex_started:
        return True
    try:
        self.flex_client.start()
        self._flex_started = True
        return True
    except Exception as exc:
        print(f"FlexClient start error: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            self.flex_client.stop()
        except Exception:
            pass
        return False


def _stop_flex_source(self):
    if not self._flex_started:
        return
    try:
        self.flex_client.stop()
    except Exception:
        pass
    self._flex_started = False


def _reset_wav_timeline(self):
    self._wav_time_cursor = 0.0
    self.sample_buffer = np.array([], dtype=np.complex64)

    self.spectrogram_data = np.full((self.max_history, self.fft_size), -130.0)
    self.spec_staging = np.full((self.max_history, self.fft_size), -130.0)
    self.realtime_data = np.full((self.max_history, self.fft_size), -130.0)

    self.spec_staging_filled = False
    self.realtime_filled = False
    self.accumulated_noise_floor = np.full(self.fft_size, -125.0)
    self.realtime_noise_floor = np.full(self.fft_size, -125.0)

    self.realtime_energy_buffer = np.full(self.max_history, np.nan)
    self.accumulated_energy_buffer = np.full(self.max_history, np.nan)
    self.accumulated_energy_filled = False

    self.spec_boundary = 0
    self._realtime_boundary = 0
    self.energy_boundary = 0

    self.spec_write_index = 0
    self.realtime_write_index = 0
    self.energy_write_index = 0

    self.sq_spectrogram_data = np.full((self.max_history, self.fft_size), -130.0)
    self.sq_spec_staging = np.full((self.max_history, self.fft_size), -130.0)
    self.sq_spec_staging_filled = False
    self.sq_realtime_data = np.full((self.max_history, self.fft_size), -130.0)
    self.sq_realtime_filled = False
    self.sq_spec_boundary = 0
    self._sq_realtime_boundary = 0
    self.sq_spec_write_index = 0
    self.sq_realtime_write_index = 0

    self.time_in_window = 0.0
    self.next_boundary = self.history_secs


def _process_wav_source_step(self):
    wav_path = self.selected_wav_path
    if not wav_path:
        time.sleep(0.1)
        return

    if self._wav_samples is None or self._wav_path_loaded != wav_path:
        try:
            samples, _ = _load_wav_complex(wav_path, self.sample_rate)
            self._wav_samples = samples
            self._wav_path_loaded = wav_path
            self._wav_index = 0
            _reset_wav_timeline(self)
            print(f"Loaded WAV source: {wav_path} ({len(samples)} samples @ {self.sample_rate} Hz)", flush=True)
        except Exception as exc:
            print(f"WAV load error: {exc}", flush=True)
            time.sleep(0.5)
            return

    if self._wav_samples is None or len(self._wav_samples) == 0:
        time.sleep(0.1)
        return

    chunk_size = self.fft_size * 4
    start = self._wav_index
    end = start + chunk_size
    if end <= len(self._wav_samples):
        chunk = self._wav_samples[start:end]
        self._wav_index = end % len(self._wav_samples)
    else:
        first = self._wav_samples[start:]
        remain = end - len(self._wav_samples)
        second = self._wav_samples[:remain]
        chunk = np.concatenate([first, second]).astype(np.complex64)
        self._wav_index = remain

    # WAV ingress conversion into internal full-scale standard.
    chunk = (chunk * WAV_INPUT_SCALE).astype(np.complex64)

    wav_seconds = float(self._wav_time_cursor)
    ts_int = int(wav_seconds)
    ts_frac = int((wav_seconds - ts_int) * 1e12)
    self.process_iq_data(chunk, ts_int, ts_frac)
    time.sleep(chunk_size / self.sample_rate)


def run_flex_client(self):
    """Run selected source (Flex Radio or WAV file) and feed processing pipeline."""
    while self.running:
        try:
            if self.source_mode == "wav":
                if self._flex_started:
                    _stop_flex_source(self)
                _process_wav_source_step(self)
                continue

            if not _start_flex_source(self):
                time.sleep(1.0)
                continue

            try:
                packet = self.flex_client.sample_queue.get(timeout=1.0)
                # Flex ingress conversion into internal full-scale standard.
                raw_chunk = np.asarray(packet.samples, dtype=np.complex64)
                if self._flex_ingress_scale is None:
                    max_abs = float(np.max(np.abs(raw_chunk))) if raw_chunk.size else 0.0
                    # If samples are normalized (~[-1,1]), promote to internal full scale.
                    # If already near full scale, leave unchanged.
                    if max_abs > 0.0 and max_abs <= 4.0:
                        self._flex_ingress_scale = INTERNAL_FULL_SCALE
                    else:
                        self._flex_ingress_scale = FLEX_INPUT_SCALE
                    print(
                        f"Flex ingress scale selected: {self._flex_ingress_scale:.1f} "
                        f"(initial max |sample|={max_abs:.3f})",
                        flush=True,
                    )

                chunk = (raw_chunk * self._flex_ingress_scale).astype(np.complex64)
                self.process_iq_data(chunk, packet.timestamp_int, packet.timestamp_frac)
            except queue.Empty:
                continue
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

    _stop_flex_source(self)


def _get_tuned_frequency_mhz(self):
    """Return current tuned frequency and source label from Flex client status."""
    if self.source_mode == "wav":
        return self.center_freq_mhz, "WAV", None

    tuned_freq_mhz = None
    tuned_source = None
    tuned_bandwidth_hz = None
    if hasattr(self, 'flex_client') and self.flex_client:
        dax_setup = getattr(self.flex_client, '_dax_setup', None)
        if dax_setup:
            slice_freq = getattr(dax_setup, 'slice_frequency_mhz', None)
            pan_freq = getattr(dax_setup, 'pan_frequency_mhz', None)
            pan_bw = getattr(dax_setup, 'pan_bandwidth_hz', None)
            if slice_freq is not None:
                tuned_freq_mhz = slice_freq
                tuned_source = 'Slice'
            elif pan_freq is not None:
                tuned_freq_mhz = pan_freq
                tuned_source = 'Pan'
                tuned_bandwidth_hz = pan_bw
    return tuned_freq_mhz, tuned_source, tuned_bandwidth_hz


def closeEvent(self, event):
    """Clean up on window close."""
    print("Shutting down...")
    self.running = False

    if hasattr(self, 'update_timer'):
        self.update_timer.stop()

    if hasattr(self, 'flex_client'):
        _stop_flex_source(self)

    if hasattr(self, 'client_thread') and self.client_thread.isRunning():
        self.client_thread.quit()
        if not self.client_thread.wait(2000):
            print("Thread did not exit cleanly, terminating...")
            self.client_thread.terminate()
            self.client_thread.wait()

    print("Shutdown complete")
    event.accept()
