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
"""DSP engine — no Qt imports.  Base class for MAP144Visualizer(Engine, QMainWindow)."""

import datetime
import logging
import os
import queue
from pathlib import Path

import numpy as np

from .channelizer import design_channelizer_filter, make_channelizer_state, N_CHANNELS
from .processing import (
    process_iq_data as _process_iq_data,
    N_SNR_HIST, CH_DETECT_SIZE, _METRIC_HIST_DEPTH,
)

log = logging.getLogger(__name__)


# Phase 4 cut-over feature flag (option 1: shadow mode behind a flag).
#
# Set this env var to 1 to enable a parallel Block-runtime that consumes
# the same IQ stream the legacy code does and writes its decodes to a
# separate JSONL (``MSK144/detections/decodes_blocks.jsonl``) for offline
# A/B comparison against the legacy ``decodes.jsonl``.
#
# When OFF (the default) the engine behaves exactly as before — no
# Block runtime is started, no overhead.  Easy rollback: unset the flag.
#
# This is the SHADOW mode, the lowest-risk first step toward the full
# cut-over.  After bake-time confidence the next commit promotes the
# Block path to primary (legacy detect/decode bypassed), and Phase 5
# cleanup (#13–#16) deletes the legacy paths altogether.
#
# See:
#   - docs/TODO.md §13.7 cut-over strategy options
#   - project_phase3_progress memory for detailed rationale
_USE_BLOCKS_ENV = "MAP144_USE_BLOCKS"


def _env_use_blocks() -> bool:
    raw = os.environ.get(_USE_BLOCKS_ENV, "")
    if not raw:
        return False
    return raw.strip().lower() in ("1", "true", "yes", "on")


class Engine:
    """DSP engine: holds all numpy state and the IQ processing pipeline.

    No PyQt5 imports.  ``MAP144Visualizer`` subclasses this for the GUI; IQ
    ingress calls ``process_iq_data`` (implemented in ``processing.py``).

    Phase 4 cut-over (option 1, shadow mode)
    ----------------------------------------
    When ``self.use_blocks`` is True (set via ``MAP144_USE_BLOCKS=1`` env var
    or directly by callers / tests), an internal Block runtime is started
    on first IQ tick and runs *alongside* the legacy DSP path.  Both
    consume the same IQ; the legacy path remains authoritative for GUI
    state / decode reporting; the Block path writes its own decodes.jsonl
    into ``MSK144/detections/blocks/`` for offline comparison.

    This is the lowest-risk first step toward full cut-over — easy rollback,
    no behaviour change when the flag is off.
    """

    # Class-level default.  ``__init__`` overrides from the env var so a
    # process started with ``MAP144_USE_BLOCKS=1`` automatically enables
    # shadow mode without any code change.
    use_blocks: bool = False

    def __init__(self, center_freq_mhz=50.260, sample_rate=48000, fft_size=2048,
                 bind_client_id=None, nb_factor=6.0, calling_freq_mhz=50.260):
        self.center_freq_mhz = center_freq_mhz
        self.calling_freq_mhz = calling_freq_mhz
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.bind_client_id = bind_client_id
        self.nb_factor   = nb_factor   # K threshold, H channel
        self.nb_factor_v = nb_factor   # K threshold, V channel (dual-pol)
        self.history_secs = 15
        self.blocks_per_sec = self.sample_rate / (self.fft_size // 2)
        self.max_history = int(round(self.history_secs * self.blocks_per_sec))
        self.max_time = self.history_secs
        self.running = True
        self.source_mode = "idle"
        self.selected_wav_path = None
        self._radio_started = False
        self._wav_samples = None
        self._wav_path_loaded = None
        self._wav_index = 0
        self._wav_time_cursor = 0.0
        self._wav_done = False
        self._wav_load_nonce = 0
        self._wav_nonce_loaded = -1
        self._wav_run_start_time = None
        self.radio_client = None
        self.airspy_client = None
        self._airspy_started = False
        self.rtlsdr_client = None
        self._rtlsdr_started = False
        self.sdrangel_client   = None
        self.sdrangel_client_v = None
        self._sdrangel_started = False

        # Overlap-add buffer for waterfall FFT (H, and V for dual-pol)
        _sbuf_cap = self.fft_size * 6
        self._sbuf = np.zeros(_sbuf_cap, dtype=np.complex64)
        self._sbuf_end = 0
        self._sbuf_t0  = 0.0   # wall-clock time of _sbuf[0], anchored to VITA timestamps
        self._sbuf_v = np.zeros(_sbuf_cap, dtype=np.complex64)
        self._sbuf_v_end = 0

        # Noise blanker state — H channel
        self._nb_env           = None   # running mean magnitude (signal+noise, display)
        self._nb_floor         = None   # noise floor from per-bin mean (display)
        self._nb_wideband_floor = None  # noise floor from median across bins (robust estimator)
        self._nb_lowlevel_frac  = 0.75  # fraction of bins below threshold (estimate quality)
        self._nb_spec_avg = None   # per-bin running average power, shape (NB_FFT_SIZE,)
        self._nb_last_P   = None   # per-bin power of the most recent block (display)
        self._nb_last_hot = 0      # hot-bin count of the most recent block (display)
        self._nb_blanked_count = 0
        self._nb_total_count = 0
        # Running median of per-block RMS amplitudes — updated unconditionally from every
        # block (impulse or noise) so it converges to the true noise RMS regardless of how
        # many impulse blocks are present.  Used as the broadband-gate reference and TD
        # amplitude threshold; immune to the per-bin average inflation problem.
        self._nb_blkrms_median = None

        # Noise blanker state — V channel (dual-pol, independent detection).
        # Mirrors every per-channel field above with a ``_v`` suffix so the
        # Linrad and NR0V backends can run parallel H and V detection
        # pipelines without hidden state sharing.
        self._nb_env_v            = None
        self._nb_floor_v          = None
        self._nb_wideband_floor_v = None
        self._nb_lowlevel_frac_v  = 0.75
        self._nb_spec_avg_v       = None
        self._nb_last_P_v         = None
        self._nb_last_hot_v       = 0
        self._nb_blkrms_median_v  = None
        self._nb_last_blanked_P_v = None

        # Selected noise-blanker backend.  State above is owned by the engine
        # so display/reset paths are unchanged across backends; the backend
        # itself is swappable at runtime via ``set_blanker(name)``.
        from .noise_blanker import LinradBlanker
        self.blanker = LinradBlanker()

        # Time-domain magnitude display buffer (200 ms circular) — H and V
        # _td_mag_buf*     : post-blanker (cleaned signal, for waveform display)
        # _td_mag_buf_raw* : pre-blanker  (raw ADC amplitude, for peak/headroom display)
        _td_n = int(0.200 * self.sample_rate)
        self._td_mag_buf     = np.zeros(_td_n, dtype=np.float32)
        self._td_mag_pos     = 0
        self._td_mag_buf_raw = np.zeros(_td_n, dtype=np.float32)
        self._td_mag_pos_raw = 0
        self._td_mag_buf_v   = np.zeros(_td_n, dtype=np.float32)
        self._td_mag_pos_v   = 0
        self._td_mag_buf_raw_v = np.zeros(_td_n, dtype=np.float32)
        self._td_mag_pos_raw_v = 0

        current_time = datetime.datetime.now().timestamp()
        self.time_in_window = current_time % self.history_secs
        self.next_boundary = current_time + (self.history_secs - self.time_in_window)

        # Accumulated spectrogram buffers — H (and V for dual-pol)
        self.spectrogram_data = np.full((self.max_history, self.fft_size), -130.0)
        self.spec_staging = np.full((self.max_history, self.fft_size), -130.0)
        self.spec_boundary = int(current_time / self.history_secs)
        self.spec_staging_filled = False
        self.spectrogram_data_v = np.full((self.max_history, self.fft_size), -130.0)
        self.spec_staging_v     = np.full((self.max_history, self.fft_size), -130.0)

        # Real-time spectrogram buffers — H (and V for dual-pol)
        self.realtime_data = np.full((self.max_history, self.fft_size), -130.0)
        self.realtime_time = self.history_secs
        self.realtime_filled = False
        self._realtime_dirty = False
        self._realtime_boundary = self.spec_boundary
        self.realtime_data_v = np.full((self.max_history, self.fft_size), -130.0)

        # Gap-fill cursors — track last written row index for each spectrogram.
        # Reset to None at each boundary so gap-fill doesn't bridge across windows.
        self._prev_staging_idx = None   # accumulated spectrogram
        self._prev_rt_idx      = None   # real-time spectrogram

        # Noise floor arrays
        self.accumulated_noise_floor = np.full(self.fft_size, -125.0)
        self.realtime_noise_floor = np.full(self.fft_size, -125.0)

        # Channelizer state.
        # channel_offset_hz = (calling_freq - pan_center) + 1500 Hz.
        #
        # In USB radio, signal energy is 1500 Hz above the dial frequency.
        # A station with dial at calling_freq + k kHz transmits at
        # calling_freq + k*1000 + 1500 Hz.  The +1500 places each channel NCO
        # exactly where that energy arrives in the IF.
        #
        # (calling_freq - pan_center) compensates when the panadapter isn't
        # exactly on the calling frequency.
        #
        # If center_freq_mhz is updated at runtime (Flex pan moves), call
        # _rebuild_channelizer_state() to recompute the NCO.
        # dual_pol: True when the source delivers H and V channels.
        # Set by the source adapter before IQ ingress begins.
        self.dual_pol = False

        self._ch_taps = design_channelizer_filter(self.sample_rate)
        _ch_offset_hz = (self.calling_freq_mhz - self.center_freq_mhz) * 1e6 + 1500.0
        self._ch_state = make_channelizer_state(
            N_CHANNELS, self._ch_taps, self.sample_rate,
            channel_offset_hz=_ch_offset_hz,
        )
        self._ch_state_v = make_channelizer_state(
            N_CHANNELS, self._ch_taps, self.sample_rate,
            channel_offset_hz=_ch_offset_hz,
        )
        _ch_buf_cap = CH_DETECT_SIZE + 2048
        self._ch_buf   = np.zeros((N_CHANNELS, _ch_buf_cap), dtype=np.complex64)
        self._ch_buf_end = 0
        self._ch_buf_v = np.zeros((N_CHANNELS, _ch_buf_cap), dtype=np.complex64)
        self._ch_buf_v_end = 0

        # Metric history (rolling 25th-percentile baseline) — independent per pol
        self._metric_hist_buf   = np.zeros((_METRIC_HIST_DEPTH, N_CHANNELS), dtype=np.float32)
        self._metric_hist_idx   = 0
        self._metric_hist_cnt   = 0
        self._metric_hist_buf_v = np.zeros((_METRIC_HIST_DEPTH, N_CHANNELS), dtype=np.float32)
        self._metric_hist_idx_v = 0
        self._metric_hist_cnt_v = 0

        # Coherent-sync detector state (Step 1 of sensitivity-improvement plan).
        # Runs in parallel with the squared-FFT detector; pair_metric is the
        # max of (squared-domain dB, sync-correlator dB) so a hit on either
        # path triggers the cluster gate.  Coherent integration over the
        # 16-chip MSK144 sync buys ~6 dB over the squared-domain ±1 kHz tone
        # detector at low SNR (closes the 6 dB gap measured against WSJT-X
        # in the ramp-test framework).
        # Dedicated per-channel rolling ring of NSPM samples for the sync
        # correlator.  Updated each input chunk (FIFO shift-and-append) so it
        # always holds the most recent 72 ms regardless of how the per-channel
        # detection buffer (_ch_buf) is being drained.  This is essential for
        # live sources with small chunks (Flex VITA packets, ~60 channelized
        # samples per call): _ch_buf never accumulates NSPM samples before the
        # hop loop drains it, so reading directly from _ch_buf would skip the
        # sync correlator entirely under live operation — leaving stale
        # cached metrics that render as horizontal bands in the sync heatmap.
        # The channelizer already places signal energy at DC in each channel
        # output, so no pre-mix is needed before correlation.
        from .msk144_spd import NSPM as _SYNC_NSPM
        self._sync_buf       = np.zeros((N_CHANNELS, _SYNC_NSPM), dtype=np.complex64)
        self._sync_buf_v     = np.zeros((N_CHANNELS, _SYNC_NSPM), dtype=np.complex64)
        self._sync_metric_hist_buf   = np.zeros((_METRIC_HIST_DEPTH, N_CHANNELS), dtype=np.float32)
        self._sync_metric_hist_idx   = 0
        self._sync_metric_hist_cnt   = 0
        self._sync_metric_hist_buf_v = np.zeros((_METRIC_HIST_DEPTH, N_CHANNELS), dtype=np.float32)
        self._sync_metric_hist_idx_v = 0
        self._sync_metric_hist_cnt_v = 0
        # Per-hop sync metric history for the dedicated sync-detector heatmap
        # (mirrors _ch_snr_history_h/v which the squared-FFT detector writes).
        # Same axes: rows = hops, cols = channels; row index is the hop count
        # modulo N_SNR_HIST.  Hop counter resets to 0 at each 15-s boundary
        # cross so a fresh period always starts filling at row 0.
        self._sync_snr_history_h = np.full((N_SNR_HIST, N_CHANNELS), -999.0, dtype=np.float32)
        self._sync_snr_history_v = np.full((N_SNR_HIST, N_CHANNELS), -999.0, dtype=np.float32)
        self._ch_snr_hop_count = 0

        # Pre-computed FFT window
        self._fft_window = np.hanning(self.fft_size).astype(np.float32)
        self._window_gain = float(np.sqrt(np.mean(self._fft_window ** 2)))

        # Per-channel SNR history for detection heatmap — independent H and V.
        self._ch_snr_history_h = np.full((N_SNR_HIST, N_CHANNELS), -999.0, dtype=np.float32)
        self._ch_snr_history_v = np.full((N_SNR_HIST, N_CHANNELS), -999.0, dtype=np.float32)
        self._ch_snr_write_idx = 0
        self._ch_snr_boundary  = int(current_time / self.history_secs)

        # IQ ring buffer — shape (ring_n, 2): col 0 = H, col 1 = V.
        # Single-pol sources write zeros to col 1; the ring read is shape-agnostic.
        # 30 seconds covers current 15s window + previous 15s window so both
        # Fast Graph panes can trigger analysis.
        _ring_n = int(30 * self.sample_rate)
        self._iq_ring = np.zeros((_ring_n, 2), dtype=np.complex64)
        self._iq_ring_pos = 0
        self._iq_abs_sample = 0
        # Wall-clock for ring sample 0; refreshed each chunk in process_iq_data.
        # Used by extract_and_decode to derive the exact ping epoch from the IQ
        # sample index, immune to pipeline / queue latency.
        self._iq_t0_wall: float | None = None
        self._detect_cooldowns   = {}
        self._ch_above_thresh    = {}   # ch_k -> consecutive hops above threshold
        self._iq_ring_gen = 0
        self._jt9_threads = []
        self._jt9_markers = []
        self._jt9_marker_next_id = 0
        self._decode_queue = queue.SimpleQueue()

        # Phase 4 cut-over (shadow mode).  When the env var is set we
        # turn on the parallel Block runtime; tests / direct callers can
        # set ``self.use_blocks = True`` after construction with the same
        # effect.  The Runtime itself is allocated lazily on first IQ
        # tick (inside _block_pump_iq) so process startup cost is zero
        # when the flag is off and small when on.
        self.use_blocks: bool = _env_use_blocks()
        self._block_runtime = None       # lazy-initialised Runtime
        self._block_iq_in_stream = None  # IQ input stream for the Block path
        self._block_iq_sample_counter: int = 0
        self._block_iq_anchor_wall: float = 0.0   # set on first push
        if self.use_blocks:
            log.info(
                "Engine: shadow Block runtime ENABLED (env %s); "
                "legacy DSP path remains authoritative",
                _USE_BLOCKS_ENV,
            )

        # Frequency axis
        self.fft_bin_axis_mhz = np.fft.fftshift(
            np.fft.fftfreq(self.fft_size, 1 / self.sample_rate)
        ) / 1e6
        self.freq_axis = self.fft_bin_axis_mhz + self.center_freq_mhz
        self.display_center_freq_mhz = -1.0  # force report_freq() on first display update

    def set_blanker(self, name: str) -> None:
        """Switch noise-blanker backend at runtime.

        The previous backend's warm-up state (``_nb_*``) is cleared so the new
        backend starts from a clean baseline on the next chunk.  No sample-
        stream gap is introduced — the next ``process_iq_data`` call simply
        runs under the new backend.
        """
        from .noise_blanker import make
        new = make(name)
        old = getattr(self, 'blanker', None)
        if old is not None:
            old.reset(self)     # clear prior backend's warm-up state
        self.blanker = new

    def _rebuild_channelizer_state(self):
        """Rebuild channelizer NCO after center_freq_mhz or calling_freq_mhz changes.

        Call this whenever the panadapter center is updated from the radio so that
        each channel's NCO stays aligned to the calling_freq + k x 1000 Hz grid.
        The FIR filter state is reset; any signal currently in the channelizer
        buffer will be discarded by the processing pipeline.
        """
        _ch_offset_hz = (self.calling_freq_mhz - self.center_freq_mhz) * 1e6 + 1500.0
        self._ch_state = make_channelizer_state(
            N_CHANNELS, self._ch_taps, self.sample_rate,
            channel_offset_hz=_ch_offset_hz,
        )
        self._ch_state_v = make_channelizer_state(
            N_CHANNELS, self._ch_taps, self.sample_rate,
            channel_offset_hz=_ch_offset_hz,
        )
        self._ch_buf_end   = 0
        self._ch_buf_v_end = 0
        self._sbuf_end   = 0
        self._sbuf_v_end = 0

    def setup_radio_client(self):
        """No-op base implementation — overridden by MAP144Visualizer for Qt thread."""
        self.radio_client = None

    # ──────────────────────────────────────────────────────────────────
    # Phase 4 cut-over scaffolding (option 1: shadow Block runtime)
    # ──────────────────────────────────────────────────────────────────

    def process_iq_data(self, iq_samples, timestamp_int, timestamp_frac):
        """IQ ingress dispatch — legacy ``_process_iq_data`` always runs;
        when ``self.use_blocks`` is True a parallel Block runtime
        consumes the same IQ for offline A/B validation.

        Behaviour:

        - **Flag off (default):** zero overhead, identical to the
          pre-feature-flag class-level binding ``process_iq_data =
          _process_iq_data``.
        - **Flag on:** legacy still authoritative for GUI / decode
          reporting; Block runtime writes its own decodes JSONL to
          ``MSK144/detections/blocks/decodes.jsonl`` for comparison.
          A failure on the Block side is logged but never propagates
          back into the legacy path.

        See class docstring + ``project_phase3_progress`` memory.
        """
        # Always run legacy first — it owns the engine's GUI state.
        _process_iq_data(self, iq_samples, timestamp_int, timestamp_frac)

        if not self.use_blocks:
            return

        # Then mirror the same IQ into the shadow Block runtime.
        # Failures here must not disturb the legacy path; log + continue.
        try:
            self._block_pump_iq(iq_samples)
        except Exception:
            log.exception(
                "Engine: shadow Block runtime raised on IQ pump; "
                "disabling further Block-side updates this run "
                "(legacy path unaffected)",
            )
            # Disable the flag so we stop pushing IQ; legacy continues.
            # Caller can reset the runtime via _stop_block_runtime() and
            # flip the flag back on after fixing the underlying issue.
            self.use_blocks = False

    def _ensure_block_runtime(self) -> None:
        """Lazy-init the shadow Block runtime on first IQ tick.

        Wires Source-stand-in (a manually-fed IQ stream) ▶ Tee ▶
        Channelizer ▶ Detector ▶ Decoder ▶ JsonlSink.  No Reporter — the
        live PSKReporter / DXcluster uploads come from the LEGACY path
        in shadow mode, so duplicate spots are not an issue.

        Output dir: ``MSK144/detections/blocks/`` (its own directory so
        the legacy ``decodes.jsonl`` and ``launches.jsonl`` are not
        co-mingled with the Block-path output).
        """
        if self._block_runtime is not None:
            return

        # Lazy import — keeps the blocks/ subpackage out of the import
        # graph when the flag is off.
        from .blocks import (
            ChannelizerBlock, ChannelizerBlockConfig,
            DecoderBlock, DecoderBlockConfig,
            DetectorBlock, DetectorBlockConfig,
            JsonlSink, JsonlSinkConfig,
            Runtime, Stream,
            TeeBlock, TeeBlockConfig,
            make_event_stream, make_sample_stream,
        )

        # Output directory for Block-path decodes (separate from legacy).
        block_out_dir = (
            Path(__file__).parent.parent / 'MSK144' / 'detections' / 'blocks'
        )
        block_out_dir.mkdir(parents=True, exist_ok=True)
        block_decode_jsonl = block_out_dir / 'decode_events.jsonl'

        # Channel offset — same as legacy uses for the channelizer NCO.
        ch_offset_hz = (
            (self.calling_freq_mhz - self.center_freq_mhz) * 1e6 + 1500.0
        )

        rt = Runtime()

        # No Source block — IQ is pushed externally via _block_pump_iq.
        # We wire ``iq_in`` into a Tee whose two output ports feed
        # Channelizer (for detection) and DecoderBlock (for the IQ ring
        # window read on each detection event).
        iq_tee = rt.add(
            TeeBlock("iq_tee"),
            TeeBlockConfig(
                input_port_name="iq_in",
                output_port_names=["to_chan", "to_dec"],
                get_timeout_s=0.05,
            ),
        )

        chan = rt.add(
            ChannelizerBlock("chan"),
            ChannelizerBlockConfig(
                input_port_name="iq",
                output_port_name="channels",
                sample_rate_in_hz=self.sample_rate,
                channel_offset_hz=ch_offset_hz,
                get_timeout_s=0.05,
            ),
        )

        det = rt.add(
            DetectorBlock("det"),
            DetectorBlockConfig(
                input_port_name="channels",
                detection_port_name="detections",
                channel_offset_hz=ch_offset_hz,
                get_timeout_s=0.05,
            ),
        )

        dec = rt.add(
            DecoderBlock("dec"),
            DecoderBlockConfig(
                iq_port_name="iq",
                detection_port_name="detections",
                decode_port_name="decodes",
                sample_rate_hz=self.sample_rate,
                ring_seconds=15.0,
                max_concurrent_decodes=4,
                output_dir=str(block_out_dir),
                center_freq_mhz=self.center_freq_mhz,
                get_timeout_s=0.05,
            ),
        )

        sink = rt.add(
            JsonlSink("decode_sink"),
            JsonlSinkConfig(
                input_port_name="events",
                output_path=str(block_decode_jsonl),
                get_timeout_s=0.05,
            ),
        )

        # Manual IQ-input stream — we push records onto it from
        # _block_pump_iq each time the radio source delivers IQ.
        # POLICY_DROP_OLDEST: a slow Block consumer must NEVER block
        # the legacy DSP path that just ran above.  Dropped IQ on the
        # shadow side is acceptable; a stalled GUI is not.
        iq_in = make_sample_stream(
            "engine_iq_in",
            queue_depth_seconds=2.0,
            sample_rate_hz=self.sample_rate,
            n_samples_per_record=4096,
            on_full=Stream.POLICY_DROP_OLDEST,
        )
        # Manual port assignment (no upstream block produced this).
        iq_tee.inputs["iq_in"] = iq_in

        # Wire the rest with the standard connect API.
        rt.connect(iq_tee, "to_chan", chan, "iq",
                   queue_depth_seconds=2.0,
                   sample_rate_hz=self.sample_rate,
                   n_samples_per_record=4096)
        rt.connect(iq_tee, "to_dec", dec, "iq",
                   queue_depth_seconds=2.0,
                   sample_rate_hz=self.sample_rate,
                   n_samples_per_record=4096)
        rt.connect(chan, "channels", det, "channels",
                   queue_depth_seconds=2.0,
                   sample_rate_hz=12_000,
                   n_samples_per_record=1024)
        rt.connect(det, "detections", dec, "detections",
                   kind=Stream.KIND_EVENTS, capacity=64, durable=True)
        rt.connect(dec, "decodes", sink, "events",
                   kind=Stream.KIND_EVENTS, capacity=64, durable=True)

        rt.start()

        self._block_runtime = rt
        self._block_iq_in_stream = iq_in
        self._block_iq_sample_counter = 0
        self._block_iq_anchor_wall = 0.0   # set lazily on first push

        log.info(
            "Engine: shadow Block runtime started; decodes → %s",
            block_decode_jsonl,
        )

    def _block_pump_iq(self, iq_samples: np.ndarray) -> None:
        """Push one chunk of IQ into the shadow Block runtime.

        Handles dual-pol shape (N, 2) by feeding only the H column for
        now (the production Block path is mono per docs/TODO §6.1; full
        dual-pol cut-over lands later via DualPolFlexSource +
        CombinerBlock).

        On first call, lazy-initialises the runtime via
        ``_ensure_block_runtime`` and anchors the sample-clock to
        ``time.time()`` so wall-clock-at-production stamps make sense.
        """
        import time
        from .blocks.types import Record, StreamClosed

        self._ensure_block_runtime()

        # Reshape to 1-D mono.  Dual-pol splits into H/V columns; we
        # send H here.  Pure-V testing would need a different feed point;
        # not needed for shadow-mode bake-in.
        data = iq_samples
        if data.ndim == 2 and data.shape[1] >= 1:
            data = data[:, 0]
        data = np.ascontiguousarray(data, dtype=np.complex64)
        if data.size == 0:
            return

        if self._block_iq_anchor_wall == 0.0:
            self._block_iq_anchor_wall = time.time()

        wall_now = (
            self._block_iq_anchor_wall
            + self._block_iq_sample_counter / float(self.sample_rate)
        )
        rec = Record(
            data=data,
            sample_rate_hz=int(self.sample_rate),
            start_sample=int(self._block_iq_sample_counter),
            wall_clock_at_production=wall_now,
        )
        self._block_iq_sample_counter += data.size

        # The IQ stream uses POLICY_DROP_OLDEST so put() never blocks the
        # legacy path; if the Block side is slow, IQ records get evicted
        # and the Block path simply has gaps.  StreamClosed only happens
        # if the runtime has been torn down between calls — log and
        # continue.
        try:
            self._block_iq_in_stream.put(rec)
        except StreamClosed:
            log.warning(
                "Engine: shadow Block runtime IQ stream closed; "
                "skipping push (legacy unaffected)",
            )

    def _stop_block_runtime(self, timeout_s: float = 2.0) -> None:
        """Tear down the shadow Block runtime if running.  Idempotent.

        Called from GUI shutdown / source switch / test cleanup.
        """
        rt = self._block_runtime
        if rt is None:
            return
        try:
            rt.stop(timeout_s=timeout_s)
        except Exception:
            log.exception("Engine: shadow Block runtime stop raised")
        self._block_runtime = None
        self._block_iq_in_stream = None
        self._block_iq_sample_counter = 0
        self._block_iq_anchor_wall = 0.0
