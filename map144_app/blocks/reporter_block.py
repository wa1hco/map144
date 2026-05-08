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
"""ReporterBlock — Phase 4 #12.

Durable consumer of :class:`DecodeEventStream` from
:class:`DecoderBlock`.  Wraps the existing
:class:`map144_app.reporting.Reporter` (PSKReporter IPFIX uploads +
DX cluster + WSJT-X UDP broadcast) so it slots into the block-pipeline
graph without changing any of the network protocol code.

Per design doc §9 #8: Reporter consumes a *durable* event stream
(POLICY_BLOCK on overflow, not POLICY_DROP_OLDEST).  Losing a decode
spot is worse than letting the stream backpressure briefly — operators
want every successful decode to reach PSKReporter exactly once.  In
practice the inner Reporter has its own queue + worker threads, so the
block's tick is a thin forwarder; backpressure on this block's input
is rare in real operation.

What the block does
-------------------
- Drains :class:`Event` records from its ``decodes`` input port.
- Filters: only events with ``payload['outcome'] == 'decoded'`` are
  forwarded.  no_decode / timeout / error events are silently dropped
  (those are statistics, not spots).
- Reconstructs the legacy ``decode`` dict shape that
  :meth:`Reporter.report_decode` expects, populated from the event's
  ``payload`` (which the :class:`_DecodeQueueShim` in DecoderBlock
  forwarded from ``extract_and_decode``'s queue.put dicts).
- Forwards via ``reporter.report_decode(decode_dict)``.

Settings (callsign, dial frequency, PSKReporter / DX-cluster enables,
WSJT-X UDP host/port) are applied via :meth:`apply_settings` and
:meth:`set_dial_freq` before the block runs.  These are session-level
config that the GUI populates from QSettings; we don't try to wire
them through ``BlockConfig`` because they change at runtime via UI
controls and are correctly owned by the existing Reporter class.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .block import Block
from .types import BlockConfig, Event, StreamClosed

log = logging.getLogger(__name__)


@dataclass
class ReporterBlockConfig(BlockConfig):
    """Per-ReporterBlock typed config.

    The block does NOT own callsign / network settings — those live
    on the wrapped :class:`Reporter` and are applied via
    :meth:`ReporterBlock.apply_settings` from the GUI.  This config
    only carries port names + tick-loop tuning.
    """

    decode_port_name: str = "decodes"

    # Get timeout — short enough for responsive shutdown; long enough
    # to avoid spinning when no decodes are arriving (most of the
    # time, since decodes are sparse).
    get_timeout_s: float = 0.250


class ReporterBlock(Block):
    """Forward decoded-event spots to PSKReporter / DXcluster / WSJT-X UDP.

    Inputs
    ------
    ``decodes`` : :class:`Event` (kind="decode") — emitted by
        :class:`DecoderBlock`.  Only events with
        ``payload['outcome'] == 'decoded'`` produce spots.

    Outputs
    -------
    None on the data plane (sink).

    Public methods (called from the GUI / settings layer)
    -----------------------------------------------------
    - :meth:`apply_settings` — propagates callsign, my_grid, network
      enables/host/port to the wrapped :class:`Reporter`.
    - :meth:`set_dial_freq` — propagates the radio's tuned dial
      frequency in Hz; the Reporter sends a Status update if it
      changed.
    """

    config_type = ReporterBlockConfig

    def __init__(self, name: str):
        super().__init__(name)
        # The wrapped network reporter.  Allocated in ``configure`` so
        # the blocks subpackage doesn't pull in `socket` / `random` at
        # module load.  May be ``None`` if configure() hasn't run yet.
        self._reporter = None

        # Pre-configure buffers.  apply_settings / set_dial_freq calls
        # made before configure() get queued here and replayed in
        # on_start() once the wrapped Reporter exists.  Pattern matches
        # the GUI's typical sequence: it builds the UI (which calls
        # apply_settings from saved QSettings) before the runtime is
        # started.
        self._pending_settings: dict | None = None
        self._pending_dial_freq: int | None = None

        # Stats.
        self._n_decoded_in:    int = 0
        self._n_decoded_sent:  int = 0
        self._n_other_skipped: int = 0   # no_decode / timeout / error

    # --- Lifecycle ----------------------------------------------------

    def configure(self, config: ReporterBlockConfig) -> None:  # type: ignore[override]
        super().configure(config)
        # Lazy import to keep import-time impact small for tests that
        # don't need a Reporter (e.g. the basic construction test).
        from ..reporting import Reporter
        self._reporter = Reporter()
        # Replay any pre-configure apply_settings / set_dial_freq calls
        # the GUI made before the runtime started.
        if self._pending_settings is not None:
            self._reporter.apply_settings(**self._pending_settings)
            self._pending_settings = None
        if self._pending_dial_freq is not None:
            self._reporter.report_freq(self._pending_dial_freq)
            self._pending_dial_freq = None

    def on_start(self) -> None:
        super().on_start()
        if self._reporter is None:
            raise RuntimeError(f"{self.name}: configure() must be called first")
        # If apply_settings has been called pre-start, the reporter
        # already has callsign + network state; if not, this is a
        # headless run (no reporting) — still fine, the worker just
        # won't have anything enabled.
        self._reporter.start()
        log.info(
            "%s: started.  my_call=%r wsjtx=%s pskreporter=%s dx=%s",
            self.name,
            getattr(self._reporter, "my_call", ""),
            getattr(self._reporter, "wsjtx_enabled", False),
            getattr(self._reporter, "pskreporter_enabled", False),
            getattr(self._reporter, "dx_enabled", False),
        )

    def on_stop(self) -> None:
        super().on_stop()
        if self._reporter is not None:
            try:
                self._reporter.stop()
            except Exception:
                log.exception("%s: reporter.stop() raised", self.name)
        log.info(
            "%s: decoded_in=%d, decoded_sent=%d, other_skipped=%d",
            self.name,
            self._n_decoded_in, self._n_decoded_sent, self._n_other_skipped,
        )

    # --- Run loop body ------------------------------------------------

    def tick(self) -> None:
        cfg: ReporterBlockConfig = self.config  # type: ignore[assignment]
        in_port = cfg.decode_port_name
        if in_port not in self.inputs:
            raise RuntimeError(
                f"{self.name}: input port {in_port!r} not connected"
            )

        try:
            evt = self.inputs[in_port].get(timeout=cfg.get_timeout_s)
        except TimeoutError:
            return
        # StreamClosed propagates to the run loop and exits cleanly.

        if not isinstance(evt, Event) or evt.kind != "decode":
            log.warning(
                "%s: dropped non-decode item %r on input port",
                self.name, type(evt).__name__,
            )
            return

        outcome = (evt.payload or {}).get("outcome")
        if outcome != "decoded":
            self._n_other_skipped += 1
            return

        self._n_decoded_in += 1
        decode_dict = self._build_decode_dict(evt)
        try:
            self._reporter.report_decode(decode_dict)
            self._n_decoded_sent += 1
        except Exception:
            log.exception(
                "%s: reporter.report_decode raised; spot lost", self.name,
            )

    # --- Helpers ------------------------------------------------------

    def _build_decode_dict(self, evt: Event) -> dict:
        """Reconstruct the legacy ``decode`` dict shape that
        :meth:`Reporter.report_decode` consumes.

        The ``_DecodeQueueShim`` in :class:`DecoderBlock` forwarded the
        legacy queue.put dict's fields into ``evt.payload`` minus a few
        we strip.  Reporter.report_decode reads:

          - message
          - jt9_snr
          - t_sec
          - radio_khz
          - utc_time  (optional — falls back to datetime.now in Reporter)

        Everything else is auxiliary.
        """
        p = evt.payload or {}
        return {
            "outcome":   "decoded",
            "message":   p.get("message", ""),
            "jt9_snr":   p.get("jt9_snr"),
            "t_sec":     p.get("t_sec", 0.0),
            "radio_khz": p.get("radio_khz", 0.0),
            "utc_time":  p.get("utc_time"),
            # Pass marker_id through for any GUI-overlay correlation
            # the legacy GUI does — no harm if unused.
            "marker_id": p.get("marker_id"),
        }

    # --- Public settings-passthrough API -----------------------------

    def apply_settings(self, **kwargs) -> None:
        """Forward to ``self._reporter.apply_settings``.

        Accepts the same kwargs the legacy Reporter does:
        my_call, my_grid, wsjtx_enabled, wsjtx_host, wsjtx_port,
        pskreporter_enabled, dx_enabled, dx_host, dx_port.

        Safe to call before configure() — kwargs get buffered and
        replayed inside configure() once the wrapped Reporter exists.
        Also safe at runtime to update settings while the block runs.
        """
        if self._reporter is None:
            # Buffer; replay in configure().  Merge with any prior
            # pre-configure call so multiple GUI settings updates
            # before runtime start coalesce into one apply_settings
            # call when configure() runs.
            if self._pending_settings is None:
                self._pending_settings = {}
            self._pending_settings.update(kwargs)
            return
        self._reporter.apply_settings(**kwargs)

    def set_dial_freq(self, freq_hz: int) -> None:
        """Forward dial-frequency change to the wrapped Reporter.

        The Reporter sends a Status UDP message if the frequency
        actually changed; harmless if called repeatedly with the same
        value.  Buffered if configure() hasn't run yet.
        """
        if self._reporter is None:
            self._pending_dial_freq = int(freq_hz)
            return
        self._reporter.report_freq(int(freq_hz))

    # --- Stats hook ---------------------------------------------------

    def stats(self) -> dict[str, Any]:
        base = super().stats()
        base.update({
            "decoded_in":     self._n_decoded_in,
            "decoded_sent":   self._n_decoded_sent,
            "other_skipped":  self._n_other_skipped,
        })
        # Forward selected Reporter stats so the JSONL block-stats dump
        # has visibility into the network side.
        if self._reporter is not None:
            base.update({
                "reporter_udp_sent":     getattr(self._reporter, "stat_udp_sent", 0),
                "reporter_psk_uploaded": getattr(self._reporter, "stat_psk_uploaded", 0),
                "reporter_psk_queued":   getattr(self._reporter, "stat_psk_queued", 0),
                "reporter_dx_sent":      getattr(self._reporter, "stat_dx_sent", 0),
                "reporter_dx_status":    getattr(self._reporter, "stat_dx_status", "?"),
            })
        return base
