# MAP144 Block / Stream architecture — design doc (sanity-check draft)

**Status:** sanity-check; no code yet. Goal is to agree on contracts before
TODO #6 (`Block` base class) and the Phase 3 refactors (#7–#9).

**Scope:** the runtime DSP pipeline only — Source → Channelizer → Detector →
Decoder → (Display, Reporter). Offline analysis tools (`compare_decoders.py`,
`gap*_diagnose.py`, etc.) are out of scope; they keep their current shape.

**Non-goals:**

- No algorithm changes. Same channelizer, same sync correlator, same SPD,
  same jt9 fallback. This doc is about packaging.
- No new sensitivity work. TODO #20 (frame-coherent integration) is independent
  and slots into the `Decoder` block once it lands.
- Not a multi-process plan. Threads with bounded queues; the GIL releases for
  numpy / numba / scipy / jt9-subprocess so this is enough.

---

## 1. Why bother

Today the pipeline is a single radio-source thread that runs ingestion, noise
blanking, channelization, the squared-FFT detector, and the sync detector
inline, then spawns daemon threads per detected ping for `extract_and_decode`.
Display reads three histories that the radio thread writes into. Reporter
reads a `_decode_queue` that the daemons write into. There is no explicit
boundary between any of these stages.

This works, but it has costs we have already paid:

- The "scrolling display dark-line" bug (see `project_sample_clock_anchoring`
  memory) was an instance of two stages disagreeing on what "now" means; one
  was sample-anchored, the fallback was wall-clock.
- TODO #19 was misframed (`project_todo19_misframed`) because the burst-window
  contract between sync-trigger and SPD was implicit.
- The radio-thread CPU budget is shared between ingest, channelizer, detector,
  and detector bookkeeping; back-to-back pings can stall ingest. Phase 2's
  numba vectorization bought headroom but did not change the topology.
- Adding a new source (USRP / SDRangel / WAV / RTL-SDR / Airspy) currently
  requires editing [runtime.py:1101–1280](../map144_app/runtime.py); each new
  source duplicates the "drain queue → call `process_iq_data`" pattern.

The argument for blocks is **observability and testability**, not throughput.
Each block boundary becomes a place we can:

- record a typed event stream (for offline replay and analysis #25–#28);
- swap implementations (sim source for tests; alternate detector);
- attach instrumentation without touching the producer or consumer.

If we cannot articulate a block boundary that gives us one of those wins, we
should not introduce it.

---

## 2. Glossary — the words we will use everywhere

The TODO calls this out specifically because today the same concept has
different names in different files. The standardisation:

| Term | Meaning | Units / scope | Replaces |
|---|---|---|---|
| **sample** | One complex IQ value at the stream's native rate | scalar | — |
| **record** | The unit of transfer on a stream. Sample-bearing record = a contiguous run of samples + time anchor; event record = one event. Time-ordered within a stream. | varies | (new) |
| **window** | A run of samples large enough for a stage's algorithm to run once (e.g. NSPM=864 for sync; 512 for squared FFT; 1.7 s for SPD extraction). A window is assembled from one or more records by the consumer. | n samples | informal "window" use in `processing.py:51,141,586`, `detection.py:72` |
| **stride** | Time advance between successive windows of the same stage (e.g. 256 samples @ 12 kHz between detector evaluations) | n samples | `_CH_DETECT_HOP` and "hop" in `processing.py:166,647,691,761,776` |
| **tick** | One scheduled iteration of a block's run loop. A tick may consume several records and emit zero, one, or several output records | dimensionless | (new — replaces "iteration"/"loop body") |
| **decimation** | Integer sample-rate reduction (today: 4× from 48 kHz → 12 kHz) | ratio | `decimate`, `DECIMATE_FACTOR` (kept as-is) |
| **frame** | An MSK144 protocol frame: 864 samples @ 12 kHz = 72 ms. Always means MSK144 frames in this codebase | 864 samp @ 12 kHz | `frame` in `msk144_spd.py:45,863`, `detection.py:126` (kept) |
| **burst window** | The IQ window handed to the decoder around a triggered detection (currently 1.7 s = 500 ms pre + 1200 ms post) | ~20 400 samp @ 12 kHz | "extract window" |

**Why `record` and not `frame` / `slice` / `chunk` / `waveform`:**

- `frame` (Simulink, MATLAB DSP / Comms) — the natural DSP term, but **collides with MSK144 frame** which is entrenched throughout the codebase.
- `slice` (informal name we tried first) — collides with Flex SmartSDR's `slice` (a virtual receiver: A/B/C, frequency + mode + filter; see [flexclient/client.py:122](../flexclient/client.py)). Reader switching files would get whiplash.
- `waveform` (LabVIEW) — closest semantic match (LabVIEW Waveform = `t0 + dt + Y + attributes`), but LabVIEW's `t0` is wall-clock; ours is sample-counter (see §3.1, `project_sample_clock_anchoring`). Borrowing the name with different semantics invites bugs.
- `chunk` — already used informally in [channelizer.py:101](../map144_app/channelizer.py); fine but weak time-ordered connotation.
- `record` (chosen) — generic streaming / database / event-log vocabulary (Kafka, Beam, Reactive Streams). Zero domain collisions in this codebase, and §3 was already using "time-ordered records" as the umbrella description.

Conventions:

- "Hop" goes away in new code. Existing constants (`_CH_DETECT_HOP`,
  `_hops_in_call`) get migrated when their consumer code moves into a block,
  per TODO #16. We do not do a rename pass on the legacy code first.
- "Step" is ambiguous (NCO phase step vs detector advance) and is banned in
  block-level interfaces. NCO code may still call its phasor `phase_step`
  internally because that is unambiguous in context.
- "Frame" never means "FFT frame" or "DSP frame" or "audio frame" in this
  codebase. If you mean an FFT window, write **window**.
- An **event** is a discrete record on a stream (a detection, a decode);
  a **record** is a span of samples on a stream. Both flow through the same
  Stream abstraction (§3) but they are different shapes.

---

## 3. Streams — the typed protocol

A **Stream** carries time-ordered records from one block to one or more
consumers. There are two flavours, decided at the producer:

### 3.1 Sample streams (continuous, sample-anchored)

Carry IQ or per-channel IQ. Each record:

```
Record  (sample-bearing)
  data                      : np.ndarray  # complex64, shape (..., n_samples)
  sample_rate_hz            : int
  start_sample              : int         # sample-clock anchor (§3.5)
  wall_clock_at_production  : float       # wall-clock anchor (§3.5)
  metadata                  : dict        # producer-specific (gain, packet seq, etc.)
```

**Invariants — non-negotiable:**

- `start_sample` is the canonical **sample-clock** time. Strictly monotonic
  within a stream connection. The only time used by DSP, display indexing,
  decimation alignment, NCO phase, FIR state, coherent averaging, and H/V
  pair coherence (§3.2). This is what `project_sample_clock_anchoring`
  enforces — moving the anchor to a stream contract makes it unforgeable
  downstream.
- `wall_clock_at_production` is the canonical **wall-clock** time for this
  record. Captured at the Source by reading `time.time()` *independently*
  from the sample counter (see §3.5). It is **not** derived from
  `start_sample / sample_rate_hz`. The two clocks drift over long runs
  (TCXO ppm error); the architecture is designed for that drift.
- Records are **contiguous and gap-free** within a single stream connection.
  A gap means the producer produces a new connection (or marks a `break_after`
  flag and the consumer drops its accumulated state). Detected sample drops
  are visible, not silently filled.
- `data.shape[-1]` is the record length and may vary tick-to-tick. Consumers
  that need a fixed window assemble it themselves with a small accumulator.

Concrete sample-stream types (named for clarity, but they are all `Record`):

- `IQStream` — raw IQ at source rate (48 kHz today). Shape `(n,)` mono or
  `(2, n)` for dual-pol H/V.
- `ChannelStream` — channelizer output. Shape `(N_CHANNELS=48, n)` at 12 kHz.
  Optionally `(2, 48, n)` for dual-pol.

### 3.2 Coherent pairs (dual-polarization)

Some sample streams must stay sample-aligned and phase-aligned with another
sample stream. The H and V polarization streams from a dual-pol Source are
the canonical case: they are coherent at acquisition (sampled by the same
ADC clock with deterministic phase relationship between RF1 and RF2) and
the polarization-combination math

> `combined = H·cosθ + V·exp(jΔφ)·sinθ`

is meaningless if the two streams have drifted relative to each other by
even one sample or have accumulated different filter group delays.

A **coherent pair** is two sample streams declared at the producer to be
coherent. The runtime guarantees, for any pair of records `(record_H, record_V)`
that originated from the same producer tick:

1. **Sample alignment** — `record_H.start_sample == record_V.start_sample`
   (they ride the same sample-counter; the producer emits both at once).
2. **Phase coherence** — the producer is responsible for not introducing a
   path-length difference between H and V that varies tick-to-tick. Static
   path-length differences are part of `Δφ` calibration; tick-jittered ones
   would be a bug.
3. **Lockstep backpressure** — see §5. If the consumer of one stream blocks,
   the producer of the *paired* stream also blocks. Dropping a record on H
   while admitting it on V would silently desync the pair, which is exactly
   the failure mode this invariant exists to prevent.
4. **Lockstep filter / decimation** — any block that processes one half of
   a coherent pair must process the other half with bit-identical taps,
   identical decimation alignment, and identical NCO phase initialisation.
   This is checked by construction: a `Channelizer` instance is created
   per polarization, but both share the same `ChannelizerState` *taps* (not
   *state* — H and V each get their own NCO and FIR memory).

Between two well-defined merge points, the H and V streams of a coherent
pair may take separate paths through the graph (separate Channelizer
threads, separate Detector threads if it ever helps). Coherence is not
asserted across all of time — only between matching records at any merge
point. The merge point is where the polarization combination happens (the
`Combiner` block in §6).

If a coherent-pair invariant is violated at runtime, the pair is closed
with an explicit `CoherenceLost` event and downstream blocks see the same
sentinel they would see on any stream close. We do not silently re-align.

### 3.3 Event streams (discrete, sparse)

Carry detection results, decode results, source heartbeats. Each record is
self-describing:

```
Event
  kind            : str                 # "detection" | "decode" | "heartbeat" | ...
  occurred_at     : (sample_rate_hz, sample_index)   # anchored on the upstream stream
  payload         : dict                # kind-specific
  metadata        : dict
```

**Invariants:**

- `occurred_at` is anchored to a sample-stream's `start_sample` so any event
  can be correlated to IQ post-hoc.
- Events are append-only and may be dropped under backpressure (§5) for
  best-effort streams (`DetectionEventStream`) but **never** for durable
  streams (`DecodeEventStream` — we have to ship those to PSKReporter).

Concrete event types:

- `DetectionEventStream` — one event per launch decision (channel, fc_hz,
  detector_metric_db, sync_metric_db, cluster geometry).
- `DecodeEventStream` — one event per successful decode (callsign, message,
  snr_db, source: SPD vs jt9, navg).
- `SourceMetadataStream` — best-effort heartbeat (#27): IQ-derived metrics
  every launch + every 15 s during quiet.

### 3.4 Why two flavours and not one

A sample stream carries *all* the data: dropping a record silently loses IQ
and breaks downstream invariants (the channelizer state, the rolling pct25
baseline). It must use a **block-on-full** queue or break the connection
explicitly.

An event stream is *commentary about* sample data and can be coalesced or
dropped without breaking downstream invariants (Display will draw the next
detection a few ticks later; it does not care). It uses **drop-oldest** or
**drop-newest** queues.

Mixing those two policies in one type forces every consumer to think about
which policy applies. Splitting the types puts the policy at the producer
where the producer already has the information.

### 3.5 Two clocks, by construction

The system has **two independent clocks** and both are first-class.

1. **Sample clock** — the radio's ADC clock. Strictly monotonic, deterministic,
   drives all DSP. One ADC tick = one sample. Time in sample clock is
   `start_sample / sample_rate_hz`.
2. **Wall clock** — OS time (`CLOCK_REALTIME`, NTP-disciplined). Drives
   reporting, logging, period-edge detection, file naming, cross-station
   correlation. Whatever `time.time()` returns when read.

These drift relative to each other. A typical TCXO-equipped ham radio drifts
~0.5–2 ppm; at 1 ppm:

| Run duration | Implied drift |
|---|---|
| 1 hour | 3.6 ms |
| 1 day | 86 ms |
| 1 week | 600 ms |
| 1 month | 2.6 s |
| 1 year | 31 s |

For a "weak-signal monitor running unattended for months" use case, deriving
wall clock from sample count produces nonsense PSKReporter timestamps and
breaks cross-station log correlation. The architecture must not assume
`start_sample / sample_rate_hz ≈ wall_clock_elapsed`.

#### Operations split by which clock they care about

| Sample-clock operations (deterministic, monotonic) | Wall-clock operations (UTC) |
|---|---|
| FFT / correlation / FIR filter state | PSKReporter timestamps |
| Decimation alignment | DXcluster timestamps |
| NCO mixing phase | Log entries shown to operator |
| Coherent averaging in SPD | "Is this a 15-s WSJT-X period?" |
| H/V coherent-pair alignment (§3.2) | File names with date stamps |
| Spectrogram column placement | Cross-station log correlation |
| MSK144 frame boundaries | Scheduled start/stop, daily rotation |
| Reproducible replay from saved IQ | Heartbeat timestamps (TODO #27 analysis) |

These are not the same time measured two ways. They would be if the radio's
TCXO and the OS's NTP-disciplined oscillator were the same source. They
aren't.

#### Records carry both anchors

Every sample-bearing Record has both `start_sample` (sample clock) and
`wall_clock_at_production` (wall clock). Neither is derived from the other.

#### Producer rules

- **Source** captures both anchors at production time by reading two
  independent observations: `start_sample` from the radio's sample counter
  (or a Source-maintained counter if the radio doesn't expose one), and
  `wall_clock_at_production = time.time()`. They are produced as a pair but
  computed independently.
- **Downstream blocks producing new Records** (Channelizer, Combiner)
  **inherit** `wall_clock_at_production` from the upstream record unchanged.
  These blocks are deterministic functions of their input; the Source is the
  authoritative wall-clock anchor for that signal.
- **Events** captured by a Block (`DetectionEvent`, `DecodeEvent`) carry their
  own `wall_clock_at_production` representing **when the event happened**
  (not when the underlying signal was acquired). For "when did the ping
  happen?" semantics, the event also carries `signal_wall_clock` inherited
  from the upstream record's wall clock. The difference is the block's
  processing latency, exposed by construction.

#### Consumer rules

- **DSP code** reads `start_sample` only. Wall clock is invisible.
- **Reporting code** reads `wall_clock_at_production` only (or `signal_wall_clock`
  for "when the ping happened" semantics). Sample counter is invisible.
- **Replay code** reads `start_sample` for reproducibility. The recorded
  `wall_clock_at_production` is preserved as-was (it's part of the recorded
  stream, not re-derived).
- **Cross-station / cross-system correlation** uses wall-clock fields
  exclusively.

#### Period anchoring — timing events within a 15-s period

Keeping the clocks separate leaves one question: what absolute UTC do we stamp
on an event detected mid-period? Two naive answers are both wrong:

- `time.time()` read at *event-processing* time — contaminated by per-event
  processing latency (detection/decode/log lag), so the stamp is later than when
  the ping actually arrived.
- `period_start_wall + event_sample / rate` from a *single session-start* anchor
  — accumulates TCXO drift over the whole run. This is the legacy 1970-bug shape.

The rule: **at each 15-s period boundary, capture one synchronized
`(period_start_sample, period_start_wall)` pair** — both clocks read together,
once per period. An event at `event_sample` within that period is stamped:

    utc(event) = period_start_wall + (event_sample - period_start_sample) / sample_rate

Why this is correct:

- **Intra-period timing is sample-precise and monotonic.** The offset
  `(event_sample - period_start_sample)` is exact, so sub-burst timing and event
  ordering within the period are exact — which is what burst-mode placement and
  coherent-integration frame selection need.
- **The period's absolute UTC is fresh.** `period_start_wall` is re-read each
  period, so cross-period drift cannot accumulate.
- **No event-latency jitter.** The stamp reflects when the *samples* arrived, not
  when a block got around to processing them.

Drift *within* one period is negligible (15 s × 1 ppm = 15 µs), so the short
sample-derived interval is safe; only *cross-period* extrapolation is unsafe, and
the per-period re-pair eliminates it. This is "compare the two clocks once per
period," **not** "slew either clock" — the clocks stay independent and
uncorrected (next subsection). `TimeBase` owns this: `mark_period_start(sample)`
captures the synchronized pair; `utc_at(event_sample)` applies the formula above.

#### Drift is observable, not corrected

The Source publishes a `ClockHealth` event roughly once per second:

```
ClockHealth event payload:
  sample_counter         : int      # current sample-clock position
  wall_clock_at_check    : float    # time.time() at the same instant
  initial_anchor         : (sample=int, wall_clock=float)   # at session start
  implied_drift_seconds  : float    # wall_clock_at_check - (counter/rate + initial_anchor.wall_clock - initial_anchor.sample/rate)
  drift_ppm              : float    # implied_drift_seconds / elapsed_wall * 1e6
```

The drift is **logged and made available to TODO #27's analysis bag** but
**not acted on**:

- Slewing the sample clock to match wall clock would corrupt DSP phase
  coherence and break decimation alignment.
- Slewing wall clock to match sample clock would corrupt reporting and break
  cross-station correlation.

They drift independently and that is the truth. NTP step-corrections to the
OS wall clock are absorbed naturally — the next `wall_clock_at_production`
reading reflects the post-step value, no special handling required.

#### Drift thresholds (sensitive by design, so the path gets exercised)

Drift handling is the canonical "can't reach the failure state in test time"
problem. You can't wait days to validate the drift code path. The defaults
below are picked so the **characterization** path is exercised within any
realistic development session on healthy hardware — without dressing routine
TCXO drift up as a warning:

| Tier | Threshold | Log level | When it fires on a typical Flex TCXO (~1 ppm) |
|---|---|---|---|
| **OBSERVE** | always (every `ClockHealth` event, 1 Hz) | DEBUG | every second — measurement path always exercised |
| **CHARACTERIZE** | `|drift| > 2 ms`, rate-limited to once per 60 s | INFO | within ~33 minutes — system characterization, expected |
| **WARN** | `|drift| > 100 ms`, rate-limited to once per 60 s | WARNING | should never fire in healthy operation |

CHARACTERIZE is **not** a warning. It is the system describing its own
TCXO drift behaviour for the operator and for offline analysis. Calling it
WARN would create alert fatigue and miscommunicate intent — it fires on
healthy hardware by design. INFO is the right log level: visible at default
verbosity, not escalated by monitoring.

Picking 2 ms for CHARACTERIZE means: **on every healthy multi-hour session,
the drift-measurement code path runs at least once in production**. We do
not have to remember to test it. If CHARACTERIZE ever stops firing on a
long-running daemon, that itself is information (TCXO is suspiciously
stable, or the drift logic is broken).

WARN is "something is actually off" — failing TCXO, NTP slewing
pathologically, or a wall-clock hardware fault. It should not fire in
normal operation; if it does, the operator should look.

Rate-limiting is "emit once per 60 s after threshold crossed", not
"sustained for N seconds." TCXO drift is monotonic on second-timescales
(no flapping to de-bounce); a simple time-since-last-emit rate limit is
sufficient for both tiers.

#### Test-injection hook

A runtime flag — `MAP144_TEST_CLOCK_DRIFT_PPM` env var or
`--test-clock-drift-ppm` CLI — adds synthetic drift to the Source's
`wall_clock_at_production` reading at a controlled rate (in ppm relative to
sample clock). With `--test-clock-drift-ppm 1000` (1 ms/s), the CHARACTERIZE
threshold (2 ms) is hit in 2 s and the WARN threshold (100 ms) in 100 s.
Both code paths reachable in a unit test.

This is a Source-internal detail: the synthetic offset is added inside the
Source's wall-clock-anchor capture, so downstream blocks see a consistent
(if intentionally drifting) wall clock and the dual-clock contract is
preserved. DSP behaviour is unchanged — `start_sample` is untouched.

The hook is **not** a production feature. It is the only practical way to
write an automated test for clock-drift handling.

#### Cautionary tale — the legacy monolith violates this section

The pre-refactor `process_iq_data` (`processing.py`) **derives wall clock from
sample count**, exactly what the rule above (this §3.5, "must not assume
`start_sample / sample_rate_hz ≈ wall_clock_elapsed`") forbids:
`_loop_wall = _iq_t0_wall + _iq_abs_sample / sample_rate`, then
`period_idx = int(_loop_wall / 15)`. It keeps a single session-start anchor
(`_iq_t0_wall`) as mutable engine state, reset in only one place
(`_reset_wav_timeline`, the WAV path).

On 2026-05-31 this produced the **1970-epoch TFMF-timestamp bug**: a session
that ran WAV replays (anchor driven to ≈0) then switched to live radio
inherited the stale ≈0 anchor — the live-start path never reset it — so
`_loop_wall` became seconds-since-start, `period_idx` went tiny, and
`datetime.fromtimestamp(period_idx*15)` landed in 1970. ~200k overnight
candidates were unusable for time-windowed comparison; "TFMF saw 0/378 signals"
was a measurement artifact of the broken timestamp, not a detection failure.

The architecture keeps the two clocks independent and never slews either one.
The *only* place they are compared is a synchronized re-pair at each period
boundary ("Period anchoring" above), used to anchor intra-period event timing —
once every 15 s, the natural cadence, never extrapolated across periods. The
legacy bug is the opposite shape: a single session-start anchor, wall *derived*
from sample, mutated as engine state, leaking across a source switch.

The fix is **not** another anchor-reset patch (see CLAUDE.md "Refactor /
migration policy") — it is the `TimeBase` migration: one owner (the Source),
one reset point (source open) with an explicit per-source anchor policy
(radio = `time.time()`; WAV = a chosen policy, not ≈0 by accident). **The 1970
bug is `TimeBase`'s acceptance test.**

---

## 4. The Block contract

A `Block` is a unit of computation that owns one OS thread, reads from zero
or more input streams, and writes to zero or more output streams. The
contract:

```
class Block:
    name : str
    config_type : type[BlockConfig]           # subclass-defined dataclass

    # declared at construction; immutable after start()
    inputs  : list[Stream]
    outputs : list[Stream]

    # lifecycle; called by the runtime, not by other blocks
    def configure(self, config: BlockConfig) -> None  # typed dataclass per block
    def start(self) -> None                           # spawn the run thread
    def stop(self, timeout_s: float) -> None          # cooperative shutdown; drain outputs
    def join(self) -> None

    # the run loop body — implemented per-block
    def tick(self) -> None                    # one iteration; called repeatedly while running

    # observability
    def stats(self) -> dict                   # see §9 #8 for the field set
```

**Threading rule:** one thread per block. The run loop is

```
def _run(self):
    self.on_start()
    try:
        while not self._stopping:
            self.tick()
    finally:
        self.on_stop()
```

`tick()` is allowed to block on its inputs; the runtime has nothing to
schedule. This is deliberately the simplest possible model. We are not
introducing async/await, we are not introducing a cooperative scheduler,
we are not introducing actor mailboxes. Python threads + bounded queues is
sufficient because every heavy-CPU stage releases the GIL (numpy / numba /
scipy / jt9 subprocess).

**Why a thread per block instead of a thread pool:**

- Each block has a clear identity in `top` / `py-spy` output.
- Per-block CPU profiling is just attaching to the block's PID/TID.
- A misbehaving block (e.g. SPD on a pathological window) does not steal
  ticks from the radio thread. Today it can.

**Configuration:** `configure()` receives a per-block typed dataclass
(`config_type`). The runtime composes a `RuntimeConfig` from per-block
sections at graph build time. Stream types and queue depths live on the
block's connection graph, not on individual blocks; see §6.

**Single instance vs multi-instance:** every block is a Python object that
can be instantiated multiple times. Dual-pol becomes "two `Channelizer`
instances feeding one `Detector` that takes both" rather than dual-pol code
inside `Channelizer`. This is a refactor we want, not a contract requirement.

---

## 5. Backpressure model

The default queue between two blocks is bounded. The default size is declared
at the connection, not the block:

```
runtime.connect(src, "iq", dst, "iq", queue_depth_seconds=1.0, on_full="block")
```

(See §9 #3 — `runtime.connect` was chosen over operator-overload syntax for greppability.)

**Policies, by stream type:**

| Stream | `queue_depth_seconds` | `on_full` | Reason |
|---|---|---|---|
| IQStream (Source → Channelizer) | 1.0 s | **block** | Dropping IQ silently is the bug class we are trying to make impossible |
| IQStream (Source → Decoder, on-trigger window-read) | n/a | shared ring | Decoder reads a fixed-length window from `_iq_ring`-equivalent; not a queue |
| ChannelStream (Channelizer → Detector) | 1.0 s | block | Same — channelizer state is sample-exact |
| DetectionEventStream (Detector → Decoder, Display) | 16 events | drop-oldest | Decoder is rate-limited by its semaphore; if it cannot keep up, dropping the oldest is correct (the new ping is more interesting than the old one) |
| DecodeEventStream (Decoder → Reporter) | 256 events | block | We must not lose decodes; PSKReporter is durable. 256 entries × ~5 min upload interval is way more than we will ever queue |
| SampleStream (anything → Display) | 1 record | drop-oldest | Display always wants "latest" |
| SourceMetadataStream (Source → analysis sink) | 64 events | drop-oldest | Heartbeat is best-effort by definition |

**On-full = block** is what makes the radio thread visibly slow down rather
than silently corrupt. If the Channelizer cannot keep up with the Source,
the Source's `tick()` blocks on the queue. The driver's underlying socket
buffer eventually fills; we then see an explicit overrun (see TODO #4 — add
explicit overflow-drop logging). That is what we want.

**On-full = drop-oldest** for event streams is the pattern Detector and
Display use today implicitly (Detector spawns a daemon thread per launch
without checking whether the previous one finished; Display always reads the
latest history). Making it explicit means the drop is logged.

**Coherent-pair backpressure (§3.2).** When two streams are a declared
coherent pair, their queues are linked: a write blocks if *either* queue is
full, and a record is admitted to *both* or *neither*. The runtime enforces
this at the connection — a paired connection is a single object that fans
out into two queues but presents one `put()` and one `get()` semantically.
This is what guarantees H and V never drift by a record. The cost is that a
slow consumer on one polarization slows down the other; that is the right
tradeoff because the alternative is silent loss of coherence.

**Open questions in §9:** whether to expose `queue_depth_samples` instead of
seconds for IQ streams; whether to merge "block" and "block-with-warning" or
keep them separate.

---

## 6. The proposed connection graph

Today (radio path):

```
[FlexDAXIQ / USRP / SDRangel / Airspy / RTL-SDR / WAV] callback / drain loop
        │
        ▼
process_iq_data ─┬─ noise blanker
                 ├─ apply_channelizer ─ _ch_buf
                 │                       │
                 │                       ▼ (squared-FFT detector + sync correlator + cluster gate)
                 ├─ _iq_ring             │
                 │  ▲                    ▼ launch decision
                 │  │              ┌─────┴──────┐
                 │  │              ▼            ▼
                 │  │     extract_and_decode    _ch_snr_history_h ──► Display
                 │  │     (daemon thread)
                 │  │              │
                 │  └──────────────┘ window read
                 │              │
                 │              ▼
                 │     SPD → jt9 fallback → _decode_queue ──► Reporter
                 ▼
              _sbuf ──► waterfall
```

Proposed:

```
Source ── IQStream ──► Channelizer ── ChannelStream ──► Detector ── DetectionEventStream ──┐
   │                                          │                                            │
   │                                          ├──────► Display (channel power heatmap)     │
   │                                          │                                            ▼
   └──────────────────────── IQStream ────────┴──────────────► Decoder ── DecodeEventStream ──► Reporter
                                                              (consumes IQ window
                                                               on each detection event)
```

| Block | Owns today | Inputs | Outputs | Notes |
|---|---|---|---|---|
| **Source** | flex_client / usrp / sdrangel / airspy / rtl-sdr / WAV ingress in [runtime.py:1101–1280](../map144_app/runtime.py); also the `_iq_t0_wall + _iq_abs_sample / sr` fallback in [processing.py:329–339](../map144_app/processing.py) | radio handle | `IQStream` | Per-source subclass (`FlexSource`, `UsrpSource`, …). Owns the sample-clock anchor. After this block, no code looks at `time.time()` for stream-time. |
| **NoiseBlanker** | NB code path inside `process_iq_data` | `IQStream` | `IQStream` | Optional. Pass-through if disabled. Open question §9: separate block or first-stage of Source? |
| **Channelizer** | [channelizer.py](../map144_app/channelizer.py) `apply_channelizer` + the `_ch_buf` accumulator in [processing.py:455–481](../map144_app/processing.py) | `IQStream` | `ChannelStream`, `IQStream` (pass-through for waterfall + decoder window) | Note: today `_ch_buf` is also where the detector reads. Under the new graph the Channelizer emits a `ChannelStream` and the Detector buffers what it needs; the Channelizer does not own a multi-consumer accumulator. |
| **Detector** | sync correlator (`_sync_correlate_batch`) + squared-FFT detector + cluster gate in [processing.py:516–889](../map144_app/processing.py) | `ChannelStream` | `DetectionEventStream`, internal `ChannelMetricStream` for Display | Owns `_detect_cooldowns`, `_ch_above_thresh`, the rolling pct25 baseline. Independent of source thread. TODO #5 (cluster-gate tightening) and TODO #6 (noise-floor-rise hysteresis) live here. |
| **Decoder** | `extract_and_decode` ([detection.py:422–571](../map144_app/detection.py)), SPD ([msk144_spd.py](../map144_app/msk144_spd.py)), jt9 fallback | `DetectionEventStream`, `IQStream` (windowed read on event) | `DecodeEventStream` | Internally pools jt9 (existing semaphore). Reads IQ from a shared ring, not via a queue — the trigger fires, the Decoder reads a 1.7 s window from the ring at the event's `occurred_at`. |
| **Display** | heatmap rendering, marker overlay, waterfall in [displays.py](../map144_app/displays.py) and [processing.py:901–950](../map144_app/processing.py) | `ChannelMetricStream`, `DetectionEventStream`, `IQStream` (waterfall) | — | Owns `_ch_snr_boundary`, `_ch_snr_hop_count`, `_jt9_markers`, the scrolling state. TODO #14 cleanup is automatic once Display is the only writer. |
| **Reporter** | [reporting.py](../map144_app/reporting.py), `_decode_queue` consumer | `DecodeEventStream` | — | PSKReporter / DXcluster. Already mostly a block; just needs the typed input. |

### 6.1 Dual-polarization graph

For dual-pol sources (B210, SDRangel H+V), the same blocks are instantiated
twice on the polarization branches, with one new block at the merge point:

```
                                       ┌── ChannelStream_H ──► Detector_H ──┐
Source ── (IQStream_H, IQStream_V) ────┤                                    │
        coherent pair (§3.2)           │                                    ├──► Combiner ── ChannelStream ──► Decoder ──► Reporter
                                       └── ChannelStream_V ──► Detector_V ──┘
                                          (paired, lockstep backpressure)
```

The **Combiner** block consumes a coherent pair of `ChannelStream`s and
emits a single `ChannelStream` representing the polarization-combined
signal `H·cosθ + V·exp(jΔφ)·sinθ`. From the Decoder's perspective downstream,
the input is indistinguishable from a mono path. CLAUDE.md's polarization
math lives entirely inside the Combiner.

**Why the merge point is at ChannelStream and not earlier or later:**

- **Earlier (combine raw IQ):** the polarization measurement (which θ and Δφ
  to use) is per-channel — different signals at different frequencies have
  different polarization rotations. Combining at IQStream forces one global
  θ/Δφ for the whole 48 kHz band, which is the wrong physics.
- **Later (combine after decode):** loses sensitivity. The whole point of
  polarization combining is to recover SNR before detection / decode.
- **At ChannelStream:** each channel can carry its own θ/Δφ, the combination
  can be optimised per detection event, and the Decoder still sees a single
  combined input. This is also where existing experimental code combines.

The Detector branches **before** the Combiner exist so each polarization
can run its own detection (a signal that is strong on H but absent on V
should still launch a decode). The cluster gate in each Detector is
independent; the Combiner accepts detection events from either branch and
combines IQ on demand at the requested θ/Δφ.

**Polarization measurement model — v1: static config.** θ and Δφ are read
from the Combiner's config dataclass at startup (one set per channel, or a
single global set with per-channel offsets). Per-event optimisation
(Combiner sweeps θ/Δφ for each detection event, picks the one that
maximises sync metric) is a v2 enhancement — gated behind a config flag,
implemented after Phase 4 lands. An external `PolarizationModelStream` /
calibration block is deferred until experiments establish what data drives
the model; the contract for that stream is not designed yet.

This is more wiring than a `(2, 48, n)` single-stream model, but it keeps
the coherent-pair invariant out of every consumer's contract: only the
Source and the Combiner have to know about pairing. Everything between
them is just "two streams that happen to share the same `start_sample`."

### 6.2 Tap

**The Decoder's IQ access pattern needs a name.** It is not a stream-consumer
in the queue sense — it does a windowed *random read* into the IQ history.
We call this a **Tap**: a Block declares a tap on a stream, and the runtime
keeps a ring buffer at the tap point sized for the longest window the
consumer might ask for (1.7 s for the Decoder; ~10 s for the waterfall).
Open question §9 — whether Tap is a first-class concept in the protocol
or just the Decoder's private detail.

**Single-source blocks first.** Phase 3 starts with the Flex source. USRP /
SDRangel / Airspy / RTL-SDR / WAV ingestion all inherit the same `IQStream`
contract; they become subclasses of `Source` in Phase 3 cleanup, not Phase 3
core.

---

## 7. Lifecycle, in detail

The runtime owns a graph of blocks plus their connections. Startup:

1. `runtime.build_graph()` constructs blocks and connections.
2. `runtime.start()` calls `block.configure()` on every block (validates; no
   threads yet).
3. `runtime.start()` calls `block.start()` in **topological order from sinks
   to sources**: Reporter, Display, Decoder, Detector, Channelizer, Source.
   Sources are last so consumers are ready before any data flows.
4. Each block's run thread enters its `tick()` loop.

Shutdown is the reverse:

1. `runtime.stop()` calls `block.stop(timeout)` from **sources to sinks**:
   Source, Channelizer, Detector, Decoder, Display, Reporter.
2. A block's `stop()` sets `self._stopping`, then closes its output streams.
3. A consumer reading a closed stream sees a sentinel and exits its `tick()`
   loop after draining what is already queued.
4. `runtime.join()` waits for all threads.

**Pause / resume:** out of scope for v1. The current code does not need it.

**Failure handling:** if a block raises in `tick()`, the runtime logs and
shuts the whole graph down. We do not try to restart a block in v1; the
current code does not either.

**Backfill / replay:** out of scope, but the design admits it — a `Source`
that reads from a saved IQ file and emits an `IQStream` is exactly the same
contract as the live source. This is what makes #25–#28 (offline analysis)
trivial to implement on top.

---

## 8. What this preserves vs changes

**Preserved:**

- Numerical behaviour: same channelizer taps, same NCO, same sync correlator,
  same SPD, same jt9 fallback, same cluster-gate logic.
- Sample-clock anchoring: enforced at the `IQStream` contract instead of by
  convention.
- Single-writer / multi-reader on the IQ history: still true; the Decoder's
  windowed read uses a ring, not a queue.
- Detection invariants from CLAUDE.md (cluster definition, only-best cooldown,
  fresh-channel too-many gate, no proximity guard): untouched. They live
  inside the Detector block.

**Changed:**

- Concurrency: ingest, channelize, and detect run on independent threads
  instead of one. Bounded queues make backpressure visible.
- Naming: hop → stride for new code; "frame" only means MSK144 frame.
- State ownership: Display owns its scrolling state; Detector owns its
  cooldown bookkeeping; Source owns the time anchor.
- Per-source code reduces to a `Source` subclass per radio.

**Sensitivity / detection behaviour:** zero change is the goal. Validation
plan in §10.

---

## 9. Decisions (formerly open questions)

All resolved in this review round. Listed in original numbering for
traceability with the conversation history.

1. **Tap vs Stream — private.** Tap stays a Decoder-internal detail. The
   waterfall does the same windowed read but keeps its own private ring; no
   protocol concept for two users.

2. **Configuration shape — per-block typed dataclass.** Each block declares
   `config_type : type[BlockConfig]`. The runtime composes a `RuntimeConfig`
   from per-block sections at graph build time. Each block is independently
   testable with its own config.

3. **Stream-connection API — `runtime.connect(src, "out_port", dst, "in_port", **policy)`.**
   No operator overload. Greppable; explicit; easy rename refactors; works
   with editor jump-to-definition.

4. **NoiseBlanker placement — inside Source.** First stage of every Source
   subclass. Promote to a separate block later only if a non-Flex source
   needs a different blanker.

5. **Dual-pol channel-stream shape — two streams as a coherent pair.**
   See §3.2 and §6.1. Coherence is a contract on the *pair*, not on every
   block downstream. Combination at `ChannelStream` granularity in the
   Combiner so per-channel θ/Δφ is meaningful.

6. **Display update cadence — 100 ms (10 Hz) default, configurable.**
   Matches the existing `QTimer` cadence. Display block runs in its own
   thread and pushes updates to the GUI thread via Qt signals. Higher than
   10 Hz isn't worth the CPU; lower than 4 Hz feels laggy. Tunable per the
   block's config dataclass.

7. **Backpressure on `IQStream` to Source — block-on-full.** Source blocks;
   OS / radio surfaces the overflow. Silent record drops would break
   coherent pairs (§3.2) and corrupt downstream invariants.

8. **`Block.stats()` — JSONL log, no Prometheus in v1.** Returned dict has:

   ```
   {
     "name": str,
     "runtime_seconds": float,
     "ticks_executed": int,
     "samples_in":  {port_name: int},
     "samples_out": {port_name: int},
     "events_in":   {port_name: int},
     "events_out":  {port_name: int},
     "queue_full_seconds": {port_name: float},   # cumulative blocked time per output
     "tick_latency_ms_p50": float,
     "tick_latency_ms_p99": float,
     "errors_in_tick": int,
   }
   ```

   Dumped JSONL on shutdown. `--stats-jsonl-path` CLI flag for periodic
   dumps (1 line per block per minute). Prometheus / OpenMetrics endpoints
   are out of scope for v1 — they add a dependency and change the
   deployment shape; can be added later as a separate Reporter-style block
   reading the stats stream.

9. **Burst window — `burst_pre_s=0.5`, `burst_post_s=1.2` separately.**
   Different concepts (pre = sync-search slack; post = multi-frame decode
   reach). TODO #20 (frame-coherent integration) may grow `burst_post_s`
   without changing `burst_pre_s`. Keeping them separate makes the
   asymmetry observable and tunable.

10. **Replay tools — Phase 3.** `WavSource` (subclass of Source, emits
    `IQStream` from a recorded file with the recording's original
    `wall_clock_at_production` preserved) and `JsonlSink` (consumer block
    that writes any event stream to JSONL). ~200 LOC total. They are how
    we run §10's bit-exact-replay validation gate; the validation is
    unprovable without them.

11. **Polarization measurement — static config for v1.** θ and Δφ in the
    Combiner's config dataclass. Per-event sweep is a v2 enhancement
    behind a config flag, after Phase 4 lands. External
    `PolarizationModelStream` / calibration block deferred until
    experiments establish what data drives the model.

---

## 10. Validation plan

Before considering Phase 3 done, the following must hold:

- **Bit-exact replay equivalence.** Save a 30 s captured IQ session through
  the old pipeline (record `_iq_ring`, all detector decisions, all decode
  results). Replay the same captured IQ through the new pipeline.
  Tolerances:
  - Detection events: identical set (no extras, no missing). Per event:
    same channel index; `fc_hz` exact (integer Hz); `pair_metric_db` within
    ±0.01 dB; `sync_metric_db` within ±0.01 dB.
  - Decode events: identical set. Per event: callsign string-equal; `snr_db`
    within ±0.1; SPD/jt9 source field exact; `navg` exact.
- **Multi-seed ramp regression.** `run_ramp_tests.py` (TODO #23 lives here)
  on the new pipeline, compared to the same seeds on the old pipeline.
  ≥ 1000 trials per SNR point. 50 % decode-rate threshold within ±0.2 dB
  of the old-pipeline curve. (At 1000 trials a 50 % point has ~1.6 %
  binomial std, so ±0.2 dB is comfortably above noise.)
- **Live A/B.** A 30-min live capture on a known-good band, both pipelines
  fed the same IQ via a tee. Decode counts identical. Per matched decode:
  same callsign; `fc_hz` within 1 Hz; `snr_db` within ±0.1.
- **CPU regression.** New pipeline ≤ +20 % CPU vs old at idle; ≤ +10 % CPU
  during dense activity. Higher than that means a queue-handling problem,
  not "threading is expensive."
- **Drift-injection test (§3.5).** Run with `--test-clock-drift-ppm 1000`
  for 200 s. Assert: CHARACTERIZE log fires at 2 ± 0.5 s after first
  `ClockHealth` event; WARN log fires at 100 ± 1 s; DSP output (decoded
  callsigns, sync metrics, channel metrics) bit-identical (per the
  bit-exact tolerances above) to a no-drift run with the same IQ input.
  Validates that the dual-clock split actually works — drift in wall clock
  must not perturb `start_sample`-driven DSP.

Failing any of these is a stop-the-world condition; we do not ship the
refactor.

---

## 11. What I am NOT proposing

These are tempting because the refactor opens space for them. They are out
of scope for Phase 3 to keep the change reviewable.

- ❌ Changing the channelizer (e.g. variable channel count, GPU FFT).
- ❌ Changing the SPD coherent-averaging core. That is TODO #20.
- ❌ Async / await. The threads-with-queues model is enough.
- ❌ Persistent log of all stream events. JSONL recording for replay is
  scoped to debug-mode only.
- ❌ A plugin / marketplace API for external blocks. Internal use only.
- ❌ Multiprocessing. The GIL is not the bottleneck (numba, numpy, scipy,
  jt9-subprocess).

---

## 12. Decisions — all ratified

Every contract and every implementation choice this doc raises has a
position. Nothing here is "still open." Push-back goes through the doc, not
the code.

### Contracts (architectural — won't change without a doc revision)

1. ✅ **Glossary** in §2 — "stride replaces hop in new code only", "frame
   means MSK144 frame only", "step banned from block interfaces", "record"
   is the unit of transfer (chosen over `slice` / `frame` / `waveform` /
   `chunk` — see §2 rationale).
2. ✅ **Two stream flavours** in §3 — sample stream (block-on-full) vs event
   stream (drop-oldest by default).
3. ✅ **Coherent pair** invariant in §3.2 for dual-pol — sample-aligned,
   phase-aligned, lockstep backpressure, lockstep filtering between merge
   points. Combination at `ChannelStream` granularity in a `Combiner` block.
4. ✅ **Two clocks, dual-anchor Records** (§3.5). `start_sample` for DSP /
   coherence / display indexing; `wall_clock_at_production` for reporting /
   logs / cross-station correlation. Both first-class; neither derived from
   the other; drift observed via `ClockHealth` and logged but **not
   corrected**. Designed for unattended runs of months.
5. ✅ **One thread per block**, no scheduler, no async, no actors.
6. ✅ **Block list and graph in §6 / §6.1** — Source, Channelizer, Detector,
   Combiner (dual-pol only), Decoder, Display, Reporter.

### Implementation choices (tactical — can change with a small doc edit)

7. ✅ **NoiseBlanker inside Source** (not its own block) for v1.
8. ✅ **Source backpressure: block-on-full.** OS / radio surfaces the overflow.
9. ✅ **Tap as a Decoder-internal detail**, not a protocol concept.
10. ✅ **Configuration: per-block typed dataclass** composed into a
    `RuntimeConfig` at graph build time (§9 #2).
11. ✅ **Connection API: `runtime.connect(src, "out", dst, "in", **policy)`** —
    no operator overloads (§9 #3).
12. ✅ **Display tick rate: 100 ms (10 Hz) default**, configurable (§9 #6).
13. ✅ **`Block.stats()`: JSONL on shutdown; `--stats-jsonl-path` for periodic
    dumps; no Prometheus in v1** (§9 #8 for the dict shape).
14. ✅ **Burst window: `burst_pre_s=0.5`, `burst_post_s=1.2`** as separate
    Decoder config fields (§9 #9).
15. ✅ **Replay tools (`WavSource`, `JsonlSink`) land in Phase 3** — needed
    for §10 bit-exact-replay validation (§9 #10).
16. ✅ **Polarization measurement: static config for v1**, per-event sweep
    behind a v2 flag, external calibration block deferred (§9 #11, §6.1).
17. ✅ **Drift tiers: OBSERVE / CHARACTERIZE / WARN** at DEBUG / INFO /
    WARNING; thresholds 2 ms (CHARACTERIZE) and 100 ms (WARN); test-injection
    via `--test-clock-drift-ppm` (§3.5).
18. ✅ **Validation tolerances** in §10 — detection events identical,
    `pair_metric_db` ±0.01 dB; decode events identical, `snr_db` ±0.1; ramp
    50 % threshold ±0.2 dB at ≥1000 trials/SNR; CPU ≤ +20 %/+10 % idle/dense.

### Phase ordering (unchanged from previous version)

TODO #6 (`Block` base class) is unblocked. Phase 3 refactors proceed in
order: #7 Source → #8 Channelizer → #9 Detector. The Combiner is inserted
as part of #9 (dual-pol path only). Phase 4 (#10 Display, #11 Decoder, #12
Reporter) follows once the upstream side is stable. Replay tools
(`WavSource`, `JsonlSink`) land alongside #7 because §10 validation needs
them.
