"""
signal_controller.py  —  AeroFlow AI
Signal Phase State Machine  (the "Act" layer of Sense-Think-Act)

Manages the full signal cycle for one intersection:
    GREEN  →  YELLOW  →  RED  (repeat for next lane)

Rules enforced:
  - MIN_GREEN_TIME  : no green phase shorter than this (prevents starvation)
  - MAX_GREEN_TIME  : no green phase longer than this
  - YELLOW_INTERVAL : always fixed (pedestrian safety — never shortened)
  - One lane is GREEN at a time; all others are RED

Integration:
    The controller runs in a background thread.
    aeroflow_ai.py calls controller.update_scores(lane_scores) every frame.
    The controller picks the optimal phase order from the latest scores and
    manages timing autonomously.

Hardware output:
    set_signal() is the hardware interface stub.
    Replace its body with actual GPIO / relay / SCATS calls.
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

from config import (
    LANES,
    MIN_GREEN_TIME,
    MAX_GREEN_TIME,
    YELLOW_INTERVAL,
    TOTAL_CYCLE_TIME,
)


# ── Signal phase enum ─────────────────────────────────────────────────────────
class Phase(Enum):
    GREEN  = auto()
    YELLOW = auto()
    RED    = auto()


# ── Per-lane state ────────────────────────────────────────────────────────────
@dataclass
class LaneState:
    name         : str
    phase        : Phase = Phase.RED
    green_time   : int   = 0       # allocated green seconds for this cycle
    time_remaining: float = 0.0    # seconds left in current phase
    score        : float = 0.0     # latest pollution urgency score
    total_greens : int   = 0       # cumulative green cycles served
    pm25_saved   : float = 0.0     # cumulative PM2.5 avoided estimate (grams)


# ── Signal controller ─────────────────────────────────────────────────────────
class SignalController:
    """
    Runs a continuous signal cycle in a daemon thread.
    Thread-safe: use update_scores() from any thread.
    """

    def __init__(self, verbose: bool = True):
        self.lanes        : dict[str, LaneState] = {
            name: LaneState(name=name) for name in LANES
        }
        self._scores_lock = threading.Lock()
        self._latest_scores: dict[str, float] = {name: 0.0 for name in LANES}
        self._running     = False
        self._thread      : threading.Thread | None = None
        self.verbose      = verbose
        self.cycle_count  = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def update_scores(self, scores: dict[str, float]) -> None:
        """
        Thread-safe update of pollution urgency scores.
        Called by aeroflow_ai.py every time a new frame is processed.
        """
        with self._scores_lock:
            self._latest_scores.update(scores)
            for lane, score in scores.items():
                if lane in self.lanes:
                    self.lanes[lane].score = score

    def start(self) -> None:
        """Start the signal cycle in a background daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._run_cycle_loop,
            daemon=True,
            name="SignalController",
        )
        self._thread.start()
        self._log("Signal controller started.")

    def stop(self) -> None:
        """Gracefully stop the controller after the current phase completes."""
        self._running = False
        self._log("Signal controller stopping (will finish current phase).")

    def get_status(self) -> dict:
        """
        Snapshot of all lane states — safe to call from any thread.
        Returns a dict suitable for logging or dashboard consumption.
        """
        return {
            name: {
                "phase":          ls.phase.name,
                "score":          ls.score,
                "green_time":     ls.green_time,
                "time_remaining": round(ls.time_remaining, 1),
                "total_greens":   ls.total_greens,
                "pm25_saved":     round(ls.pm25_saved, 2),
            }
            for name, ls in self.lanes.items()
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_current_scores(self) -> dict[str, float]:
        with self._scores_lock:
            return dict(self._latest_scores)

    def _compute_green_splits(
        self, scores: dict[str, float]
    ) -> dict[str, int]:
        """
        Distribute green time proportional to urgency score.
        Higher score → longer green to flush out heavy-emitter clusters.
        Clamped to [MIN_GREEN_TIME, MAX_GREEN_TIME].
        """
        n           = len(scores)
        available   = TOTAL_CYCLE_TIME - n * YELLOW_INTERVAL
        total       = sum(scores.values())

        if total == 0:
            per = available // n
            return {lane: per for lane in scores}

        raw = {l: (s / total) * available for l, s in scores.items()}
        return {
            l: max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, round(t)))
            for l, t in raw.items()
        }

    def _order_lanes(self, scores: dict[str, float]) -> list[str]:
        """
        Return lane names ordered by ascending pollution score.
        Cleanest lane (lowest score) dispatched first.
        """
        return sorted(scores, key=scores.get)

    def _run_cycle_loop(self) -> None:
        """Main loop — runs one full cycle, repeats until stopped."""
        while self._running:
            self.cycle_count += 1
            scores      = self._get_current_scores()
            splits      = self._compute_green_splits(scores)
            order       = self._order_lanes(scores)

            self._log(f"\n{'='*55}")
            self._log(f"Cycle #{self.cycle_count}  |  Priority order: {order}")
            self._log(f"{'='*55}")

            for lane_name in order:
                if not self._running:
                    break

                lane_state             = self.lanes[lane_name]
                lane_state.green_time  = splits[lane_name]

                # ── GREEN phase ───────────────────────────────────────────────
                self._set_phase(lane_name, Phase.GREEN)
                lane_state.time_remaining = splits[lane_name]
                self._log(
                    f"  🟢 GREEN  → {lane_name}  "
                    f"({splits[lane_name]}s  score={scores[lane_name]})"
                )
                self._set_signal(lane_name, Phase.GREEN)
                self._countdown(lane_state, splits[lane_name])
                lane_state.total_greens += 1

                # ── YELLOW phase ──────────────────────────────────────────────
                self._set_phase(lane_name, Phase.YELLOW)
                lane_state.time_remaining = YELLOW_INTERVAL
                self._log(f"  🟡 YELLOW → {lane_name}  ({YELLOW_INTERVAL}s fixed)")
                self._set_signal(lane_name, Phase.YELLOW)
                self._countdown(lane_state, YELLOW_INTERVAL)

                # ── Back to RED ───────────────────────────────────────────────
                self._set_phase(lane_name, Phase.RED)
                self._set_signal(lane_name, Phase.RED)

                # Refresh scores for next lane
                scores  = self._get_current_scores()
                splits  = self._compute_green_splits(scores)

    def _set_phase(self, lane_name: str, phase: Phase) -> None:
        """Update all lanes: target lane = phase, all others = RED."""
        for name, ls in self.lanes.items():
            ls.phase = phase if name == lane_name else Phase.RED

    def _countdown(self, lane_state: LaneState, seconds: float) -> None:
        """Sleep in 0.1 s ticks, updating time_remaining."""
        ticks = int(seconds / 0.1)
        for _ in range(ticks):
            if not self._running:
                return
            lane_state.time_remaining = round(
                max(0.0, lane_state.time_remaining - 0.1), 1
            )
            time.sleep(0.1)

    def _set_signal(self, lane_name: str, phase: Phase) -> None:
        """
        Hardware interface stub.
        Replace this body with actual GPIO / relay / SCATS API calls.

        Example for Raspberry Pi GPIO:
            import RPi.GPIO as GPIO
            PIN_MAP = {"Lane 1": {"GREEN": 17, "YELLOW": 27, "RED": 22}, ...}
            GPIO.output(PIN_MAP[lane_name][phase.name], GPIO.HIGH)
        """
        pass   # no-op in software-only mode

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


# ── Standalone demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    ctrl = SignalController(verbose=True)
    ctrl.start()

    print("Signal controller running.  Ctrl+C to stop.\n")

    try:
        while True:
            # Simulate live score updates from aeroflow_ai.py
            mock_scores = {lane: round(random.uniform(0, 20), 1) for lane in LANES}
            ctrl.update_scores(mock_scores)
            print(f"[MOCK] Updated scores: {mock_scores}")
            print(f"[STATUS] {ctrl.get_status()}\n")
            time.sleep(8)
    except KeyboardInterrupt:
        ctrl.stop()
        print("\nStopped.")
