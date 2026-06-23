"""
pm25_estimator.py  —  AeroFlow AI
Estimates the mass of PM2.5 avoided by AeroFlow's adaptive signal control
compared to a naive round-robin (equal green time) baseline.

Methodology:
  1. For each high-emission vehicle at a red light, estimate the extra
     idle time it would have experienced under a dumb fixed-timer system.
  2. Multiply idle time by the vehicle's PM2.5 idle emission rate.
  3. Sum across all vehicles and all cycles to get cumulative savings.

Emission rates used (conservative, from ICCT / MoRTH BS-VI data):
  ┌───────────┬───────────────────────────────────────┬──────────────────┐
  │ Category  │ Vehicle type                          │ Idle PM2.5 g/min │
  ├───────────┼───────────────────────────────────────┼──────────────────┤
  │ High      │ Pre-BS-IV diesel bus / truck          │ 0.090            │
  │ Medium    │ BS-III/IV 2-wheeler (motorbike)       │ 0.012            │
  │ BS-VI     │ Post-2020 car / van                   │ 0.003            │
  │ Clean     │ EV / bicycle                          │ 0.000            │
  └───────────┴───────────────────────────────────────┴──────────────────┘

  Baseline idle time = TOTAL_CYCLE_TIME / NUM_LANES  (equal split)
  AeroFlow idle time = actual red time assigned to that lane
  Time saved = Baseline − AeroFlow idle time  (clamped to ≥ 0)
  PM2.5 saved = time_saved_minutes × idle_rate × vehicle_count

Output:
  Returns a PM25Report dataclass and optionally writes a daily savings CSV.
"""
from __future__ import annotations

import csv
import datetime
import os
from dataclasses import dataclass, field

from config import LANES, TOTAL_CYCLE_TIME, YELLOW_INTERVAL

# ── Idle emission rates  (g PM2.5 per minute of idling) ──────────────────────
IDLE_EMISSION_RATES: dict[str, float] = {
    "High":   0.090,   # Pre-BS-IV heavy diesel — ICCT estimate
    "Medium": 0.012,   # BS-III/IV 2-wheeler
    "BS-VI":  0.003,   # Post-2020 car/van with DPF
    "Clean":  0.000,   # EV / bicycle
}

# Baseline: naive equal-split green time per lane (seconds)
_BASELINE_GREEN = (TOTAL_CYCLE_TIME - len(LANES) * YELLOW_INTERVAL) // len(LANES)

# Savings log file
SAVINGS_LOG = "pm25_savings.csv"
_SAVINGS_COLS = [
    "Timestamp", "Lane",
    "HighVehicles", "MediumVehicles", "BS6Vehicles",
    "AeroFlowGreenTime", "BaselineGreenTime",
    "IdleTimeSavedSec", "PM25SavedGrams",
]


@dataclass
class PM25Report:
    """Holds PM2.5 savings for one signal cycle."""
    timestamp          : str
    per_lane           : dict[str, dict] = field(default_factory=dict)
    total_pm25_saved_g : float = 0.0
    total_idle_saved_s : float = 0.0

    def summary(self) -> str:
        lines = [
            f"[PM2.5 Report  {self.timestamp}]",
            f"  Total PM2.5 avoided : {self.total_pm25_saved_g:.3f} g",
            f"  Total idle time saved: {self.total_idle_saved_s:.1f} s",
        ]
        for lane, data in self.per_lane.items():
            lines.append(
                f"  {lane:8s}  green={data['aeroflow_green']:2d}s  "
                f"saved={data['idle_saved_s']:4.1f}s  "
                f"PM2.5={data['pm25_saved_g']:.3f}g"
            )
        return "\n".join(lines)


class PM25Estimator:
    """
    Stateful estimator — accumulates savings across the session.
    Call record_cycle() once per signal cycle from aeroflow_ai.py.
    """

    def __init__(self, write_log: bool = True):
        self.write_log           = write_log
        self.session_pm25_saved  = 0.0   # grams this session
        self.session_idle_saved  = 0.0   # seconds this session
        self.cycle_count         = 0

        if write_log and not os.path.exists(SAVINGS_LOG):
            with open(SAVINGS_LOG, "w", newline="") as f:
                csv.writer(f).writerow(_SAVINGS_COLS)

    def record_cycle(
        self,
        lane_counts  : dict[str, dict[str, int]],   # {lane: {cat: count}}
        green_splits : dict[str, int],               # {lane: green_seconds}
    ) -> PM25Report:
        """
        Compute PM2.5 savings for one completed signal cycle.

        Args:
            lane_counts  : vehicle emission counts per lane
            green_splits : actual green time allocated per lane by AeroFlow

        Returns:
            PM25Report dataclass with per-lane breakdown and totals.
        """
        ts      = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report  = PM25Report(timestamp=ts)
        rows    = []

        for lane in LANES:
            counts     = lane_counts.get(lane, {})
            aero_green = green_splits.get(lane, _BASELINE_GREEN)

            # Time this lane's vehicles spent idling under AeroFlow
            # = total cycle − this lane's green − its yellow
            aero_red    = TOTAL_CYCLE_TIME - aero_green - YELLOW_INTERVAL
            baseline_red = TOTAL_CYCLE_TIME - _BASELINE_GREEN - YELLOW_INTERVAL

            # Extra idling avoided vs baseline
            idle_saved_s = max(0.0, baseline_red - aero_red)
            idle_saved_m = idle_saved_s / 60.0

            # PM2.5 saved across all vehicle categories in this lane
            pm25_saved = sum(
                counts.get(cat, 0) * rate * idle_saved_m
                for cat, rate in IDLE_EMISSION_RATES.items()
            )

            report.per_lane[lane] = {
                "aeroflow_green"  : aero_green,
                "baseline_green"  : _BASELINE_GREEN,
                "idle_saved_s"    : idle_saved_s,
                "pm25_saved_g"    : round(pm25_saved, 4),
                "high_vehicles"   : counts.get("High",   0),
                "medium_vehicles" : counts.get("Medium", 0),
                "bs6_vehicles"    : counts.get("BS-VI",  0),
            }

            report.total_pm25_saved_g += pm25_saved
            report.total_idle_saved_s += idle_saved_s

            rows.append([
                ts, lane,
                counts.get("High",   0),
                counts.get("Medium", 0),
                counts.get("BS-VI",  0),
                aero_green, _BASELINE_GREEN,
                round(idle_saved_s, 1),
                round(pm25_saved,   4),
            ])

        report.total_pm25_saved_g = round(report.total_pm25_saved_g, 4)
        report.total_idle_saved_s = round(report.total_idle_saved_s, 1)

        # Accumulate session totals
        self.session_pm25_saved += report.total_pm25_saved_g
        self.session_idle_saved += report.total_idle_saved_s
        self.cycle_count        += 1

        # Write to savings log
        if self.write_log:
            with open(SAVINGS_LOG, "a", newline="") as f:
                csv.writer(f).writerows(rows)

        return report

    def session_summary(self) -> str:
        return (
            f"\n{'='*55}\n"
            f"  AeroFlow PM2.5 Session Summary\n"
            f"  Cycles completed  : {self.cycle_count}\n"
            f"  PM2.5 avoided     : {self.session_pm25_saved:.2f} g\n"
            f"  Idle time saved   : {self.session_idle_saved:.0f} s  "
            f"({self.session_idle_saved/60:.1f} min)\n"
            f"{'='*55}"
        )


# ── Standalone demo ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random
    from classify_emissions import compute_pollution_score

    estimator = PM25Estimator(write_log=True)

    for cycle in range(5):
        # Mock data
        counts = {
            lane: {
                "High":   random.randint(0, 4),
                "Medium": random.randint(0, 3),
                "BS-VI":  random.randint(0, 6),
                "Clean":  random.randint(0, 2),
            }
            for lane in LANES
        }
        # Mock green splits
        splits = {lane: random.randint(15, 60) for lane in LANES}

        report = estimator.record_cycle(counts, splits)
        print(report.summary())

    print(estimator.session_summary())
