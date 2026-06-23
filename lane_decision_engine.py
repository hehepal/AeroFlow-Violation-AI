"""
lane_decision_engine.py  —  AeroFlow AI
Signal control simulation / integration-test harness.

Uses SIMULATED vehicle counts + REAL AQI data to validate the scoring
and signal-allocation logic before deploying with live cameras.

Key features:
  ✅ Pollution Urgency Score per lane (emission weights × AQI multiplier)
  ✅ Priority queue — cleanest lane gets green FIRST (was: inverted)
  ✅ Adaptive green-time splits — proportional to urgency score
  ✅ Safety guardrails:
       MIN_GREEN_TIME   = 15 s  (no lane starved)
       MAX_GREEN_TIME   = 60 s  (no lane monopolises)
       YELLOW_INTERVAL  =  3 s  (fixed — never adjusted by algorithm)
  ✅ AQI cached once before the loop (CPCB updates hourly; no need to refetch)
  ✅ Green time borrowing: extra seconds from low-emission phases fund
     extended green for heavy-emitter clusters

To switch to production mode, replace simulate_lane_counts() with
real output from classify_and_count() in classify_emissions.py.
"""
import heapq
import random
import time

from aqi_reader import get_aqi, get_aqi_category
from classify_emissions import compute_pollution_score

# ── Intersection config ───────────────────────────────────────────────────────
# Map lane → nearest CPCB monitoring station (Delhi-NCR stations)
LANES: dict[str, tuple[str, str]] = {
    "Lane 1": ("ITO",           "Delhi"),
    "Lane 2": ("Anand Vihar",   "Delhi"),
    "Lane 3": ("RK Puram",      "Delhi"),
    "Lane 4": ("Punjabi Bagh",  "Delhi"),
}

# ── Signal safety guardrails ──────────────────────────────────────────────────
MIN_GREEN_TIME   = 15    # seconds — prevents queue starvation
MAX_GREEN_TIME   = 60    # seconds — prevents green-wave monopoly
YELLOW_INTERVAL  = 3     # seconds — fixed; pedestrian safety, never adjusted
TOTAL_CYCLE_TIME = 120   # seconds — full signal cycle budget
POLL_INTERVAL    = 5     # seconds — how often the engine re-evaluates


def simulate_lane_counts() -> dict[str, dict[str, int]]:
    """
    Simulate random vehicle emission counts per lane.
    REPLACE with real classify_and_count() output in production.
    """
    return {
        lane: {
            "High":   random.randint(0, 5),
            "Medium": random.randint(0, 4),
            "BS-VI":  random.randint(0, 7),
            "Clean":  random.randint(0, 2),
        }
        for lane in LANES
    }


def compute_green_splits(lane_scores: dict[str, float]) -> dict[str, int]:
    """
    Distribute green time across lanes proportional to pollution urgency.

    Higher urgency → more green time to flush out the high-emission cluster
    ("borrowing" seconds from low-urgency phases).

    Subject to MIN_GREEN_TIME and MAX_GREEN_TIME safety guardrails.

    Args:
        lane_scores: {lane_name: pollution_urgency_score}

    Returns:
        {lane_name: green_seconds}
    """
    n              = len(lane_scores)
    total_yellow   = n * YELLOW_INTERVAL
    available      = TOTAL_CYCLE_TIME - total_yellow
    total_score    = sum(lane_scores.values())

    if total_score == 0:
        # All scores zero → equal split
        per_lane = available // n
        return {lane: per_lane for lane in lane_scores}

    raw = {
        lane: (score / total_score) * available
        for lane, score in lane_scores.items()
    }
    return {
        lane: max(MIN_GREEN_TIME, min(MAX_GREEN_TIME, round(t)))
        for lane, t in raw.items()
    }


def build_priority_queue(lane_scores: dict[str, float]) -> list[tuple]:
    """
    Build a min-heap priority queue keyed by pollution score.

    Lowest score = cleanest lane = highest dispatch priority.
    The lane at pq[0] gets the green signal first.
    """
    heap = [(score, lane) for lane, score in lane_scores.items()]
    heapq.heapify(heap)
    return heap


# ── Pre-fetch AQI once (CPCB data is hourly — no benefit in re-fetching) ──────
print("=" * 60)
print("📡 AeroFlow Signal Engine  [SIMULATION MODE]")
print("=" * 60)
print("Fetching AQI data...")
aqi_per_lane: dict[str, float | None] = {
    lane: get_aqi(loc, city)
    for lane, (loc, city) in LANES.items()
}
print()

# ── Main simulation loop ───────────────────────────────────────────────────────
while True:
    print("─" * 60)
    print("🚦 Signal Cycle")
    print("─" * 60)

    lane_data   = simulate_lane_counts()
    lane_scores = {}

    for lane, counts in lane_data.items():
        aqi   = aqi_per_lane[lane]
        score = compute_pollution_score(counts, aqi)
        lane_scores[lane] = score

        cat, _  = get_aqi_category(aqi)
        aqi_str = f"{aqi} ({cat})" if aqi is not None else "N/A"
        print(
            f"  {lane:8s} | "
            f"🔴{counts['High']} 🟠{counts['Medium']} "
            f"🟢{counts['BS-VI']} 🚴{counts['Clean']} | "
            f"AQI: {aqi_str:28s}| Score: {score}"
        )

    # ── Decision (FIX: min = cleanest lane = lowest urgency = green first) ────
    pq            = build_priority_queue(lane_scores)
    green_splits  = compute_green_splits(lane_scores)
    priority_lane = pq[0][1]   # ← min score, NOT max

    print()
    print("  📊 Adaptive Signal Allocation:")
    rank = 1
    for score, lane in pq:
        g      = green_splits[lane]
        marker = f"🟢 ({rank})" if rank == 1 else f"   ({rank})"
        print(
            f"    {marker} {lane:8s}: {g:2d}s green + "
            f"{YELLOW_INTERVAL}s yellow  [score {score}]"
        )
        rank += 1

    print(f"\n  ✅ Priority GREEN  → {priority_lane} "
          f"(score {lane_scores[priority_lane]} — cleanest lane)")
    print(f"  🛡️  Guardrails: min={MIN_GREEN_TIME}s  "
          f"max={MAX_GREEN_TIME}s  yellow={YELLOW_INTERVAL}s (fixed)")

    time.sleep(POLL_INTERVAL)
