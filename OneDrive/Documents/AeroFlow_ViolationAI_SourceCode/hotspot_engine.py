"""
hotspot_engine.py  —  AeroFlow Violation AI
Transforms the violation log into actionable enforcement intelligence.

Outputs:
  1. Interactive heatmap (HTML) — which junctions need attention
  2. Time-of-day analysis — when violations peak (deploy constables smartly)
  3. Violation type concentration — what's happening where

Why this shifts from reactive to predictive enforcement:
  Traditional: officer responds to violations after the fact.
  AeroFlow:    system tells officers WHERE to stand and WHEN to be there
               before violations happen, based on historical patterns.

This is the layer that makes the system genuinely useful to a traffic
superintendent planning weekly deployment schedules.

Output files:
  evidence/reports/hotspot_map.html       ← interactive folium map
  evidence/reports/hotspot_summary.json   ← structured data for API use
"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd

from config import VIOLATIONS_LOG, EVIDENCE_REPORTS_DIR

# ── Bengaluru intersection coordinates ───────────────────────────────────────
# Real coordinates for key Bengaluru traffic hotspots.
# In production: each camera registers its GPS coordinates in config.
BENGALURU_INTERSECTIONS: dict[str, tuple[float, float]] = {
    "Silk Board"      : (12.9176, 77.6234),
    "KR Puram"        : (13.0067, 77.6952),
    "Hebbal"          : (13.0450, 77.5972),
    "Whitefield"      : (12.9698, 77.7500),
    "Marathahalli"    : (12.9590, 77.6974),
    "Electronic City" : (12.8406, 77.6669),
    "Koramangala"     : (12.9352, 77.6245),
    "Indiranagar"     : (12.9784, 77.6408),
}

# Lane → intersection mapping (matches config.py LANES, mapped to Bengaluru)
LANE_TO_INTERSECTION: dict[str, str] = {
    "Lane 1": "Silk Board",
    "Lane 2": "KR Puram",
    "Lane 3": "Hebbal",
    "Lane 4": "Marathahalli",
}

# Violation severity weights (for heatmap intensity)
SEVERITY_WEIGHTS: dict[str, float] = {
    "Red-Light Violation"    : 1.0,
    "Wrong-Side Driving"     : 1.0,
    "Triple Riding"          : 0.8,
    "Helmet Non-Compliance"  : 0.7,
    "Stop-Line Violation"    : 0.6,
    "Illegal Parking"        : 0.5,
    "Seatbelt Non-Compliance": 0.6,
}


class HotspotEngine:
    """
    Reads violation CSV and generates hotspot intelligence outputs.
    Call generate_all() to produce all outputs in one shot.
    """

    def __init__(self):
        os.makedirs(EVIDENCE_REPORTS_DIR, exist_ok=True)

    # ── Data loading ──────────────────────────────────────────────────────────

    def load_violations(self) -> pd.DataFrame:
        if not os.path.exists(VIOLATIONS_LOG):
            return pd.DataFrame()
        df = pd.read_csv(VIOLATIONS_LOG)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        return df

    def enrich_with_location(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map violations to geographic coordinates.
        Uses lane info if available, otherwise distributes across all intersections.
        """
        import random
        intersections = list(BENGALURU_INTERSECTIONS.keys())

        lats, lngs, names = [], [], []
        for _, row in df.iterrows():
            # Try to map by lane name
            lane = str(row.get("Lane", "")) if "Lane" in df.columns else ""
            intersection = LANE_TO_INTERSECTION.get(lane)

            if not intersection:
                # Fallback: distribute across intersections
                # In production, camera GPS handles this automatically
                intersection = random.choice(intersections)

            coords = BENGALURU_INTERSECTIONS[intersection]
            # Add small jitter so overlapping points spread on map
            jitter = 0.001
            lats.append(coords[0] + random.uniform(-jitter, jitter))
            lngs.append(coords[1] + random.uniform(-jitter, jitter))
            names.append(intersection)

        df = df.copy()
        df["lat"]          = lats
        df["lng"]          = lngs
        df["intersection"] = names
        return df

    # ── Hotspot map ───────────────────────────────────────────────────────────

    def generate_heatmap(self, df: pd.DataFrame) -> str:
        """
        Generate an interactive folium heatmap HTML.
        Returns path to saved HTML file.
        """
        try:
            import folium
            from folium.plugins import HeatMap, MarkerCluster
        except ImportError:
            print("[HOTSPOT] folium not installed. Run: pip install folium")
            return ""

        if df.empty:
            print("[HOTSPOT] No violation data to map.")
            return ""

        df = self.enrich_with_location(df)

        # Bengaluru centre
        m = folium.Map(
            location=[12.9716, 77.5946],
            zoom_start=12,
            tiles="CartoDB dark_matter",
        )

        # Build heatmap data: [lat, lng, weight]
        heat_data = []
        for _, row in df.iterrows():
            weight = SEVERITY_WEIGHTS.get(
                str(row.get("ViolationType", "")), 0.5
            )
            heat_data.append([row["lat"], row["lng"], weight])

        if heat_data:
            HeatMap(
                heat_data,
                min_opacity=0.3,
                radius=25,
                blur=20,
                gradient={
                    0.2: "blue", 0.4: "cyan",
                    0.6: "lime", 0.8: "yellow", 1.0: "red"
                },
            ).add_to(m)

        # Intersection markers with violation counts
        for intersection, coords in BENGALURU_INTERSECTIONS.items():
            count = len(df[df["intersection"] == intersection])
            if count == 0:
                continue
            top_vtype = (
                df[df["intersection"] == intersection]["ViolationType"]
                .value_counts().index[0]
                if "ViolationType" in df.columns else "Unknown"
            )
            color = ("red" if count > 20 else
                     "orange" if count > 10 else "green")
            folium.CircleMarker(
                location=coords,
                radius=max(8, min(25, count // 2)),
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>{intersection}</b><br>"
                    f"Violations: {count}<br>"
                    f"Top type: {top_vtype}",
                    max_width=200,
                ),
                tooltip=f"{intersection}: {count} violations",
            ).add_to(m)

        # Title overlay
        title_html = """
        <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
             z-index:1000;background:rgba(0,0,0,0.75);padding:10px 20px;
             border-radius:8px;color:white;font-family:Arial;font-size:14px;
             border:1px solid #1A56DB;">
            🚦 AeroFlow AI — Bengaluru Violation Hotspot Map
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))

        # Legend
        legend_html = """
        <div style="position:fixed;bottom:30px;right:10px;z-index:1000;
             background:rgba(0,0,0,0.75);padding:10px;border-radius:8px;
             color:white;font-family:Arial;font-size:12px;">
            <b>Hotspot Intensity</b><br>
            🔵 Low &nbsp;&nbsp; 🟢 Medium<br>🟡 High &nbsp; 🔴 Critical
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        out_path = os.path.join(EVIDENCE_REPORTS_DIR, "hotspot_map.html")
        m.save(out_path)
        print(f"[HOTSPOT] Heatmap saved: {out_path}")
        return out_path

    # ── Time-of-day analysis ──────────────────────────────────────────────────

    def time_of_day_analysis(self, df: pd.DataFrame) -> dict:
        """
        Compute peak violation hours per intersection.
        Returns deployment recommendations for traffic constables.
        """
        if df.empty or "Timestamp" not in df.columns:
            return {}

        df = df.dropna(subset=["Timestamp"])
        df = self.enrich_with_location(df)
        df["Hour"] = df["Timestamp"].dt.hour

        peak_hours   = (
            df.groupby("Hour").size()
            .sort_values(ascending=False)
            .head(3).index.tolist()
        )
        by_intersection = {}
        for intersection in BENGALURU_INTERSECTIONS:
            sub = df[df["intersection"] == intersection]
            if sub.empty:
                continue
            peak = (
                sub.groupby("Hour").size()
                .sort_values(ascending=False)
                .head(2).index.tolist()
            )
            by_intersection[intersection] = {
                "total_violations" : len(sub),
                "peak_hours"       : peak,
                "top_violation"    : (
                    sub["ViolationType"].value_counts().index[0]
                    if "ViolationType" in sub.columns and len(sub) > 0
                    else "N/A"
                ),
            }

        return {
            "overall_peak_hours"    : peak_hours,
            "by_intersection"       : by_intersection,
            "deployment_recommendation": self._deployment_advice(by_intersection),
        }

    @staticmethod
    def _deployment_advice(by_intersection: dict) -> list[str]:
        """Generate plain-English deployment advice for traffic officers."""
        advice = []
        sorted_locations = sorted(
            by_intersection.items(),
            key=lambda x: x[1]["total_violations"],
            reverse=True,
        )
        for intersection, data in sorted_locations[:3]:
            hours = ", ".join(
                f"{h:02d}:00–{h+1:02d}:00" for h in data["peak_hours"]
            )
            advice.append(
                f"Deploy at {intersection} during {hours} — "
                f"{data['total_violations']} violations recorded, "
                f"mostly {data['top_violation']}."
            )
        return advice

    # ── Summary JSON ──────────────────────────────────────────────────────────

    def generate_summary_json(self, analysis: dict) -> str:
        out_path = os.path.join(EVIDENCE_REPORTS_DIR, "hotspot_summary.json")
        with open(out_path, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"[HOTSPOT] Summary JSON saved: {out_path}")
        return out_path

    # ── Generate all outputs ──────────────────────────────────────────────────

    def generate_all(self) -> dict[str, str]:
        """
        One-call method to generate all hotspot intelligence outputs.
        Returns dict of {output_name: file_path}.
        """
        df       = self.load_violations()
        outputs  = {}

        map_path = self.generate_heatmap(df)
        if map_path:
            outputs["heatmap"] = map_path

        analysis     = self.time_of_day_analysis(df)
        summary_path = self.generate_summary_json(analysis)
        outputs["summary"] = summary_path

        # Print deployment advice to terminal
        if analysis.get("deployment_recommendation"):
            print("\n[HOTSPOT] 📍 Constable Deployment Recommendations:")
            for tip in analysis["deployment_recommendation"]:
                print(f"  → {tip}")

        return outputs
