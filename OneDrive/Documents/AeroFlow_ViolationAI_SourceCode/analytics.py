"""
analytics.py  —  AeroFlow Violation AI
Violation analytics, statistics, trends, and searchable reports.

Theme 3 Tasks covered:
  - Generate violation statistics and trends
  - Provide searchable records and summary reports
"""
from __future__ import annotations

import datetime
import os

import pandas as pd

from config import VIOLATIONS_LOG, EVIDENCE_REPORTS_DIR


# ── Analytics engine ──────────────────────────────────────────────────────────

class ViolationAnalytics:
    """
    Loads violation CSV and computes stats, trends, and reports.
    Re-reads CSV on each public method call so dashboard always has latest data.
    """

    VIOLATION_TYPES = [
        "Helmet Non-Compliance",
        "Triple Riding",
        "Wrong-Side Driving",
        "Stop-Line Violation",
        "Red-Light Violation",
        "Illegal Parking",
        "Seatbelt Non-Compliance",
    ]

    # ── Data loading ──────────────────────────────────────────────────────────

    def load(self) -> pd.DataFrame:
        """Load violations CSV. Returns empty DataFrame if not found."""
        if not os.path.exists(VIOLATIONS_LOG):
            return pd.DataFrame(columns=[
                "ViolationID", "Timestamp", "FrameID", "TrackID",
                "VehicleClass", "ViolationType", "Confidence",
                "BBoxX1", "BBoxY1", "BBoxX2", "BBoxY2",
                "PlateText", "PlateConfidence", "EvidenceFramePath",
            ])
        df = pd.read_csv(VIOLATIONS_LOG)
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        if "Confidence" in df.columns:
            df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
        return df

    # ── Summary statistics ────────────────────────────────────────────────────

    def violation_counts(self) -> dict[str, int]:
        """Total count per violation type."""
        df = self.load()
        if df.empty or "ViolationType" not in df.columns:
            return {v: 0 for v in self.VIOLATION_TYPES}
        counts = df["ViolationType"].value_counts().to_dict()
        # Normalise: strip count suffixes like "(3 persons)"
        clean = {}
        for k, v in counts.items():
            matched = next(
                (vt for vt in self.VIOLATION_TYPES if vt.lower() in k.lower()),
                k
            )
            clean[matched] = clean.get(matched, 0) + v
        return clean

    def vehicle_breakdown(self) -> dict[str, int]:
        """Violations per vehicle class."""
        df = self.load()
        if df.empty or "VehicleClass" not in df.columns:
            return {}
        return df["VehicleClass"].value_counts().to_dict()

    def hourly_trend(self) -> pd.DataFrame:
        """
        Violations grouped by hour.
        Returns DataFrame with columns: Hour, ViolationType, Count.
        """
        df = self.load()
        if df.empty or "Timestamp" not in df.columns:
            return pd.DataFrame(columns=["Hour", "ViolationType", "Count"])
        df = df.dropna(subset=["Timestamp"])
        df["Hour"] = df["Timestamp"].dt.floor("h")
        trend = (
            df.groupby(["Hour", "ViolationType"])
            .size()
            .reset_index(name="Count")
        )
        return trend

    def daily_summary(self) -> pd.DataFrame:
        """Violations per day with totals."""
        df = self.load()
        if df.empty or "Timestamp" not in df.columns:
            return pd.DataFrame()
        df = df.dropna(subset=["Timestamp"])
        df["Date"] = df["Timestamp"].dt.date
        return (
            df.groupby(["Date", "ViolationType"])
            .size()
            .reset_index(name="Count")
        )

    def search(
        self,
        plate       : str  = "",
        vtype       : str  = "",
        vehicle_cls : str  = "",
        from_dt     : str  = "",
        to_dt       : str  = "",
    ) -> pd.DataFrame:
        """
        Filter violations by any combination of:
          - plate number (partial match)
          - violation type
          - vehicle class
          - date range (YYYY-MM-DD strings)
        """
        df = self.load()
        if df.empty:
            return df

        if plate and "PlateText" in df.columns:
            df = df[df["PlateText"].astype(str).str.contains(
                plate, case=False, na=False
            )]
        if vtype and "ViolationType" in df.columns:
            df = df[df["ViolationType"].str.contains(vtype, case=False, na=False)]
        if vehicle_cls and "VehicleClass" in df.columns:
            df = df[df["VehicleClass"].str.contains(vehicle_cls, case=False, na=False)]
        if from_dt and "Timestamp" in df.columns:
            df = df[df["Timestamp"] >= pd.to_datetime(from_dt, errors="coerce")]
        if to_dt and "Timestamp" in df.columns:
            df = df[df["Timestamp"] <= pd.to_datetime(to_dt, errors="coerce")]

        return df.reset_index(drop=True)

    # ── Text report ───────────────────────────────────────────────────────────

    def generate_text_report(self, save: bool = True) -> str:
        """
        Generate a human-readable summary report.
        Optionally saves to evidence/reports/ directory.
        """
        df     = self.load()
        counts = self.violation_counts()
        total  = sum(counts.values())
        ts     = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            "=" * 60,
            "  AEROFLOW VIOLATION AI — SUMMARY REPORT",
            f"  Generated : {ts}",
            f"  Location  : ITO Crossing, Delhi",
            "=" * 60,
            "",
            f"  Total violations logged : {total}",
            f"  Total records in DB     : {len(df)}",
            "",
            "  BREAKDOWN BY VIOLATION TYPE:",
        ]
        for vtype, count in sorted(counts.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / total * 100 if total else 0
                lines.append(f"    {vtype:<35s} {count:4d}  ({pct:.1f}%)")

        lines += ["", "  BREAKDOWN BY VEHICLE CLASS:"]
        vb = self.vehicle_breakdown()
        for cls, count in sorted(vb.items(), key=lambda x: -x[1]):
            lines.append(f"    {cls:<20s} {count:4d}")

        if not df.empty and "Timestamp" in df.columns:
            df_valid = df.dropna(subset=["Timestamp"])
            if not df_valid.empty:
                lines += [
                    "",
                    f"  Date range : {df_valid['Timestamp'].min().date()} "
                    f"→ {df_valid['Timestamp'].max().date()}",
                ]

        lines += ["", "=" * 60]
        report = "\n".join(lines)

        if save:
            fname  = f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            fpath  = os.path.join(EVIDENCE_REPORTS_DIR, fname)
            os.makedirs(EVIDENCE_REPORTS_DIR, exist_ok=True)
            with open(fpath, "w") as f:
                f.write(report)
            print(f"[Analytics] Report saved: {fpath}")

        return report
