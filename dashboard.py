"""
dashboard.py  —  AeroFlow Violation AI
Real-time Streamlit dashboard with 5 tabs.

Run with:
    streamlit run dashboard.py

Tabs:
  1. 🚦 Live         — lane scores, AQI, signal allocation, PM2.5 saved
  2. 🚨 Violations   — live violation feed, counts, trends
  3. 🖼️  Evidence     — searchable annotated frame viewer
  4. 📊 Analytics    — hourly/daily trends, vehicle mix, violation breakdown
  5. 📈 Evaluation   — FPS, Precision, Recall, F1, mAP display
"""
import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analytics import ViolationAnalytics
from aqi_reader import get_aqi_category
from config import (
    EVIDENCE_FRAMES_DIR,
    VIOLATIONS_LOG,
    LOG_FILE,
    DASHBOARD_REFRESH_SECS,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AeroFlow Violation AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .block-container { padding-top: 0.8rem; }
  h1 { font-size: 1.6rem !important; }
  .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
  .metric-card {
    padding: 16px; border-radius: 12px;
    background: linear-gradient(135deg,#1a1a2e,#16213e);
    border: 1px solid #2a2a3e; margin-bottom: 8px;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚦 AeroFlow Violation AI — Traffic Monitoring Dashboard")
st.caption("Delhi-NCR  ·  YOLOv8 Computer Vision  ·  CPCB AQI  ·  Real-Time Enforcement")

# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=4)
def load_lane_logs() -> pd.DataFrame | None:
    if not os.path.exists(LOG_FILE):
        return None
    df = pd.read_csv(LOG_FILE)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for col in ["Score", "AQI", "GreenTime", "High", "Medium", "BS-VI",
                "Clean", "VehicleCount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

@st.cache_data(ttl=4)
def load_violations() -> pd.DataFrame:
    analytics = ViolationAnalytics()
    return analytics.load()

# ── Helpers ───────────────────────────────────────────────────────────────────

LANE_PALETTE = {
    "Lane 1": "#FF4C4C", "Lane 2": "#FF9900",
    "Lane 3": "#4CAF50", "Lane 4": "#2196F3",
}
VTYPE_PALETTE = {
    "Helmet Non-Compliance" : "#FF4C4C",
    "Triple Riding"         : "#FF9900",
    "Wrong-Side Driving"    : "#FF44FF",
    "Stop-Line Violation"   : "#FFFF44",
    "Red-Light Violation"   : "#FF2222",
    "Illegal Parking"       : "#FF8800",
}

def score_color(s):
    if pd.isna(s): return "#888"
    if s > 10: return "#FF4C4C"
    if s > 5:  return "#FF9900"
    return "#4CAF50"

def dark_fig():
    fig, ax = plt.subplots()
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="#aaa", labelsize=8)
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    return fig, ax

# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🚦 Live Signal",
    "🚨 Violations",
    "🖼️ Evidence",
    "📊 Analytics",
    "📈 Evaluation",
    "🗺️ Hotspot Map",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Signal
# ════════════════════════════════════════════════════════════════════════════════
with tab1:
    df_lanes = load_lane_logs()

    if df_lanes is None or df_lanes.empty:
        st.warning("⏳ No lane data yet. Start `prototype.py` to begin.")
    else:
        latest_ts = df_lanes["Timestamp"].max()
        latest    = df_lanes[df_lanes["Timestamp"] == latest_ts].reset_index(drop=True)
        priority  = latest["Priority"].iloc[0] if "Priority" in latest.columns else "N/A"

        # KPI row
        kc = st.columns(5)
        kc[0].metric("⏰ Last Update",    latest_ts.strftime("%H:%M:%S"))
        kc[1].metric("🟢 Priority Lane",  priority)
        kc[2].metric("🚗 Vehicles Now",
                     int(latest["VehicleCount"].sum())
                     if "VehicleCount" in latest.columns else "—")
        avg_aqi  = latest["AQI"].mean() if "AQI" in latest.columns else None
        aqi_cat, _ = get_aqi_category(avg_aqi)
        kc[3].metric("🌫️ Avg AQI",
                     f"{avg_aqi:.0f} {aqi_cat}" if pd.notna(avg_aqi) else "N/A")
        pm_saved = round(df_lanes["High"].sum() * 0.15, 1) \
                   if "High" in df_lanes.columns else 0
        kc[4].metric("🌿 PM2.5 Avoided", f"{pm_saved} g")

        st.divider()

        # Lane status cards
        st.subheader("📊 Lane Status")
        cols = st.columns(len(latest))
        for i, row in latest.iterrows():
            lane    = row.get("Lane", f"Lane {i+1}")
            score   = row.get("Score", 0)
            aqi_v   = row.get("AQI", None)
            green_t = row.get("GreenTime", "—")
            high    = int(row["High"]) if "High" in row and pd.notna(row["High"]) else "—"
            medium  = int(row["Medium"]) if "Medium" in row and pd.notna(row["Medium"]) else "—"
            is_prio = (lane == priority)
            border  = "3px solid #4CAF50" if is_prio else "1px solid #2a2a3e"
            badge   = "<span style='background:#4CAF50;color:#000;padding:1px 7px;border-radius:4px;font-size:10px;'>PRIORITY</span>" if is_prio else ""
            cat, chex = get_aqi_category(aqi_v if pd.notna(aqi_v) else None)
            cols[i].markdown(f"""
            <div style='padding:14px;border-radius:12px;
                background:linear-gradient(135deg,#1a1a2e,#16213e);
                border:{border};'>
                <div style='font-size:14px;font-weight:700;color:#eee;'>
                    {lane} {badge}</div>
                <div style='font-size:28px;font-weight:900;
                    color:{score_color(score)};'>{score:.1f}</div>
                <div style='font-size:11px;color:{chex};'>
                    AQI: {aqi_v:.0f if pd.notna(aqi_v) else "N/A"} — {cat}</div>
                <hr style='border-color:#333;margin:5px 0;'>
                <div style='font-size:11px;color:#bbb;'>
                    🚌 High: <b>{high}</b> &nbsp;
                    🛵 Medium: <b>{medium}</b><br>
                    🚦 Green: <b>{green_t:.0f if isinstance(green_t,float) else green_t}s</b>
                </div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        cl, cr = st.columns(2)

        with cl:
            st.subheader("📈 Urgency Score Trend")
            fig, ax = dark_fig()
            for lane in sorted(df_lanes["Lane"].unique()):
                ld = df_lanes[df_lanes["Lane"] == lane].sort_values("Timestamp")
                ax.plot(ld["Timestamp"], ld["Score"],
                        label=lane, color=LANE_PALETTE.get(lane, "#aaa"), lw=1.8)
            ax.set_ylabel("Score", color="#aaa", fontsize=9)
            ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        with cr:
            st.subheader("🌫️ AQI Trend")
            if df_lanes["AQI"].notna().any():
                fig2, ax2 = dark_fig()
                for lo, hi, col in [(0,50,"#00B050"),(50,100,"#92D050"),
                                    (100,200,"#FFFF00"),(200,300,"#FF9900"),
                                    (300,500,"#FF0000")]:
                    ax2.axhspan(lo, hi, alpha=0.07, color=col)
                for lane in sorted(df_lanes["Lane"].unique()):
                    ld = df_lanes[df_lanes["Lane"] == lane].dropna(
                        subset=["AQI"]).sort_values("Timestamp")
                    if not ld.empty:
                        ax2.plot(ld["Timestamp"], ld["AQI"],
                                 label=lane, color=LANE_PALETTE.get(lane, "#aaa"), lw=1.5)
                ax2.set_ylabel("AQI", color="#aaa", fontsize=9)
                ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
                plt.tight_layout(); st.pyplot(fig2); plt.close(fig2)
            else:
                st.info("AQI data not available.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — Violations
# ════════════════════════════════════════════════════════════════════════════════
with tab2:
    df_vio = load_violations()
    analytics = ViolationAnalytics()
    counts    = analytics.violation_counts()
    total_v   = sum(counts.values())

    # KPI
    vc = st.columns(4)
    vc[0].metric("🚨 Total Violations",  total_v)
    vc[1].metric("🏍️ Helmet Issues",     counts.get("Helmet Non-Compliance", 0))
    vc[2].metric("🚗 Red-Light",         counts.get("Red-Light Violation", 0))
    vc[3].metric("🅿️ Illegal Parking",   counts.get("Illegal Parking", 0))

    st.divider()

    if df_vio.empty:
        st.info("No violations recorded yet. Run `prototype.py` to start detection.")
    else:
        vl, vr = st.columns(2)

        with vl:
            st.subheader("🚨 Violations by Type")
            fig3, ax3 = dark_fig()
            labels = [k for k, v in counts.items() if v > 0]
            values = [counts[k] for k in labels]
            colors = [VTYPE_PALETTE.get(k, "#888") for k in labels]
            ax3.barh(labels, values, color=colors, height=0.5)
            ax3.set_xlabel("Count", color="#aaa", fontsize=9)
            for i, v in enumerate(values):
                ax3.text(v + 0.1, i, str(v), va="center",
                         color="#eee", fontsize=8)
            plt.tight_layout(); st.pyplot(fig3); plt.close(fig3)

        with vr:
            st.subheader("🚗 Violations by Vehicle Class")
            vb = analytics.vehicle_breakdown()
            if vb:
                fig4, ax4 = dark_fig()
                ax4.pie(
                    list(vb.values()), labels=list(vb.keys()),
                    autopct="%1.1f%%",
                    colors=["#FF4C4C","#FF9900","#4CAF50","#2196F3","#FF44FF"],
                    textprops={"color": "#ddd", "fontsize": 9},
                )
                plt.tight_layout(); st.pyplot(fig4); plt.close(fig4)

        st.divider()
        st.subheader("📋 Recent Violations")
        display_cols = ["Timestamp","VehicleClass","ViolationType",
                        "Confidence","PlateText","EvidenceFramePath"]
        show_cols    = [c for c in display_cols if c in df_vio.columns]
        st.dataframe(
            df_vio[show_cols].tail(50).sort_values(
                "Timestamp", ascending=False
            ) if "Timestamp" in df_vio.columns else df_vio[show_cols].tail(50),
            use_container_width=True,
        )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — Evidence viewer
# ════════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("🖼️ Evidence Frame Viewer")

    # Search bar
    sc1, sc2, sc3 = st.columns(3)
    search_plate = sc1.text_input("🔍 Search by Plate", placeholder="e.g. DL 3C")
    search_type  = sc2.selectbox("Filter by Violation", ["All"] + [
        "Helmet Non-Compliance", "Triple Riding", "Wrong-Side Driving",
        "Stop-Line Violation", "Red-Light Violation", "Illegal Parking"
    ])
    max_show = sc3.slider("Max frames to show", 4, 20, 8)

    df_vio2 = load_violations()
    vt_filter = "" if search_type == "All" else search_type
    filtered  = analytics.search(
        plate=search_plate, vtype=vt_filter
    ) if not df_vio2.empty else pd.DataFrame()

    if filtered.empty:
        st.info("No evidence frames match your filter.")
    else:
        frame_paths = (
            filtered["EvidenceFramePath"].dropna().unique()
            if "EvidenceFramePath" in filtered.columns else []
        )
        frame_paths = [p for p in frame_paths if os.path.exists(p)][:max_show]

        if not frame_paths:
            st.info("Evidence frames not yet saved or path not found.")
        else:
            cols_img = st.columns(min(4, len(frame_paths)))
            for i, fpath in enumerate(frame_paths):
                cols_img[i % 4].image(
                    fpath,
                    caption=os.path.basename(fpath),
                    use_container_width=True,
                )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — Analytics
# ════════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("📊 Violation Analytics")
    trend_df = analytics.hourly_trend()

    if trend_df.empty:
        st.info("Not enough data for trend analysis yet.")
    else:
        st.subheader("📈 Hourly Violation Trend")
        fig5, ax5 = dark_fig()
        for vtype in trend_df["ViolationType"].unique():
            td = trend_df[trend_df["ViolationType"] == vtype]
            ax5.plot(td["Hour"], td["Count"],
                     label=vtype[:22],
                     color=VTYPE_PALETTE.get(vtype, "#aaa"), lw=1.5)
        ax5.set_ylabel("Violations", color="#aaa", fontsize=9)
        ax5.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=7,
                   loc="upper left")
        plt.tight_layout(); st.pyplot(fig5); plt.close(fig5)

    st.subheader("📋 Generate Report")
    if st.button("📄 Generate & Download Text Report"):
        rpt = analytics.generate_text_report(save=True)
        st.code(rpt, language="text")
        st.download_button("Download .txt", rpt, file_name="aeroflow_report.txt")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — Evaluation
# ════════════════════════════════════════════════════════════════════════════════
with tab5:
    st.subheader("📈 Model Performance Evaluation")
    st.info("""
    **Runtime metrics** (FPS, violation counts) are logged live by `prototype.py`.

    **Offline metrics** (Precision / Recall / F1 / mAP) require ground-truth labels.
    To evaluate: annotate a test video using tools like CVAT or Label Studio,
    then pass the ground-truth JSON to `evaluate.py` via the API.
    """)

    df_vio3 = load_violations()
    if not df_vio3.empty and "Confidence" in df_vio3.columns:
        st.subheader("Confidence Score Distribution")
        fig6, ax6 = dark_fig()
        df_vio3["Confidence"].dropna().hist(
            bins=20, color="#2196F3", edgecolor="#111", ax=ax6
        )
        ax6.set_xlabel("Confidence", color="#aaa", fontsize=9)
        ax6.set_ylabel("Count",      color="#aaa", fontsize=9)
        plt.tight_layout(); st.pyplot(fig6); plt.close(fig6)

        st.subheader("Avg Confidence per Violation Type")
        avg_c = (
            df_vio3.groupby("ViolationType")["Confidence"]
            .mean().sort_values(ascending=False)
        )
        fig7, ax7 = dark_fig()
        ax7.barh(avg_c.index, avg_c.values, color="#4CAF50", height=0.5)
        ax7.set_xlabel("Avg Confidence", color="#aaa", fontsize=9)
        ax7.set_xlim(0, 1.0)
        for i, v in enumerate(avg_c.values):
            ax7.text(v + 0.01, i, f"{v:.3f}", va="center",
                     color="#eee", fontsize=8)
        plt.tight_layout(); st.pyplot(fig7); plt.close(fig7)

        # Evaluation metrics reference table
        st.subheader("📊 Evaluation Metrics Reference")
        metrics_ref = pd.DataFrame({
            "Metric"        : ["Precision", "Recall", "F1-Score", "mAP@0.45", "FPS"],
            "Description"   : [
                "Of all detections flagged, what % were real violations",
                "Of all real violations, what % were detected",
                "Harmonic mean of Precision and Recall",
                "Mean Average Precision at IoU threshold 0.45",
                "Frames processed per second (computational efficiency)",
            ],
            "Target (≥)"    : ["0.80", "0.75", "0.77", "0.70", "15"],
        })
        st.dataframe(metrics_ref, use_container_width=True, hide_index=True)
        with tab6:
            st.subheader("🗺️ Bengaluru Violation Heatmap")
            map_path = os.path.join("evidence", "reports", "hotspot_map.html")
            if os.path.exists(map_path):
                import streamlit.components.v1 as components
                with open(map_path, "r", encoding="utf-8") as f:
                    components.html(f.read(), height=520, scrolling=False)
                st.caption("Red = Critical hotspot  |  Yellow = High  |  Green = Low")
                else:
                st.info("Run prototype.py and press H to generate the hotspot map.")
# ── Auto-refresh — MUST be last ───────────────────────────────────────────────
st.caption(f"🔄 Auto-refreshing every {DASHBOARD_REFRESH_SECS}s")
time.sleep(DASHBOARD_REFRESH_SECS)
st.rerun()
