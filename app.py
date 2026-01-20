import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
import swisseph as swe
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ============================================================
# Planetary Edge Finder (Correct bifurcation)
#   FAST planets -> Level-1 overextension events -> reversion success
#   SLOW planets (Jupiter/Saturn/Rahu/Ketu) -> EMA200 regime -> trend bias
#
# Updates in this version:
# - Occurrence charts zoom-out context (1M/3M/6M/1Y/custom) around the placement period
# - Each occurrence chart highlights the actual placement window as a shaded region
# - Each occurrence chart includes a compact "other-planet legend" summarizing what other planets did
#   during that placement period (top values + coverage)
# ============================================================

# -------------------- Page / Style --------------------
st.set_page_config(page_title="Planetary Edge Finder", layout="wide")
st.markdown(
    """
<style>
.big-verdict {
    padding: 18px 18px;
    border-radius: 14px;
    color: white;
    text-align: left;
    font-size: 18px;
    margin: 14px 0;
}
.badge {
    display:inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    font-weight: 700;
    margin-right: 10px;
}
.bg-green { background: #10b981; }
.bg-yellow{ background: #f59e0b; }
.bg-red   { background: #ef4444; }
.smallmuted { color: #6b7280; font-size: 13px; }
.legendbox {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 10px 12px;
    margin-top: 8px;
    background: #ffffff;
}
.legendtitle { font-weight: 700; margin-bottom: 6px; }
.legenditem { margin: 2px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- Constants --------------------
SIGNS = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

ASSET_PRESETS = {
    "Nifty 50 (^NSEI)": "^NSEI",
    "Silver (SI=F)": "SI=F",
    "Gold (GC=F)": "GC=F",
    "Crude (CL=F)": "CL=F",
    "Bitcoin (BTC-USD)": "BTC-USD",
    "S&P 500 (^GSPC)": "^GSPC",
}

# Swiss Ephemeris (Lahiri sidereal)
swe.set_sid_mode(swe.SIDM_LAHIRI)

# FAST features for event/reversion analysis
FAST_FEATURES = [
    "Moon_sign", "Mercury_sign", "Venus_sign", "Mars_sign",
    "Phase", "Mercury_retro", "Moon_Node",
]

# SLOW features for trend/EMA200 analysis
SLOW_FEATURES = [
    "Jupiter_sign", "Saturn_sign", "Rahu_sign", "Ketu_sign",
    "Jupiter_retro", "Saturn_retro",
    "JS_aspect",
]

ALL_FEATURES = list(dict.fromkeys(FAST_FEATURES + SLOW_FEATURES))

# -------------------- Helpers --------------------
def sign_from_lon(lon_deg: float) -> str:
    return SIGNS[int((lon_deg % 360.0) // 30)]

def julday_fast(d, hour_ist: int = 10) -> float:
    dt_local = datetime(d.year, d.month, d.day, hour_ist, 0, 0, tzinfo=IST)
    dt_utc = dt_local.astimezone(UTC)
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour + dt_utc.minute/60.0 + dt_utc.second/3600.0)

def julday_slow(d, hour_utc: int = 12) -> float:
    dt_utc = datetime(d.year, d.month, d.day, hour_utc, 0, 0, tzinfo=UTC)
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour)

def sidereal_lon_speed(jd_ut: float, body: int) -> tuple[float, float]:
    flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL
    xx, _ = swe.calc_ut(jd_ut, body, flags)
    lon = float(xx[0]) % 360.0
    spd = float(xx[3])
    return lon, spd

def sign_distance(a: str, b: str) -> int:
    ia = SIGNS.index(a)
    ib = SIGNS.index(b)
    d = (ib - ia) % 12
    return min(d, 12 - d)

def verdict_class_from_edge(edge_pp: float, n: int, min_n: int) -> tuple[str, str]:
    if n < min_n:
        return "INSUFFICIENT SAMPLE", "bg-red"
    if edge_pp >= 10:
        return "STRONG EDGE", "bg-green"
    if edge_pp >= 3:
        return "WEAK EDGE", "bg-yellow"
    return "NO EDGE", "bg-red"

def safe_pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x*100:.1f}%"

def safe_pp(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{x:+.1f}pp"

def match_combo(df: pd.DataFrame, combo: dict) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for feat, val in combo.items():
        if feat not in df.columns:
            m &= False
            continue
        if isinstance(val, (list, tuple, set)):
            m &= df[feat].isin(list(val))
        else:
            m &= (df[feat] == val)
    return m

def contiguous_blocks(mask: pd.Series) -> list[tuple[int,int]]:
    arr = mask.fillna(False).astype(bool).values
    idx = np.where(arr)[0]
    if len(idx) == 0:
        return []
    blocks = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            blocks.append((start, prev))
            start = i
            prev = i
    blocks.append((start, prev))
    return blocks

def context_days_from_mode(mode: str, custom_days: int) -> int:
    if mode == "None":
        return 0
    if mode.startswith("1M"):
        return 30
    if mode.startswith("3M"):
        return 90
    if mode.startswith("6M"):
        return 180
    if mode.startswith("1Y"):
        return 365
    return int(custom_days)

def summarize_other_planets(df_period: pd.DataFrame, features: list[str], top_k: int = 2) -> dict:
    """
    Summarize what other planets were doing during the actual placement period.
    For each feature:
      - list the top_k most common values with % coverage.
    """
    out = {}
    for feat in features:
        if feat not in df_period.columns:
            continue
        s = df_period[feat].dropna()
        if len(s) == 0:
            continue
        vc = s.value_counts(dropna=True)
        total = vc.sum()
        items = []
        for val, cnt in vc.head(top_k).items():
            pct = (cnt / total) * 100.0
            items.append((val, pct))
        out[feat] = items
    return out

def render_legend_box(summary: dict):
    if not summary:
        return
    lines = []
    for feat, items in summary.items():
        parts = [f"{val} ({pct:.0f}%)" for val, pct in items]
        lines.append(f"<div class='legenditem'><b>{feat}</b>: " + ", ".join(parts) + "</div>")
    html = "<div class='legendbox'><div class='legendtitle'>Other planets during this placement</div>" + "".join(lines) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

# -------------------- Data fetch / normalization --------------------
@st.cache_data(show_spinner=False)
def fetch_price(symbol: str, period: str, interval: str = "1d") -> pd.DataFrame:
    raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().copy()

    # yfinance might use Date or Datetime
    if "Date" in df.columns:
        dt_col = "Date"
    elif "Datetime" in df.columns:
        dt_col = "Datetime"
    else:
        dt_col = df.columns[0]

    df = df.rename(columns={dt_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    df = df[pd.notna(df["Date"])].copy()
    return df

# -------------------- Tech computations --------------------
def compute_tech_fast_events(
    df_px: pd.DataFrame,
    bb_w: int,
    bb_k: float,
    margin: float,
    lookahead: int,
    stopout: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df_px.copy()

    df["TP"] = (df["High"] + df["Low"]) / 2.0
    df["MA20"] = df["TP"].rolling(bb_w).mean()
    df["STD20"] = df["TP"].rolling(bb_w).std()
    df["Upper"] = df["MA20"] + (bb_k - margin) * df["STD20"]
    df["Lower"] = df["MA20"] - (bb_k - margin) * df["STD20"]

    df["ext_dir"] = np.where(df["TP"] >= df["Upper"], "U", np.where(df["TP"] <= df["Lower"], "D", "N"))
    df["event_start"] = (df["ext_dir"] != df["ext_dir"].shift(1)) & (df["ext_dir"] != "N")

    y_revert = [np.nan] * len(df)
    for i in range(len(df)):
        if not bool(df.loc[i, "event_start"]):
            continue
        side = df.loc[i, "ext_dir"]
        if side not in ("U", "D"):
            continue

        ref_high = float(df.loc[i, "High"])
        ref_low = float(df.loc[i, "Low"])

        success = 0
        for j in range(i + 1, min(i + 1 + lookahead, len(df))):
            tp_j = float(df.loc[j, "TP"])
            ma_j = df.loc[j, "MA20"]
            if pd.isna(ma_j):
                continue
            ma_j = float(ma_j)

            if side == "U":
                if tp_j <= ma_j:
                    success = 1
                    break
                if tp_j > ref_high * (1.0 + stopout):
                    success = 0
                    break
            else:
                if tp_j >= ma_j:
                    success = 1
                    break
                if tp_j < ref_low * (1.0 - stopout):
                    success = 0
                    break

        y_revert[i] = success

    df["y_revert"] = y_revert
    events = df[df["event_start"] & df["ext_dir"].isin(["U","D"])].copy()
    return df, events

def compute_tech_slow_trend(df_px: pd.DataFrame) -> pd.DataFrame:
    df = df_px.copy()
    df["ema200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["trend"] = np.where(df["Close"] >= df["ema200"], "UP", "DOWN")
    df["is_uptrend"] = (df["trend"] == "UP").astype(int)
    df["is_downtrend"] = (df["trend"] == "DOWN").astype(int)
    return df

# -------------------- Astro (always returns Date column) --------------------
@st.cache_data(show_spinner=False)
def compute_astro(dates: list) -> pd.DataFrame:
    cols = ["Date"] + ALL_FEATURES
    if dates is None or len(dates) == 0:
        return pd.DataFrame(columns=cols)

    norm_dates = pd.to_datetime(pd.Series(dates), errors="coerce").dt.date
    norm_dates = [d for d in norm_dates.tolist() if pd.notna(d)]
    if len(norm_dates) == 0:
        return pd.DataFrame(columns=cols)

    rows = []
    for d in norm_dates:
        jd_f = julday_fast(d, 10)
        jd_s = julday_slow(d, 12)

        # FAST planets + lunar constructs
        moon_lon, _ = sidereal_lon_speed(jd_f, swe.MOON)
        merc_lon, merc_sp = sidereal_lon_speed(jd_f, swe.MERCURY)
        ven_lon, _ = sidereal_lon_speed(jd_f, swe.VENUS)
        mars_lon, _ = sidereal_lon_speed(jd_f, swe.MARS)
        sun_lon, _ = sidereal_lon_speed(jd_f, swe.SUN)

        phase_angle = (moon_lon - sun_lon) % 360.0
        phase = "Waxing" if phase_angle < 180.0 else "Waning"

        # nodes at slow-time for regime consistency
        rahu_lon, _ = sidereal_lon_speed(jd_s, swe.MEAN_NODE)
        ketu_lon = (rahu_lon + 180.0) % 360.0

        d_moon_rahu = abs((moon_lon - rahu_lon + 180.0) % 360.0 - 180.0)
        moon_node = "Conjunct" if (d_moon_rahu <= 15.0 or abs(d_moon_rahu - 180.0) <= 15.0) else "None"

        jup_lon, jup_sp = sidereal_lon_speed(jd_s, swe.JUPITER)
        sat_lon, sat_sp = sidereal_lon_speed(jd_s, swe.SATURN)

        jup_sign = sign_from_lon(jup_lon)
        sat_sign = sign_from_lon(sat_lon)
        js_d = sign_distance(jup_sign, sat_sign)
        if js_d == 0:
            js_aspect = "CONJUNCTION"
        elif js_d == 6:
            js_aspect = "OPPOSITION"
        else:
            js_aspect = "NONE"

        rows.append({
            "Date": d,

            # fast
            "Moon_sign": sign_from_lon(moon_lon),
            "Mercury_sign": sign_from_lon(merc_lon),
            "Venus_sign": sign_from_lon(ven_lon),
            "Mars_sign": sign_from_lon(mars_lon),
            "Phase": phase,
            "Mercury_retro": bool(merc_sp < 0),
            "Moon_Node": moon_node,

            # slow
            "Rahu_sign": sign_from_lon(rahu_lon),
            "Ketu_sign": sign_from_lon(ketu_lon),
            "Jupiter_sign": jup_sign,
            "Saturn_sign": sat_sign,
            "Jupiter_retro": bool(jup_sp < 0),
            "Saturn_retro": bool(sat_sp < 0),
            "JS_aspect": js_aspect,
        })

    out = pd.DataFrame(rows)
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]

# -------------------- Lift lookup tables --------------------
def lift_lookup_fast(events_side: pd.DataFrame, min_n_events: int) -> tuple[float, dict, pd.DataFrame]:
    base = float(events_side["y_revert"].mean())
    rows = []
    lk = {}
    for feat in FAST_FEATURES + ["Rahu_sign","Ketu_sign"]:
        if feat not in events_side.columns:
            continue
        g = events_side.groupby(feat)["y_revert"].agg(["count","mean"]).reset_index()
        for _, r in g.iterrows():
            val = r[feat]
            if pd.isna(val):
                continue
            n = int(r["count"])
            if n < min_n_events:
                continue
            p = float(r["mean"])
            lift_pp = (p - base) * 100.0
            lk[(feat, val)] = {"n": n, "p": p, "lift_pp": lift_pp}
            rows.append({"Feature": feat, "Value": val, "n_events": n, "P(revert|value)": p, "Lift_vs_baseline_pp": lift_pp})
    df = pd.DataFrame(rows).sort_values(["Lift_vs_baseline_pp","n_events"], ascending=[False, False]).reset_index(drop=True) if rows else pd.DataFrame()
    return base, lk, df

def lift_lookup_slow(df_trend: pd.DataFrame, min_n_days: int) -> tuple[float, dict, pd.DataFrame]:
    base_up = float(df_trend["is_uptrend"].mean())
    rows = []
    lk = {}
    for feat in SLOW_FEATURES:
        if feat not in df_trend.columns:
            continue
        g = df_trend.groupby(feat).agg(n_days=("Date","count"), p_up=("is_uptrend","mean"), p_dn=("is_downtrend","mean")).reset_index()
        for _, r in g.iterrows():
            val = r[feat]
            if pd.isna(val):
                continue
            n = int(r["n_days"])
            if n < min_n_days:
                continue
            p_up = float(r["p_up"])
            p_dn = float(r["p_dn"])
            lift_pp = (p_up - base_up) * 100.0
            lk[(feat, val)] = {"n": n, "p_up": p_up, "p_dn": p_dn, "lift_pp": lift_pp}
            rows.append({"Feature": feat, "Value": val, "n_days": n, "P(UP|value)": p_up, "P(DOWN|value)": p_dn, "Lift_UP_vs_baseline_pp": lift_pp})
    df = pd.DataFrame(rows).sort_values(["Lift_UP_vs_baseline_pp","n_days"], ascending=[False, False]).reset_index(drop=True) if rows else pd.DataFrame()
    return base_up, lk, df

# -------------------- Charting --------------------
def plot_slice(
    df_slice: pd.DataFrame,
    title: str,
    show_bb: bool,
    show_ema200: bool,
    vlines: list | None = None,
    highlight_window: tuple | None = None,
):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_slice["Date"].astype(str),
        open=df_slice["Open"], high=df_slice["High"], low=df_slice["Low"], close=df_slice["Close"],
        name="Price"
    ))

    if show_bb and all(c in df_slice.columns for c in ["MA20","Upper","Lower"]):
        fig.add_trace(go.Scatter(x=df_slice["Date"].astype(str), y=df_slice["MA20"], mode="lines", name="MA20"))
        fig.add_trace(go.Scatter(x=df_slice["Date"].astype(str), y=df_slice["Upper"], mode="lines", name="Upper"))
        fig.add_trace(go.Scatter(x=df_slice["Date"].astype(str), y=df_slice["Lower"], mode="lines", name="Lower"))

    if show_ema200 and "ema200" in df_slice.columns:
        fig.add_trace(go.Scatter(x=df_slice["Date"].astype(str), y=df_slice["ema200"], mode="lines", name="EMA200"))

    # Highlight actual placement window inside context
    if highlight_window is not None:
        w0, w1 = highlight_window
        fig.add_vrect(x0=str(w0), x1=str(w1), opacity=0.14, line_width=0)

    # Event vlines
    if vlines:
        for x in vlines:
            d = x.get("date")
            if d is None:
                continue
            if x.get("outcome") is None:
                fig.add_vline(x=str(d), opacity=0.25, line_width=1)
            else:
                color = "#10b981" if int(x["outcome"]) == 1 else "#ef4444"
                fig.add_vline(x=str(d), opacity=0.45, line_width=2, line_color=color)

    fig.update_layout(
        title=title,
        height=430,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_future_lifts(df_future: pd.DataFrame, base_prob: float, lk: dict, feature_list: list, title_prefix: str, prob_label: str):
    if df_future.empty:
        st.info("No future dates available.")
        return

    out = pd.DataFrame({"Date": df_future["Date"]})
    for feat in feature_list:
        vals = df_future[feat] if feat in df_future.columns else pd.Series([np.nan]*len(df_future))
        lifts = []
        for v in vals.tolist():
            if (feat, v) in lk:
                lifts.append(float(lk[(feat, v)]["lift_pp"]))
            else:
                lifts.append(0.0)
        out[feat] = lifts

    out["TotalLift_pp"] = out[feature_list].sum(axis=1)
    out["ForceProb"] = np.clip(base_prob + out["TotalLift_pp"]/100.0, 0.0, 1.0)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=out["Date"].astype(str), y=out["ForceProb"], mode="lines", name=prob_label))
    fig1.add_hline(y=base_prob, line_dash="dash", opacity=0.5)
    fig1.update_layout(
        title=f"{title_prefix} — Total Force (baseline + summed lifts)",
        height=360,
        xaxis_title="Date",
        yaxis_title=prob_label,
        hovermode="x unified",
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = go.Figure()
    for feat in feature_list:
        fig2.add_trace(go.Scatter(x=out["Date"].astype(str), y=out[feat], mode="lines", name=feat))
    fig2.add_trace(go.Scatter(x=out["Date"].astype(str), y=out["TotalLift_pp"], mode="lines", name="TotalLift_pp"))
    fig2.update_layout(
        title=f"{title_prefix} — Lift contribution by placement (pp)",
        height=430,
        xaxis_title="Date",
        yaxis_title="Lift (percentage-points)",
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# UI
# ============================================================
st.title("Planetary Edge Finder")
st.caption("FAST: Level-1 peaks/bottoms reversion • SLOW: EMA200 trend regimes • Occurrence charts include zoom-out + other-planet legend")

with st.sidebar:
    st.header("Asset")
    preset = st.selectbox("Preset", list(ASSET_PRESETS.keys()), index=0)
    symbol_default = ASSET_PRESETS[preset]
    symbol = st.text_input("Or type Yahoo symbol", value=symbol_default).strip() or symbol_default

    st.header("History")
    period = st.selectbox("Period", ["max", "20y", "10y", "5y", "2y"], index=0)
    interval = "1d"

    st.header("FAST event settings (Level-1)")
    bb_w = st.slider("BB Window", 10, 40, 20)
    bb_k = st.slider("BB K", 1.0, 3.0, 2.0, 0.1)
    margin = st.slider("Margin Sigma", 0.0, 1.0, 0.25, 0.05)
    lookahead = st.slider("Lookahead days", 5, 30, 15)
    stopout = st.slider("Stopout %", 0.5, 5.0, 2.0, 0.1) / 100.0

    st.header("Robustness")
    min_n_events = st.slider("FAST min events (n)", 5, 80, 15)
    min_n_days = st.slider("SLOW min days (n)", 50, 1600, 200)

    st.header("Charting")
    show_bb = st.checkbox("Show Bollinger thresholds (FAST charts)", value=True)
    show_ema = st.checkbox("Show EMA200 (SLOW charts)", value=True)
    max_occ_charts = st.slider("Max occurrence charts to render", 5, 80, 25)

    st.header("Occurrence chart context")
    context_mode = st.selectbox(
        "Zoom-out context",
        ["None", "1M before/after", "3M before/after", "6M before/after", "1Y before/after", "Custom days"],
        index=1
    )
    custom_days = st.number_input("Custom days (each side)", 7, 2000, 90, step=1)
    pad_days = context_days_from_mode(context_mode, int(custom_days))

    show_planet_legend = st.checkbox("Show other-planet legend per occurrence chart", value=True)

    legend_planets_fast = st.multiselect(
        "Legend planets (FAST occurrence charts)",
        ["Moon_sign","Mercury_sign","Venus_sign","Mars_sign","Phase","Mercury_retro","Moon_Node","Rahu_sign","Ketu_sign"],
        default=["Mercury_sign","Venus_sign","Moon_sign","Phase","Rahu_sign","Ketu_sign"]
    )
    legend_planets_slow = st.multiselect(
        "Legend planets (SLOW occurrence charts)",
        ["Jupiter_sign","Saturn_sign","Rahu_sign","Ketu_sign","Jupiter_retro","Saturn_retro","JS_aspect"],
        default=["Jupiter_sign","Saturn_sign","Rahu_sign","Ketu_sign","JS_aspect"]
    )

tabs = st.tabs(["FAST: Reversion Edge (Level-1)", "SLOW: Trend Edge (EMA200)", "Future Force"])

# ============================================================
# Load shared data
# ============================================================
with st.spinner("Loading market data…"):
    df_px = fetch_price(symbol, period, interval)

if df_px.empty:
    st.error("No price data returned. Try a different symbol or smaller period.")
    st.stop()

with st.spinner("Computing astro features…"):
    df_astro = compute_astro(df_px["Date"].tolist())

# FAST prep
df_fast_all, events = compute_tech_fast_events(df_px, bb_w, bb_k, margin, lookahead, stopout)
df_fast_all = df_fast_all.merge(df_astro, on="Date", how="left")
events = events.merge(df_astro, on="Date", how="left")

# SLOW prep
df_slow_all = compute_tech_slow_trend(df_px)
df_slow_all = df_slow_all.merge(df_astro, on="Date", how="left")

# ============================================================
# TAB 1: FAST (events + reversion)
# ============================================================
with tabs[0]:
    st.subheader("FAST: Reversion Edge on Level-1 Overextension Events")

    colA, colB = st.columns([1, 2])
    with colA:
        side = st.radio("Event side", ["U", "D"], horizontal=True)
    with colB:
        st.markdown(
            """
<div class="smallmuted">
U = overextension above Upper threshold (peaky zone). D = overextension below Lower threshold (bottomy zone).<br>
Occurrence charts show the placement period, plus your chosen zoom-out context. Actual placement window is shaded.
</div>
""",
            unsafe_allow_html=True,
        )

    events_side = events[(events["ext_dir"] == side) & pd.notna(events["y_revert"])].copy()
    if len(events_side) == 0:
        st.error("No labeled events found for this side with current settings.")
        st.stop()

    base_fast, lk_fast, disc_fast = lift_lookup_fast(events_side, min_n_events=min_n_events)

    # Combo builder (FAST + optional node filters)
    with st.expander("FAST combo builder (placements)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            moon_signs = st.multiselect("Moon sign", SIGNS, default=[])
        with c2:
            merc_signs = st.multiselect("Mercury sign", SIGNS, default=[])
        with c3:
            venus_signs = st.multiselect("Venus sign", SIGNS, default=[])
        with c4:
            mars_signs = st.multiselect("Mars sign", SIGNS, default=[])

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            phases = st.multiselect("Phase", ["Waxing", "Waning"], default=[])
        with c6:
            moon_node = st.multiselect("Moon-Node", ["Conjunct", "None"], default=[])
        with c7:
            merc_retro = st.selectbox("Mercury retro", ["Any", "True", "False"], index=0)
        with c8:
            include_nodes = st.checkbox("Include Rahu/Ketu sign filters", value=False)

        rahu_signs = []
        ketu_signs = []
        if include_nodes:
            c9, c10 = st.columns(2)
            with c9:
                rahu_signs = st.multiselect("Rahu sign", SIGNS, default=[])
            with c10:
                ketu_signs = st.multiselect("Ketu sign", SIGNS, default=[])

    fast_combo = {}
    if moon_signs: fast_combo["Moon_sign"] = moon_signs
    if merc_signs: fast_combo["Mercury_sign"] = merc_signs
    if venus_signs: fast_combo["Venus_sign"] = venus_signs
    if mars_signs: fast_combo["Mars_sign"] = mars_signs
    if phases: fast_combo["Phase"] = phases
    if moon_node: fast_combo["Moon_Node"] = moon_node
    if merc_retro != "Any":
        fast_combo["Mercury_retro"] = (merc_retro == "True")
    if include_nodes:
        if rahu_signs: fast_combo["Rahu_sign"] = rahu_signs
        if ketu_signs: fast_combo["Ketu_sign"] = ketu_signs

    baseline_n = int(len(events_side))
    if fast_combo:
        m_ev = match_combo(events_side, fast_combo)
        combo_events = events_side[m_ev].copy()
    else:
        combo_events = events_side.copy()

    combo_n = int(len(combo_events))
    combo_p = float(combo_events["y_revert"].mean()) if combo_n else np.nan
    edge_pp = (combo_p - base_fast) * 100.0 if np.isfinite(combo_p) else np.nan

    verdict_text, verdict_class = verdict_class_from_edge(edge_pp if np.isfinite(edge_pp) else -999, combo_n, min_n_events)

    st.markdown(
        f"""
<div class="big-verdict {verdict_class}">
  <span class="badge">FAST</span>
  <b>{verdict_text}</b><br>
  Baseline P(revert) for side <b>{side}</b>: <b>{safe_pct(base_fast)}</b> (n={baseline_n})<br>
  Selected combo P(revert): <b>{safe_pct(combo_p)}</b> (n={combo_n}) • Edge: <b>{safe_pp(edge_pp)}</b>
</div>
""",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline P(revert)", safe_pct(base_fast))
    k2.metric("Combo P(revert)", safe_pct(combo_p))
    k3.metric("Edge vs baseline", safe_pp(edge_pp))
    k4.metric("Combo n (events)", f"{combo_n}")

    # Occurrence charts with context + legend
    st.subheader("FAST occurrence charts (one per placement-period)")
    if not fast_combo:
        st.info("Select at least one placement in the FAST combo builder to generate occurrence charts.")
    else:
        mask_full = match_combo(df_fast_all, fast_combo)
        blocks = contiguous_blocks(mask_full)

        if len(blocks) == 0:
            st.info("No historical placement periods found for this combo.")
        else:
            block_ranges = []
            for a, b in blocks:
                block_ranges.append((df_fast_all.loc[a, "Date"], df_fast_all.loc[b, "Date"], a, b))
            block_ranges.sort(key=lambda x: x[0], reverse=True)

            shown = 0
            for start_d, end_d, a, b in block_ranges:
                if shown >= max_occ_charts:
                    break

                # context slice
                start_ctx = start_d - timedelta(days=pad_days)
                end_ctx = end_d + timedelta(days=pad_days)
                df_slice = df_fast_all[(df_fast_all["Date"] >= start_ctx) & (df_fast_all["Date"] <= end_ctx)].copy()

                # actual placement period (for legend + shading)
                df_period = df_fast_all.iloc[a:b+1].copy()

                # event lines inside the actual placement period
                vlines = []
                ev_slice = combo_events[(combo_events["Date"] >= start_d) & (combo_events["Date"] <= end_d)].copy()
                if len(ev_slice):
                    for _, r in ev_slice.iterrows():
                        vlines.append({"date": r["Date"], "outcome": int(r["y_revert"])})

                title = f"{start_d} → {end_d} | placement-days={len(df_period)} | {side}-events={len(ev_slice)} | context=±{pad_days}d"
                with st.expander(title, expanded=(shown < 3)):
                    plot_slice(
                        df_slice,
                        title=title,
                        show_bb=show_bb,
                        show_ema200=False,
                        vlines=vlines,
                        highlight_window=(start_d, end_d),
                    )
                    if show_planet_legend:
                        summary = summarize_other_planets(df_period, legend_planets_fast, top_k=2)
                        render_legend_box(summary)

                shown += 1

    st.subheader("FAST discovery: single placements ranked by lift (pp)")
    if disc_fast is None or len(disc_fast) == 0:
        st.info("No FAST single-feature signals meet the minimum event count.")
    else:
        df_show = disc_fast.copy()
        df_show["P(revert|value)"] = df_show["P(revert|value)"].map(lambda x: round(float(x) * 100, 1))
        df_show["Lift_vs_baseline_pp"] = df_show["Lift_vs_baseline_pp"].map(lambda x: round(float(x), 1))
        st.dataframe(df_show, use_container_width=True, height=420)

# ============================================================
# TAB 2: SLOW (trend vs EMA200)
# ============================================================
with tabs[1]:
    st.subheader("SLOW: Trend Edge using EMA200 Regime (Jupiter/Saturn/Rahu/Ketu)")

    base_up, lk_slow, disc_slow = lift_lookup_slow(df_slow_all, min_n_days=min_n_days)
    base_dn = float(df_slow_all["is_downtrend"].mean())
    base_days = int(len(df_slow_all))

    with st.expander("SLOW combo builder (placements)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            jup_signs = st.multiselect("Jupiter sign", SIGNS, default=[])
        with c2:
            sat_signs = st.multiselect("Saturn sign", SIGNS, default=[])
        with c3:
            rahu_signs = st.multiselect("Rahu sign", SIGNS, default=[])
        with c4:
            ketu_signs = st.multiselect("Ketu sign", SIGNS, default=[])

        c5, c6, c7 = st.columns(3)
        with c5:
            jup_retro = st.selectbox("Jupiter retro", ["Any", "True", "False"], index=0)
        with c6:
            sat_retro = st.selectbox("Saturn retro", ["Any", "True", "False"], index=0)
        with c7:
            js_aspect = st.multiselect("Jupiter–Saturn aspect", ["CONJUNCTION", "OPPOSITION", "NONE"], default=[])

    slow_combo = {}
    if jup_signs: slow_combo["Jupiter_sign"] = jup_signs
    if sat_signs: slow_combo["Saturn_sign"] = sat_signs
    if rahu_signs: slow_combo["Rahu_sign"] = rahu_signs
    if ketu_signs: slow_combo["Ketu_sign"] = ketu_signs
    if js_aspect: slow_combo["JS_aspect"] = js_aspect
    if jup_retro != "Any":
        slow_combo["Jupiter_retro"] = (jup_retro == "True")
    if sat_retro != "Any":
        slow_combo["Saturn_retro"] = (sat_retro == "True")

    if slow_combo:
        m_day = match_combo(df_slow_all, slow_combo)
        slow_sub = df_slow_all[m_day].copy()
    else:
        slow_sub = df_slow_all.copy()

    n_days = int(len(slow_sub))
    p_up = float(slow_sub["is_uptrend"].mean()) if n_days else np.nan
    p_dn = float(slow_sub["is_downtrend"].mean()) if n_days else np.nan
    lift_up_pp = (p_up - base_up) * 100.0 if np.isfinite(p_up) else np.nan

    verdict_text, verdict_class = verdict_class_from_edge(lift_up_pp if np.isfinite(lift_up_pp) else -999, n_days, min_n_days)

    st.markdown(
        f"""
<div class="big-verdict {verdict_class}">
  <span class="badge">SLOW</span>
  <b>{verdict_text}</b><br>
  Baseline P(UP): <b>{safe_pct(base_up)}</b> • P(DOWN): <b>{safe_pct(base_dn)}</b> (days={base_days})<br>
  Selected combo P(UP): <b>{safe_pct(p_up)}</b> • P(DOWN): <b>{safe_pct(p_dn)}</b> (days={n_days}) • Lift in UP odds: <b>{safe_pp(lift_up_pp)}</b>
</div>
""",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline P(UP)", safe_pct(base_up))
    k2.metric("Combo P(UP)", safe_pct(p_up))
    k3.metric("Lift in P(UP)", safe_pp(lift_up_pp))
    k4.metric("Combo n (days)", f"{n_days}")

    st.subheader("SLOW occurrence charts (one per placement-period)")
    if not slow_combo:
        st.info("Select at least one placement in the SLOW combo builder to generate occurrence charts.")
    else:
        mask_full = match_combo(df_slow_all, slow_combo)
        blocks = contiguous_blocks(mask_full)

        if len(blocks) == 0:
            st.info("No historical placement periods found for this combo.")
        else:
            block_ranges = []
            for a, b in blocks:
                block_ranges.append((df_slow_all.loc[a, "Date"], df_slow_all.loc[b, "Date"], a, b))
            block_ranges.sort(key=lambda x: x[0], reverse=True)

            shown = 0
            for start_d, end_d, a, b in block_ranges:
                if shown >= max_occ_charts:
                    break

                start_ctx = start_d - timedelta(days=pad_days)
                end_ctx = end_d + timedelta(days=pad_days)
                df_slice = df_slow_all[(df_slow_all["Date"] >= start_ctx) & (df_slow_all["Date"] <= end_ctx)].copy()
                df_period = df_slow_all.iloc[a:b+1].copy()

                title = f"{start_d} → {end_d} | placement-days={len(df_period)} | P(UP) in period={df_period['is_uptrend'].mean()*100:.1f}% | context=±{pad_days}d"
                with st.expander(title, expanded=(shown < 3)):
                    plot_slice(
                        df_slice,
                        title=title,
                        show_bb=False,
                        show_ema200=show_ema,
                        vlines=None,
                        highlight_window=(start_d, end_d),
                    )
                    if show_planet_legend:
                        summary = summarize_other_planets(df_period, legend_planets_slow, top_k=2)
                        render_legend_box(summary)

                shown += 1

    st.subheader("SLOW discovery: single placements ranked by UP-regime lift (pp)")
    if disc_slow is None or len(disc_slow) == 0:
        st.info("No SLOW single-feature signals meet the minimum day count.")
    else:
        df_show = disc_slow.copy()
        df_show["P(UP|value)"] = df_show["P(UP|value)"].map(lambda x: round(float(x) * 100, 1))
        df_show["P(DOWN|value)"] = df_show["P(DOWN|value)"].map(lambda x: round(float(x) * 100, 1))
        df_show["Lift_UP_vs_baseline_pp"] = df_show["Lift_UP_vs_baseline_pp"].map(lambda x: round(float(x), 1))
        st.dataframe(df_show, use_container_width=True, height=440)

# ============================================================
# TAB 3: Future Force
# ============================================================
with tabs[2]:
    st.subheader("Future Force: Lift Timeseries (next N days)")

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        future_days = st.selectbox("Forecast window (days)", [30, 60, 90, 120, 180], index=0)
    with col2:
        future_mode = st.selectbox("Mode", ["FAST (reversion lifts)", "SLOW (trend lifts)"], index=0)
    with col3:
        st.markdown('<div class="smallmuted">Total Force = baseline + sum of lifts (pp) from all placements that have stable historical estimates (filtered by min-n).</div>', unsafe_allow_html=True)

    start = datetime.now(tz=UTC).date()
    future_dates = [start + timedelta(days=i) for i in range(1, int(future_days) + 1)]
    df_future = compute_astro(future_dates)

    if df_future.empty:
        st.info("No future astro data returned.")
        st.stop()

    if future_mode.startswith("FAST"):
        side_ff = st.radio("FAST side for lift mapping", ["U", "D"], horizontal=True, key="future_fast_side")
        events_side = events[(events["ext_dir"] == side_ff) & pd.notna(events["y_revert"])].copy()
        if len(events_side) == 0:
            st.info("No labeled events available for this side under current settings.")
            st.stop()

        base_fast, lk_fast, _ = lift_lookup_fast(events_side, min_n_events=min_n_events)

        include_moon = st.checkbox("Include Moon_sign (very noisy)", value=False)
        ff_features = ["Mercury_sign", "Venus_sign", "Mars_sign", "Phase", "Mercury_retro", "Moon_Node", "Rahu_sign", "Ketu_sign"]
        if include_moon:
            ff_features = ["Moon_sign"] + ff_features

        plot_future_lifts(
            df_future=df_future,
            base_prob=base_fast,
            lk=lk_fast,
            feature_list=ff_features,
            title_prefix=f"FAST Future Force (side={side_ff}) — {symbol}",
            prob_label="P(revert)",
        )

    else:
        base_up, lk_slow, _ = lift_lookup_slow(df_slow_all, min_n_days=min_n_days)
        sf_features = ["Jupiter_sign", "Saturn_sign", "Rahu_sign", "Ketu_sign", "Jupiter_retro", "Saturn_retro", "JS_aspect"]

        plot_future_lifts(
            df_future=df_future,
            base_prob=base_up,
            lk=lk_slow,
            feature_list=sf_features,
            title_prefix=f"SLOW Future Force (EMA200 UP bias) — {symbol}",
            prob_label="P(UP)",
        )
