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
#   FAST planets -> Level-1 overextension events -> "reversion-to-mean" success
#   SLOW planets (Jupiter/Saturn/Rahu/Ketu) -> daily regime vs EMA200 -> trend bias
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
.kpi-title { color:#6b7280; font-size:12px; }
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

# Swiss Ephemeris (Lahiri sidereal) — required for your deployment
swe.set_sid_mode(swe.SIDM_LAHIRI)

# FAST features for event/reversion analysis
FAST_FEATURES = [
    "Moon_sign", "Mercury_sign", "Venus_sign", "Mars_sign",
    "Phase", "Mercury_retro", "Moon_Node",
]

# SLOW features for trend/EMA200 analysis (as per your bifurcation)
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
    # FAST convention: 10:00 IST -> UTC
    dt_local = datetime(d.year, d.month, d.day, hour_ist, 0, 0, tzinfo=IST)
    dt_utc = dt_local.astimezone(UTC)
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour + dt_utc.minute/60.0 + dt_utc.second/3600.0)

def julday_slow(d, hour_utc: int = 12) -> float:
    # SLOW convention: 12:00 UTC
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
    """
    edge_pp = (p_cond - p_base)*100 in percentage-points.
    Provide a simple banding + respect min_n.
    """
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

# -------------------- Data fetch / normalization (Streamlit Cloud-safe) --------------------
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

def compute_tech_fast_events(
    df_px: pd.DataFrame,
    bb_w: int,
    bb_k: float,
    margin: float,
    lookahead: int,
    stopout: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    FAST logic (Gemini):
      - Overextension streak starts => Level-1 events
      - y_revert success if price returns to MA20 before stopout extension
    """
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
    """
    SLOW logic (Long planets):
      - Trend label based on EMA200 regime
    """
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

        # SLOW planets (and nodes) computed at slow-time for regime consistency
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

# -------------------- Metrics tables --------------------
def discovery_table_fast(events_side: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """
    For FAST event analysis:
      baseline = P(revert) on event days for a given side (U or D).
      For each feature/value:
         P(revert|feature=value), n, lift_pp = (p_cond - baseline)*100
    """
    base = float(events_side["y_revert"].mean())
    rows = []
    for feat in FAST_FEATURES + ["Rahu_sign","Ketu_sign"]:  # allow nodes to be used as event filters if desired
        if feat not in events_side.columns:
            continue
        for val, sub in events_side.groupby(feat):
            if pd.isna(val):
                continue
            n = len(sub)
            if n < min_n:
                continue
            p = float(sub["y_revert"].mean())
            rows.append({
                "Feature": feat,
                "Value": val,
                "n_events": n,
                "P(revert|value)": p,
                "Lift_vs_baseline_pp": (p - base) * 100.0,
            })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values(["Lift_vs_baseline_pp","n_events"], ascending=[False, False]).reset_index(drop=True)

def discovery_table_slow(df_trend: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """
    For SLOW trend analysis:
      baseline = P(UP) over all days.
      For each feature/value:
         P(UP|feature=value), n, lift_up_pp = (p_up_cond - baseline_up)*100
      Also show P(DOWN|value) for readability.
    """
    base_up = float(df_trend["is_uptrend"].mean())
    rows = []
    for feat in SLOW_FEATURES:
        if feat not in df_trend.columns:
            continue
        for val, sub in df_trend.groupby(feat):
            if pd.isna(val):
                continue
            n = len(sub)
            if n < min_n:
                continue
            p_up = float(sub["is_uptrend"].mean())
            p_dn = float(sub["is_downtrend"].mean())
            rows.append({
                "Feature": feat,
                "Value": val,
                "n_days": n,
                "P(UP|value)": p_up,
                "P(DOWN|value)": p_dn,
                "Lift_UP_vs_baseline_pp": (p_up - base_up) * 100.0,
            })
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values(["Lift_UP_vs_baseline_pp","n_days"], ascending=[False, False]).reset_index(drop=True)

# -------------------- Charting --------------------
def add_sign_zone(fig, df: pd.DataFrame, feat: str, value):
    """
    Shade contiguous ranges where df[feat] == value.
    """
    if feat not in df.columns:
        return
    mask = (df[feat] == value).fillna(False).values
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return
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
    for a, b in blocks:
        x0 = str(df.loc[a, "Date"])
        x1 = str(df.loc[b, "Date"])
        fig.add_vrect(x0=x0, x1=x1, opacity=0.14, line_width=0)

def plot_price(df: pd.DataFrame, title: str, show_bb: bool, show_ema200: bool, vlines: list, zones: list):
    """
    zones: list of (feat, value)
    vlines: list of dict {date, kind, outcome?}
    """
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["Date"].astype(str),
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ))

    if show_bb and all(c in df.columns for c in ["MA20","Upper","Lower"]):
        fig.add_trace(go.Scatter(x=df["Date"].astype(str), y=df["MA20"], mode="lines", name="MA20"))
        fig.add_trace(go.Scatter(x=df["Date"].astype(str), y=df["Upper"], mode="lines", name="Upper"))
        fig.add_trace(go.Scatter(x=df["Date"].astype(str), y=df["Lower"], mode="lines", name="Lower"))

    if show_ema200 and "ema200" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"].astype(str), y=df["ema200"], mode="lines", name="EMA200"))

    for feat, val in zones:
        add_sign_zone(fig, df, feat, val)

    # vlines: red/green for outcomes when available
    for x in vlines:
        d = x.get("date")
        if d is None:
            continue
        if x.get("outcome") is None:
            fig.add_vline(x=str(d), opacity=0.25, line_width=1)
        else:
            # 1 = win (green), 0 = loss (red)
            color = "#10b981" if int(x["outcome"]) == 1 else "#ef4444"
            fig.add_vline(x=str(d), opacity=0.45, line_width=2, line_color=color)

    fig.update_layout(
        title=title,
        height=650,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# UI
# ============================================================
st.title("Planetary Edge Finder")
st.caption("FAST planets: peaks/bottoms episodes (Level-1) • SLOW planets (Jupiter/Saturn/Rahu/Ketu): EMA200 trend regimes")

with st.sidebar:
    st.header("Asset")
    preset = st.selectbox("Preset", list(ASSET_PRESETS.keys()), index=0)
    symbol_default = ASSET_PRESETS[preset]
    symbol = st.text_input("Or type Yahoo symbol", value=symbol_default).strip() or symbol_default

    st.header("History")
    # As far back as possible by default
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
    min_n_days = st.slider("SLOW min days (n)", 50, 1200, 200)

    st.header("Chart overlays")
    show_bb = st.checkbox("Show Bollinger thresholds (MA20/Upper/Lower)", value=True)
    show_ema = st.checkbox("Show EMA200", value=True)

tabs = st.tabs(["FAST: Reversion Edge (Peaks/Bottoms)", "SLOW: Trend Edge (EMA200 Regime)"])

# ============================================================
# Load data once (shared)
# ============================================================
with st.spinner("Loading market data…"):
    df_px = fetch_price(symbol, period, interval)

if df_px.empty:
    st.error("No price data returned. Try a different symbol or a smaller period.")
    st.stop()

# Compute shared astro once over all dates
with st.spinner("Computing astro features…"):
    df_astro = compute_astro(df_px["Date"].tolist())

# -------------------- FAST prep --------------------
df_fast_all, events = compute_tech_fast_events(df_px, bb_w, bb_k, margin, lookahead, stopout)
df_fast_all = df_fast_all.merge(df_astro, on="Date", how="left")
events = events.merge(df_astro, on="Date", how="left")

# -------------------- SLOW prep --------------------
df_slow_all = compute_tech_slow_trend(df_px)
df_slow_all = df_slow_all.merge(df_astro, on="Date", how="left")

# ============================================================
# TAB 1: FAST (events + reversion)
# ============================================================
with tabs[0]:
    st.subheader("FAST: Reversion Edge on Level-1 Overextension Events")

    st.markdown(
        """
**What is measured here (FAST planets)?**
- We only look at **Level-1 overextension starts** (first day price pushes beyond the adjusted Bollinger threshold).
- Each event is labeled **WIN (1)** if price returns to **MA20** within your lookahead window *before* a stop-out extension.
- This measures: **“When a peak/bottom episode starts, does it mean-revert?”**
"""
    )

    c0, c1, c2 = st.columns([1, 1, 2])
    with c0:
        side = st.radio("Event side", ["U", "D"], horizontal=True)
    with c1:
        show_outcome_lines = st.checkbox("Color event lines by outcome (green=win/red=loss)", value=True)
    with c2:
        st.markdown(
            """
<div class="smallmuted">
U = overextension above Upper threshold (peaky zone).<br>
D = overextension below Lower threshold (bottomy zone).<br>
“Lift” is shown in percentage-points vs the baseline for the chosen side.
</div>
""",
            unsafe_allow_html=True,
        )

    # Filter to selected side and valid labels
    events_side = events[(events["ext_dir"] == side) & pd.notna(events["y_revert"])].copy()
    if len(events_side) == 0:
        st.error("No labeled events found for this side with current settings.")
        st.stop()

    baseline = float(events_side["y_revert"].mean())
    baseline_n = int(len(events_side))

    # FAST combo builder
    with st.expander("FAST combo builder (optional)", expanded=True):
        colA, colB, colC, colD = st.columns(4)
        with colA:
            moon_signs = st.multiselect("Moon sign", SIGNS, default=[])
        with colB:
            merc_signs = st.multiselect("Mercury sign", SIGNS, default=[])
        with colC:
            mars_signs = st.multiselect("Mars sign", SIGNS, default=[])
        with colD:
            venus_signs = st.multiselect("Venus sign", SIGNS, default=[])

        colE, colF, colG = st.columns(3)
        with colE:
            phases = st.multiselect("Phase", ["Waxing", "Waning"], default=[])
        with colF:
            merc_retro = st.selectbox("Mercury retro", ["Any", "True", "False"], index=0)
        with colG:
            moon_node = st.multiselect("Moon-Node", ["Conjunct", "None"], default=[])

        fast_combo = {}
        if moon_signs: fast_combo["Moon_sign"] = moon_signs
        if merc_signs: fast_combo["Mercury_sign"] = merc_signs
        if mars_signs: fast_combo["Mars_sign"] = mars_signs
        if venus_signs: fast_combo["Venus_sign"] = venus_signs
        if phases: fast_combo["Phase"] = phases
        if moon_node: fast_combo["Moon_Node"] = moon_node
        if merc_retro != "Any":
            fast_combo["Mercury_retro"] = (merc_retro == "True")

    # Apply combo
    if fast_combo:
        m = match_combo(events_side, fast_combo)
        combo_events = events_side[m].copy()
    else:
        combo_events = events_side.copy()

    combo_n = int(len(combo_events))
    combo_p = float(combo_events["y_revert"].mean()) if combo_n else np.nan
    edge_pp = (combo_p - baseline) * 100.0 if np.isfinite(combo_p) else np.nan

    verdict_text, verdict_class = verdict_class_from_edge(edge_pp if np.isfinite(edge_pp) else -999, combo_n, min_n_events)

    st.markdown(
        f"""
<div class="big-verdict {verdict_class}">
  <span class="badge">FAST</span>
  <b>{verdict_text}</b><br>
  Baseline P(revert) for side <b>{side}</b>: <b>{safe_pct(baseline)}</b> (n={baseline_n})<br>
  Selected combo P(revert): <b>{safe_pct(combo_p)}</b> (n={combo_n}) • Edge: <b>{safe_pp(edge_pp)}</b>
</div>
""",
        unsafe_allow_html=True,
    )

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline P(revert)", safe_pct(baseline))
    k2.metric("Combo P(revert)", safe_pct(combo_p))
    k3.metric("Edge vs baseline", safe_pp(edge_pp))
    k4.metric("Combo n (events)", f"{combo_n}")

    # Chart: full price + BB + EMA200 + vlines on matched events
    vlines = []
    if combo_n:
        for _, r in combo_events.iterrows():
            vlines.append({
                "date": r["Date"],
                "outcome": int(r["y_revert"]) if show_outcome_lines else None
            })

    # Optional zones (keep simple: allow shading only for a single selected placement)
    zones = []
    with st.expander("FAST chart zones (optional)", expanded=False):
        zone_feat = st.selectbox("Shade where", ["None"] + FAST_FEATURES, index=0)
        zone_val = None
        if zone_feat != "None":
            zone_val = st.selectbox("Value", sorted([v for v in df_fast_all[zone_feat].dropna().unique().tolist()]))
            zones = [(zone_feat, zone_val)]

    plot_price(
        df_fast_all,
        title=f"{symbol} — FAST events (side={side}) as vertical lines",
        show_bb=show_bb,
        show_ema200=show_ema,
        vlines=vlines,
        zones=zones,
    )

    # Occurrence table (lightweight)
    if combo_n:
        st.subheader("FAST occurrences (event days)")
        show_cols = ["Date", "ext_dir", "y_revert"] + [c for c in FAST_FEATURES if c in combo_events.columns]
        st.dataframe(combo_events.sort_values("Date", ascending=False)[show_cols], use_container_width=True, height=320)

    # Discovery (single-feature edge table) with explanation
    st.subheader("FAST discovery: which single placements improve reversion odds?")
    st.markdown(
        """
**How to read this table**
- **P(revert|value)**: success rate when that placement is true (within the chosen side U/D).
- **Lift (pp)**: *(P(revert|value) − baseline)* in percentage-points.
- **n_events**: sample size; small n is often unstable. Use the min-events slider to filter.
"""
    )
    disc_fast = discovery_table_fast(events_side, min_n=min_n_events)
    if disc_fast.empty:
        st.info("No FAST single-feature signals meet the minimum event count.")
    else:
        df_show = disc_fast.copy()
        df_show["P(revert|value)"] = df_show["P(revert|value)"].map(lambda x: round(float(x)*100, 1))
        df_show["Lift_vs_baseline_pp"] = df_show["Lift_vs_baseline_pp"].map(lambda x: round(float(x), 1))
        st.dataframe(df_show, use_container_width=True, height=420)

# ============================================================
# TAB 2: SLOW (daily trend vs EMA200)
# ============================================================
with tabs[1]:
    st.subheader("SLOW: Trend Edge using EMA200 Regime (Jupiter/Saturn/Rahu/Ketu)")

    st.markdown(
        """
**What is measured here (SLOW planets)?**
- Every trading day is labeled **UP** if Close ≥ EMA200, else **DOWN**.
- This measures: **“When a slow-planet placement is active, is the market more likely to be in an UP regime?”**
- This is intentionally **trend-based**, not reversal-based.
"""
    )

    base_up = float(df_slow_all["is_uptrend"].mean())
    base_dn = float(df_slow_all["is_downtrend"].mean())
    base_days = int(len(df_slow_all))

    # SLOW combo builder (Jup/Sat/Rahu/Ketu)
    with st.expander("SLOW combo builder (optional)", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            jup_signs = st.multiselect("Jupiter sign", SIGNS, default=[])
        with col2:
            sat_signs = st.multiselect("Saturn sign", SIGNS, default=[])
        with col3:
            rahu_signs = st.multiselect("Rahu sign", SIGNS, default=[])
        with col4:
            ketu_signs = st.multiselect("Ketu sign", SIGNS, default=[])

        col5, col6, col7 = st.columns(3)
        with col5:
            jup_retro = st.selectbox("Jupiter retro", ["Any", "True", "False"], index=0)
        with col6:
            sat_retro = st.selectbox("Saturn retro", ["Any", "True", "False"], index=0)
        with col7:
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
        m = match_combo(df_slow_all, slow_combo)
        slow_sub = df_slow_all[m].copy()
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

    # SLOW chart: show EMA200 and shade placement zones (most useful here)
    zones = []
    with st.expander("SLOW chart zones (recommended)", expanded=True):
        zone_feat = st.selectbox("Shade where", ["None"] + SLOW_FEATURES, index=0)
        zone_val = None
        if zone_feat != "None":
            zone_val = st.selectbox("Value", sorted([v for v in df_slow_all[zone_feat].dropna().unique().tolist()]))
            zones = [(zone_feat, zone_val)]

    plot_price(
        df_slow_all,
        title=f"{symbol} — SLOW regimes (EMA200) with shaded zones",
        show_bb=False,
        show_ema200=show_ema,
        vlines=[],
        zones=zones,
    )

    # Discovery for slow planets (single feature)
    st.subheader("SLOW discovery: which placements shift EMA200 trend odds?")
    st.markdown(
        """
**How to read this table**
- **P(UP|value)**: fraction of days the market is above EMA200 while that placement is active.
- **Lift_UP (pp)**: *(P(UP|value) − baseline P(UP))* in percentage-points.
- **n_days**: sample size; require large n for stability (use min-days slider).
"""
    )
    disc_slow = discovery_table_slow(df_slow_all, min_n=min_n_days)
    if disc_slow.empty:
        st.info("No SLOW single-feature signals meet the minimum day count.")
    else:
        df_show = disc_slow.copy()
        df_show["P(UP|value)"] = df_show["P(UP|value)"].map(lambda x: round(float(x)*100, 1))
        df_show["P(DOWN|value)"] = df_show["P(DOWN|value)"].map(lambda x: round(float(x)*100, 1))
        df_show["Lift_UP_vs_baseline_pp"] = df_show["Lift_UP_vs_baseline_pp"].map(lambda x: round(float(x), 1))
        st.dataframe(df_show, use_container_width=True, height=440)
