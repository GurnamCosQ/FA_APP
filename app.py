import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
import swisseph as swe
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# -------------------- Global config --------------------
st.set_page_config(page_title="Planetary Force (Level-1)", layout="wide")

SIGNS = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

ASSET_PRESETS = {
    "Nifty 50 (^NSEI)": "^NSEI",
    "Silver (SI=F)": "SI=F",
    "Gold (GC=F)": "GC=F",
    "Crude (CL=F)": "CL=F",
}

# Swiss Ephemeris: Lahiri sidereal
swe.set_sid_mode(swe.SIDM_LAHIRI)

# Features used in the "brain" and force model
FEATURES = [
    # fast
    "Moon_sign", "Mercury_sign", "Venus_sign", "Mars_sign",
    "Phase", "Mercury_retro", "Moon_Node", "Rahu_sign", "Ketu_sign",
    # slow
    "Jupiter_sign", "Saturn_sign", "Jupiter_retro", "Saturn_retro",
]

# -------------------- Utility helpers --------------------
def sign_from_lon(lon_deg: float) -> str:
    return SIGNS[int((lon_deg % 360.0) // 30)]

def julday_fast(d, hour_ist: int = 10) -> float:
    # Gemini convention: 10:00 IST -> UTC
    dt_local = datetime(d.year, d.month, d.day, hour_ist, 0, 0, tzinfo=IST)
    dt_utc = dt_local.astimezone(UTC)
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour + dt_utc.minute/60.0 + dt_utc.second/3600.0)

def julday_slow(d, hour_utc: int = 12) -> float:
    # Long planets convention: 12:00 UTC
    dt_utc = datetime(d.year, d.month, d.day, hour_utc, 0, 0, tzinfo=UTC)
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour)

def sidereal_lon_speed(jd_ut: float, body: int) -> tuple[float, float]:
    flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL
    xx, _ = swe.calc_ut(jd_ut, body, flags)
    lon = float(xx[0]) % 360.0
    spd = float(xx[3])
    return lon, spd

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

# -------------------- Data layer (cached) --------------------
@st.cache_data(show_spinner=False)
def fetch_price(symbol: str, period: str, interval: str) -> pd.DataFrame:
    raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)

    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close","Volume"])

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index().copy()

    # Normalize yfinance date column name
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

def compute_tech_labels(
    df_px: pd.DataFrame,
    bb_window: int,
    bb_k: float,
    margin_sigma: float,
    lookahead_days: int,
    stop_out_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df_px.copy()

    # TP + BB thresholds (same event logic)
    df["TP"] = (df["High"] + df["Low"]) / 2.0
    df["MA20"] = df["TP"].rolling(bb_window).mean()
    df["STD20"] = df["TP"].rolling(bb_window).std()
    df["UpperThreshold"] = df["MA20"] + (bb_k - margin_sigma) * df["STD20"]
    df["LowerThreshold"] = df["MA20"] - (bb_k - margin_sigma) * df["STD20"]

    df["ext_dir"] = np.where(
        df["TP"] >= df["UpperThreshold"], "U",
        np.where(df["TP"] <= df["LowerThreshold"], "D", "N")
    )
    df["event_start"] = (df["ext_dir"] != df["ext_dir"].shift(1)) & (df["ext_dir"] != "N")

    # y_revert on event days only (Gemini logic)
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
        for j in range(i + 1, min(i + 1 + lookahead_days, len(df))):
            tp_j = float(df.loc[j, "TP"])
            ma_j = df.loc[j, "MA20"]
            if pd.isna(ma_j):
                continue
            ma_j = float(ma_j)

            if side == "U":
                if tp_j <= ma_j:
                    success = 1
                    break
                if tp_j > ref_high * (1.0 + stop_out_pct):
                    success = 0
                    break
            else:
                if tp_j >= ma_j:
                    success = 1
                    break
                if tp_j < ref_low * (1.0 - stop_out_pct):
                    success = 0
                    break

        y_revert[i] = success

    df["y_revert"] = y_revert

    # EMA200 trend (context; not used for force score here)
    df["ema200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["trend"] = np.where(df["Close"] >= df["ema200"], "UP", "DOWN")

    level1 = df[df["event_start"] & df["ext_dir"].isin(["U", "D"])].copy()
    return df, level1

@st.cache_data(show_spinner=False)
def compute_astro(dates: list) -> pd.DataFrame:
    # Always return DF with Date + expected columns (fix for merge KeyError)
    cols = ["Date"] + FEATURES
    if dates is None or len(dates) == 0:
        return pd.DataFrame(columns=cols)

    norm_dates = pd.to_datetime(pd.Series(dates), errors="coerce").dt.date
    norm_dates = [d for d in norm_dates.tolist() if pd.notna(d)]
    if len(norm_dates) == 0:
        return pd.DataFrame(columns=cols)

    rows = []
    for d in norm_dates:
        jd_f = julday_fast(d, hour_ist=10)
        jd_s = julday_slow(d, hour_utc=12)

        # Fast set
        moon_lon, _ = sidereal_lon_speed(jd_f, swe.MOON)
        merc_lon, merc_sp = sidereal_lon_speed(jd_f, swe.MERCURY)
        ven_lon, _ = sidereal_lon_speed(jd_f, swe.VENUS)
        mars_lon, _ = sidereal_lon_speed(jd_f, swe.MARS)
        sun_lon, _ = sidereal_lon_speed(jd_f, swe.SUN)

        rahu_lon, _ = sidereal_lon_speed(jd_f, swe.MEAN_NODE)
        ketu_lon = (rahu_lon + 180.0) % 360.0

        phase_angle = (moon_lon - sun_lon) % 360.0
        phase = "Waxing" if phase_angle < 180.0 else "Waning"

        d_moon_rahu = abs((moon_lon - rahu_lon + 180.0) % 360.0 - 180.0)
        moon_node = "Conjunct" if (d_moon_rahu <= 15.0 or abs(d_moon_rahu - 180.0) <= 15.0) else "None"

        # Slow set
        jup_lon, jup_sp = sidereal_lon_speed(jd_s, swe.JUPITER)
        sat_lon, sat_sp = sidereal_lon_speed(jd_s, swe.SATURN)

        rows.append({
            "Date": d,

            "Moon_sign": sign_from_lon(moon_lon),
            "Mercury_sign": sign_from_lon(merc_lon),
            "Venus_sign": sign_from_lon(ven_lon),
            "Mars_sign": sign_from_lon(mars_lon),
            "Phase": phase,
            "Mercury_retro": bool(merc_sp < 0),
            "Rahu_sign": sign_from_lon(rahu_lon),
            "Ketu_sign": sign_from_lon(ketu_lon),
            "Moon_Node": moon_node,

            "Jupiter_sign": sign_from_lon(jup_lon),
            "Saturn_sign": sign_from_lon(sat_lon),
            "Jupiter_retro": bool(jup_sp < 0),
            "Saturn_retro": bool(sat_sp < 0),
        })

    out = pd.DataFrame(rows)
    # Ensure exact column order
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]

# -------------------- Brain + force model --------------------
def build_brain(df_events: pd.DataFrame, min_n: int = 15) -> dict:
    brain = {"min_n": min_n, "U": {"baseline": None, "table": {}}, "D": {"baseline": None, "table": {}}}

    for side in ["U", "D"]:
        sub = df_events[(df_events["ext_dir"] == side) & pd.notna(df_events["y_revert"])].copy()
        if len(sub) == 0:
            brain[side]["baseline"] = np.nan
            continue

        baseline = float(sub["y_revert"].mean())
        brain[side]["baseline"] = baseline

        for feat in FEATURES:
            if feat not in sub.columns:
                continue
            g = sub.groupby(feat)["y_revert"].agg(["count", "mean"]).reset_index()
            for _, r in g.iterrows():
                n = int(r["count"])
                if n < min_n:
                    continue
                val = r[feat]
                p = float(r["mean"])
                lift_pp = (p - baseline) * 100.0
                brain[side]["table"][(feat, val)] = {"n": n, "p": p, "lift_pp": lift_pp}

    return brain

def force_score(row: pd.Series, brain: dict, side: str, k_shrink: float = 50.0) -> tuple[float, list]:
    base = brain.get(side, {}).get("baseline", np.nan)
    table = brain.get(side, {}).get("table", {})
    if not np.isfinite(base):
        return np.nan, []

    lifts = []
    weights = []
    breakdown = []

    for feat in FEATURES:
        key = (feat, row.get(feat))
        if key not in table:
            continue
        n = table[key]["n"]
        lift_pp = table[key]["lift_pp"]
        w = n / (n + k_shrink)
        lifts.append(lift_pp)
        weights.append(w)
        breakdown.append({"feature": feat, "value": row.get(feat), "lift_pp": lift_pp, "n": n, "weight": w})

    if not weights:
        return float(base), []

    total_lift_pp = float(np.dot(weights, lifts) / np.sum(weights))
    score = float(base + total_lift_pp / 100.0)
    score = float(max(0.0, min(1.0, score)))
    breakdown = sorted(breakdown, key=lambda x: abs(x["lift_pp"]), reverse=True)
    return score, breakdown

# -------------------- Chart helpers --------------------
def add_sign_zone(fig, df: pd.DataFrame, feat: str, sign: str):
    # Shade contiguous ranges where df[feat] == sign
    if feat not in df.columns:
        return
    z = (df[feat] == sign).fillna(False).values
    idx = np.where(z)[0]
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

def plot_price_with_overlays(df: pd.DataFrame, hit_dates: list, zone_feat: str | None, zone_sign: str | None, title: str):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["Date"].astype(str),
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ))

    fig.add_trace(go.Scatter(x=df["Date"].astype(str), y=df["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df["Date"].astype(str), y=df["UpperThreshold"], mode="lines", name="UpperThreshold"))
    fig.add_trace(go.Scatter(x=df["Date"].astype(str), y=df["LowerThreshold"], mode="lines", name="LowerThreshold"))

    # Optional shaded sign zone (contiguous)
    if zone_feat and zone_sign:
        add_sign_zone(fig, df, zone_feat, zone_sign)

    # Vertical lines for combo hits
    for d in hit_dates:
        fig.add_vline(x=str(d), opacity=0.30, line_width=1)

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        height=680,
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------- UI --------------------
st.title("Planetary Force Dashboard — Level-1 Events (Fast + Slow Planets)")
st.caption("Fast planets: top/bottom episodes (Level-1 overextensions). Slow planets: included as trend-setter context within the same force model.")

with st.sidebar:
    st.header("Asset")
    asset_label = st.selectbox("Preset", list(ASSET_PRESETS.keys()), index=0)
    symbol_default = ASSET_PRESETS[asset_label]
    symbol = st.text_input("Yahoo symbol", value=symbol_default).strip() or symbol_default

    st.header("Event settings")
    period = st.selectbox("History window", ["10y","5y","2y","max"], index=0)
    interval = "1d"

    bb_window = st.slider("BB Window", 10, 40, 20)
    bb_k = st.slider("BB K", 1.0, 3.0, 2.0, 0.1)
    margin_sigma = st.slider("Margin Sigma", 0.0, 1.0, 0.25, 0.05)
    lookahead_days = st.slider("Lookahead days", 5, 30, 15)
    stop_out_pct = st.slider("Stop-out %", 0.5, 5.0, 2.0, 0.1) / 100.0
    side = st.radio("Side", ["U", "D"], horizontal=True)

    st.header("Robustness")
    min_n = st.slider("Min sample size (n)", 5, 100, 15)
    k_shrink = st.slider("Shrinkage k", 10, 200, 50)

    st.header("Combo (placements)")
    st.caption("Leave empty = no filter (just show force + baseline).")

    moon_sign = st.multiselect("Moon sign", SIGNS, default=[])
    merc_sign = st.multiselect("Mercury sign", SIGNS, default=[])
    venus_sign = st.multiselect("Venus sign", SIGNS, default=[])
    mars_sign = st.multiselect("Mars sign", SIGNS, default=[])
    jup_sign = st.multiselect("Jupiter sign", SIGNS, default=[])
    sat_sign = st.multiselect("Saturn sign", SIGNS, default=[])

    phase = st.multiselect("Phase", ["Waxing", "Waning"], default=[])
    moon_node = st.multiselect("Moon-Node", ["Conjunct", "None"], default=[])

    merc_retro = st.selectbox("Mercury retro", ["Any", "True", "False"], index=0)
    jup_retro = st.selectbox("Jupiter retro", ["Any", "True", "False"], index=0)
    sat_retro = st.selectbox("Saturn retro", ["Any", "True", "False"], index=0)

    st.header("Chart overlay")
    zone_feat = st.selectbox(
        "Shade planet-in-sign zone",
        ["None", "Mars_sign", "Mercury_sign", "Venus_sign", "Moon_sign", "Jupiter_sign", "Saturn_sign", "Rahu_sign", "Ketu_sign"],
        index=0
    )
    zone_sign = None
    if zone_feat != "None":
        zone_sign = st.selectbox("Zone sign", SIGNS, index=0)

    st.header("Future scan (minimal)")
    future_days = st.slider("Scan next N days", 7, 90, 30)
    top_k = st.slider("Top K dates", 3, 20, 5)

# -------------------- Build pipeline --------------------
df_px = fetch_price(symbol, period, interval)
if df_px.empty:
    st.error("No price data returned for this symbol/period. Try a different symbol or shorter history window.")
    st.stop()

df_all, level1 = compute_tech_labels(df_px, bb_window, bb_k, margin_sigma, lookahead_days, stop_out_pct)

df_astro = compute_astro(df_all["Date"].tolist())
# Merge (KeyError-safe due to Date normalization + compute_astro always returning Date)
df_all = df_all.merge(df_astro, on="Date", how="left")
level1_all = level1.merge(df_astro, on="Date", how="left")

# Brain train split (same as your earlier approach)
split_date = datetime(2022, 1, 1).date()
train = level1_all[level1_all["Date"] < split_date].copy()
brain = build_brain(train, min_n=min_n)

# -------------------- Combo dict --------------------
combo = {}
if moon_sign: combo["Moon_sign"] = moon_sign
if merc_sign: combo["Mercury_sign"] = merc_sign
if venus_sign: combo["Venus_sign"] = venus_sign
if mars_sign: combo["Mars_sign"] = mars_sign
if jup_sign: combo["Jupiter_sign"] = jup_sign
if sat_sign: combo["Saturn_sign"] = sat_sign
if phase: combo["Phase"] = phase
if moon_node: combo["Moon_Node"] = moon_node

if merc_retro != "Any":
    combo["Mercury_retro"] = (merc_retro == "True")
if jup_retro != "Any":
    combo["Jupiter_retro"] = (jup_retro == "True")
if sat_retro != "Any":
    combo["Saturn_retro"] = (sat_retro == "True")

# -------------------- Metrics --------------------
today = df_all.iloc[-1].copy()
today_force, today_breakdown = force_score(today, brain, side=side, k_shrink=k_shrink)

level1_side = level1_all[(level1_all["ext_dir"] == side) & pd.notna(level1_all["y_revert"])].copy()
baseline = brain[side]["baseline"]

occ = pd.DataFrame()
if combo:
    sub = level1_side.copy()
    m = match_combo(sub, combo)
    occ = sub[m].copy()

combo_n = int(len(occ)) if combo else 0
combo_p = float(occ["y_revert"].mean()) if len(occ) else np.nan

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Asset", symbol)
c2.metric("Side", side)
c3.metric("Baseline P(revert)", f"{baseline*100:.2f}%" if np.isfinite(baseline) else "n/a")
c4.metric("Today's Force", f"{today_force*100:.2f}%" if np.isfinite(today_force) else "n/a")
c5.metric("Combo n / P(revert)", f"{combo_n} / {combo_p*100:.1f}%" if np.isfinite(combo_p) else f"{combo_n} / n/a")

# -------------------- Chart: vlines for combo hits, optional sign zone shading --------------------
hit_dates = []
if len(occ):
    hit_dates = occ["Date"].tolist()

zone_feat_resolved = None if zone_feat == "None" else zone_feat
plot_price_with_overlays(
    df_all,
    hit_dates=hit_dates,
    zone_feat=zone_feat_resolved,
    zone_sign=zone_sign if zone_feat_resolved else None,
    title=f"{symbol} — Candles + MA20/Thresholds + Combo Hit Lines ({side})",
)

# -------------------- Breakdown + Occurrence table --------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Today's force breakdown")
    if not today_breakdown:
        st.info("No contributing placements met min_n for today. Force = baseline.")
    else:
        bd = pd.DataFrame(today_breakdown)
        bd["lift_pp"] = bd["lift_pp"].map(lambda x: round(float(x), 2))
        bd["weight"] = bd["weight"].map(lambda x: round(float(x), 3))
        st.dataframe(bd, use_container_width=True, height=320)

with right:
    st.subheader("Combo definition")
    if combo:
        st.json(combo)
    else:
        st.info("No combo selected. Add placements in the sidebar to highlight historical hits.")

st.subheader("Historical occurrences (Level-1 event starts)")
if len(occ) == 0:
    st.info("No historical matches for the selected combo (with current robustness threshold).")
else:
    px = df_all.set_index("Date")

    def fwd_ret(d, h):
        if d not in px.index:
            return np.nan
        i = px.index.get_loc(d)
        j = i + h
        if j >= len(px):
            return np.nan
        c0 = float(px.iloc[i]["Close"])
        c1 = float(px.iloc[j]["Close"])
        return (c1 / c0 - 1.0) * 100.0

    occ = occ.sort_values("Date", ascending=False).copy()
    for h in [1, 3, 5, 10]:
        occ[f"ret_{h}d_pct"] = occ["Date"].apply(lambda d: fwd_ret(d, h))

    # Show key columns + whichever features exist
    cols_show = ["Date", "ext_dir", "y_revert"] + [c for c in FEATURES if c in occ.columns] + ["ret_1d_pct","ret_3d_pct","ret_5d_pct","ret_10d_pct"]
    view = occ[cols_show].copy()
    for c in ["ret_1d_pct","ret_3d_pct","ret_5d_pct","ret_10d_pct"]:
        view[c] = view[c].map(lambda x: round(float(x), 3) if np.isfinite(x) else np.nan)

    st.dataframe(view, use_container_width=True, height=420)

# -------------------- Minimal future scan (top K + breakdown) --------------------
st.subheader("Future force (top dates)")
start = datetime.now(tz=UTC).date()
future_dates = [start + timedelta(days=i) for i in range(1, future_days + 1)]
df_future = compute_astro(future_dates)

if df_future.empty or not np.isfinite(baseline):
    st.info("Future scan unavailable (missing baseline or astro data).")
else:
    scores = []
    breakdown_map = {}
    for _, r in df_future.iterrows():
        p, bd = force_score(r, brain, side=side, k_shrink=k_shrink)
        scores.append(p)
        breakdown_map[r["Date"]] = bd

    df_future = df_future.copy()
    df_future["pred_prob"] = scores
    top = df_future.sort_values("pred_prob", ascending=False).head(top_k).copy()

    st.dataframe(
        top[["Date","pred_prob","Moon_sign","Mercury_sign","Venus_sign","Mars_sign","Jupiter_sign","Saturn_sign","Phase","Mercury_retro","Moon_Node"]],
        use_container_width=True
    )

    if len(top):
        chosen = st.selectbox("Inspect breakdown for date", top["Date"].astype(str).tolist(), index=0)
        chosen_date = datetime.strptime(chosen, "%Y-%m-%d").date()
        bd = breakdown_map.get(chosen_date, [])
        if not bd:
            st.info("No contributing placements met min_n for this date. Score = baseline.")
        else:
            bdf = pd.DataFrame(bd)
            bdf["lift_pp"] = bdf["lift_pp"].map(lambda x: round(float(x), 2))
            bdf["weight"] = bdf["weight"].map(lambda x: round(float(x), 3))
            st.dataframe(bdf, use_container_width=True, height=300)
