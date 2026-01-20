import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
import swisseph as swe
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# -------------------- Constants --------------------
SIGNS = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

FAST_FEATURES = [
    "Moon_sign", "Mercury_sign", "Venus_sign", "Mars_sign",
    "Phase", "Mercury_retro", "Moon_Node", "Rahu_sign", "Ketu_sign"
]

ASSET_PRESETS = {
    "Nifty 50 (^NSEI)": "^NSEI",
    "Silver (SI=F)": "SI=F",
    "Gold (GC=F)": "GC=F",
    "Crude (CL=F)": "CL=F",
}

# Swiss Ephemeris: Lahiri sidereal
swe.set_sid_mode(swe.SIDM_LAHIRI)

# -------------------- Helpers --------------------
def sign_from_lon(lon_deg: float) -> str:
    return SIGNS[int((lon_deg % 360) // 30)]

def julday_from_date_fast(d, hour_ist=10):
    dt_local = datetime(d.year, d.month, d.day, hour_ist, 0, 0, tzinfo=IST)
    dt_utc = dt_local.astimezone(UTC)
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour + dt_utc.minute/60.0 + dt_utc.second/3600.0)

def sidereal_lon_speed(jd_ut: float, body: int):
    flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL
    xx, _ = swe.calc_ut(jd_ut, body, flags)
    lon = float(xx[0]) % 360.0
    speed = float(xx[3])
    return lon, speed

@st.cache_data(show_spinner=False)
def fetch_price(symbol: str, period: str, interval: str) -> pd.DataFrame:
    raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
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
        ref_low  = float(df.loc[i, "Low"])

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

    # EMA200 trend (for context charting / regime)
    df["ema200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["trend"] = np.where(df["Close"] >= df["ema200"], "UP", "DOWN")
    df["is_uptrend"] = (df["trend"] == "UP").astype(int)
    df["is_downtrend"] = (df["trend"] == "DOWN").astype(int)

    level1 = df[df["event_start"] & df["ext_dir"].isin(["U","D"])].copy()
    return df, level1

@st.cache_data(show_spinner=False)
def compute_astro(dates: list) -> pd.DataFrame:
    rows = []
    for d in dates:
        jd = julday_from_date_fast(d, hour_ist=10)

        moon_lon, _ = sidereal_lon_speed(jd, swe.MOON)
        merc_lon, merc_sp = sidereal_lon_speed(jd, swe.MERCURY)
        ven_lon, _ = sidereal_lon_speed(jd, swe.VENUS)
        mars_lon, _ = sidereal_lon_speed(jd, swe.MARS)
        sun_lon, _ = sidereal_lon_speed(jd, swe.SUN)

        rahu_lon, _ = sidereal_lon_speed(jd, swe.MEAN_NODE)
        ketu_lon = (rahu_lon + 180.0) % 360.0

        phase_angle = (moon_lon - sun_lon) % 360.0
        phase = "Waxing" if phase_angle < 180.0 else "Waning"

        d_moon_rahu = abs((moon_lon - rahu_lon + 180.0) % 360.0 - 180.0)
        moon_node = "Conjunct" if (d_moon_rahu <= 15.0 or abs(d_moon_rahu - 180.0) <= 15.0) else "None"

        rows.append({
            "Date": d,
            "Moon_sign": sign_from_lon(moon_lon),
            "Mercury_sign": sign_from_lon(merc_lon),
            "Venus_sign": sign_from_lon(ven_lon),
            "Mars_sign": sign_from_lon(mars_lon),
            "Mercury_retro": bool(merc_sp < 0),
            "Phase": phase,
            "Rahu_sign": sign_from_lon(rahu_lon),
            "Ketu_sign": sign_from_lon(ketu_lon),
            "Moon_Node": moon_node,
        })
    return pd.DataFrame(rows)

def build_brain(df_events: pd.DataFrame, min_n: int = 15) -> dict:
    brain = {"min_n": min_n, "U": {"baseline": None, "table": {}}, "D": {"baseline": None, "table": {}}}
    for side in ["U","D"]:
        sub = df_events[(df_events["ext_dir"] == side) & pd.notna(df_events["y_revert"])].copy()
        if len(sub) == 0:
            brain[side]["baseline"] = np.nan
            continue
        baseline = float(sub["y_revert"].mean())
        brain[side]["baseline"] = baseline
        for feat in FAST_FEATURES:
            g = sub.groupby(feat)["y_revert"].agg(["count","mean"]).reset_index()
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
    """
    Returns (score_prob, breakdown list of dicts).
    Breakdown includes each contributing feature with (lift_pp, n, weight).
    """
    base = brain.get(side, {}).get("baseline", np.nan)
    table = brain.get(side, {}).get("table", {})
    if not np.isfinite(base):
        return np.nan, []

    lifts = []
    weights = []
    breakdown = []

    for feat in FAST_FEATURES:
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
    return score, sorted(breakdown, key=lambda x: abs(x["lift_pp"]), reverse=True)

def match_combo(df: pd.DataFrame, combo: dict) -> pd.Series:
    """
    combo = {feature: value} where value can be:
      - a scalar (exact match)
      - list/tuple/set of allowed values
      - True/False for booleans
    """
    m = pd.Series(True, index=df.index)
    for feat, val in combo.items():
        if isinstance(val, (list, tuple, set)):
            m &= df[feat].isin(list(val))
        else:
            m &= (df[feat] == val)
    return m

def plot_chart(df: pd.DataFrame, hits: pd.Series, title: str, shade_feat: str | None = None, shade_value: str | None = None):
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["Date"].astype(str),
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ))

    # Bollinger-ish overlays used in your event logic
    fig.add_trace(go.Scatter(x=df["Date"].astype(str), y=df["MA20"], mode="lines", name="MA20"))
    fig.add_trace(go.Scatter(x=df["Date"].astype(str), y=df["UpperThreshold"], mode="lines", name="UpperThreshold"))
    fig.add_trace(go.Scatter(x=df["Date"].astype(str), y=df["LowerThreshold"], mode="lines", name="LowerThreshold"))

    # Highlight combo hit days with markers
    hit_dates = df.loc[hits, "Date"].astype(str).tolist()
    hit_prices = df.loc[hits, "Close"].tolist()
    fig.add_trace(go.Scatter(
        x=hit_dates, y=hit_prices,
        mode="markers", name="Combo hits",
        marker=dict(size=8, symbol="circle")
    ))

    # Optional shaded zones for "planet in sign"
    if shade_feat and shade_value and shade_feat in df.columns:
        z = (df[shade_feat] == shade_value)
        # draw vertical rectangles for contiguous stretches
        idx = np.where(z.values)[0]
        if len(idx) > 0:
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
                fig.add_vrect(x0=x0, x1=x1, opacity=0.15, line_width=0)

    fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Price", height=650)
    st.plotly_chart(fig, use_container_width=True)

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="Planetary Force Dashboard", layout="wide")
st.title("Planetary Force Dashboard (Level-1 Tops/Bottoms + Force Model)")

with st.sidebar:
    st.header("Controls")

    asset_label = st.selectbox("Asset", list(ASSET_PRESETS.keys()), index=0)
    symbol = ASSET_PRESETS[asset_label]
    symbol = st.text_input("Or type any Yahoo symbol", value=symbol).strip() or symbol

    period = st.selectbox("History window", ["max","10y","5y","2y"], index=0)
    interval = st.selectbox("Interval", ["1d"], index=0)

    st.subheader("Event logic")
    bb_window = st.slider("BB Window", 10, 40, 20)
    bb_k = st.slider("BB K", 1.0, 3.0, 2.0, 0.1)
    margin_sigma = st.slider("Margin Sigma", 0.0, 1.0, 0.25, 0.05)
    lookahead_days = st.slider("Lookahead days", 5, 30, 15)
    stop_out_pct = st.slider("Stop-out %", 0.5, 5.0, 2.0, 0.1) / 100.0

    st.subheader("Statistical robustness")
    min_n = st.slider("Min sample size (n)", 5, 100, 15)
    k_shrink = st.slider("Shrinkage k (bigger = more conservative)", 10, 200, 50)

    st.subheader("Combo builder")
    side = st.radio("Event side", ["U","D"], horizontal=True)

    # Multi-select placements (simple, direct)
    moon_sign = st.multiselect("Moon sign", SIGNS, default=[])
    mars_sign = st.multiselect("Mars sign", SIGNS, default=[])
    merc_sign = st.multiselect("Mercury sign", SIGNS, default=[])
    venus_sign = st.multiselect("Venus sign", SIGNS, default=[])
    phase = st.multiselect("Phase", ["Waxing","Waning"], default=[])
    merc_retro = st.selectbox("Mercury retro", ["Any", "True", "False"], index=0)
    moon_node = st.multiselect("Moon-Node", ["Conjunct","None"], default=[])

    st.subheader("Future scan")
    future_days = st.slider("Scan next N days", 7, 90, 30)
    top_k = st.slider("Top K future dates", 3, 20, 5)

# -------------------- Build data --------------------
df_px = fetch_price(symbol, period, interval)
df_all, level1 = compute_tech_labels(df_px, bb_window, bb_k, margin_sigma, lookahead_days, stop_out_pct)

df_astro = compute_astro(df_all["Date"].tolist())
df_all = df_all.merge(df_astro, on="Date", how="left")
level1_all = level1.merge(df_astro, on="Date", how="left")

# Train/test split for "brain"
split_date = datetime(2022, 1, 1).date()
df_train = level1_all[level1_all["Date"] < split_date].copy()
df_test  = level1_all[level1_all["Date"] >= split_date].copy()

brain = build_brain(df_train, min_n=min_n)

# -------------------- Combo dict from sidebar --------------------
combo = {}
if moon_sign: combo["Moon_sign"] = moon_sign
if mars_sign: combo["Mars_sign"] = mars_sign
if merc_sign: combo["Mercury_sign"] = merc_sign
if venus_sign: combo["Venus_sign"] = venus_sign
if phase: combo["Phase"] = phase
if moon_node: combo["Moon_Node"] = moon_node
if merc_retro != "Any":
    combo["Mercury_retro"] = (merc_retro == "True")

# -------------------- KPI: today's force (based on today's astro, not necessarily an event day) --------------------
today_row = df_all.iloc[-1].copy()
today_force, today_breakdown = force_score(today_row, brain, side=side, k_shrink=k_shrink)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Asset", symbol)
c2.metric("Brain baseline P(revert)", f"{brain[side]['baseline']*100:.2f}%" if np.isfinite(brain[side]["baseline"]) else "n/a")
c3.metric("Today's Total Force", f"{today_force*100:.2f}%" if np.isfinite(today_force) else "n/a")
c4.metric("Level-1 events (train)", f"{len(df_train)}")

# -------------------- Tabs --------------------
tab1, tab2, tab3 = st.tabs(["Historical Combo Backtest", "Blind Test Force Bins", "Future Force Scanner"])

with tab1:
    st.subheader("Historical combo occurrences on Level-1 event days")
    if combo:
        hits = match_combo(level1_all[level1_all["ext_dir"] == side], combo)
        occ = level1_all[level1_all["ext_dir"] == side].loc[hits].copy()
    else:
        hits = pd.Series(False, index=level1_all.index)
        occ = pd.DataFrame()

    st.write("Combo:", combo if combo else "(none selected)")

    # Chart: show combo hits on full price chart; optional shading for one selected sign-zone
    shade_feat = None
    shade_value = None
    # If user picked exactly one Mars sign, shade Mars zone as an example
    if "Mars_sign" in combo and isinstance(combo["Mars_sign"], list) and len(combo["Mars_sign"]) == 1:
        shade_feat, shade_value = "Mars_sign", combo["Mars_sign"][0]

    # Build hit series aligned to df_all dates (mark hit dates on the price chart)
    hit_dates_set = set(occ["Date"].tolist()) if len(occ) else set()
    hits_on_all = df_all["Date"].apply(lambda d: d in hit_dates_set)

    plot_chart(
        df_all,
        hits_on_all,
        title=f"{symbol} — Candles + MA20/Thresholds + Combo Hits ({side})",
        shade_feat=shade_feat,
        shade_value=shade_value,
    )

    if len(occ) == 0:
        st.info("No historical matches for the selected combo on Level-1 event days.")
    else:
        # Add realized forward return columns (optional but useful in your “nature of move” inspection)
        # Uses Close-to-Close returns from event day.
        for h in [1, 3, 5, 10]:
            occ[f"ret_{h}d_pct"] = (
                level1_all.set_index("Date").loc[occ["Date"], "Close"].values
            )  # placeholder, replaced below

        # compute properly using df_all indexed by Date
        px_idx = df_all.set_index("Date")
        def fwd_ret(d, h):
            if d not in px_idx.index:
                return np.nan
            i = px_idx.index.get_loc(d)
            j = i + h
            if j >= len(px_idx):
                return np.nan
            c0 = float(px_idx.iloc[i]["Close"])
            c1 = float(px_idx.iloc[j]["Close"])
            return (c1 / c0 - 1.0) * 100.0

        for h in [1, 3, 5, 10]:
            occ[f"ret_{h}d_pct"] = occ["Date"].apply(lambda d: fwd_ret(d, h))

        occ_view = occ[["Date","ext_dir","y_revert","Moon_sign","Mercury_sign","Venus_sign","Mars_sign","Phase","Mercury_retro","Moon_Node",
                        "ret_1d_pct","ret_3d_pct","ret_5d_pct","ret_10d_pct"]].sort_values("Date", ascending=False)

        st.dataframe(occ_view, use_container_width=True)

        st.write("Occurrence stats:")
        st.write({
            "n": int(len(occ)),
            "P(revert)": float(occ["y_revert"].mean()),
            "avg_5d_return_%": float(np.nanmean(occ["ret_5d_pct"])),
        })

with tab2:
    st.subheader("Blind test: predicted force bins vs actual reversal rate (Level-1 events)")
    if len(df_test) == 0:
        st.info("No test data in the selected window.")
    else:
        # score each test event
        sub = df_test[df_test["ext_dir"] == side].copy()
        if len(sub) == 0:
            st.info("No test events for selected side.")
        else:
            sub["pred_prob"] = sub.apply(lambda r: force_score(r, brain, side=side, k_shrink=k_shrink)[0], axis=1)
            sub["prob_bin"] = pd.qcut(sub["pred_prob"], q=4, labels=["Low","Mid","High","Extreme"])
            res = sub.groupby("prob_bin")["y_revert"].agg(["mean","count"]).reset_index()

            fig = go.Figure()
            fig.add_bar(x=res["prob_bin"].astype(str), y=res["mean"])
            fig.add_hline(y=float(sub["y_revert"].mean()), line_dash="dash", annotation_text="Baseline")
            fig.update_layout(
                title=f"Blind Test — Actual reversal rate vs predicted force ({symbol}, side={side})",
                xaxis_title="Predicted force bin",
                yaxis_title="Actual P(revert)"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(res, use_container_width=True)

with tab3:
    st.subheader("Future Force Scanner")
    # build future astro rows (no price needed)
    start = datetime.now(tz=UTC).date()
    future_dates = [start + timedelta(days=i) for i in range(1, future_days + 1)]
    df_future = compute_astro(future_dates)

    # score each future date using today's "side" brain
    scores = []
    breakdown_map = {}
    for _, r in df_future.iterrows():
        p, bd = force_score(r, brain, side=side, k_shrink=k_shrink)
        scores.append(p)
        breakdown_map[r["Date"]] = bd

    df_future["pred_prob"] = scores
    top = df_future.sort_values("pred_prob", ascending=False).head(top_k).copy()

    st.write(f"Top {top_k} dates in next {future_days} days (side={side}):")
    st.dataframe(top[["Date","pred_prob","Moon_sign","Mercury_sign","Venus_sign","Mars_sign","Phase","Mercury_retro","Moon_Node"]], use_container_width=True)

    if len(top) > 0:
        chosen = st.selectbox("Inspect breakdown for date", top["Date"].astype(str).tolist(), index=0)
        chosen_date = datetime.strptime(chosen, "%Y-%m-%d").date()
        bd = breakdown_map.get(chosen_date, [])
        if not bd:
            st.info("No contributing signals met min_n for this date; score is baseline.")
        else:
            st.write("Force breakdown (largest absolute lifts first):")
            st.dataframe(pd.DataFrame(bd), use_container_width=True)
