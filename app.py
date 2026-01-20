import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import swisseph as swe
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# -------------------- Global config --------------------
st.set_page_config(page_title="Planetary Force Analytics", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better UI
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)

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
    "NASDAQ (^IXIC)": "^IXIC",
}

swe.set_sid_mode(swe.SIDM_LAHIRI)

FEATURES = [
    "Moon_sign", "Mercury_sign", "Venus_sign", "Mars_sign",
    "Phase", "Mercury_retro", "Moon_Node", "Rahu_sign", "Ketu_sign",
    "Jupiter_sign", "Saturn_sign", "Jupiter_retro", "Saturn_retro",
]

# -------------------- Utility helpers --------------------
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

def compute_tech_labels(df_px: pd.DataFrame, bb_window: int, bb_k: float, margin_sigma: float, lookahead_days: int, stop_out_pct: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df_px.copy()
    df["TP"] = (df["High"] + df["Low"]) / 2.0
    df["MA20"] = df["TP"].rolling(bb_window).mean()
    df["STD20"] = df["TP"].rolling(bb_window).std()
    df["UpperThreshold"] = df["MA20"] + (bb_k - margin_sigma) * df["STD20"]
    df["LowerThreshold"] = df["MA20"] - (bb_k - margin_sigma) * df["STD20"]
    df["ext_dir"] = np.where(df["TP"] >= df["UpperThreshold"], "U", np.where(df["TP"] <= df["LowerThreshold"], "D", "N"))
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
    df["ema200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["trend"] = np.where(df["Close"] >= df["ema200"], "UP", "DOWN")
    level1 = df[df["event_start"] & df["ext_dir"].isin(["U", "D"])].copy()
    return df, level1

@st.cache_data(show_spinner=False)
def compute_astro(dates: list) -> pd.DataFrame:
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

# -------------------- Enhanced Chart Functions --------------------
def create_main_chart(df: pd.DataFrame, hit_dates: list, side: str, symbol: str):
    """Enhanced multi-panel chart with price, volume, force score, and events"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxis=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.20, 0.15],
        specs=[[{"type": "candlestick"}],
               [{"type": "bar"}],
               [{"type": "scatter"}],
               [{"type": "scatter"}]]
    )

    # Panel 1: Candlestick with BB
    fig.add_trace(go.Candlestick(
        x=df["Date"].astype(str),
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["Date"].astype(str), y=df["MA20"],
        mode="lines", name="MA20",
        line=dict(color='#ffa726', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["Date"].astype(str), y=df["UpperThreshold"],
        mode="lines", name="Upper",
        line=dict(color='rgba(255,0,0,0.3)', dash='dash'),
        fill=None
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["Date"].astype(str), y=df["LowerThreshold"],
        mode="lines", name="Lower",
        line=dict(color='rgba(0,255,0,0.3)', dash='dash'),
        fill='tonexty',
        fillcolor='rgba(100,100,100,0.1)'
    ), row=1, col=1)

    # Panel 2: Volume
    colors = ['#ef5350' if df.loc[i, 'Close'] < df.loc[i, 'Open'] else '#26a69a' for i in df.index]
    fig.add_trace(go.Bar(
        x=df["Date"].astype(str), y=df["Volume"],
        name="Volume",
        marker_color=colors,
        showlegend=False
    ), row=2, col=1)

    # Panel 3: Force Score Timeline (if available)
    if 'force_score' in df.columns:
        fig.add_trace(go.Scatter(
            x=df["Date"].astype(str), y=df["force_score"],
            mode="lines+markers",
            name="Force Score",
            line=dict(color='#667eea', width=2),
            marker=dict(size=4)
        ), row=3, col=1)
        
        # Add baseline reference
        if 'baseline' in df.columns:
            fig.add_trace(go.Scatter(
                x=df["Date"].astype(str), y=df["baseline"],
                mode="lines",
                name="Baseline",
                line=dict(color='gray', dash='dot', width=1)
            ), row=3, col=1)

    # Panel 4: Event markers
    event_df = df[df["event_start"] == True].copy()
    if len(event_df):
        up_events = event_df[event_df["ext_dir"] == "U"]
        down_events = event_df[event_df["ext_dir"] == "D"]
        
        fig.add_trace(go.Scatter(
            x=up_events["Date"].astype(str),
            y=[1] * len(up_events),
            mode="markers",
            name="Upper Events",
            marker=dict(symbol='triangle-up', size=10, color='red')
        ), row=4, col=1)
        
        fig.add_trace(go.Scatter(
            x=down_events["Date"].astype(str),
            y=[0] * len(down_events),
            mode="markers",
            name="Lower Events",
            marker=dict(symbol='triangle-down', size=10, color='green')
        ), row=4, col=1)

    # Add combo hit vertical lines
    for d in hit_dates:
        fig.add_vline(x=str(d), line_width=2, line_dash="dash", 
                     line_color="rgba(102, 126, 234, 0.5)")

    fig.update_layout(
        title=dict(text=f"{symbol} - Planetary Force Analysis ({side} Side)", font=dict(size=20)),
        height=1000,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white'
    )
    
    # Add subplot titles as annotations
    fig.add_annotation(text="Price & Bollinger Bands", xref="paper", yref="paper",
                      x=0.5, y=0.98, showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(text="Volume", xref="paper", yref="paper",
                      x=0.5, y=0.63, showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(text="Force Score Timeline", xref="paper", yref="paper",
                      x=0.5, y=0.38, showarrow=False, font=dict(size=12, color="gray"))
    fig.add_annotation(text="Event Distribution", xref="paper", yref="paper",
                      x=0.5, y=0.13, showarrow=False, font=dict(size=12, color="gray"))

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

    return fig

def create_force_breakdown_chart(breakdown: list):
    """Horizontal bar chart for force breakdown"""
    if not breakdown:
        return None
    
    df = pd.DataFrame(breakdown)
    df = df.sort_values('lift_pp')
    
    colors = ['#ef5350' if x < 0 else '#26a69a' for x in df['lift_pp']]
    
    fig = go.Figure(go.Bar(
        x=df['lift_pp'],
        y=[f"{row['feature']}: {row['value']}" for _, row in df.iterrows()],
        orientation='h',
        marker_color=colors,
        text=df['lift_pp'].round(2),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Lift: %{x:.2f}pp<br>Weight: %{customdata[0]:.3f}<br>Samples: %{customdata[1]}<extra></extra>',
        customdata=df[['weight', 'n']].values
    ))
    
    fig.update_layout(
        title="Force Components (Lift %)",
        xaxis_title="Lift (percentage points)",
        yaxis_title="",
        height=max(400, len(breakdown) * 35),
        showlegend=False,
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white'
    )
    
    fig.add_vline(x=0, line_width=2, line_color='black')
    
    return fig

def create_performance_heatmap(occ: pd.DataFrame):
    """Heatmap of returns by planetary placement"""
    if len(occ) == 0:
        return None
    
    # Create pivot for Moon sign vs returns
    ret_cols = [c for c in occ.columns if c.startswith('ret_') and c.endswith('_pct')]
    if not ret_cols or 'Moon_sign' not in occ.columns:
        return None
    
    pivot = occ.groupby('Moon_sign')[ret_cols].mean()
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[c.replace('ret_', '').replace('_pct', '') for c in pivot.columns],
        y=pivot.index,
        colorscale='RdYlGn',
        zmid=0,
        text=np.round(pivot.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Return %")
    ))
    
    fig.update_layout(
        title="Average Returns by Moon Sign",
        xaxis_title="Holding Period",
        yaxis_title="Moon Sign",
        height=400,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig

# -------------------- UI --------------------
st.title("🌟 Planetary Force Analytics Dashboard")
st.markdown("Advanced astrological market analysis with enhanced visualizations")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    with st.expander("📊 Asset Selection", expanded=True):
        asset_label = st.selectbox("Preset", list(ASSET_PRESETS.keys()), index=0)
        symbol_default = ASSET_PRESETS[asset_label]
        symbol = st.text_input("Yahoo symbol", value=symbol_default).strip() or symbol_default
        period = st.selectbox("History", ["10y","5y","2y","1y"], index=0)

    with st.expander("🎯 Event Parameters", expanded=True):
        bb_window = st.slider("BB Window", 10, 40, 20)
        bb_k = st.slider("BB K", 1.0, 3.0, 2.0, 0.1)
        margin_sigma = st.slider("Margin Sigma", 0.0, 1.0, 0.25, 0.05)
        lookahead_days = st.slider("Lookahead days", 5, 30, 15)
        stop_out_pct = st.slider("Stop-out %", 0.5, 5.0, 2.0, 0.1) / 100.0
        side = st.radio("Side", ["U", "D"], horizontal=True, help="U=Upper, D=Lower")

    with st.expander("🧠 Model Settings"):
        min_n = st.slider("Min sample size", 5, 100, 15)
        k_shrink = st.slider("Shrinkage k", 10, 200, 50)

    with st.expander("🌙 Planetary Filters"):
        st.caption("Select placements to filter historical events")
        
        col1, col2 = st.columns(2)
        with col1:
            moon_sign = st.multiselect("Moon", SIGNS, default=[])
            venus_sign = st.multiselect("Venus", SIGNS, default=[])
            mars_sign = st.multiselect("Mars", SIGNS, default=[])
        with col2:
            merc_sign = st.multiselect("Mercury", SIGNS, default=[])
            jup_sign = st.multiselect("Jupiter", SIGNS, default=[])
            sat_sign = st.multiselect("Saturn", SIGNS, default=[])
        
        phase = st.multiselect("Lunar Phase", ["Waxing", "Waning"], default=[])
        moon_node = st.multiselect("Moon-Node", ["Conjunct", "None"], default=[])
        
        merc_retro = st.selectbox("Mercury Rx", ["Any", "True", "False"], index=0)
        jup_retro = st.selectbox("Jupiter Rx", ["Any", "True", "False"], index=0)
        sat_retro = st.selectbox("Saturn Rx", ["Any", "True", "False"], index=0)

    with st.expander("🔮 Future Scan"):
        future_days = st.slider("Scan days ahead", 7, 90, 30)
        top_k = st.slider("Show top dates", 3, 20, 5)

# -------------------- Build pipeline --------------------
with st.spinner("Loading data..."):
    df_px = fetch_price(symbol, period, "1d")
    
if df_px.empty:
    st.error("❌ No price data available. Try a different symbol or period.")
    st.stop()

with st.spinner("Computing technical indicators..."):
    df_all, level1 = compute_tech_labels(df_px, bb_window, bb_k, margin_sigma, lookahead_days, stop_out_pct)

with st.spinner("Computing planetary positions..."):
    df_astro = compute_astro(df_all["Date"].tolist())
    df_all = df_all.merge(df_astro, on="Date", how="left")
    level1_all = level1.merge(df_astro, on="Date", how="left")

# Train brain
split_date = datetime(2022, 1, 1).date()
train = level1_all[level1_all["Date"] < split_date].copy()
brain = build_brain(train, min_n=min_n)

# Build combo filter
combo = {}
if moon_sign: combo["Moon_sign"] = moon_sign
if merc_sign: combo["Mercury_sign"] = merc_sign
if venus_sign: combo["Venus_sign"] = venus_sign
if mars_sign: combo["Mars_sign"] = mars_sign
if jup_sign: combo["Jupiter_sign"] = jup_sign
if sat_sign: combo["Saturn_sign"] = sat_sign
if phase: combo["Phase"] = phase
if moon_node: combo["Moon_Node"] = moon_node
if merc_retro != "Any": combo["Mercury_retro"] = (merc_retro == "True")
if jup_retro != "Any": combo["Jupiter_retro"] = (jup_retro == "True")
if sat_retro != "Any": combo["Saturn_retro"] = (sat_retro == "True")

# Compute force scores for all data
baseline = brain[side]["baseline"]
df_all['force_score'] = df_all.apply(lambda row: force_score(row, brain, side, k_shrink)[0], axis=1)
df_all['baseline'] = baseline

# Get today's analysis
today = df_all.iloc[-1].copy()
today_force, today_breakdown = force_score(today, brain, side=side, k_shrink=k_shrink)

# Filter combo occurrences
level1_side = level1_all[(level1_all["ext_dir"] == side) & pd.notna(level1_all["y_revert"])].copy()
occ = pd.DataFrame()
if combo:
    m = match_combo(level1_side, combo)
    occ = level1_side[m].copy()

combo_n = len(occ) if combo else 0
combo_p = float(occ["y_revert"].mean()) if len(occ) else np.nan

# -------------------- Key Metrics Display --------------------
st.markdown("### 📊 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Asset", symbol)
    st.metric("Side", side, help="U=Upper extreme, D=Lower extreme")

with col2:
    st.metric("Total Events", f"{len(level1_side):,}")
    st.metric("Combo Matches", f"{combo_n:,}")

with col3:
    if np.isfinite(baseline):
        st.metric("Baseline P(revert)", f"{baseline*100:.1f}%")
    else:
        st.metric("Baseline P(revert)", "N/A")
    
    if np.isfinite(combo_p):
        delta = (combo_p - baseline) * 100 if np.isfinite(baseline) else None
        st.metric("Combo P(revert)", f"{combo_p*100:.1f}%", 
                 delta=f"{delta:+.1f}pp" if delta else None)
    else:
        st.metric("Combo P(revert)", "N/A")

with col4:
    if np.isfinite(today_force):
        delta_today = (today_force - baseline) * 100 if np.isfinite(baseline) else None
        st.metric("Today's Force", f"{today_force*100:.1f}%",
                 delta=f"{delta_today:+.1f}pp" if delta_today else None)
    else:
        st.metric("Today's Force", "N/A")
    
    st.metric("Latest Date", str(today["Date"]))

with col5:
    latest_price = float(df_all.iloc[-1]["Close"])
    prev_price = float(df_all.iloc[-2]["Close"]) if len(df_all) > 1 else latest_price
    price_change = ((latest_price / prev_price) - 1) * 100
    st.metric("Latest Price", f"${latest_price:.2f}", 
             delta=f"{price_change:+.2f}%")
    
    if 'trend' in df_all.columns:
        trend = df_all.iloc[-1]['trend']
        st.metric("Trend (EMA200)", trend)

# -------------------- Main Chart --------------------
st.markdown("---")
st.markdown("### 📈 Price & Force Analysis")

hit_dates = occ["Date"].tolist() if len(occ) else []
main_chart = create_main_chart(df_all, hit_dates, side, symbol)
st.plotly_chart(main_chart, use_container_width=True)

# -------------------- Force Breakdown & Performance --------------------
st.markdown("---")
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### 🔍 Today's Force Breakdown")
    if today_breakdown:
        breakdown_chart = create_force_breakdown_chart(today_breakdown)
        if breakdown_chart:
            st.plotly_chart(breakdown_chart, use_container_width=True)
        
        # Show detailed table
        with st.expander("📋 Detailed Breakdown Table"):
            bd_df = pd.DataFrame(today_breakdown)
            bd_df['lift_pp'] = bd_df['lift_pp'].round(2)
            bd_df['weight'] = bd_df['weight'].round(3)
            st.dataframe(bd_df, use_container_width=True, height=300)
    else:
        st.info("ℹ️ No contributing placements met min_n threshold. Force score equals baseline.")

with col_right:
    st.markdown("### 🎯 Performance Heatmap")
    if len(occ) > 0:
        perf_heatmap = create_performance_heatmap(occ)
        if perf_heatmap:
            st.plotly_chart(perf_heatmap, use_container_width=True)
        else:
            st.info("ℹ️ Insufficient data for heatmap visualization.")
    else:
        st.info("ℹ️ Select planetary filters to see performance patterns.")

# -------------------- Historical Occurrences --------------------
st.markdown("---")
st.markdown("### 📚 Historical Occurrences Analysis")

if len(occ) == 0:
    st.info("ℹ️ No historical matches for the selected combo. Adjust filters or reduce min_n threshold.")
else:
    # Calculate forward returns
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
    
    # Summary statistics
    st.markdown("#### Summary Statistics")
    sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
    
    with sum_col1:
        win_rate = float(occ["y_revert"].mean() * 100) if len(occ) else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    with sum_col2:
        avg_1d = float(occ["ret_1d_pct"].mean()) if "ret_1d_pct" in occ.columns else 0
        st.metric("Avg 1-Day Return", f"{avg_1d:.2f}%")
    
    with sum_col3:
        avg_5d = float(occ["ret_5d_pct"].mean()) if "ret_5d_pct" in occ.columns else 0
        st.metric("Avg 5-Day Return", f"{avg_5d:.2f}%")
    
    with sum_col4:
        avg_10d = float(occ["ret_10d_pct"].mean()) if "ret_10d_pct" in occ.columns else 0
        st.metric("Avg 10-Day Return", f"{avg_10d:.2f}%")
    
    # Detailed table
    st.markdown("#### Event Details")
    
    # Select columns to display
    base_cols = ["Date", "ext_dir", "y_revert"]
    planet_cols = [c for c in FEATURES if c in occ.columns]
    ret_cols = ["ret_1d_pct", "ret_3d_pct", "ret_5d_pct", "ret_10d_pct"]
    
    display_cols = base_cols + planet_cols + ret_cols
    view = occ[display_cols].copy()
    
    # Format return columns
    for c in ret_cols:
        if c in view.columns:
            view[c] = view[c].round(2)
    
    # Add filters
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        show_winners_only = st.checkbox("Show winners only (y_revert=1)")
    with col_filter2:
        min_return = st.number_input("Min 5-day return %", value=-100.0, step=1.0)
    
    filtered_view = view.copy()
    if show_winners_only:
        filtered_view = filtered_view[filtered_view["y_revert"] == 1]
    if "ret_5d_pct" in filtered_view.columns:
        filtered_view = filtered_view[filtered_view["ret_5d_pct"] >= min_return]
    
    st.dataframe(filtered_view, use_container_width=True, height=400)
    
    # Download button
    csv = filtered_view.to_csv(index=False)
    st.download_button(
        label="📥 Download filtered data as CSV",
        data=csv,
        file_name=f"planetary_force_{symbol}_{side}.csv",
        mime="text/csv"
    )

# -------------------- Active Combo Display --------------------
if combo:
    st.markdown("---")
    st.markdown("### 🌙 Active Planetary Filter")
    combo_display = {}
    for k, v in combo.items():
        if isinstance(v, list):
            combo_display[k] = ", ".join(map(str, v))
        else:
            combo_display[k] = str(v)
    
    cols = st.columns(len(combo_display))
    for i, (key, val) in enumerate(combo_display.items()):
        with cols[i]:
            st.info(f"**{key}**\n\n{val}")

# -------------------- Future Scan --------------------
st.markdown("---")
st.markdown("### 🔮 Future Force Predictions")

with st.spinner("Calculating future planetary positions..."):
    start = datetime.now(tz=UTC).date()
    future_dates = [start + timedelta(days=i) for i in range(1, future_days + 1)]
    df_future = compute_astro(future_dates)

if df_future.empty or not np.isfinite(baseline):
    st.warning("⚠️ Future scan unavailable (missing baseline or astro data).")
else:
    scores = []
    breakdown_map = {}
    
    for _, r in df_future.iterrows():
        p, bd = force_score(r, brain, side=side, k_shrink=k_shrink)
        scores.append(p)
        breakdown_map[r["Date"]] = bd
    
    df_future = df_future.copy()
    df_future["pred_prob"] = scores
    df_future["lift_from_baseline"] = (df_future["pred_prob"] - baseline) * 100
    
    # Show top predictions
    top = df_future.sort_values("pred_prob", ascending=False).head(top_k).copy()
    
    st.markdown(f"#### Top {top_k} Dates by Force Score")
    
    # Display top dates with metrics
    top_cols = st.columns(min(top_k, 5))
    for i, (_, row) in enumerate(top.head(5).iterrows()):
        with top_cols[i]:
            delta_val = float(row["lift_from_baseline"])
            st.metric(
                str(row["Date"]),
                f"{row['pred_prob']*100:.1f}%",
                delta=f"{delta_val:+.1f}pp"
            )
    
    # Detailed table
    display_future_cols = ["Date", "pred_prob", "lift_from_baseline", "Moon_sign", "Mercury_sign", 
                          "Venus_sign", "Mars_sign", "Jupiter_sign", "Saturn_sign", 
                          "Phase", "Mercury_retro", "Moon_Node"]
    
    future_display = top[[c for c in display_future_cols if c in top.columns]].copy()
    future_display["pred_prob"] = (future_display["pred_prob"] * 100).round(1)
    future_display["lift_from_baseline"] = future_display["lift_from_baseline"].round(1)
    
    st.dataframe(future_display, use_container_width=True)
    
    # Timeline chart of future force scores
    st.markdown("#### Future Force Score Timeline")
    
    future_chart = go.Figure()
    
    future_chart.add_trace(go.Scatter(
        x=df_future["Date"].astype(str),
        y=df_future["pred_prob"] * 100,
        mode="lines+markers",
        name="Predicted Force",
        line=dict(color='#667eea', width=3),
        marker=dict(size=6)
    ))
    
    future_chart.add_hline(
        y=baseline * 100,
        line_dash="dash",
        line_color="gray",
        annotation_text="Baseline"
    )
    
    # Highlight top K dates
    for _, row in top.iterrows():
        future_chart.add_vline(
            x=str(row["Date"]),
            line_dash="dot",
            line_color="rgba(255,0,0,0.3)",
            line_width=2
        )
    
    future_chart.update_layout(
        title="Predicted Force Scores (Next " + str(future_days) + " Days)",
        xaxis_title="Date",
        yaxis_title="Force Score (%)",
        height=400,
        hovermode='x unified',
        plot_bgcolor='#f8f9fa',
        paper_bgcolor='white'
    )
    
    st.plotly_chart(future_chart, use_container_width=True)
    
    # Breakdown inspector
    st.markdown("#### Inspect Date Breakdown")
    chosen = st.selectbox(
        "Select a date to see force breakdown:",
        top["Date"].astype(str).tolist(),
        index=0
    )
    
    if chosen:
        chosen_date = datetime.strptime(chosen, "%Y-%m-%d").date()
        bd = breakdown_map.get(chosen_date, [])
        
        if not bd:
            st.info("ℹ️ No contributing placements met min_n for this date. Score = baseline.")
        else:
            # Show breakdown chart
            bd_chart = create_force_breakdown_chart(bd)
            if bd_chart:
                st.plotly_chart(bd_chart, use_container_width=True)
            
            # Show breakdown table
            with st.expander("📋 Detailed Breakdown"):
                bdf = pd.DataFrame(bd)
                bdf["lift_pp"] = bdf["lift_pp"].round(2)
                bdf["weight"] = bdf["weight"].round(3)
                st.dataframe(bdf, use_container_width=True, height=300)

# -------------------- Footer --------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🌟 Planetary Force Analytics Dashboard</p>
    <p style='font-size: 0.9em;'>Combining technical analysis with planetary positions for market insights</p>
</div>
""", unsafe_allow_html=True)
