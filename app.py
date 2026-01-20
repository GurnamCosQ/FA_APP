import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
import swisseph as swe
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# -------------------- Config --------------------
st.set_page_config(page_title="Planetary Edge Finder", layout="wide")

st.markdown("""
<style>
.big-verdict {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    color: white;
    text-align: center;
    font-size: 24px;
    margin: 20px 0;
}
.signal-green { background: #10b981 !important; }
.signal-yellow { background: #f59e0b !important; }
.signal-red { background: #ef4444 !important; }
</style>
""", unsafe_allow_html=True)

SIGNS = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
IST = ZoneInfo("Asia/Kolkata")
UTC = ZoneInfo("UTC")

ASSET_PRESETS = {
    "Nifty 50": "^NSEI",
    "Silver": "SI=F",
    "Gold": "GC=F",
    "Crude": "CL=F",
    "Bitcoin": "BTC-USD",
    "S&P 500": "^GSPC",
}

swe.set_sid_mode(swe.SIDM_LAHIRI)

FEATURES = [
    "Moon_sign", "Mercury_sign", "Venus_sign", "Mars_sign",
    "Phase", "Mercury_retro", "Moon_Node", "Rahu_sign", "Ketu_sign",
    "Jupiter_sign", "Saturn_sign", "Jupiter_retro", "Saturn_retro",
]

# -------------------- Utilities --------------------
def sign_from_lon(lon_deg: float) -> str:
    return SIGNS[int((lon_deg % 360.0) // 30)]

def julday_fast(d, hour_ist: int = 10) -> float:
    dt_local = datetime(d.year, d.month, d.day, hour_ist, 0, 0, tzinfo=IST)
    dt_utc = dt_local.astimezone(UTC)
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour + dt_utc.minute/60.0)

def julday_slow(d, hour_utc: int = 12) -> float:
    dt_utc = datetime(d.year, d.month, d.day, hour_utc, 0, 0, tzinfo=UTC)
    return swe.julday(dt_utc.year, dt_utc.month, dt_utc.day, dt_utc.hour)

def sidereal_lon_speed(jd_ut: float, body: int) -> tuple[float, float]:
    flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL
    xx, _ = swe.calc_ut(jd_ut, body, flags)
    return float(xx[0]) % 360.0, float(xx[3])

@st.cache_data(show_spinner=False)
def fetch_price(symbol: str, period: str) -> pd.DataFrame:
    raw = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
    if raw is None or len(raw) == 0:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index().copy()
    dt_col = "Date" if "Date" in df.columns else "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={dt_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    return df[pd.notna(df["Date"])].copy()

def compute_tech(df_px: pd.DataFrame, bb_w: int, bb_k: float, margin: float, lookahead: int, stopout: float):
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
        if not df.loc[i, "event_start"]:
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
                    break
            else:
                if tp_j >= ma_j:
                    success = 1
                    break
                if tp_j < ref_low * (1.0 - stopout):
                    break
        y_revert[i] = success
    
    df["y_revert"] = y_revert
    df["ema200"] = df["Close"].ewm(span=200, adjust=False).mean()
    events = df[df["event_start"] & df["ext_dir"].isin(["U", "D"])].copy()
    return df, events

@st.cache_data(show_spinner=False)
def compute_astro(dates: list) -> pd.DataFrame:
    if not dates:
        return pd.DataFrame(columns=["Date"] + FEATURES)
    rows = []
    for d in dates:
        jd_f = julday_fast(d, 10)
        jd_s = julday_slow(d, 12)
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
    return pd.DataFrame(rows)

def match_combo(df: pd.DataFrame, combo: dict) -> pd.Series:
    m = pd.Series(True, index=df.index)
    for feat, val in combo.items():
        if feat not in df.columns:
            m &= False
            continue
        if isinstance(val, (list, tuple)):
            m &= df[feat].isin(list(val))
        else:
            m &= (df[feat] == val)
    return m

def get_signal_color(success_rate: float) -> str:
    if success_rate >= 0.65:
        return "signal-green"
    elif success_rate >= 0.50:
        return "signal-yellow"
    else:
        return "signal-red"

def create_occurrence_chart(df_all: pd.DataFrame, event_date, window_before: int = 30, window_after: int = 30):
    """Create a single chart for one occurrence with placement zones"""
    # Find index
    df_all = df_all.reset_index(drop=True)
    idx = df_all[df_all["Date"] == event_date].index
    if len(idx) == 0:
        return None
    
    idx = idx[0]
    start_idx = max(0, idx - window_before)
    end_idx = min(len(df_all) - 1, idx + window_after)
    
    df_slice = df_all.iloc[start_idx:end_idx+1].copy()
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=df_slice["Date"].astype(str),
        y=df_slice["Close"],
        mode="lines",
        name="Price",
        line=dict(color='#2962ff', width=2)
    ))
    
    # MA20
    fig.add_trace(go.Scatter(
        x=df_slice["Date"].astype(str),
        y=df_slice["MA20"],
        mode="lines",
        name="MA20",
        line=dict(color='#ff6d00', width=1.5, dash='dash')
    ))
    
    # EMA200
    if 'ema200' in df_slice.columns:
        fig.add_trace(go.Scatter(
            x=df_slice["Date"].astype(str),
            y=df_slice["ema200"],
            mode="lines",
            name="EMA200",
            line=dict(color='#aa00ff', width=1, dash='dot'),
            opacity=0.6
        ))
    
    # BB bands
    fig.add_trace(go.Scatter(
        x=df_slice["Date"].astype(str),
        y=df_slice["Upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_slice["Date"].astype(str),
        y=df_slice["Lower"],
        mode="lines",
        line=dict(width=0),
        fillcolor='rgba(68, 138, 255, 0.1)',
        fill='tonexty',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Event vertical line
    fig.add_vline(x=str(event_date), line_width=3, line_color="red", line_dash="solid")
    
    fig.update_layout(
        title=f"Event on {event_date}",
        xaxis_title="Date",
        yaxis_title="Price",
        height=400,
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True,
        legend=dict(orientation="h", y=1.1)
    )
    
    return fig

# -------------------- UI --------------------
st.title("🎯 Planetary Edge Finder")
st.caption("Simple. Effective. Find your alpha.")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Setup")
    
    asset = st.selectbox("Asset", list(ASSET_PRESETS.keys()))
    symbol = ASSET_PRESETS[asset]
    period = st.selectbox("History", ["10y","5y","2y"], index=1)
    
    st.divider()
    st.subheader("Technical Settings")
    bb_w = st.slider("BB Window", 10, 40, 20)
    bb_k = st.slider("BB K", 1.0, 3.0, 2.0, 0.1)
    margin = st.slider("Margin Sigma", 0.0, 1.0, 0.25, 0.05)
    lookahead = st.slider("Lookahead Days", 5, 30, 15)
    stopout = st.slider("Stopout %", 0.5, 5.0, 2.0, 0.1) / 100.0
    side = st.radio("Event Side", ["U", "D"], horizontal=True)
    
    st.divider()
    st.subheader("Discovery Filters")
    min_sample = st.slider("Min Sample Size", 5, 50, 15, help="Only show combos that happened at least this many times")
    
    st.divider()
    st.subheader("🌙 Select Planets to Test")
    st.caption("Pick which planetary placements you want to analyze")
    
    test_moon = st.checkbox("Moon Sign", value=False)
    if test_moon:
        moon_signs = st.multiselect("Which Moon signs?", SIGNS, default=[])
    
    test_mercury = st.checkbox("Mercury Sign", value=False)
    if test_mercury:
        mercury_signs = st.multiselect("Which Mercury signs?", SIGNS, default=[])
    
    test_venus = st.checkbox("Venus Sign", value=False)
    if test_venus:
        venus_signs = st.multiselect("Which Venus signs?", SIGNS, default=[])
    
    test_mars = st.checkbox("Mars Sign", value=True)
    if test_mars:
        mars_signs = st.multiselect("Which Mars signs?", SIGNS, default=["Taurus"])
    
    test_jupiter = st.checkbox("Jupiter Sign", value=False)
    if test_jupiter:
        jupiter_signs = st.multiselect("Which Jupiter signs?", SIGNS, default=[])
    
    test_saturn = st.checkbox("Saturn Sign", value=False)
    if test_saturn:
        saturn_signs = st.multiselect("Which Saturn signs?", SIGNS, default=[])
    
    test_phase = st.checkbox("Lunar Phase", value=False)
    if test_phase:
        phases = st.multiselect("Which phases?", ["Waxing", "Waning"], default=[])

# -------------------- Load Data --------------------
with st.spinner("Loading market data..."):
    df_px = fetch_price(symbol, period)
    
if df_px.empty:
    st.error("No data available")
    st.stop()

with st.spinner("Computing technical signals..."):
    df_all, events = compute_tech(df_px, bb_w, bb_k, margin, lookahead, stopout)

with st.spinner("Computing planetary positions..."):
    df_astro = compute_astro(df_all["Date"].tolist())
    df_all = df_all.merge(df_astro, on="Date", how="left")
    events = events.merge(df_astro, on="Date", how="left")

# Filter by side and valid outcomes
events_side = events[(events["ext_dir"] == side) & pd.notna(events["y_revert"])].copy()

if len(events_side) == 0:
    st.error("No events found for this configuration")
    st.stop()

# Calculate baseline
baseline_success = float(events_side["y_revert"].mean())
baseline_n = len(events_side)

# -------------------- Build Combo --------------------
combo = {}
if test_moon and moon_signs:
    combo["Moon_sign"] = moon_signs
if test_mercury and mercury_signs:
    combo["Mercury_sign"] = mercury_signs
if test_venus and venus_signs:
    combo["Venus_sign"] = venus_signs
if test_mars and mars_signs:
    combo["Mars_sign"] = mars_signs
if test_jupiter and jupiter_signs:
    combo["Jupiter_sign"] = jupiter_signs
if test_saturn and saturn_signs:
    combo["Saturn_sign"] = saturn_signs
if test_phase and phases:
    combo["Phase"] = phases

# -------------------- Analysis --------------------
if not combo:
    st.info("👈 Select planetary placements in the sidebar to begin analysis")
    st.metric("Baseline Success Rate (All Events)", f"{baseline_success*100:.1f}%", help=f"Based on {baseline_n} total events")
    st.stop()

# Match combo
matches = match_combo(events_side, combo)
combo_events = events_side[matches].copy()

combo_n = len(combo_events)
combo_success = float(combo_events["y_revert"].mean()) if combo_n > 0 else 0.0
edge = (combo_success - baseline_success) * 100

# Signal strength
if combo_success >= 0.65:
    signal = "🟢 STRONG EDGE"
    signal_class = "signal-green"
elif combo_success >= 0.50:
    signal = "🟡 WEAK EDGE"
    signal_class = "signal-yellow"
else:
    signal = "🔴 NO EDGE"
    signal_class = "signal-red"

# -------------------- Verdict Box --------------------
st.markdown(f"""
<div class="big-verdict {signal_class}">
    <div style="font-size: 32px; font-weight: bold; margin-bottom: 10px;">{signal}</div>
    <div style="font-size: 18px;">
        Success Rate: {combo_success*100:.1f}% | Edge: {edge:+.1f}pp | Sample Size: {combo_n} events
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------- Metrics --------------------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Success Rate", f"{combo_success*100:.1f}%", 
             help="What % of the time did this signal lead to a win?")

with col2:
    st.metric("Edge", f"{edge:+.1f}pp",
             help="How much better than random? (percentage points)")

with col3:
    st.metric("Sample Size", combo_n,
             help="How many times has this combo occurred?")

with col4:
    st.metric("Baseline", f"{baseline_success*100:.1f}%",
             help="Success rate of all events (no filter)")

st.divider()

# -------------------- Active Combo Display --------------------
st.subheader("📋 Active Combination")
combo_display = []
for k, v in combo.items():
    if isinstance(v, list):
        combo_display.append(f"**{k}**: {', '.join(map(str, v))}")
    else:
        combo_display.append(f"**{k}**: {v}")

st.info(" | ".join(combo_display))

# -------------------- Show Individual Occurrences --------------------
if combo_n == 0:
    st.warning("⚠️ No historical occurrences found for this combination. Try different planets or reduce min sample size.")
else:
    st.subheader(f"📊 Individual Occurrences ({combo_n} charts)")
    st.caption("Each chart shows 30 days before and after the event")
    
    # Sort by date descending
    combo_events = combo_events.sort_values("Date", ascending=False)
    
    # Show charts in grid
    for idx, (_, event) in enumerate(combo_events.iterrows()):
        event_date = event["Date"]
        outcome = "✅ WIN" if event["y_revert"] == 1 else "❌ LOSS"
        
        with st.expander(f"Event #{idx+1}: {event_date} - {outcome}", expanded=(idx < 3)):
            chart = create_occurrence_chart(df_all, event_date, window_before=30, window_after=30)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Show planetary positions on this date
            planets_info = []
            for feat in FEATURES:
                if feat in event.index:
                    planets_info.append(f"{feat}: {event[feat]}")
            
            st.caption(" | ".join(planets_info[:6]))

# -------------------- Future Predictions --------------------
st.divider()
st.subheader("🔮 Future Scan")

future_days = st.slider("Scan next N days", 7, 90, 30)

with st.spinner("Computing future positions..."):
    start = datetime.now(tz=UTC).date()
    future_dates = [start + timedelta(days=i) for i in range(1, future_days + 1)]
    df_future = compute_astro(future_dates)

if not df_future.empty:
    # Check which future dates match our combo
    future_matches = match_combo(df_future, combo)
    matching_dates = df_future[future_matches].copy()
    
    if len(matching_dates) > 0:
        st.success(f"🎯 Found {len(matching_dates)} matching dates in the next {future_days} days!")
        
        for _, row in matching_dates.iterrows():
            st.info(f"**{row['Date']}** - Predicted Success Rate: {combo_success*100:.1f}% (based on {combo_n} historical occurrences)")
    else:
        st.info(f"No matching dates found in the next {future_days} days for this combination.")
else:
    st.warning("Unable to compute future positions")

# -------------------- Discovery Table --------------------
st.divider()
st.subheader("🔍 Discovery: All Combinations")
st.caption("Explore what worked historically (sorted by edge)")

# Build all combos from single features
discovery_rows = []

for feat in FEATURES:
    if feat not in events_side.columns:
        continue
    
    unique_vals = events_side[feat].unique()
    
    for val in unique_vals:
        if pd.isna(val):
            continue
        
        subset = events_side[events_side[feat] == val]
        n = len(subset)
        
        if n < min_sample:
            continue
        
        success = float(subset["y_revert"].mean())
        edge_pp = (success - baseline_success) * 100
        
        discovery_rows.append({
            "Feature": feat,
            "Value": val,
            "Sample Size": n,
            "Success Rate %": round(success * 100, 1),
            "Edge (pp)": round(edge_pp, 1),
            "Signal": "🟢" if success >= 0.65 else "🟡" if success >= 0.50 else "🔴"
        })

discovery_df = pd.DataFrame(discovery_rows)

if len(discovery_df) > 0:
    discovery_df = discovery_df.sort_values("Edge (pp)", ascending=False)
    st.dataframe(discovery_df, use_container_width=True, height=400)
else:
    st.info("No combinations meet the minimum sample size threshold")

st.divider()
st.caption("💡 Tip: Use the Discovery table to find high-edge combinations, then test them in the sidebar!")
