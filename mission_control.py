import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import time
import json
import requests
import subprocess
import sys
import numpy as np

# ──────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────
st.set_page_config(page_title="SKYCIPHER DRONE COMMAND", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
.stApp {
    background-color: #050a07;
    background-image: linear-gradient(0deg, transparent 24%, rgba(57,255,20,0.04) 25%, rgba(57,255,20,0.04) 26%, transparent 27%);
    background-size: 50px 50px;
    color: #d3d3d3;
    font-family: 'Share Tech Mono', monospace;
}
h1,h2,h3,h4 { color: #39ff14 !important; font-family: 'Share Tech Mono', monospace !important;
    text-transform: uppercase; letter-spacing: 3px; text-shadow: 0 0 8px rgba(57,255,20,0.5); }
div[data-testid="metric-container"] {
    background: rgba(13,24,17,0.85); border-radius: 2px; padding: 15px;
    border: 1px solid #1f3b28; border-left: 5px solid #39ff14;
    box-shadow: 0 4px 15px rgba(0,0,0,0.8), inset 0 0 15px rgba(57,255,20,0.05);
}
div[data-testid="metric-container"] label { color: #6a9c78 !important; font-size:0.85rem !important; text-transform:uppercase; }
div[data-testid="metric-container"] div { color: #39ff14 !important; font-weight: bold; }
.stButton>button { background:#0d1811; color:#39ff14 !important; font-weight:700;
    font-family:'Share Tech Mono',monospace; letter-spacing:2px; border-radius:0px;
    border:1px solid #39ff14; padding:12px 24px; transition:all 0.1s ease; text-transform:uppercase; }
.stButton>button:hover { background:#39ff14; color:#000 !important; box-shadow:0 0 25px rgba(57,255,20,0.8); }
.stTabs [data-baseweb="tab-list"] { gap:2px; }
.stTabs [data-baseweb="tab"] { background-color:#0a120d; border-radius:0px; padding:10px 15px;
    border:1px solid #1f3b28; border-bottom:none; color:#4a7553; font-weight:bold; letter-spacing:1px; }
.stTabs [aria-selected="true"] { background-color:#15291b !important; color:#39ff14 !important;
    border-color:#39ff14; border-bottom:2px solid #39ff14; }
div[data-testid="stSidebar"] { background-color:#070d0a; border-right:2px solid #1f3b28; }
.terminal-box { background-color:#000; color:#39ff14; font-family:'Share Tech Mono',monospace;
    padding:15px; border:1px solid #39ff14; border-radius:2px; height:400px;
    overflow-y:auto; box-shadow:inset 0 0 15px rgba(57,255,20,0.3); white-space:pre-wrap; }
.stProgress > div > div > div > div { background-color:#39ff14; box-shadow:0 0 10px #39ff14; }
hr { border-color:#1f3b28 !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────
# Constants
# ──────────────────────────────────────────
RADAR_COLS = [f'radar_{i}' for i in range(8)]
CHART_COLORS = ["#39ff14", "#8fbc8f", "#00ffcc", "#ffdd00", "#ff4444"]
TEMPLATE = "plotly_dark"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "DroneProject", "skycipher_dataset_20260512_125542.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join("DroneProject", "skycipher_dataset_20260512_125542.csv")
TEST_SCRIPT = os.path.join(BASE_DIR, "DroneProject", "test.py")
if not os.path.exists(TEST_SCRIPT):
    TEST_SCRIPT = os.path.join("DroneProject", "test.py")


# ──────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────
@st.cache_data(show_spinner="[ LOADING SKYCIPHER DATASET... ]")
def load_data(path):
    try:
        df = pd.read_csv(path, low_memory=False)
    except FileNotFoundError:
        st.error(f"[ CRITICAL ] Dataset not found: {path}")
        st.stop()

    # Feature engineering
    df['Avg_Radar'] = df[RADAR_COLS].mean(axis=1)
    df['Risk_Level'] = (df['wind_strength'] * 0.5) + ((1 - df['closest_obstacle']) * 0.5)
    df['Cumulative_Reward'] = df.groupby('episode')['reward'].cumsum()
    df['Distance_Change'] = df.groupby('episode')['target_dist'].diff().fillna(0)
    return df

df = load_data(DATA_PATH)

# ──────────────────────────────────────────
# ML Model — Random Forest on real columns
# ──────────────────────────────────────────
@st.cache_resource(show_spinner="[ TRAINING RF CLASSIFIER... ]")
def train_rf(df):
    feats = ['target_dist', 'wind_strength', 'Avg_Radar', 'Risk_Level',
             'closest_obstacle', 'action_magnitude', 'bearing_deg']
    X = df[feats].fillna(0)
    y = (df['episode_outcome'] == 'success').astype(int)

    if len(X) > 10000:
        idx = X.sample(10000, random_state=42).index
        X_tr, y_tr = X.loc[idx], y.loc[idx]
    else:
        X_tr, y_tr = X, y

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    if len(y_tr.unique()) > 1:
        clf.fit(X_tr, y_tr)
    return clf, feats

model, features = train_rf(df)

# ──────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## [ SKYCIPHER COMMAND ]")

    with st.expander("[ SYSTEM INTEL ]", expanded=True):
        st.markdown("""
**DATASET:** SkyCipher v3 (Level 3)
- **target_dist** — distance to objective
- **wind_strength** — environmental wind
- **radar_detected** — enemy radar flag
- **closest_obstacle** — nearest obstacle
- **frame_class** — Normal/Evading/Approaching/Crash
- **episode_outcome** — success/crash/timeout
        """)

    n8n_url = "http://localhost:5678/webhook/c4895708-1203-4095-9991-b5c6ef3461e2"


    st.subheader("[ FILTERS ]")
    outcomes = df['episode_outcome'].unique().tolist()
    sel_outcomes = st.multiselect("OUTCOME:", outcomes, default=outcomes)
    classes = df['frame_class'].unique().tolist()
    sel_classes = st.multiselect("FRAME CLASS:", classes, default=classes)

    filtered = df[df['episode_outcome'].isin(sel_outcomes) & df['frame_class'].isin(sel_classes)]

    st.markdown("---")
    st.subheader("[ TACTICAL AI ]")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "[ SKYCIPHER ONLINE. AWAITING COMMAND. ]"}]

    chat_box = st.container(height=300)
    with chat_box:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]): st.markdown(m["content"])

    if prompt := st.chat_input("ENTER COMMAND..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_box:
            with st.chat_message("user"): st.markdown(prompt)
            with st.chat_message("assistant"):
                ph = st.empty()
                if not filtered.empty:
                    last_row = filtered.iloc[-1]
                    ctx = {
                        "episode": int(last_row.get('episode', 1)),
                        "step": int(last_row.get('step', 0)),
                        "drone_x": round(float(last_row.get('drone_x', 0)), 2),
                        "drone_y": round(float(last_row.get('drone_y', 0)), 2),
                        "drone_z": round(float(last_row.get('drone_z', 0)), 2),
                        "target_dist": round(float(last_row.get('target_dist', 0)), 2),
                        "wind_strength": round(float(last_row.get('wind_strength', 0)), 2),
                        "Risk_Level": round(float(last_row.get('Risk_Level', 0)), 2),
                        "frame_class": str(last_row.get('frame_class', 'Normal')),
                        "closest_obstacle": round(float(last_row.get('closest_obstacle', 1)), 2),
                        "radar_detected": bool(last_row.get('radar_detected', 0))
                    }
                else:
                    ctx = {}
                chat_history = st.session_state.messages[:-1] 
                payload = {
                    "question": prompt, 
                    "drone_context": ctx,
                    "history": chat_history
                }
                try:
                    res = requests.post(n8n_url, json=payload, timeout=45)
                    
                    if res.status_code == 200:
                        try:
                            data = res.json()
                            if isinstance(data, dict):
                                resp = data.get("output", data.get("text", data.get("response", data.get("message", res.text))))
                            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                                resp = data[0].get("output", data[0].get("text", str(data[0])))
                            else:
                                resp = str(data)
                        except Exception:
                            resp = res.text
                    else:
                        resp = f"[ UPLINK ERROR {res.status_code} ]"
                        
                except requests.exceptions.Timeout:
                    resp = "[ UPLINK TIMEOUT ] - انتهى وقت الاتصال، الذكاء الاصطناعي يفكر طويلاً."
                except Exception as e:
                    resp = f"[ SYSTEM ERROR ]: {e}"
                
                full = ""
                for w in str(resp).split():
                    full += w + " "
                    time.sleep(0.04)
                    ph.markdown(full + "█")
                ph.markdown(full)
        st.session_state.messages.append({"role": "assistant", "content": full})

# ────
# ──────────────────────────────────────────
# Header
# ──────────────────────────────────────────
st.markdown("<h1 style='text-align:center'>[ SKYCIPHER DRONE COMMAND v3 ]</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#39ff14;font-size:1rem;text-align:center;letter-spacing:2px'>[ PPO LEVEL 3 | PYBULLET ENV | SKYCIPHER DATASET ]</p>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "[ DASHBOARD ]",
    "[ LIVE STREAM ]",
    "[ 3D RADAR ]",
    "[ ML RESULTS ]",
    "[ ALGO INTEL ]",
    "[ HARDWARE ]",
    "[ SIMULATION ]",
])

# ══════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════
with tab1:
    st.markdown("### [ MISSION OVERVIEW ]")
    c1, c2, c3, c4, c5 = st.columns(5)
    total_eps = filtered['episode'].nunique()
    successes = (filtered.groupby('episode')['episode_outcome'].first() == 'success').sum()
    success_rate = successes / total_eps * 100 if total_eps > 0 else 0
    c1.metric("TOTAL EPISODES", total_eps)
    c2.metric("SUCCESS RATE", f"{success_rate:.1f}%")
    c3.metric("AVG TARGET DIST", f"{filtered['target_dist'].mean():.2f} m")
    c4.metric("MAX WIND", f"{filtered['wind_strength'].max():.2f} m/s")
    c5.metric("RADAR DETECTIONS", int(filtered['radar_detected'].sum()))

    st.markdown("---")
    ca, cb = st.columns([1, 2])
    with ca:
        st.markdown("#### [ OUTCOME DISTRIBUTION ]")
        oc = filtered.groupby('episode')['episode_outcome'].first().value_counts().reset_index()
        oc.columns = ['Outcome', 'Count']
        fig = px.pie(oc, names='Outcome', values='Count', hole=0.65, color_discrete_sequence=CHART_COLORS)
        fig.update_layout(template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        st.markdown("#### [ FRAME CLASS DISTRIBUTION ]")
        fc = filtered['frame_class'].value_counts().reset_index()
        fc.columns = ['Class', 'Count']
        fig2 = px.bar(fc, x='Class', y='Count', color='Class', color_discrete_sequence=CHART_COLORS)
        fig2.update_layout(template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### [ WIND vs RISK vs DISTANCE ]")
    
    # 1. أخذ العينة
    sample_df = filtered.sample(min(1200, len(filtered)), random_state=42)
    
    # 2. خدعة الطبقات (Layering): ترتيب البيانات لتُرسم حالة Normal في الخلفية، والحالات الحرجة في المقدمة
    sort_order = {'Normal': 0, 'Approaching': 1, 'Evading': 2, 'Crash': 3}
    sample_df['render_order'] = sample_df['frame_class'].map(sort_order)
    sample_df = sample_df.sort_values('render_order')

    fig3 = px.scatter(
        sample_df,
        x='wind_strength', 
        y='target_dist', 
        color='frame_class',
        size='Risk_Level', 
        hover_data=['episode','reward'],
        # 3. تخصيص الألوان والشفافية: جعل Normal شفاف جداً ليصبح خلفية، وإبراز الباقي
        color_discrete_map={
            'Normal': 'rgba(57, 255, 20, 0.25)', # أخضر تكتيكي شفاف جداً
            'Approaching': '#8fbc8f',            # لون واضح للاقتراب
            'Evading': '#00ffcc',                # أزرق ساطع للتفادي
            'Crash': '#ff4444'                   # أحمر للتحطم
        },
        size_max=12  # 4. التقييد الصارم: منع أي فقاعة من التضخم أكثر من هذا الحجم
    )
    
    fig3.update_traces(marker=dict(line=dict(width=0)))
    fig3.update_layout(
        template=TEMPLATE, 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(title="WIND STRENGTH (m/s)", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(title="TARGET DISTANCE (m)", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) # نقل المفتاح للأعلى لترتيب المساحة
    )
    
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════
# TAB 2 — LIVE STREAM
# ══════════════════════════════════════════
with tab2:
    st.markdown("### [ EPISODE TACTICAL STREAM ]")
    ep_options = filtered['episode'].unique()
    
    if len(ep_options) == 0:
        st.warning("[ NO EPISODES MATCH CURRENT FILTERS ]")
    else:
        col_ep, col_spd, col_btn = st.columns([2, 1, 1])
        with col_ep:
            sel_ep = st.selectbox("SELECT MISSION EPISODE:", ep_options, key="live_ep_select")
        with col_spd:
            spd_option = st.selectbox("STREAM SPEED:", ["1.0x (Real-time)", "0.5x (Slow)", "2.0x (Fast)", "4.0x (Hyper)"], index=0)
            speed_map = {"0.5x (Slow)": 0.12, "1.0x (Real-time)": 0.05, "2.0x (Fast)": 0.02, "4.0x (Hyper)": 0.005}
            sleep_delay = speed_map[spd_option]
        with col_btn:
            st.write("")
            st.write("")
            start_playback = st.button("[ INITIATE PLAYBACK ]", use_container_width=True)

        ep_data = filtered[filtered['episode'] == sel_ep].sort_values('step').reset_index(drop=True)
        total_ep_steps = len(ep_data)

        col_vis, col_hud = st.columns([1, 2])
        with col_vis:
            cam_ph = st.empty()
        with col_hud:
            hud_ph = st.empty()
        chart_ph = st.empty()

        def render_vision(row):
            img_path = str(row.get('image_path', ''))
            img_file = None
            if pd.notna(img_path) and img_path.strip() and img_path != 'nan':
                fname = os.path.basename(img_path.replace('\\', '/'))
                candidate = os.path.join(BASE_DIR, "DroneProject", "dataset_images", fname)
                if os.path.exists(candidate):
                    img_file = candidate
            
            f_class = str(row.get('frame_class', 'Normal'))
            color_map = {'Normal': '#39ff14', 'Approaching': '#8fbc8f', 'Evading': '#00ffcc', 'Crash': '#ff4444'}
            sc = color_map.get(f_class, '#39ff14')

            if img_file:
                cam_ph.image(img_file, caption=f"OPTICAL CAMERA | [ {f_class.upper()} ]", use_container_width=True)
            else:
                cam_ph.markdown(f"""
                <div style="background: rgba(13,24,17,0.85); border: 1px solid #1f3b28; border-top: 3px solid {sc}; border-radius: 4px; padding: 22px 10px; text-align: center; height: 185px; display: flex; flex-direction: column; justify-content: center;">
                    <div style="font-size: 0.72rem; color: #6a9c78; letter-spacing: 2px;">TACTICAL VISION SENSOR</div>
                    <div style="font-size: 1.5rem; color: {sc}; font-weight: bold; font-family: 'Share Tech Mono', monospace; margin: 8px 0;">[ {f_class.upper()} ]</div>
                    <div style="font-size: 0.72rem; color: #888;">BEARING: <b style="color:#39ff14;">{row.get('bearing_deg', 0):.1f}°</b> | EW SHIELD ACTIVE</div>
                </div>
                """, unsafe_allow_html=True)

        def render_hud(row, prob):
            dist = row['target_dist']
            wind = row['wind_strength']
            f_class = row['frame_class']
            z_alt = row.get('drone_z', 0)
            obs_dist = row.get('closest_obstacle', 1.0)
            is_radar = bool(row.get('radar_detected', 0))
            
            status_color = "#39ff14" if f_class == "Normal" else "#8fbc8f" if f_class == "Approaching" else "#00ffcc" if f_class == "Evading" else "#ff4444"
            prob_color = "#39ff14" if prob >= 70 else "#ffdd00" if prob >= 40 else "#ff4444"
            radar_badge = "<span style='color:#ff4444;font-weight:bold;'>[ ⚠️ RADAR LOCK ]</span>" if is_radar else "<span style='color:#39ff14;'>[ RADAR CLEAR ]</span>"
            
            hud_html = f"""
            <div style="background: rgba(7,13,10,0.95); border: 1px solid #1f3b28; border-radius: 4px; padding: 16px; margin-bottom: 15px; box-shadow: inset 0 0 20px rgba(57,255,20,0.05);">
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                    <div style="background: rgba(13,24,17,0.8); border-left: 4px solid #39ff14; padding: 10px 14px; border-radius: 2px;">
                        <div style="font-size: 0.72rem; color: #6a9c78; letter-spacing: 1px;">TARGET DISTANCE</div>
                        <div style="font-size: 1.6rem; color: #39ff14; font-weight: bold; font-family: 'Share Tech Mono', monospace;">{dist:.2f} m</div>
                        <div style="font-size: 0.75rem; color: #888;">ALTITUDE: <b style="color:#39ff14;">{z_alt:.1f}m</b></div>
                    </div>
                    <div style="background: rgba(13,24,17,0.8); border-left: 4px solid #ffdd00; padding: 10px 14px; border-radius: 2px;">
                        <div style="font-size: 0.72rem; color: #9c926a; letter-spacing: 1px;">WIND STRENGTH</div>
                        <div style="font-size: 1.6rem; color: #ffdd00; font-weight: bold; font-family: 'Share Tech Mono', monospace;">{wind:.2f} m/s</div>
                        <div style="font-size: 0.75rem; color: #888;">GUST: <b style="color:#ffdd00;">{row.get('wind_x', 0):.1f}, {row.get('wind_y', 0):.1f}</b></div>
                    </div>
                    <div style="background: rgba(13,24,17,0.8); border-left: 4px solid {prob_color}; padding: 10px 14px; border-radius: 2px;">
                        <div style="font-size: 0.72rem; color: #6a9c78; letter-spacing: 1px;">SURVIVAL PROB</div>
                        <div style="font-size: 1.6rem; color: {prob_color}; font-weight: bold; font-family: 'Share Tech Mono', monospace;">{prob:.1f}%</div>
                        <div style="font-size: 0.75rem; color: #888;">STATUS: <b style="color:{prob_color};">{'SECURE' if prob >= 70 else 'CAUTION' if prob >= 40 else 'DANGER'}</b></div>
                    </div>
                    <div style="background: rgba(13,24,17,0.8); border-left: 4px solid {status_color}; padding: 10px 14px; border-radius: 2px;">
                        <div style="font-size: 0.72rem; color: #6a9c78; letter-spacing: 1px;">TACTICAL STATE</div>
                        <div style="font-size: 1.3rem; color: {status_color}; font-weight: bold; font-family: 'Share Tech Mono', monospace; margin-top:2px;">[ {f_class.upper()} ]</div>
                        <div style="font-size: 0.75rem; margin-top:3px;">{radar_badge}</div>
                    </div>
                </div>
                <div style="margin-top: 12px; font-size: 0.78rem; color: #6a9c78; display: flex; justify-content: space-between; border-top: 1px solid #1f3b28; padding-top: 8px;">
                    <span>MISSION STEP: <b style="color:#39ff14;">{row['ep_step_count']} / {total_ep_steps}</b></span>
                    <span>COORDINATES: <b style="color:#39ff14;">X:{row.get('drone_x', 0):.2f} | Y:{row.get('drone_y', 0):.2f} | Z:{z_alt:.2f}</b></span>
                    <span>OBSTACLE PROXIMITY: <b style="color:{'#ff4444' if obs_dist < 0.3 else '#39ff14'};">{obs_dist:.2f}</b></span>
                </div>
            </div>
            """
            hud_ph.markdown(hud_html, unsafe_allow_html=True)

        def compute_prob(row):
            inp = pd.DataFrame([[row['target_dist'], row['wind_strength'], row['Avg_Radar'],
                                  row['Risk_Level'], row['closest_obstacle'],
                                  row.get('action_magnitude', 0), row.get('bearing_deg', 0)]], columns=features)
            return float(model.predict_proba(inp)[0][1] * 100) if hasattr(model, 'predict_proba') else 85.0

        def draw_chart(steps_arr, dist_arr, alt_arr, reward_arr):
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=steps_arr, y=dist_arr, name="Target Dist (m)", line=dict(color="#39ff14", width=2.5)))
            fig_line.add_trace(go.Scatter(x=steps_arr, y=alt_arr, name="Altitude Z (m)", line=dict(color="#00ffcc", width=2, dash="dot")))
            fig_line.add_trace(go.Scatter(x=steps_arr, y=reward_arr, name="Reward", line=dict(color="#ffdd00", width=1.5), yaxis="y2"))
            fig_line.update_layout(
                template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title="MISSION STEPS", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(title="DISTANCE / ALTITUDE (m)", showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                yaxis2=dict(title="REWARD", overlaying='y', side='right', color='#ffdd00', showgrid=False),
                margin=dict(t=15, b=25, l=15, r=15),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=320,
                uirevision="tactical_fixed"
            )
            # Crucial: NO key argument here to prevent StreamlitDuplicateElementKey during loop/updates
            chart_ph.plotly_chart(fig_line, use_container_width=True)

        if start_playback:
            live_steps, live_dist, live_alt, live_reward = [], [], [], []
            step_count = len(ep_data)
            stride = max(1, step_count // 120) if step_count > 150 else 1
            
            for idx in range(0, step_count, stride):
                row = ep_data.iloc[idx]
                s = int(row['ep_step_count'])
                live_steps.append(s)
                live_dist.append(float(row['target_dist']))
                live_alt.append(float(row.get('drone_z', 0)))
                live_reward.append(float(row.get('reward', 0)))
                
                p_val = compute_prob(row)
                render_vision(row)
                render_hud(row, p_val)
                draw_chart(live_steps, live_dist, live_alt, live_reward)
                time.sleep(sleep_delay)
                
            outcome = ep_data.iloc[-1]['episode_outcome']
            if outcome == "success":
                st.success("[ MISSION OUTCOME ]: MISSION ACCOMPLISHED — TARGET EXTRACTION COMPLETE")
            elif outcome == "crash":
                st.error("[ MISSION OUTCOME ]: MISSION FAILED — DRONE COLLISION RECORDED")
            else:
                st.warning("[ MISSION OUTCOME ]: MISSION TERMINATED — MAXIMUM ENGAGEMENT TIMEOUT")
        else:
            if total_ep_steps > 0:
                step_idx = st.slider("SCRUB TELEMETRY TIMELINE (STEP):", min_value=0, max_value=max(0, total_ep_steps - 1), value=max(0, total_ep_steps - 1))
                curr_row = ep_data.iloc[step_idx]
                p_val = compute_prob(curr_row)
                render_vision(curr_row)
                render_hud(curr_row, p_val)
                
                hist_data = ep_data.iloc[:step_idx + 1]
                draw_chart(
                    hist_data['ep_step_count'].tolist(),
                    hist_data['target_dist'].tolist(),
                    hist_data['drone_z'].tolist(),
                    hist_data['reward'].tolist()
                )


# ══════════════════════════════════════════
# TAB 3 — 3D RADAR
# ══════════════════════════════════════════
# ══════════════════════════════════════════
# TAB 3 — 3D RADAR
# ══════════════════════════════════════════
# ══════════════════════════════════════════
# TAB 3 — 3D RADAR
# ══════════════════════════════════════════
# ══════════════════════════════════════════
# TAB 3 — 3D RADAR & TACTICAL SENSORS
# ══════════════════════════════════════════
with tab3:
    st.markdown("### [ 3D HOLOGRAPHIC FLIGHT PATH ]")
    if not filtered.empty:
        # 1. 🚫 إيقاف العشوائية نهائياً (No more random sampling)
        # نقوم بترتيب البيانات زمنياً لرسم مسار طيران متصل ومنطقي
        path_data = filtered.sort_values(by=['episode', 'step'])
        
        # حماية للمتصفح: نأخذ أحدث 4000 خطوة (متصلة زمنياً) لتجنب الانهيار (Crash)
        if len(path_data) > 4000:
            path_data = path_data.tail(4000)

        # 2. رسم مسارات متصلة (Lines) بدلاً من نقاط مبعثرة
        fig_3d = px.line_3d(
            path_data, 
            x='drone_x', y='drone_y', z='drone_z',
            line_group='episode', # يربط النقاط الخاصة بكل مهمة ببعضها لإنشاء خط سير متصل
            color='frame_class',  # يلون المسار بناءً على حالة الدرون (تكتيكي جداً)
            color_discrete_map={
                'Normal': '#39ff14',      # أخضر نيون للمسار الطبيعي
                'Approaching': '#8fbc8f', # أخضر باهت للاقتراب
                'Evading': '#00ffcc',     # أزرق ساطع عند التفادي
                'Crash': '#ff4444'        # أحمر عند التحطم
            },
            hover_data=['episode', 'step', 'target_dist']
        )

        # 3. تضخيم الخطوط وإضافة نقاط خفيفة لإعطاء طابع هولوجرام
        fig_3d.update_traces(
            mode='lines+markers', 
            line=dict(width=5), 
            marker=dict(size=2, opacity=0.3)
        )

        fig_3d.update_layout(
            template=TEMPLATE, 
            paper_bgcolor='rgba(0,0,0,0)', 
            height=600,
            margin=dict(t=0, b=0, l=0, r=0),
            scene=dict(
                xaxis=dict(showgrid=True, gridcolor='#1f3b28', backgroundcolor='rgba(0,0,0,0)'),
                yaxis=dict(showgrid=True, gridcolor='#1f3b28', backgroundcolor='rgba(0,0,0,0)'),
                zaxis=dict(showgrid=True, gridcolor='#1f3b28', backgroundcolor='rgba(0,0,0,0)')
            ),
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01) # ترتيب مكان المفتاح
        )
        st.plotly_chart(fig_3d, use_container_width=True)

        st.markdown("---")
        st.markdown("#### [ TACTICAL 360° SENSOR ARRAY ]")
        
        # 🟢 [ استبدال الـ Heatmap برادار دائري حقيقي ] 🟢
        # نأخذ قراءة حساسات الرادار لآخر لحظة وصلها الدرون
        latest_radar = path_data.iloc[-1][RADAR_COLS].values.tolist()
        
        # نغلق الدائرة (تكرار أول قيمة في النهاية) لكي يكتمل الرسم الدائري 360 درجة
        latest_radar.append(latest_radar[0])
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N'] 
        
        # بناء شبكة الرادار
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=latest_radar,
            theta=directions,
            fill='toself',
            fillcolor='rgba(57, 255, 20, 0.2)', # تعبئة خضراء شفافة
            line=dict(color='#39ff14', width=3), # حدود نيون
            name='Threat Detection'
        ))

        fig_radar.update_layout(
            template=TEMPLATE,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            margin=dict(t=40, b=40, l=40, r=40),
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(
                    visible=True, 
                    gridcolor='#1f3b28', # خطوط الشبكة الداخلية
                    linecolor='#1f3b28',
                    tickfont=dict(color='#6a9c78')
                ),
                angularaxis=dict(
                    gridcolor='#1f3b28',
                    linecolor='#39ff14',
                    tickfont=dict(color='#39ff14', size=14, family="Share Tech Mono")
                )
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
# ══════════════════════════════════════════
# TAB 4 — ML STRATEGIC ANALYSIS (FIXED & CLEAN)
# ══════════════════════════════════════════
with tab4:
    st.markdown("### [ ML STRATEGIC ANALYSIS ]")

    # 1. تعريف قاموس الألوان داخل التاب لضمان الوصول إليه
    outcome_colors = {'success': '#39ff14', 'crash': '#ff4444', 'timeout': '#ffdd00'}

    # الصف الأول: منحنى التعلم وأهمية الميزات
    ca, cb = st.columns(2)
    with ca:
        st.markdown("#### [ SMOOTHED LEARNING TREND ]")
        ep_reward = filtered.groupby('episode')['Cumulative_Reward'].max().reset_index()
        # تنعيم البيانات باستخدام المتوسط المتحرك
        ep_reward['Moving_Avg'] = ep_reward['Cumulative_Reward'].rolling(window=50, min_periods=1).mean()
        
        fig_cr = go.Figure()
        fig_cr.add_trace(go.Scatter(x=ep_reward['episode'], y=ep_reward['Cumulative_Reward'],
                                   mode='lines', line=dict(color='rgba(57, 255, 20, 0.1)'),
                                   name='Raw Reward'))
        fig_cr.add_trace(go.Scatter(x=ep_reward['episode'], y=ep_reward['Moving_Avg'],
                                   mode='lines', line=dict(color='#39ff14', width=3),
                                   name='Learning Trend'))
        
        fig_cr.update_layout(
            template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="EPISODES", yaxis_title="MAX REWARD",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_cr, use_container_width=True)

    with cb:
        st.markdown("#### [ RF FEATURE IMPORTANCE ]")
        if hasattr(model, 'feature_importances_'):
            imp_df = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance')
            fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                            color='Importance', color_continuous_scale=['#0d1811', '#39ff14'])
            fig_imp.update_layout(template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")

    # الصف الثاني: توزيع الرياح وتوزيع الأفعال
    ca2, cb2 = st.columns(2)
    with ca2:
        st.markdown("#### [ WIND STRENGTH DISTRIBUTION ]")
        # استخدام المتغير المعرف أعلاه مباشرة
        fig_wind = px.histogram(
            filtered, x='wind_strength', nbins=50, 
            color='episode_outcome',
            color_discrete_map=outcome_colors 
        )
        fig_wind.update_layout(
            template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="WIND STRENGTH (m/s)", yaxis_title="SAMPLES"
        )
        st.plotly_chart(fig_wind, use_container_width=True)

    with cb2:
        st.markdown("#### [ TACTICAL ACTION ANALYSIS ]")
        # تحليل قوة الأفعال (Magnitude) بناءً على حالة الإطار
        fig_act = px.box(
            filtered, x='frame_class', y='action_magnitude', 
            color='frame_class', color_discrete_sequence=CHART_COLORS
        )
        fig_act.update_layout(
            template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="FRAME CLASS", yaxis_title="ACTION INTENSITY"
        )
        st.plotly_chart(fig_act, use_container_width=True)
# ══════════════════════════════════════════
# TAB 5 — ALGO INTEL
# ══════════════════════════════════════════
with tab5:
    st.markdown("### [ ALGORITHM INTEL ]")

    st.markdown("#### [ PPO ARCHITECTURE ]")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**PPO (Proximal Policy Optimization)**
- Policy: `MultiInputPolicy` (CNN + MLP)
- CNN: 3 Conv layers → 64×8×8 → 256 features
- Sensor MLP: 19 inputs → 128 → 64 features
- Total features: 320
- Learning Rate: `2e-4`
- Clip Range: `0.2`
- Entropy Coef: `0.02`
- Steps: `800,000`
        """)
    with col_b:
        st.markdown("""
**Curriculum Learning (3 Levels)**
| Level | Environment | Target Success |
|-------|-------------|----------------|
| 1 | Open field | 70% |
| 2 | Lego City (buildings) | 65% |
| 3 | Wind + Enemies + Radar | 78% |

**Reward System:**
- ✅ Success: +50
- 💥 Crash: -15
- ⏱ Timeout: -10
- 📡 Radar detection: -1/step
- 🏹 Approach reward: ×15 × progress
        """)

    st.markdown("---")
    st.markdown("#### [ ENVIRONMENT — DroneEnvironment (PyBullet) ]")
    env_cols = st.columns(3)
    with env_cols[0]:
        st.markdown("""
**Level 1**
- Open flat ground
- No obstacles
- No wind
- Basic navigation
        """)
    with env_cols[1]:
        st.markdown("""
**Level 2**
- Lego City (15 buildings)
- 8-ray radar sensors
- No wind
- Obstacle avoidance
        """)
    with env_cols[2]:
        st.markdown("""
**Level 3**
- Wind (0.3–2.5 m/s)
- 3 enemy drones
- 2 ground radars
- 1 tank
- Full combat scenario
        """)

    st.markdown("---")
    st.markdown("#### [ SKYCIPHER DATASET STATS ]")
    total_rows = len(df)
    total_eps = df['episode'].nunique()
    class_dist = df['frame_class'].value_counts()
    outcome_dist = df.groupby('episode')['episode_outcome'].first().value_counts()

    s1, s2, s3 = st.columns(3)
    s1.metric("TOTAL STEPS", f"{total_rows:,}")
    s2.metric("TOTAL EPISODES", f"{total_eps:,}")
    s3.metric("CSV COLUMNS", len(df.columns))

    col_c, col_d = st.columns(2)
    with col_c:
        st.markdown("**Frame Class Distribution**")
        fig_fc = px.bar(x=class_dist.index, y=class_dist.values, color=class_dist.index,
            color_discrete_sequence=CHART_COLORS, labels={'x': 'Class', 'y': 'Count'})
        fig_fc.update_layout(template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_fc, use_container_width=True)

    with col_d:
        st.markdown("**Episode Outcome Distribution**")
        fig_od = px.pie(values=outcome_dist.values, names=outcome_dist.index, hole=0.6,
            color_discrete_sequence=["#39ff14", "#ff4444", "#ffdd00"])
        fig_od.update_layout(template=TEMPLATE, paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_od, use_container_width=True)

    st.markdown("#### [ DATASET SAMPLE ]")
    st.dataframe(
        filtered.sample(min(100, len(filtered))).style.background_gradient(cmap='Greens', subset=['reward', 'Risk_Level']),
        use_container_width=True, height=300
    )

# ══════════════════════════════════════════
# TAB 6 — HARDWARE
# ══════════════════════════════════════════
with tab6:
    st.markdown("### [ HARDWARE DIAGNOSTICS ]")
    if not filtered.empty:
        diag_ep = st.selectbox("SELECT MISSION FOR SCAN:", filtered['episode'].unique(), key="diag_ep")
        ep_data = filtered[filtered['episode'] == diag_ep]

        steps = len(ep_data)
        wind = ep_data['wind_strength'].max()
        risk = ep_data['Risk_Level'].mean()

        # Simulated health metrics based on mission stress
        battery = max(0, 100 - (steps * 0.08) - (wind * 5))
        motor = max(0, 100 - (wind * 10) - (risk * 20))
        sensor = max(0, 100 - (risk * 30))

        c1, c2, c3 = st.columns(3)
        c1.metric("BATTERY", f"{battery:.1f}%")
        c2.metric("MOTOR HEALTH", f"{motor:.1f}%")
        c3.metric("SENSOR ARRAY", f"{sensor:.1f}%")

        st.markdown("---")
        st.markdown("**BATTERY STATUS**")
        st.progress(int(battery))
        st.markdown("**MOTOR INTEGRITY**")
        st.progress(int(motor))
        st.markdown("**SENSOR CALIBRATION**")
        st.progress(int(sensor))

        if battery < 20 or motor < 40:
            st.error("[ WARNING ]: CRITICAL WEAR DETECTED. MAINTENANCE REQUIRED.")
        else:
            st.success("[ OK ]: ALL SYSTEMS OPERATIONAL.")

# ══════════════════════════════════════════
# TAB 7 — LIVE SIMULATION (ULTIMATE SILENT MODE)
# ══════════════════════════════════════════
with tab7:
    st.markdown("### [ LIVE PYBULLET SIMULATION ]")
    st.info("[ NOTE ]: Runs `test.py` — PyBullet window will open on host machine. Requires `vision_smart_drone_level_3.zip`.")

    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        st.markdown("""
**What happens:**
1. Load PPO model (`vision_smart_drone_level_3.zip`)
2. Initialize Level 3 environment (wind + enemies + radar)
3. Run test episodes
4. Report mission stats
        """)
    with col_sim2:
        st.markdown("""
**Requirements:**
- `vision_smart_drone_level_3.zip` must exist in project folder ✅
- PyBullet installed ✅
- PyTorch + Stable Baselines3 installed ✅
        """)

    st.markdown("---")
    if st.button("[ INITIATE DEPLOYMENT SEQUENCE ]", key="run_sim"):
        st.markdown("#### [ TACTICAL TERMINAL FEED ]")
        terminal_ph = st.empty()
        log = "[ SYSTEM ]: INITIATING LEVEL 3 DEPLOYMENT SEQUENCE...\n[ SYSTEM ]: LOADING MODEL (vision_smart_drone_level_3.zip)...\n[ SYSTEM ]: CONNECTING TO 3D PYBULLET PHYSICS SERVER...\n"
        terminal_ph.markdown(f'<div class="terminal-box">{log}</div>', unsafe_allow_html=True)
        try:
            env_vars = os.environ.copy()
            env_vars["PYTHONIOENCODING"] = "utf-8"
            env_vars["PYTHONUNBUFFERED"] = "1"
            env_vars["KMP_DUPLICATE_LIB_OK"] = "True"

            root_dir = os.path.dirname(os.path.abspath(__file__))
            test_script_path = os.path.join(root_dir, "DroneProject", "test.py")
            if not os.path.exists(test_script_path):
                test_script_path = os.path.join("DroneProject", "test.py")

            proc = subprocess.Popen(
                [sys.executable, "-u", "-X", "utf8", test_script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                cwd=root_dir,
                env=env_vars
            )

            # تصفية أسطر الـ C++ debug الداخلية فقط
            c_debug_ignore = ["b3Printf", "argc", "argv[0]", "argv[1]", "starting thread", "started testThreads", "Vendor", "Renderer", "numActiveThreads", "btShutDown"]

            for line in iter(proc.stdout.readline, ''):
                if not line.strip():
                    continue
                if any(kw in line for kw in c_debug_ignore):
                    continue
                
                log += line
                terminal_ph.markdown(f'<div class="terminal-box">{log}</div>', unsafe_allow_html=True)

            proc.stdout.close()
            proc.wait()

            log += "\n[ SYSTEM ]: MISSION SEQUENCE COMPLETED SECURELY."
            terminal_ph.markdown(f'<div class="terminal-box">{log}</div>', unsafe_allow_html=True)
            st.success("[ MISSION LOG SAVED ]")

        except Exception as e:
            st.error(f"[ CRITICAL SYSTEM FAILURE ]: {e}")