import streamlit as st
import pickle
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Crypto Forensics AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. THEME LOGIC ---
# We use Session State to remember the user's choice
if 'theme' not in st.session_state:
    st.session_state.theme = 'Dark'  # Default


# Function to toggle theme
def toggle_theme():
    if st.session_state.theme == 'Dark':
        st.session_state.theme = 'Light'
    else:
        st.session_state.theme = 'Dark'


# Sidebar Toggle
with st.sidebar:
    st.title("Settings")
    # This button triggers a rerun with the new state
    st.button(f"Switch to {('Light' if st.session_state.theme == 'Dark' else 'Dark')} Mode", on_click=toggle_theme)
    st.caption(f"Current Theme: **{st.session_state.theme}**")

# Define Color Palette based on selection
if st.session_state.theme == 'Dark':
    bg_color = '#0e1117'
    card_bg = 'rgba(255, 255, 255, 0.05)'
    text_color = '#ffffff'
    border_color = 'rgba(255, 255, 255, 0.2)'
    chart_template = 'plotly_dark'
else:
    bg_color = '#ffffff'
    card_bg = 'rgba(0, 0, 0, 0.05)'
    text_color = '#000000'
    border_color = 'rgba(0, 0, 0, 0.1)'
    chart_template = 'plotly_white'

# --- 3. DYNAMIC CSS INJECTION ---
# We inject CSS variables that change based on the Python variable above
st.markdown(f"""
    <style>
    /* Force Background Color (Optional - usually better to let Streamlit handle main bg) */
    /* .stApp {{ background-color: {bg_color}; }} */

    /* 1. Card Styling & Hover */
    div[data-testid="stMetric"] {{
        background-color: {card_bg};
        border: 1px solid {border_color};
        color: {text_color};
        padding: 15px;
        border-radius: 10px;
        transition: all 0.3s ease-in-out;
    }}

    div[data-testid="stMetric"]:hover {{
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border-color: #FF4B4B;
    }}

    /* 2. Text Colors */
    h1, h2, h3, p, span {{
        color: {text_color} !important;
    }}

    /* 3. Button Hover */
    div.stButton > button {{
        transition: all 0.2s;
    }}
    div.stButton > button:hover {{
        transform: scale(1.05);
    }}
    </style>
    """, unsafe_allow_html=True)


# --- 4. LOAD ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    try:
        with open('unsupervised_pack.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


artifacts = load_artifacts()

if not artifacts:
    st.error(" Error: `unsupervised_pack.pkl` not found. Run training script first.")
    st.stop()

kmeans = artifacts['kmeans']
scaler = artifacts['scaler']
pca = artifacts['pca']
viz_data = artifacts['viz_data']


# --- 5. LOGIC ---
def get_features(address, chain_input):
    entropy = 0
    if address:
        for x in set(address):
            p_x = address.count(x) / len(address)
            entropy += - p_x * math.log2(p_x)

    btc_type = 3
    if str(address).startswith('1'):
        btc_type = 0
    elif str(address).startswith('3'):
        btc_type = 1
    elif str(address).startswith('bc1'):
        btc_type = 2

    chain_val = 1 if chain_input == 'Ethereum' else 0

    return pd.DataFrame([[len(address), entropy, chain_val, btc_type]],
                        columns=['length', 'entropy', 'chain', 'btc_type'])


# --- 6. UI LAYOUT ---
st.title("🛡️ Crypto Wallet Forensics")
st.markdown("**Interactive Anomaly Detection System**")
st.markdown("---")

# Input Section
col_input, col_btn = st.columns([3, 1])
with col_input:
    c1, c2 = st.columns([1, 3])
    with c1:
        chain_choice = st.selectbox("Network", ["Bitcoin", "Ethereum"], label_visibility="collapsed")
    with c2:
        user_address = st.text_input("Address", placeholder="Paste wallet address...", label_visibility="collapsed")
with col_btn:
    analyze_btn = st.button("🔍 Analyze Pattern", type="primary", use_container_width=True)

# --- 7. RESULTS ---
if analyze_btn:
    if user_address:
        with st.spinner("Analyzing structure..."):
            # Math
            raw_features = get_features(user_address, chain_choice)
            scaled_features = scaler.transform(raw_features)
            cluster_id = kmeans.predict(scaled_features)[0]
            pca_coords = pca.transform(scaled_features)
            user_x, user_y = pca_coords[0]

            # Metrics
            st.write("### 📊 Forensic Data")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Entropy", f"{raw_features['entropy'][0]:.2f}", delta="Complexity")
            m2.metric("Length", f"{raw_features['length'][0]}", delta="Chars")
            m3.metric("Cluster Group", f"#{cluster_id}", delta="Match")

            dist = math.sqrt(user_x ** 2 + user_y ** 2)
            status = "Normal" if dist < 3 else "Anomaly"
            color = "normal" if dist < 3 else "inverse"
            m4.metric("Deviation Score", f"{dist:.2f}", delta=status, delta_color=color)

            # Chart
            st.write("### 🗺️ Live Cluster Map")

            # Base Chart
            fig = px.scatter(
                viz_data, x='x', y='y', color='cluster',
                color_continuous_scale='Viridis', opacity=0.5,
                title="Latent Space Projection",
                template=chart_template  # USES THE THEME VARIABLE
            )

            # User Point
            fig.add_trace(go.Scatter(
                x=[user_x], y=[user_y], mode='markers',
                marker=dict(size=25, color='#FF4B4B', symbol='star', line=dict(width=2, color='white')),
                name='TARGET', hoverinfo="text",
                hovertext=f"Address: {user_address[:8]}...<br>Cluster: {cluster_id}"
            ))

            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=20, r=20, t=50, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👆 Enter an address to begin.")

# Footer
st.markdown("---")
st.markdown(
    f"<center style='color:{text_color}; opacity: 0.5'>AI Forensics Tool • {st.session_state.theme} Mode Active</center>",
    unsafe_allow_html=True)