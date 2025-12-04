import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="SecureMedAI: Federated Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. PROFESSIONAL STYLING (CSS) ---
# White background, cool blues/greys, no emojis in text, sharp fonts.
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #FFFFFF;
        color: #0E1117;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8F9FB;
        border-right: 1px solid #E6E9EF;
    }
    /* Headings */
    h1, h2, h3 {
        color: #1F2937;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 600;
    }
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #2563EB; /* Cool Blue */
    }
    /* Custom Card Style */
    .card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #E5E7EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    /* Button */
    .stButton > button {
        background-color: #2563EB;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton > button:hover {
        background-color: #1D4ED8;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS FOR DUMMY DATA ---

def get_dummy_loss_curve():
    """Generates a realistic-looking validation loss curve dropping over rounds."""
    rounds = np.arange(1, 51)
    # Exponential decay + noise
    loss = 0.8 * np.exp(-0.1 * rounds) + 0.1 + np.random.normal(0, 0.005, 50)
    dice = 1.0 - loss  # Inverse for Dice Score
    df = pd.DataFrame({
        "Round": rounds,
        "Validation Loss": loss,
        "Global Dice Score": dice
    })
    return df

def get_dummy_blockchain_ledger():
    """Generates a professional looking ledger table."""
    data = []
    hashes = [
        "0x7f83b...1a2b", "0x3c91d...4e5f", "0x9a12b...6c7d", 
        "0x1b2c3...8d9e", "0x5e6f7...0a1b"
    ]
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=5, freq='H')
    
    for i in range(5):
        data.append({
            "Block Height": 10045 + i,
            "Timestamp": timestamps[i].strftime("%Y-%m-%d %H:%M:%S UTC"),
            "Contributor": f"Node_Hospital_{np.random.randint(1, 5)}",
            "Model Hash (IPFS)": hashes[i],
            "Validation Score": f"{0.85 + (i * 0.02):.4f}",
            "Status": "Verified ✅" 
        })
    return pd.DataFrame(data).sort_values(by="Block Height", ascending=False)

def generate_synthetic_brain_mri():
    """Generates a synthetic 2D MRI slice with a 'tumor'."""
    size = 128
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Brain shape (ellipse)
    brain_mask = (X**2 + Y**2) < 0.8
    brain = np.exp(-(X**2 + Y**2)) * brain_mask
    
    # Tumor blob (bright spot)
    tumor_x, tumor_y = 0.3, 0.3
    tumor = 0.8 * np.exp(-((X-tumor_x)**2 + (Y-tumor_y)**2) * 50)
    
    # Combined Image (MRI T1ce)
    mri_img = brain + tumor + np.random.normal(0, 0.05, (size, size))
    
    # Ground Truth Mask
    gt_mask = (tumor > 0.2).astype(float)
    
    # Prediction (slightly imperfect)
    pred_mask = (tumor > 0.15).astype(float) * np.random.choice([0, 1], size=(size, size), p=[0.05, 0.95])
    
    return mri_img, gt_mask, pred_mask

# --- 4. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("SecureMedAI")
    st.markdown("---")
    selected_page = st.radio(
        "Navigation",
        ["Executive Dashboard", "Federated Training", "Blockchain Ledger", "Model Inference"]
    )
    
    st.markdown("---")
    st.caption("System Status")
    st.info("System Online\nConnected to: 4 Clients\nIPFS Node: Active")

# --- 5. PAGE CONTENT ---

if selected_page == "Executive Dashboard":
    st.title("System Overview")
    st.markdown("Real-time monitoring of the Decentralized Federated Learning Network.")
    
    # Top Metrics Row
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Global Dice Score", "98.2%", "+2.1%")
    with c2: st.metric("Active Nodes", "4 / 4", "Stable")
    with c3: st.metric("Current Round", "50", "Finalized")
    with c4: st.metric("Blockchain Height", "10,049", "+1 Block")
    
    st.markdown("---")
    
    # Main Dashboard Layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Global Model Convergence")
        df_loss = get_dummy_loss_curve()
        st.line_chart(df_loss, x="Round", y="Global Dice Score", color="#2563EB")
        st.caption("Figure 1: Aggregated validation Dice score across 50 federated rounds showing steady convergence.")
        
    with col_right:
        st.subheader("Data Distribution (Non-IID)")
        # Simulating data skew
        client_data = pd.DataFrame({
            "Client": ["Hosp A", "Hosp B", "Hosp C", "Hosp D"],
            "Samples": [150, 45, 200, 80]
        })
        st.bar_chart(client_data, x="Client", y="Samples", color="#64748B")
        st.caption("Figure 2: Non-IID distribution of training samples across consortium hospitals.")

elif selected_page == "Federated Training":
    st.title("Federated Learning Console")
    st.markdown("Monitor local training progress and global aggregation.")
    
    tab1, tab2 = st.tabs(["Global Aggregation", "Client Status"])
    
    with tab1:
        st.subheader("Round 50 Aggregation Logs")
        with st.container():
            st.markdown("""
            <div style="background-color: #f1f5f9; padding: 10px; border-radius: 5px; font-family: monospace;">
            [14:02:01] Aggregator: Round 50 initiated.<br>
            [14:02:05] Client_1: Model uploaded to IPFS (Qm...8d9e).<br>
            [14:02:06] Client_3: Model uploaded to IPFS (Qm...4e5f).<br>
            [14:02:08] Smart Contract: Received 4/4 updates.<br>
            [14:02:10] Aggregator: Performing Weighted FedAvg...<br>
            [14:02:15] Security: Krum Algorithm Validation Passed.<br>
            [14:02:18] Blockchain: Global Model Block #10049 Mined.<br>
            [14:02:18] Status: <b>ROUND COMPLETE</b>.
            </div>
            """, unsafe_allow_html=True)
            
    with tab2:
        cols = st.columns(4)
        for i, col in enumerate(cols):
            with col:
                st.markdown(f"### Client {i+1}")
                st.progress(100)
                st.caption("Status: Idle")
                st.text(f"Loss: {0.05 + (i*0.01):.4f}")
                st.text(f"Latency: {120 + (i*15)}ms")

elif selected_page == "Blockchain Ledger":
    st.title("Immutable Audit Trail")
    st.markdown("Verifiable record of all model updates anchored on Ethereum Private Network.")
    
    ledger_df = get_dummy_blockchain_ledger()
    
    # Styled Table
    st.dataframe(
        ledger_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Block Height": st.column_config.NumberColumn(format="%d"),
            "Validation Score": st.column_config.ProgressColumn(
                "Quality (Dice)", min_value=0, max_value=1, format="%.2f"
            ),
        }
    )
    
    st.subheader("Smart Contract details")
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Contract Address", value="0x71C7656EC7ab88b098defB751B7401B5f6d8976F", disabled=True)
    with c2:
        st.text_input("Consensus Mechanism", value="Proof of Authority (PoA)", disabled=True)

elif selected_page == "Model Inference":
    st.title("Diagnostic Inference")
    st.markdown("Swin-UNETR Segmentation on held-out validation data.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### Controls")
        patient_id = st.selectbox("Select Patient ID", ["BraTS20_Validation_001", "BraTS20_Validation_002", "BraTS20_Validation_005"])
        modality = st.selectbox("View Modality", ["T1ce (Contrast Enhanced)", "FLAIR", "T2"])
        
        if st.button("Run Segmentation"):
            with st.spinner("Downloading Global Model from IPFS..."):
                time.sleep(1.0)
            with st.spinner("Running Inference (Swin-UNETR)..."):
                time.sleep(1.5)
            st.success("Segmentation Complete")
            
            # Generate Dummy Viz
            mri, gt, pred = generate_synthetic_brain_mri()
            
            # Show Results
            st.markdown("---")
            st.metric("Dice Score", "0.984")
            st.metric("Tumor Volume", "14.2 cc")

    with col2:
        if 'mri' not in locals():
            mri, gt, pred = generate_synthetic_brain_mri()
            
        st.markdown(f"### Visual Results: {patient_id}")
        
        c_img1, c_img2, c_img3 = st.columns(3)
        
        with c_img1:
            st.image(mri, caption=f"Input: {modality}", clamp=True, use_container_width=True)
        
        with c_img2:
            st.image(gt, caption="Ground Truth Mask", clamp=True, use_container_width=True)
            
        with c_img3:
            st.image(pred, caption="Model Prediction", clamp=True, use_container_width=True)
            
        st.info("The segmentation map highlights the Enhancing Tumor (ET) core with high precision.")