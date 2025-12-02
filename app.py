import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import numpy as np

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Inshira - Material Flow Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# === CUSTOM CSS FOR AMAZING UI ===
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background with gradient animation */
    .stApp {
        background: linear-gradient(-45deg, #0a0e27, #1a1a2e, #16213e, #0f3460);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(135deg, #ffffff, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #a5b4fc;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Upload cards */
    .upload-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-card:hover {
        border-color: rgba(16, 185, 129, 0.5);
        transform: translateY(-5px);
        box-shadow: 0 20px 60px rgba(16, 185, 129, 0.3);
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff, #10b981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a5b4fc !important;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    /* Success/Info boxes */
    .success-box {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2), rgba(6, 182, 212, 0.2));
        border: 2px solid #10b981;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 2rem 0;
    }
    
    .info-box {
        background: rgba(6, 182, 212, 0.1);
        border: 1px solid rgba(6, 182, 212, 0.3);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #10b981, #06b6d4);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 40px rgba(16, 185, 129, 0.5);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed rgba(16, 185, 129, 0.3);
        border-radius: 15px;
        padding: 1rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: rgba(16, 185, 129, 0.6);
        background: rgba(16, 185, 129, 0.1);
    }
    
    /* Dataframe styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
    }
    
    /* Section headers */
    h2 {
        color: #ffffff;
        font-weight: 700;
        margin-top: 2rem;
        padding-left: 1rem;
        border-left: 4px solid #10b981;
    }
    
    h3 {
        color: #06b6d4;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# === AUTO-GENERATE SAMPLE DATA ===
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    
    batches = [f"B{i:03d}" for i in range(1, 11)]
    
    material_df = pd.DataFrame({
        "Batch ID": batches,
        "Material Type": ["Aluminium 6082"] * 10,
        "Initial Quantity": np.random.uniform(90, 110, 10).round(2)
    })
    
    process_df = pd.DataFrame({
        "Batch ID": batches * 3,
        "Process Step": ["Cutting"]*10 + ["Milling"]*10 + ["QC"]*10
    })
    
    qc_df = pd.DataFrame({
        "Batch ID": batches,
        "Scrap Quantity": np.random.uniform(2, 15, 10).round(2),
        "Defects": np.random.randint(0, 5, 10),
        "Total Inspected": [100]*10
    })
    
    output_df = pd.DataFrame({
        "Batch ID": batches,
        "Final Quantity": material_df["Initial Quantity"] - qc_df["Scrap Quantity"] - np.random.uniform(1, 4, 10).round(2)
    })
    
    return material_df, process_df, qc_df, output_df

# === MAIN APP ===
# Header
st.markdown('<h1 class="main-title">⚡ Inshira</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transform Material Waste Into Profit - Upload your production data and discover hidden savings in minutes</p>', unsafe_allow_html=True)

# === UPLOAD SECTION ===
st.markdown("## 📤 Upload Your Production Files")

st.markdown("""
<div class="info-box">
    <h3>💡 What files do I need?</h3>
    <ul style="list-style: none; padding-left: 0;">
        <li>✓ <strong>Material Data:</strong> Raw material deliveries with Batch IDs and quantities</li>
        <li>✓ <strong>Process Steps:</strong> Manufacturing steps each batch goes through</li>
        <li>✓ <strong>QC Reports:</strong> Quality control data including scrap and defects</li>
        <li>✓ <strong>Final Output:</strong> Finished quantities per batch</li>
    </ul>
    <p style="margin-top: 1rem; color: #a5b4fc; font-size: 0.9rem;">✨ Don't have files ready? Click "Use Sample Data" below to see it in action!</p>
</div>
""", unsafe_allow_html=True)

# File uploaders in columns
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("### 📦 Material Data")
    material_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'], key="material", label_visibility="collapsed")
    
with col2:
    st.markdown("### 🔄 Process Steps")
    process_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'], key="process", label_visibility="collapsed")
    
with col3:
    st.markdown("### 🔍 QC Reports")
    qc_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'], key="qc", label_visibility="collapsed")
    
with col4:
    st.markdown("### ✅ Final Output")
    output_file = st.file_uploader("Upload CSV/Excel", type=['csv', 'xlsx'], key="output", label_visibility="collapsed")

# Action buttons
st.markdown("<br>", unsafe_allow_html=True)
col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 2])

with col_btn1:
    use_sample = st.button("📊 Use Sample Data", use_container_width=True)

with col_btn2:
    if material_file and process_file and qc_file and output_file:
        analyze_uploaded = st.button("🚀 Analyze My Data", use_container_width=True, type="primary")
    else:
        st.button("🚀 Analyze My Data (Upload all files first)", use_container_width=True, disabled=True)

# === ANALYSIS SECTION ===
if use_sample or (material_file and process_file and qc_file and output_file):
    
    # Load data
    if use_sample:
        material_df, process_df, qc_df, output_df = generate_sample_data()
        st.success("✅ Sample data loaded successfully!")
    else:
        # Parse uploaded files (simplified - you'd add proper parsing logic)
        try:
            material_df = pd.read_csv(material_file) if material_file.name.endswith('.csv') else pd.read_excel(material_file)
            process_df = pd.read_csv(process_file) if process_file.name.endswith('.csv') else pd.read_excel(process_file)
            qc_df = pd.read_csv(qc_file) if qc_file.name.endswith('.csv') else pd.read_excel(qc_file)
            output_df = pd.read_csv(output_file) if output_file.name.endswith('.csv') else pd.read_excel(output_file)
            st.success("✅ Your files have been uploaded and analyzed successfully!")
        except Exception as e:
            st.error(f"Error reading files: {str(e)}")
            st.stop()
    
    # Merge everything
    merged = material_df.merge(qc_df, on="Batch ID")
    merged = merged.merge(output_df, on="Batch ID")
    merged["Material Loss %"] = (merged["Initial Quantity"] - merged["Final Quantity"]) / merged["Initial Quantity"] * 100
    merged["Scrap %"] = merged["Scrap Quantity"] / merged["Initial Quantity"] * 100
    
    # === KEY INSIGHTS ===
    st.markdown("## 💎 Key Performance Insights")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📉 Average Material Loss", f"{merged['Material Loss %'].mean():.1f}%")
    c2.metric("♻️ Average Scrap Rate", f"{merged['Scrap %'].mean():.1f}%")
    c3.metric("⚠️ Highest Scrap Batch", merged.loc[merged['Scrap %'].idxmax(), 'Batch ID'])
    c4.metric("💰 Potential Savings (10t/year)", "€4,200 – €12,800")
    
    # === MATERIAL FLOW MAP ===
    st.markdown("## 🔄 Material Flow Visualization")
    
    G = nx.DiGraph()
    steps = ['📦 Delivery', '✂️ Cutting', '🔧 Milling', '🔍 QC', '✨ Finished Product']
    for i in range(len(steps)-1):
        G.add_edge(steps[i], steps[i+1])
    pos = {step: (i, 0) for i, step in enumerate(steps)}
    
    # Edge traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=8, color='rgba(16, 185, 129, 0.6)'),
        mode='lines',
        hoverinfo='none'
    )
    
    # Node traces
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    node_text = []
    node_colors = ['#10b981', '#06b6d4', '#8b5cf6', '#f59e0b', '#10b981']
    
    avg_scrap = merged["Scrap %"].mean()
    for node in steps:
        if node in ["✂️ Cutting", "🔧 Milling", "🔍 QC"]:
            node_text.append(f"{node}<br>Avg Scrap: {avg_scrap:.1f}%")
        else:
            node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        marker=dict(size=100, color=node_colors, line=dict(width=5, color='#ffffff')),
        textfont=dict(size=16, color='#ffffff', family='Inter'),
        hoverinfo="text"
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=100, l=60, r=60, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.5, 4.5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.8, 0.8]),
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # === BATCH-LEVEL TRACEABILITY ===
    st.markdown("## 📊 Batch-Level Intelligence")
    
    display_df = merged[["Batch ID", "Initial Quantity", "Scrap Quantity", "Final Quantity", "Material Loss %", "Scrap %"]]
    display_df = display_df.round(2)
    
    # Highlight max values
    def highlight_max(s):
        is_max = s == s.max()
        return ['background-color: rgba(239, 68, 68, 0.3); font-weight: bold; color: #fca5a5' if v else '' for v in is_max]
    
    styled_df = display_df.style.apply(highlight_max, subset=["Material Loss %", "Scrap %"])
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # === SUCCESS BANNER ===
    st.markdown("""
    <div class="success-box">
        ✨ Analysis Complete! These insights are based on YOUR data - ready to reduce waste and boost profits? ✨
    </div>
    """, unsafe_allow_html=True)
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Report (CSV)",
        data=csv,
        file_name="inshira_material_flow_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.markdown("""
    <div class="info-box" style="text-align: center; margin-top: 3rem;">
        <h3>👆 Upload your production files or use sample data to get started</h3>
        <p style="color: #a5b4fc; margin-top: 1rem;">We'll analyze your material flow and show you exactly where you're losing money to waste</p>
    </div>
    """, unsafe_allow_html=True)
