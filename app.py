import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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

# === CONSTANTS AND HELPERS ===
REQUIRED_SCHEMAS = {
    "Material Data": ["Batch ID", "Initial Quantity"],
    "Process Steps": ["Batch ID", "Process Step"],
    "QC Reports": ["Batch ID", "Scrap Quantity"],
    "Final Output": ["Batch ID", "Final Quantity"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.strip() for col in df.columns]
    return df


def read_uploaded_table(file) -> pd.DataFrame:
    if file is None:
        raise ValueError("No file provided")
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)


def validate_schema(df: pd.DataFrame, dataset_name: str) -> list[str]:
    expected = REQUIRED_SCHEMAS[dataset_name]
    missing = [col for col in expected if col not in df.columns]
    return missing


@st.cache_data
def generate_sample_data():
    np.random.seed(42)

    batches = [f"B{i:03d}" for i in range(1, 11)]

    material_df = pd.DataFrame(
        {
            "Batch ID": batches,
            "Material Type": ["Aluminium 6082"] * 10,
            "Initial Quantity": np.random.uniform(90, 110, 10).round(2),
        }
    )

    process_df = pd.DataFrame(
        {
            "Batch ID": batches * 3,
            "Process Step": ["Cutting"] * 10 + ["Milling"] * 10 + ["QC"] * 10,
            "Step Order": list(range(1, 4)) * 10,
        }
    )

    qc_df = pd.DataFrame(
        {
            "Batch ID": batches,
            "Scrap Quantity": np.random.uniform(2, 15, 10).round(2),
            "Defects": np.random.randint(0, 5, 10),
            "Total Inspected": [100] * 10,
        }
    )

    output_df = pd.DataFrame(
        {
            "Batch ID": batches,
            "Final Quantity": material_df["Initial Quantity"]
            - qc_df["Scrap Quantity"]
            - np.random.uniform(1, 4, 10).round(2),
        }
    )

    return material_df, process_df, qc_df, output_df


@st.cache_data
def sample_templates() -> dict[str, str]:
    material, process, qc, output = generate_sample_data()
    return {
        "Material Data": material.to_csv(index=False),
        "Process Steps": process[["Batch ID", "Process Step", "Step Order"]].to_csv(index=False),
        "QC Reports": qc.to_csv(index=False),
        "Final Output": output.to_csv(index=False),
    }


def warn_missing_process_steps(process_df: pd.DataFrame | None) -> pd.DataFrame:
    if process_df is not None and not process_df.empty:
        return process_df

    st.warning(
        "Process steps file is missing or empty. Using default steps: Delivery → Cutting → Milling → QC → Finished Product."
    )
    batches = ["Unknown"]
    default_steps = ["Delivery", "Cutting", "Milling", "QC", "Finished Product"]
    return pd.DataFrame(
        {
            "Batch ID": batches * len(default_steps),
            "Process Step": default_steps,
            "Step Order": list(range(len(default_steps))),
        }
    )


def build_transition_table(process_df: pd.DataFrame) -> pd.DataFrame:
    order_col = "Step Order" if "Step Order" in process_df.columns else None
    transitions = []

    for batch_id, group in process_df.groupby("Batch ID"):
        ordered = group.sort_values(order_col) if order_col else group
        steps = ordered["Process Step"].tolist()
        for i in range(len(steps) - 1):
            transitions.append((steps[i], steps[i + 1], batch_id))

    if not transitions:
        return pd.DataFrame(columns=["source", "target", "Batch ID"])

    transition_df = pd.DataFrame(transitions, columns=["source", "target", "Batch ID"])
    return transition_df


def sankey_from_process(process_df: pd.DataFrame, merged: pd.DataFrame) -> go.Figure:
    transition_df = build_transition_table(process_df)
    if transition_df.empty:
        st.info("Not enough process steps to render a flow map.")
        return go.Figure()

    link_stats = (
        transition_df.groupby(["source", "target"])["Batch ID"]
        .agg(["nunique", list])
        .reset_index()
        .rename(columns={"nunique": "count", "list": "batches"})
    )

    scrap_lookup = merged.set_index("Batch ID")[["Scrap %", "Material Loss %"]]

    link_stats["scrap_pct"] = link_stats["batches"].apply(
        lambda ids: scrap_lookup.loc[scrap_lookup.index.intersection(ids)]["Scrap %"].mean()
    )
    link_stats["loss_pct"] = link_stats["batches"].apply(
        lambda ids: scrap_lookup.loc[scrap_lookup.index.intersection(ids)]["Material Loss %"].mean()
    )

    nodes = sorted(set(link_stats["source"]).union(set(link_stats["target"])))
    node_indices = {name: idx for idx, name in enumerate(nodes)}

    colors = ["rgba(16, 185, 129, 0.8)", "rgba(6, 182, 212, 0.8)", "rgba(139, 92, 246, 0.8)", "rgba(245, 158, 11, 0.8)"]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=20,
                    thickness=20,
                    line=dict(color="white", width=0.5),
                    label=nodes,
                    color=[colors[idx % len(colors)] for idx in range(len(nodes))],
                ),
                link=dict(
                    source=[node_indices[s] for s in link_stats["source"]],
                    target=[node_indices[t] for t in link_stats["target"]],
                    value=link_stats["count"],
                    color=[
                        f"rgba(239,68,68,{min(0.9, (loss or 0)/100 + 0.1)})"
                        for loss in link_stats["loss_pct"]
                    ],
                    hovertemplate=
                    "<b>%{source.label} → %{target.label}</b><br>"
                    "Batches: %{value}<br>"
                    "Avg scrap: %{customdata[0]:.2f}%<br>"
                    "Avg loss: %{customdata[1]:.2f}%<extra></extra>",
                    customdata=link_stats[["scrap_pct", "loss_pct"]].values,
                ),
            )
        ]
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=550,
    )
    return fig

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

template_map = sample_templates()
template_cols = st.columns(4)
for idx, (dataset, template) in enumerate(template_map.items()):
    with template_cols[idx]:
        st.download_button(
            label=f"📥 {dataset} Template",
            data=template,
            file_name=f"{dataset.replace(' ', '_').lower()}_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

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

    try:
        if use_sample:
            material_df, process_df, qc_df, output_df = generate_sample_data()
            st.success("✅ Sample data loaded successfully!")
        else:
            material_df = normalize_columns(read_uploaded_table(material_file))
            process_df = normalize_columns(read_uploaded_table(process_file))
            qc_df = normalize_columns(read_uploaded_table(qc_file))
            output_df = normalize_columns(read_uploaded_table(output_file))

        # Validate schemas
        errors = []
        for name, df in zip(
            ["Material Data", "Process Steps", "QC Reports", "Final Output"],
            [material_df, process_df, qc_df, output_df],
        ):
            missing = validate_schema(df, name)
            if missing:
                errors.append(f"{name}: missing columns {', '.join(missing)}")
        if errors:
            st.error("We couldn't process your files. Please fix the following issues:")
            for err in errors:
                st.markdown(f"- {err}")
            st.stop()

    except Exception as e:
        st.error(f"Error reading files: {str(e)}")
        st.stop()

    process_df = warn_missing_process_steps(process_df)

    # Merge everything
    merged = material_df.merge(qc_df, on="Batch ID", how="inner")
    merged = merged.merge(output_df, on="Batch ID", how="inner")
    merged["Material Loss %"] = (
        (merged["Initial Quantity"] - merged["Final Quantity"]) / merged["Initial Quantity"] * 100
    )
    merged["Scrap %"] = merged["Scrap Quantity"] / merged["Initial Quantity"] * 100

    # === KEY INSIGHTS ===
    st.markdown("## 💎 Key Performance Insights")

    avg_loss = merged["Material Loss %"].mean()
    avg_scrap = merged["Scrap %"].mean()
    worst_batch = merged.loc[merged["Scrap %"].idxmax(), "Batch ID"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📉 Average Material Loss", f"{avg_loss:.1f}%")
    c2.metric("♻️ Average Scrap Rate", f"{avg_scrap:.1f}%")
    c3.metric("⚠️ Highest Scrap Batch", worst_batch)

    material_cost = st.number_input("Material cost per unit (€/kg or equivalent)", min_value=0.0, value=1500.0)
    annual_batches = st.number_input("Annual batches", min_value=1, value=500, step=10)
    avg_initial = merged["Initial Quantity"].mean()
    annual_loss_units = avg_initial * (avg_scrap / 100) * annual_batches
    estimated_savings = annual_loss_units * material_cost
    c4.metric("💰 Potential Annual Savings", f"€{estimated_savings:,.0f}")

    # === MATERIAL FLOW MAP ===
    st.markdown("## 🔄 Material Flow Visualization")
    sankey_fig = sankey_from_process(process_df, merged)
    st.plotly_chart(sankey_fig, use_container_width=True)

    # === BENCHMARKING & ROI ===
    st.markdown("## 🎯 Benchmarks & ROI")
    reduction_target = st.slider("Target scrap reduction (%)", min_value=0, max_value=50, value=15, step=5)
    reduced_scrap_rate = max(avg_scrap - reduction_target, 0)
    annual_savings_if_reduced = avg_initial * (avg_scrap - reduced_scrap_rate) / 100 * annual_batches * material_cost

    bench_text = (
        "World-class (top 20%)" if avg_scrap < 3 else "Competitive (middle 50%)" if avg_scrap < 7 else "Needs attention (bottom 30%)"
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Current scrap benchmark", bench_text)
    col_b.metric("Annual scrap volume", f"{annual_loss_units:,.1f} units")
    col_c.metric("Savings if achieved target", f"€{annual_savings_if_reduced:,.0f}")

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
    kpi_summary = pd.DataFrame(
        {
            "Metric": [
                "Average Material Loss %",
                "Average Scrap %",
                "Potential Annual Savings (€)",
                "Target Scrap Reduction %",
                "Savings at Target (€)",
            ],
            "Value": [
                round(avg_loss, 2),
                round(avg_scrap, 2),
                round(estimated_savings, 2),
                reduction_target,
                round(annual_savings_if_reduced, 2),
            ],
        }
    )

    buffer = io.StringIO()
    buffer.write("KPI Summary\n")
    kpi_summary.to_csv(buffer, index=False)
    buffer.write("\nBatch Details\n")
    display_df.to_csv(buffer, index=False)
    csv = buffer.getvalue()
    st.download_button(
        label="📥 Download Full Report (CSV)",
        data=csv,
        file_name="inshira_material_flow_analysis.csv",
        mime="text/csv",
        use_container_width=True
    )

    # === FEEDBACK CAPTURE ===
    with st.expander("💬 Tell us what’s missing"):
        with st.form("feedback_form"):
            usefulness = st.radio("How useful was this analysis?", ["Very useful", "Somewhat useful", "Needs work"], index=1)
            missing = st.text_area("What would you change or add?", placeholder="List the decisions you can/can’t make with this dashboard…")
            contact = st.text_input("Your email (optional)")
            submitted = st.form_submit_button("Submit feedback")
            if submitted:
                feedback_row = pd.DataFrame(
                    [
                        {
                            "timestamp": datetime.utcnow().isoformat(),
                            "use_sample_data": bool(use_sample),
                            "usefulness": usefulness,
                            "comment": missing,
                            "contact": contact,
                            "avg_scrap_pct": avg_scrap,
                            "estimated_savings": estimated_savings,
                        }
                    ]
                )

                feedback_file = "feedback_responses.csv"
                header = not os.path.exists(feedback_file)
                feedback_row.to_csv(feedback_file, mode="a", header=header, index=False)
                st.success("Thanks for your feedback! We’ll use it to shape the product roadmap.")

else:
    st.markdown("""
    <div class="info-box" style="text-align: center; margin-top: 3rem;">
        <h3>👆 Upload your production files or use sample data to get started</h3>
        <p style="color: #a5b4fc; margin-top: 1rem;">We'll analyze your material flow and show you exactly where you're losing money to waste</p>
    </div>
    """, unsafe_allow_html=True)
