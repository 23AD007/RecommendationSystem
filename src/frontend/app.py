import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Packaging Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM THEME (UNIQUE – NOT STREAMLIT DEFAULT)
# --------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #0F172A;
    color: #E5E7EB;
    font-family: "Inter", sans-serif;
}

#MainMenu, footer, header {visibility: hidden;}

h1, h2, h3 {color: #D4A373;}

.stButton>button {
    background: linear-gradient(135deg,#40916C,#1B4332);
    color: white;
    border-radius: 999px;
    font-weight: 600;
    padding: 0.6rem 1.5rem;
}

.stMetric {
    background: linear-gradient(135deg,#1B4332,#40916C);
    padding: 1.2rem;
    border-radius: 16px;
}

.card {
    background:#111827;
    padding:1rem;
    border-radius:16px;
    border-left:5px solid #40916C;
    margin-bottom:0.6rem;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# BACKEND CONFIG
# --------------------------------------------------
BACKEND_URL = "http://localhost:5000/api/product/recommend-materials"
API_KEY = "packaging-api-key-2024"

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("""
<h1>📦 AI Packaging Intelligence</h1>
<p style="color:#9CA3AF; max-width:720px;">
Explainable AI system for sustainable packaging decisions using
cost, durability, sustainability, and innovation intelligence.
</p>
<hr/>
""", unsafe_allow_html=True)

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
st.markdown("🧾 Product Configuration")

c1, c2, c3 = st.columns(3)

with c1:
    product_category = st.selectbox(
        "Product Category",
        ["electronics","food","glassware","pharmaceutical","cosmetics","household"]
    )
    fragility_score = st.slider("Fragility", 0.0, 1.0, 0.7)

with c2:
    sustainability_priority = st.slider("Sustainability Priority", 0.0, 1.0, 0.8)
    durability_requirement = st.slider("Durability Requirement", 0.0, 1.0, 0.6)

with c3:
    material_cost = st.number_input("Material Cost", value=40.0)
    max_packaging_cost = st.number_input("Max Packaging Cost", value=100.0)
    innovation_level = st.number_input("Innovation Level", value=3.0)

# --------------------------------------------------
# SUBMIT
# --------------------------------------------------
if st.button("🚀 Generate AI Recommendation", use_container_width=True):

    payload = {
        "product_category": product_category,
        "fragility_score": fragility_score,
        "sustainability_priority": sustainability_priority,
        "durability_requirement": durability_requirement,
        "material_cost": material_cost,
        "max_packaging_cost": max_packaging_cost,
        "innovation_level": innovation_level
    }

    headers = {"X-API-Key": API_KEY}

    try:
        response = requests.post(BACKEND_URL, json=payload, headers=headers, timeout=10)
        data = response.json()
    except Exception as e:
        st.error("❌ Backend not reachable. Make sure Flask is running.")
        st.stop()

    if data.get("status") != "success":
        st.error(data.get("message", "Unknown backend error"))
        st.stop()

    # --------------------------------------------------
    # TABS
    # --------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "📦 Recommendations",
        "📊 BI Dashboard",
        "🧠 Model Reasoning"
    ])

    # ==================================================
    # TAB 1 – RECOMMENDATIONS
    # ==================================================
    with tab1:
        confidence = data["confidence_score"]

        st.metric(
            "Overall Confidence Score",
            f"{confidence*100:.1f}%"
        )

        for rec in data["recommendations"]:
            st.markdown(f"""
            <div class="card">
                <h3>{rec['material']}</h3>
                <b>Confidence:</b> {rec['confidence']}%<br/>
                <span style="color:#9CA3AF;">{rec['reason']}</span>
            </div>
            """, unsafe_allow_html=True)

    # ==================================================
    # TAB 2 – BI DASHBOARD
    # ==================================================
    with tab2:
        rec_df = pd.DataFrame(data["recommendations"])

        # ---- BAR CHART
        fig_bar = px.bar(
            rec_df,
            x="material",
            y="confidence",
            color="material",
            template="plotly_dark",
            title="Material Confidence Comparison"
        )
        fig_bar.update_layout(
            paper_bgcolor="#0F172A",
            plot_bgcolor="#0F172A",
            font_color="#E5E7EB"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ---- RADAR (SPIDER)
        radar_labels = ["Sustainability","Durability","Cost Efficiency","Innovation"]
        radar_values = [
            sustainability_priority,
            durability_requirement,
            1 - (material_cost / max_packaging_cost),
            min(innovation_level / 5, 1)
        ]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name="Product Profile"
        ))
        fig_radar.update_layout(
            polar=dict(bgcolor="#0F172A"),
            paper_bgcolor="#0F172A",
            font_color="#E5E7EB",
            title="Product Constraint Radar"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # ---- HEATMAP
        heat_df = pd.DataFrame({
            "Factor": radar_labels,
            "Score": radar_values
        })

        fig_heat = px.imshow(
            heat_df[["Score"]].T,
            x=heat_df["Factor"],
            color_continuous_scale="Viridis",
            title="Constraint Influence Heatmap"
        )
        fig_heat.update_layout(
            paper_bgcolor="#0F172A",
            font_color="#E5E7EB"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    # ==================================================
    # TAB 3 – MODEL REASONING
    # ==================================================
    with tab3:
        st.markdown("## 🧠 Why the AI made this decision")

        for k, v in data["decision_summary"].items():
            st.markdown(f"""
            <div class="card">
                <strong>{k}</strong><br/>
                <span style="color:#E5E7EB;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 🔍 Model Metadata")
        st.json(data["model_info"])
