import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="AI Packaging Recommendation", layout="wide")

BACKEND_URL = "https://recommendationsystem-txbe.onrender.com/api/product/recommend-materials"

st.title("📦 AI Packaging Recommendation System")
st.caption("Explainable, sustainable packaging decisions using ML")

# ==================================================
# INPUTS
# ==================================================
st.subheader("🧾 Product & Sustainability Inputs")

c1, c2, c3 = st.columns(3)

with c1:
    product_category = st.selectbox(
        "Product Category",
        ["electronics", "food", "pharmaceutical", "cosmetics", "household"]
    )
    fragility_score = st.slider("Fragility Score", 0.0, 1.0, 0.6)
    durability_requirement = st.slider("Durability Requirement", 0.0, 1.0, 0.7)

with c2:
    sustainability_priority = st.slider("Sustainability Priority", 0.0, 1.0, 0.8)
    eco_pressure = st.slider("Environmental Pressure", 0.0, 1.0, 0.75)
    innovation_level = st.slider("Innovation Level", 0.0, 5.0, 3.0)

with c3:
    material_cost = st.number_input("Material Cost", value=40.0)
    max_packaging_cost = st.number_input("Max Packaging Budget", value=100.0)
    cost_efficiency = st.slider("Cost Efficiency Priority", 0.0, 1.0, 0.6)

# ==================================================
# SUBMIT
# ==================================================
if st.button("🚀 Generate AI Recommendation", use_container_width=True):

    payload = {
        "product_category": product_category,
        "fragility_score": fragility_score,
        "durability_requirement": durability_requirement,
        "sustainability_priority": sustainability_priority,
        "material_cost": material_cost,
        "max_packaging_cost": max_packaging_cost,
        "innovation_level": innovation_level,
        "eco_pressure": eco_pressure,
        "cost_efficiency": cost_efficiency,
    }

    r = requests.post(
        BACKEND_URL,
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=15
    )

    data = r.json()

    if data.get("status") != "success":
        st.error("❌ Backend returned an error")
        st.json(data)
        st.stop()

    rec_df = pd.DataFrame(data["recommendations"])

    tab1, tab2, tab3 = st.tabs(
        ["📦 Recommendations", "📊 Analytics Dashboard", "🧠 Model Reasoning"]
    )

    # ==================================================
    # TAB 1 – RECOMMENDATIONS
    # ==================================================
    with tab1:
        st.metric("Overall Confidence", f"{data['confidence_score']}%")
        for r in data["recommendations"]:
            st.success(f"**{r['material']}** — {r['confidence']}%\n\n{r['reason']}")

    # ==================================================
    # TAB 2 – MULTI-CHART DASHBOARD
    # ==================================================
    with tab2:
        st.subheader("📊 Decision Analytics")

        # 1. Bar chart
        st.plotly_chart(
            px.bar(
                rec_df,
                x="material",
                y="confidence",
                title="Material Confidence Comparison",
                template="plotly_dark"
            ),
            use_container_width=True
        )

        # 2. Radar chart
        radar_df = pd.DataFrame({
            "Metric": [
                "Fragility",
                "Durability",
                "Sustainability",
                "Cost Efficiency",
                "Innovation",
                "Eco Pressure"
            ],
            "Score": [
                fragility_score,
                durability_requirement,
                sustainability_priority,
                cost_efficiency,
                innovation_level / 5,
                eco_pressure
            ]
        })

        fig_radar = px.line_polar(
            radar_df,
            r="Score",
            theta="Metric",
            line_close=True,
            title="Input Constraint Radar",
            template="plotly_dark"
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # 3. Heatmap
        fig_heat = px.imshow(
            radar_df.set_index("Metric").T,
            aspect="auto",
            title="Constraint Influence Heatmap",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # 4. Budget Utilization Gauge
        budget_ratio = material_cost / max_packaging_cost
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=budget_ratio * 100,
            title={"text": "Budget Utilization (%)"},
            gauge={"axis": {"range": [0, 100]}}
        ))
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ==================================================
    # TAB 3 – MODEL REASONING
    # ==================================================
    with tab3:
        for k, v in data["decision_summary"].items():
            st.info(f"**{k.replace('_',' ').title()}**: {v}")

        st.json(data["model_info"])
