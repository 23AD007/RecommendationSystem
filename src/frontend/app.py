import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------------------------------------------------
# APP CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Packaging Recommendation System",
    layout="wide"
)

BACKEND_URL = "http://localhost:5000/api/product/recommend-materials"
API_KEY = "packaging-api-key-2024"

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📦 AI-Powered Packaging Recommendation System")
st.caption(
    "An explainable AI system for recommending sustainable packaging materials "
    "with integrated sustainability and cost analytics."
)

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
with st.sidebar:
    st.header("🧾 Product Inputs")

    product_category = st.selectbox(
        "Product Category",
        ["electronics", "food", "glassware", "pharmaceutical", "cosmetics", "household"]
    )

    fragility_score = st.slider(
        "Fragility Score", 0.0, 1.0, 0.8,
        help="Higher value means more fragile product"
    )

    sustainability_priority = st.slider(
        "Sustainability Priority", 0.0, 1.0, 0.7
    )

    durability_requirement = st.slider(
        "Durability Requirement", 0.0, 1.0, 0.6
    )

    material_cost = st.number_input(
        "Material Cost", min_value=0.0, value=40.0
    )

    max_packaging_cost = st.number_input(
        "Max Packaging Cost", min_value=0.0, value=100.0
    )

    innovation_level = st.number_input(
        "Innovation Level", min_value=0.0, value=3.0
    )

    run_btn = st.button("🚀 Get Recommendation")

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab_rec, tab_bi = st.tabs(["📦 Recommendations", "📊 BI Dashboard"])

if run_btn:
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

    response = requests.post(
        BACKEND_URL,
        json=payload,
        headers=headers,
        timeout=10
    )

    if response.status_code != 200:
        st.error(response.text)
        st.stop()

    data = response.json()

    # ==================================================
    # TAB 1 — RECOMMENDATIONS + REASONING
    # ==================================================
    with tab_rec:
        confidence = data["confidence_score"]

        st.metric(
            "Overall Confidence Score",
            f"{confidence * 100:.1f}%"
        )

        # -------------------------------
        # STRONG HUMAN-READABLE REASONING
        # -------------------------------
        st.subheader("🧠 Why this recommendation?")

        reasoning_map = {
            "Fragility": "High fragility increases the need for protective packaging materials.",
            "Sustainability": "Higher sustainability priority favors eco-friendly materials.",
            "Cost": "Cost constraints limit the selection of premium materials.",
            "Durability": "Durability requirements influence material strength and longevity.",
            "Innovation": "Innovation preference promotes modern, alternative materials."
        }

        decision_summary = data.get("decision_summary", {})

        if decision_summary:
            for feature, _ in decision_summary.items():
                explanation = "Balanced contribution to the final decision."
                for key, text in reasoning_map.items():
                    if key.lower() in feature.lower():
                        explanation = text
                st.write(f"• **{feature}** — {explanation}")
        else:
            st.info(
                "The model arrived at this decision based on an overall balance "
                "of product constraints without a dominant factor."
            )

        # -------------------------------
        # MATERIAL RECOMMENDATIONS
        # -------------------------------
        st.subheader("📦 Recommended Materials")

        for rec in data["recommendations"]:
            with st.container():
                st.markdown(f"### {rec['material']}")
                st.write(f"**Confidence:** {rec['confidence']}%")
                st.caption(rec["reason"])
                st.divider()

    # ==================================================
    # TAB 2 — BI DASHBOARD
    # ==================================================
    with tab_bi:
        st.subheader("📊 Business Intelligence Dashboard")

        rec_df = pd.DataFrame(data["recommendations"])

        # -------------------------------
        # Synthetic BI Metrics (Industry Assumptions)
        # -------------------------------
        BI_METRICS = {
            "Recycled Cardboard": [85, 70, 80],
            "Bamboo Fiber": [90, 65, 75],
            "Hemp Fiber": [88, 60, 78],
            "Cork": [70, 55, 85],
            "Sustainable Composite": [75, 68, 82],
            "Biodegradable Plastic": [65, 50, 70],
            "Review Requirements": [40, 30, 40]
        }

        metric_labels = [
            "CO₂ Reduction (%)",
            "Cost Savings (%)",
            "Durability Index"
        ]

        bi_rows = []
        for mat in rec_df["material"]:
            co2, cost, dur = BI_METRICS.get(mat, [50, 50, 50])
            bi_rows.append([mat, co2, cost, dur])

        bi_df = pd.DataFrame(
            bi_rows,
            columns=["Material"] + metric_labels
        )

        # ==================================================
        # 🕸 INTERACTIVE RADAR (SPIDER) CHART
        # ==================================================
        st.markdown("### 🕸 Material Performance Radar")

        selected_material = st.selectbox(
            "Select material",
            bi_df["Material"]
        )

        radar_values = bi_df[
            bi_df["Material"] == selected_material
        ][metric_labels].values.flatten().tolist()

        radar_fig = go.Figure(
            data=go.Scatterpolar(
                r=radar_values,
                theta=metric_labels,
                fill="toself",
                name=selected_material
            )
        )

        radar_fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=False,
            height=400
        )

        st.plotly_chart(radar_fig, use_container_width=True)

        # ==================================================
        # 🔥 INTERACTIVE HEATMAP
        # ==================================================
        st.markdown("### 🔥 Sustainability & Cost Heatmap")

        heatmap_fig = px.imshow(
            bi_df.set_index("Material"),
            color_continuous_scale="YlGn",
            aspect="auto"
        )

        heatmap_fig.update_layout(height=350)
        st.plotly_chart(heatmap_fig, use_container_width=True)

        # ==================================================
        # 📊 CO₂ vs COST SAVINGS (GROUPED BAR)
        # ==================================================
        st.markdown("### 📊 CO₂ Reduction vs Cost Savings")

        bar_fig = px.bar(
            bi_df,
            x="Material",
            y=["CO₂ Reduction (%)", "Cost Savings (%)"],
            barmode="group",
            height=350
        )

        bar_fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(bar_fig, use_container_width=True)

        # ==================================================
        # 📈 CONFIDENCE TREND (INTERACTIVE)
        # ==================================================
        st.markdown("### 📈 Recommendation Confidence Trend")

        trend_fig = px.line(
            rec_df,
            x="material",
            y="confidence",
            markers=True,
            height=350
        )

        trend_fig.update_layout(
            yaxis_title="Confidence (%)",
            xaxis_title="Material"
        )

        st.plotly_chart(trend_fig, use_container_width=True)
