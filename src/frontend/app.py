import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(
    page_title="AI Packaging Recommendation System",
    layout="wide"
)

USE_MOCK_BACKEND = True  # ✅ REQUIRED FOR STREAMLIT CLOUD

# ==================================================
# HEADER
# ==================================================
st.title("📦 AI-Powered Packaging Recommendation System")
st.caption(
    "Explainable AI system for sustainable packaging recommendations "
    "with integrated Business Intelligence analytics."
)

tabs = st.tabs(["📦 Recommendations", "📊 BI Dashboard"])

# ==================================================
# INPUT SIDEBAR
# ==================================================
with st.sidebar:
    st.header("🧾 Product Inputs")

    product_category = st.selectbox(
        "Product Category",
        ["electronics", "food", "glassware", "pharmaceutical", "cosmetics", "household"]
    )

    fragility_score = st.slider("Fragility", 0.0, 1.0, 0.7)
    sustainability_priority = st.slider("Sustainability Priority", 0.0, 1.0, 0.8)
    durability_requirement = st.slider("Durability Requirement", 0.0, 1.0, 0.6)

    material_cost = st.number_input("Material Cost", value=40.0)
    max_packaging_cost = st.number_input("Max Packaging Cost", value=100.0)
    innovation_level = st.slider("Innovation Level", 0.0, 5.0, 3.0)

    run = st.button("🚀 Run Recommendation", use_container_width=True)

# ==================================================
# MOCK BACKEND (STREAMLIT SAFE)
# ==================================================
def mock_backend(payload):
    confidence = round(
        0.35 * payload["sustainability_priority"]
        + 0.30 * payload["fragility_score"]
        + 0.20 * payload["durability_requirement"]
        + 0.15 * (1 - payload["material_cost"] / payload["max_packaging_cost"]),
        3
    )

    recommendations = [
        {
            "material": "Recycled Cardboard",
            "confidence": round(confidence * 100, 1),
            "reason": "High sustainability priority combined with low material cost"
        },
        {
            "material": "Bamboo Fiber",
            "confidence": round(confidence * 92, 1),
            "reason": "Renewable material offering durability with eco-benefits"
        },
        {
            "material": "Cork",
            "confidence": round(confidence * 88, 1),
            "reason": "Excellent shock absorption for fragile products"
        }
    ]

    decision_summary = {
        "Sustainability Impact":
            f"Sustainability priority ({payload['sustainability_priority']}) strongly influenced material choice.",
        "Fragility Impact":
            f"Fragility score ({payload['fragility_score']}) increased preference for cushioning materials.",
        "Cost Constraint":
            f"Material cost ({payload['material_cost']}) within budget ({payload['max_packaging_cost']}).",
        "Innovation Influence":
            f"Innovation level ({payload['innovation_level']}) supported modern eco-materials."
    }

    return {
        "confidence_score": confidence,
        "recommendations": recommendations,
        "decision_summary": decision_summary
    }

# ==================================================
# RUN PIPELINE
# ==================================================
if run:
    payload = {
        "product_category": product_category,
        "fragility_score": fragility_score,
        "sustainability_priority": sustainability_priority,
        "durability_requirement": durability_requirement,
        "material_cost": material_cost,
        "max_packaging_cost": max_packaging_cost,
        "innovation_level": innovation_level
    }

    data = mock_backend(payload)

    rec_df = pd.DataFrame(data["recommendations"])

    # ==================================================
    # TAB 1 — RECOMMENDATIONS
    # ==================================================
    with tabs[0]:
        st.subheader("✅ Recommendation Confidence")
        st.metric("Overall Confidence", f"{data['confidence_score']*100:.1f}%")

        st.subheader("🧠 Why this recommendation?")
        for k, v in data["decision_summary"].items():
            st.info(f"**{k}** — {v}")

        st.subheader("📦 Recommended Materials")
        for _, r in rec_df.iterrows():
            st.markdown(
                f"""
                **{r['material']}**  
                Confidence: **{r['confidence']}%**  
                _{r['reason']}_
                """
            )
            st.divider()

    # ==================================================
    # TAB 2 — BI DASHBOARD
    # ==================================================
    with tabs[1]:
        st.subheader("📊 Business Intelligence Dashboard")

        col1, col2 = st.columns(2)

        # ---------- Radar (Spider) ----------
        with col1:
            radar_fig = go.Figure()
            radar_fig.add_trace(go.Scatterpolar(
                r=[
                    sustainability_priority,
                    durability_requirement,
                    1 - material_cost / max_packaging_cost,
                    fragility_score
                ],
                theta=[
                    "Sustainability",
                    "Durability",
                    "Cost Efficiency",
                    "Fragility"
                ],
                fill="toself",
                name="Product Profile"
            ))
            radar_fig.update_layout(height=380)
            st.plotly_chart(radar_fig, use_container_width=True)

        # ---------- Heatmap ----------
        with col2:
            heatmap_data = np.array([
                [sustainability_priority, durability_requirement],
                [fragility_score, 1 - material_cost / max_packaging_cost]
            ])

            heatmap_fig = px.imshow(
                heatmap_data,
                labels=dict(x="Factors", y="Metrics", color="Impact"),
                x=["Sustainability", "Durability"],
                y=["Fragility", "Cost Efficiency"],
                color_continuous_scale="Viridis"
            )
            heatmap_fig.update_layout(height=380)
            st.plotly_chart(heatmap_fig, use_container_width=True)

        st.divider()

        # ---------- CO₂ Reduction ----------
        st.subheader("🌱 Sustainability Impact")

        co2_df = pd.DataFrame({
            "Material": rec_df["material"],
            "CO₂ Reduction (%)": [28, 35, 22]
        })

        st.plotly_chart(
            px.bar(co2_df, x="Material", y="CO₂ Reduction (%)", height=350),
            use_container_width=True
        )

        # ---------- Cost Savings ----------
        cost_df = pd.DataFrame({
            "Material": rec_df["material"],
            "Cost Savings (%)": [18, 12, 9]
        })

        st.plotly_chart(
            px.bar(cost_df, x="Material", y="Cost Savings (%)", height=350),
            use_container_width=True
        )

        # ---------- Material Usage Trends ----------
        trend_df = pd.DataFrame({
            "Year": ["2021", "2022", "2023", "2024"],
            "Recycled Cardboard": [40, 48, 55, 63],
            "Bamboo Fiber": [20, 28, 36, 45],
            "Cork": [15, 18, 22, 26]
        })

        st.plotly_chart(
            px.line(
                trend_df,
                x="Year",
                y=trend_df.columns[1:],
                markers=True,
                height=380
            ),
            use_container_width=True
        )
