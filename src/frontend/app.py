import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Packaging Recommendation",
    layout="wide"
)

BACKEND_URL = "https://recommendationsystem-txbe.onrender.com/api/product/recommend-materials"

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📦 AI Packaging Recommendation System")
st.caption(
    "Sustainable packaging decisions using cost, durability, innovation, "
    "and environmental intelligence"
)

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
st.subheader("🧾 Product & Sustainability Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    product_category = st.selectbox(
        "Product Category",
        ["electronics", "food", "pharmaceutical", "cosmetics", "household"]
    )

    fragility_score = st.slider(
        "Fragility Score",
        0.0, 1.0, 0.6,
        help="How fragile the product is"
    )

    durability_requirement = st.slider(
        "Durability Requirement",
        0.0, 1.0, 0.7,
        help="Required packaging strength"
    )

with col2:
    sustainability_priority = st.slider(
        "Sustainability Priority",
        0.0, 1.0, 0.8,
        help="Importance of eco-friendly materials"
    )

    eco_pressure = st.slider(
        "Environmental Pressure",
        0.0, 1.0, 0.75,
        help="Regulatory & environmental constraints"
    )

    innovation_level = st.slider(
        "Innovation Level",
        0.0, 5.0, 3.0,
        help="Preference for innovative packaging materials"
    )

with col3:
    material_cost = st.number_input(
        "Material Cost",
        value=40.0,
        help="Cost of selected material"
    )

    max_packaging_cost = st.number_input(
        "Maximum Packaging Budget",
        value=100.0,
        help="Budget constraint"
    )

    cost_efficiency = st.slider(
        "Cost Efficiency Priority",
        0.0, 1.0, 0.6,
        help="Importance of low-cost packaging"
    )

# --------------------------------------------------
# SUBMIT
# --------------------------------------------------
if st.button("🚀 Generate AI Recommendation", use_container_width=True):

    payload = {
        "product_category": product_category,
        "fragility_score": fragility_score,
        "durability_requirement": durability_requirement,
        "sustainability_priority": sustainability_priority,
        "material_cost": material_cost,
        "max_packaging_cost": max_packaging_cost,
        "innovation_level": innovation_level,

        # extra raw signals (derive_features will handle them)
        "eco_pressure": eco_pressure,
        "cost_efficiency": cost_efficiency
    }

    try:
        response = requests.post(
            BACKEND_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        data = response.json()
    except Exception as e:
        st.error(f"❌ Backend not reachable: {e}")
        st.stop()

    if data.get("status") != "success":
        st.error("❌ Backend returned an error")
        st.json(data)
        st.stop()

    # --------------------------------------------------
    # SAFE PARSING
    # --------------------------------------------------
    recommendations = data.get("recommendations", [])

    if not recommendations:
        st.error("❌ No recommendations returned")
        st.json(data)
        st.stop()

    rec_df = pd.DataFrame(recommendations)

    # --------------------------------------------------
    # TABS
    # --------------------------------------------------
    tab1, tab2, tab3 = st.tabs(
        ["📦 Recommendations", "📊 Dashboard", "🧠 Model Reasoning"]
    )

    # ---------------- TAB 1 ----------------
    with tab1:
        st.metric(
            "Overall Confidence Score",
            f"{data['confidence_score']}%"
        )

        for rec in recommendations:
            st.success(
                f"**{rec['material']}** — {rec['confidence']}%\n\n"
                f"{rec['reason']}"
            )

    # ---------------- TAB 2 ----------------
    with tab2:
        fig = px.bar(
            rec_df,
            x="material",
            y="confidence",
            title="Material Confidence Comparison",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Excel
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            rec_df.to_excel(writer, index=False)

        st.download_button(
            "📥 Download Excel Report",
            excel_buffer.getvalue(),
            "packaging_recommendations.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # PDF
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer)
        styles = getSampleStyleSheet()

        content = [Paragraph("Packaging Recommendation Report", styles["Title"])]
        for rec in recommendations:
            content.append(
                Paragraph(
                    f"{rec['material']} — {rec['confidence']}%",
                    styles["Normal"]
                )
            )

        doc.build(content)

        st.download_button(
            "📄 Download PDF Report",
            pdf_buffer.getvalue(),
            "packaging_report.pdf",
            "application/pdf"
        )

    # ---------------- TAB 3 ----------------
    with tab3:
        for k, v in data["decision_summary"].items():
            st.info(f"**{k.replace('_', ' ').title()}**: {v}")

        st.json(data["model_info"])
