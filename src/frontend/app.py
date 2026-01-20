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
st.set_page_config(page_title="AI Packaging Recommendation", layout="wide")

BACKEND_URL = "https://recommendationsystem-txbe.onrender.com/api/product/recommend-materials"

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📦 AI Packaging Recommendation System")
st.caption("Explainable ML-based sustainable packaging decisions")

# --------------------------------------------------
# INPUTS
# --------------------------------------------------
c1, c2, c3 = st.columns(3)

with c1:
    product_category = st.selectbox(
        "Product Category",
        ["electronics", "food", "pharmaceutical", "cosmetics", "household"]
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
if st.button("🚀 Generate AI Recommendation"):

    payload = {
        "product_category": product_category,
        "fragility_score": fragility_score,
        "sustainability_priority": sustainability_priority,
        "durability_requirement": durability_requirement,
        "material_cost": material_cost,
        "max_packaging_cost": max_packaging_cost,
        "innovation_level": innovation_level
    }

    try:
        r = requests.post(
            BACKEND_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=15
        )
        data = r.json()
    except Exception as e:
        st.error(f"❌ Backend not reachable: {e}")
        st.stop()

    if data.get("status") != "success":
        st.error("❌ Backend returned an error")
        st.json(data)
        st.stop()

    # --------------------------------------------------
    # 🔒 DEFENSIVE PARSING (NO KeyError EVER)
    # --------------------------------------------------
    recommendations = data.get("recommendations", [])

    if not recommendations:
        st.error("❌ No recommendations returned by backend")
        st.json(data)
        st.stop()

    rec_df = pd.DataFrame(recommendations)

    # --------------------------------------------------
    # TABS
    # --------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📦 Recommendations", "📊 Dashboard", "🧠 Reasoning"])

    # ---------------- TAB 1 ----------------
    with tab1:
        st.metric("Confidence Score", f"{data['confidence_score']}%")
        for r in recommendations:
            st.success(f"**{r['material']}** — {r['confidence']}%")

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

        # Excel export
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            rec_df.to_excel(writer, index=False)

        st.download_button(
            "📥 Download Excel",
            excel_buffer.getvalue(),
            "recommendations.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # PDF export
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer)
        styles = getSampleStyleSheet()

        content = [Paragraph("Packaging Recommendation Report", styles["Title"])]
        for r in recommendations:
            content.append(
                Paragraph(f"{r['material']} — {r['confidence']}%", styles["Normal"])
            )

        doc.build(content)

        st.download_button(
            "📄 Download PDF",
            pdf_buffer.getvalue(),
            "recommendations.pdf",
            "application/pdf"
        )

    # ---------------- TAB 3 ----------------
    with tab3:
        for k, v in data["decision_summary"].items():
            st.info(f"**{k.capitalize()}**: {v}")
        st.json(data["model_info"])
