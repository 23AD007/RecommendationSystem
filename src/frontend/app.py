import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Packaging Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM THEME
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
# BACKEND CONFIG (CHANGE AFTER DEPLOYMENT)
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
    except Exception:
        st.error("❌ Backend not reachable.")
        st.stop()

    if data.get("status") != "success":
        st.error(data.get("message", "Unknown backend error"))
        st.stop()

    # --------------------------------------------------
    # DATAFRAME
    # --------------------------------------------------
    rec_df = pd.DataFrame(data["recommendations"])

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
        st.metric(
            "Overall Confidence Score",
            f"{data['confidence_score']*100:.1f}%"
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
    # TAB 2 – BI DASHBOARD + DOWNLOADS
    # ==================================================
    with tab2:
        # BAR CHART
        fig_bar = px.bar(
            rec_df,
            x="material",
            y="confidence",
            color="material",
            template="plotly_dark",
            title="Material Confidence Comparison"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # RADAR
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
            fill="toself"
        ))
        st.plotly_chart(fig_radar, use_container_width=True)

        # HEATMAP
        heat_df = pd.DataFrame({"Factor": radar_labels, "Score": radar_values})
        fig_heat = px.imshow(
            heat_df[["Score"]].T,
            x=heat_df["Factor"],
            title="Constraint Influence Heatmap"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # -------------------------
        # EXCEL DOWNLOAD
        # -------------------------
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            rec_df.to_excel(writer, index=False, sheet_name="Recommendations")

        st.download_button(
            "📥 Download Excel Report",
            data=excel_buffer.getvalue(),
            file_name="packaging_recommendations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # -------------------------
        # PDF DOWNLOAD
        # -------------------------
        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer)
        styles = getSampleStyleSheet()

        content = [
            Paragraph("AI Packaging Recommendation Report", styles["Title"]),
            Paragraph(f"Product Category: {product_category}", styles["Normal"]),
            Paragraph("<br/>", styles["Normal"]),
        ]

        for rec in data["recommendations"]:
            content.append(
                Paragraph(
                    f"{rec['material']} — Confidence: {rec['confidence']}%",
                    styles["Normal"]
                )
            )

        doc.build(content)

        st.download_button(
            "📄 Download PDF Report",
            data=pdf_buffer.getvalue(),
            file_name="packaging_report.pdf",
            mime="application/pdf"
        )

    # ==================================================
    # TAB 3 – MODEL REASONING
    # ==================================================
    with tab3:
        for k, v in data["decision_summary"].items():
            st.markdown(f"""
            <div class="card">
                <strong>{k}</strong><br/>
                <span>{v}</span>
            </div>
            """, unsafe_allow_html=True)

        st.json(data["model_info"])
