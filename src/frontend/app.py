import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io

# PDF generation
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="AI Packaging Recommendation",
    layout="wide"
)

# --------------------------------------------------
# Backend API URL
# --------------------------------------------------
BACKEND_URL = "https://recommendationsystem-txbe.onrender.com/api/product/recommend-materials"

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("📦 AI Packaging Recommendation System")
st.caption("Explainable, Sustainable, Learning-to-Rank based Decision Support")

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------
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

# --------------------------------------------------
# GENERATE RECOMMENDATION
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
        "eco_pressure": eco_pressure,
        "cost_efficiency": cost_efficiency,
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
        st.error("❌ Backend service unavailable")
        st.stop()

    if data.get("status") != "success":
        st.error("❌ Backend returned an error")
        st.json(data)
        st.stop()

    # --------------------------------------------------
    # SAFE DATA EXTRACTION
    # --------------------------------------------------
    recommendations = data.get("recommendations", [])
    decision_summary = data.get("decision_summary", {})
    model_info = data.get("model_info", {})

    if not recommendations:
        st.error("❌ No recommendations returned")
        st.stop()

    rec_df = pd.DataFrame(recommendations)

    # --------------------------------------------------
    # TABS
    # --------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📦 Recommendations",
            "📊 Analytics Dashboard",
            "🧠 Model Reasoning",
            "⬇️ Export Reports",
        ]
    )

    # ==================================================
    # TAB 1 – RECOMMENDATIONS
    # ==================================================
    with tab1:
        st.metric("Overall Confidence", f"{data.get('confidence_score', 0)}%")

        for r in recommendations:
            st.success(
                f"**{r['material']}** — {r['confidence']}%\n\n{r['reason']}"
            )

    # ==================================================
    # TAB 2 – ANALYTICS DASHBOARD
    # ==================================================
    with tab2:
        st.subheader("📊 Decision Analytics")

        # Bar Chart
        st.plotly_chart(
            px.bar(
                rec_df,
                x="material",
                y="confidence",
                title="Material Confidence Comparison",
                template="plotly_dark",
            ),
            use_container_width=True,
        )

        # Radar Chart
        radar_df = pd.DataFrame({
            "Metric": [
                "Fragility",
                "Durability",
                "Sustainability",
                "Cost Efficiency",
                "Innovation",
                "Eco Pressure",
            ],
            "Score": [
                fragility_score,
                durability_requirement,
                sustainability_priority,
                cost_efficiency,
                innovation_level / 5,
                eco_pressure,
            ],
        })

        fig_radar = px.line_polar(
            radar_df,
            r="Score",
            theta="Metric",
            line_close=True,
            title="Input Constraint Radar",
            template="plotly_dark",
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # Heatmap
        fig_heat = px.imshow(
            radar_df.set_index("Metric").T,
            aspect="auto",
            title="Constraint Influence Heatmap",
            color_continuous_scale="Viridis",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        # Budget Utilization Gauge
        budget_ratio = material_cost / max_packaging_cost
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=budget_ratio * 100,
                title={"text": "Budget Utilization (%)"},
                gauge={"axis": {"range": [0, 100]}},
            )
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ==================================================
    # TAB 3 – MODEL REASONING
    # ==================================================
    with tab3:
        for k, v in decision_summary.items():
            st.info(f"**{k.replace('_', ' ').title()}**: {v}")

        st.json(model_info)

    # ==================================================
    # TAB 4 – EXPORT REPORTS
    # ==================================================
    with tab4:
        st.subheader("⬇️ Download Reports")

        # -------------------------
        # EXCEL EXPORT
        # -------------------------
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            rec_df.to_excel(writer, index=False, sheet_name="Recommendations")

        st.download_button(
            label="📊 Download Excel Report",
            data=excel_buffer.getvalue(),
            file_name="packaging_recommendations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        # -------------------------
        # PDF EXPORT
        # -------------------------
        pdf_buffer = io.BytesIO()
        styles = getSampleStyleSheet()

        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        elements = []

        elements.append(Paragraph(
            "<b>AI Packaging Recommendation Report</b>",
            styles["Title"]
        ))
        elements.append(Spacer(1, 12))

        table_data = [["Material", "Confidence (%)", "Reason"]]
        for r in recommendations:
            table_data.append([
                r["material"],
                f"{r['confidence']}%",
                r["reason"],
            ])

        table = Table(table_data, colWidths=[120, 120, 250])
        elements.append(table)

        doc.build(elements)

        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_buffer.getvalue(),
            file_name="packaging_recommendations.pdf",
            mime="application/pdf",
        )
