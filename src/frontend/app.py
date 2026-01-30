import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import io
from fpdf import FPDF

# --------------------------------------------------
# Streamlit config (MUST be first)
# --------------------------------------------------
st.set_page_config(
    page_title="AI Packaging Recommendation System",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("📦 AI Packaging Recommendation System")
st.caption("Explainable, sustainable, data-driven packaging decisions")

# --------------------------------------------------
# Sidebar – Inputs
# --------------------------------------------------
st.sidebar.header("🧪 Product & Sustainability Inputs")

product_category = st.sidebar.selectbox(
    "Product Category",
    ["electronics", "food", "cosmetics", "pharmaceuticals", "industrial"]
)

fragility_score = st.sidebar.slider("Fragility Score", 0.0, 1.0, 0.5)
durability_requirement = st.sidebar.slider("Durability Requirement", 0.0, 1.0, 0.5)
sustainability_priority = st.sidebar.slider("Sustainability Priority", 0.0, 1.0, 0.5)
environmental_pressure = st.sidebar.slider("Environmental Pressure", 0.0, 1.0, 0.5)
innovation_level = st.sidebar.slider("Innovation Level", 0.0, 5.0, 2.5)

material_cost = st.sidebar.number_input("Material Cost", min_value=0.0, value=40.0)
max_packaging_cost = st.sidebar.number_input("Max Packaging Budget", min_value=0.0, value=100.0)
cost_efficiency = st.sidebar.slider("Cost Efficiency Priority", 0.0, 1.0, 0.5)

# --------------------------------------------------
# API Payload
# --------------------------------------------------
payload = {
    "product_category": product_category,
    "fragility_score": fragility_score,
    "durability_requirement": durability_requirement,
    "sustainability_priority": sustainability_priority,
    "environmental_pressure": environmental_pressure,
    "innovation_level": innovation_level,
    "material_cost": material_cost,
    "max_packaging_cost": max_packaging_cost,
    "cost_efficiency": cost_efficiency
}

# --------------------------------------------------
# Button
# --------------------------------------------------
if st.button("🚀 Get Recommendation"):
    try:
        res = requests.post(
            "https://recommendationsystem-txbe.onrender.com",
            json=payload,
            timeout=10
        )

        if res.status_code != 200:
            st.error(f"Backend error ({res.status_code}): {res.text}")
            st.stop()

        data = res.json()

        if "recommendations" not in data:
            st.error("Backend response missing 'recommendations'")
            st.stop()

        df = pd.DataFrame(data["recommendations"])

    except Exception as e:
        st.error(f"Connection error: {e}")
        st.stop()

    # --------------------------------------------------
    # Validate & Fix DataFrame Columns
    # --------------------------------------------------
    base_cols = ["material", "confidence", "model"]
    for col in base_cols:
        if col not in df.columns:
            df[col] = "N/A" if col != "confidence" else 0.0

    score_cols = [
        "eco_score",
        "durability_score",
        "cost_score",
        "fragility_support"
    ]

    # If backend didn't send these columns → generate demo scores
    for col in score_cols:
        if col not in df.columns:
            df[col] = (df["confidence"] * 100).round(2)

    # Convert numeric columns safely
    numeric_cols = ["confidence"] + score_cols
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # --------------------------------------------------
    # Recommendations Table
    # --------------------------------------------------
    st.subheader("✅ Recommended Materials")

    st.dataframe(
        df[["material", "confidence", "model"]],
        use_container_width=True
    )

    # --------------------------------------------------
    # BI DASHBOARD
    # --------------------------------------------------
    st.subheader("📊 BI Dashboard")

    col1, col2 = st.columns(2)

    # 1️⃣ Confidence Bar Chart
    with col1:
        st.markdown("**Confidence by Material**")
        fig = px.bar(
            df,
            x="material",
            y="confidence",
            color="material"
        )
        st.plotly_chart(fig, use_container_width=True)

    # 2️⃣ Confidence Distribution
    with col2:
        st.markdown("**Confidence Distribution**")
        fig = px.histogram(
            df,
            x="confidence",
            nbins=10
        )
        st.plotly_chart(fig, use_container_width=True)

    # 3️⃣ Sustainability vs Cost
    st.markdown("### 🌱 Sustainability vs 💰 Cost Trade-off")
    fig = px.scatter(
        df,
        x="cost_score",
        y="eco_score",
        size="confidence",
        color="material",
        hover_name="material"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 4️⃣ Durability vs Fragility
    st.markdown("### 📦 Durability vs Fragility Support")
    fig = px.scatter(
        df,
        x="durability_score",
        y="fragility_support",
        size="confidence",
        color="material",
        hover_name="material"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 5️⃣ Radar Chart
    st.markdown("### 🧭 Feature Comparison")

    radar_df = df.melt(
        id_vars=["material"],
        value_vars=score_cols,
        var_name="feature",
        value_name="score"
    )

    fig = px.line_polar(
        radar_df,
        r="score",
        theta="feature",
        color="material",
        line_close=True
    )
    st.plotly_chart(fig, use_container_width=True)

    # --------------------------------------------------
    # EXPORTS
    # --------------------------------------------------
    st.subheader("📤 Export Results")

    # Excel export
    excel_buffer = io.BytesIO()
    df.to_excel(excel_buffer, index=False)

    st.download_button(
        "📥 Download Excel",
        excel_buffer.getvalue(),
        file_name="packaging_recommendations.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # PDF export
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    for _, row in df.iterrows():
        text = f"{row['material']} | Confidence: {row['confidence']:.3f}"
        pdf.multi_cell(0, 8, text.encode("latin-1", "ignore").decode("latin-1"))

    pdf_bytes = pdf.output(dest="S").encode("latin-1", "ignore")

    st.download_button(
        "📄 Download PDF",
        pdf_bytes,
        file_name="packaging_recommendations.pdf",
        mime="application/pdf"
    )
