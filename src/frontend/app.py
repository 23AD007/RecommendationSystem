import streamlit as st
import requests

# --------------------------------------------------
# App config
# --------------------------------------------------
st.set_page_config(
    page_title="Packaging Recommendation System",
    layout="centered"
)

BACKEND_URL = "http://localhost:5000/api/product/recommend-materials"

# --------------------------------------------------
# UI
# --------------------------------------------------
st.title("📦 Packaging Recommendation System")

st.markdown(
    """
This tool recommends **sustainable packaging materials**
based on **raw product requirements**.

All intelligence and explanations come directly from the backend model.
"""
)

# --------------------------------------------------
# Input section (RAW FEATURES ONLY)
# --------------------------------------------------
st.header("Product Inputs")
product_category= st.selectbox(
    "Product Category",
    options=[
        "electronics",
        "food",
        "glassware",
        "pharmaceutical",
        "cosmetics",
        "household"
    ],
    index=0
)
fragility_score = st.slider(
    "Fragility Score",
    min_value=0.0,
    max_value=1.0,
    value=0.8,
    step=0.01
)

sustainability_priority = st.slider(
    "Sustainability Priority",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    step=0.01
)

durability_requirement = st.slider(
    "Durability Requirement",
    min_value=0.0,
    max_value=1.0,
    value=0.6,
    step=0.01
)

material_cost = st.number_input(
    "Material Cost",
    min_value=0.0,
    value=40.0
)

max_packaging_cost = st.number_input(
    "Max Packaging Cost",
    min_value=0.0,
    value=100.0
)

innovation_level = st.number_input(
    "Innovation Level",
    min_value=0.0,
    value=3.0,
    step=0.1
)

# --------------------------------------------------
# Submit
# --------------------------------------------------
if st.button("Get Recommendation"):
    payload = {
        "product_category": product_category,
        "fragility_score": fragility_score,
        "sustainability_priority": sustainability_priority,
        "durability_requirement": durability_requirement,
        "material_cost": material_cost,
        "max_packaging_cost": max_packaging_cost,
        "innovation_level": innovation_level
    }

    with st.spinner("Contacting recommendation engine..."):
        try:
            response = requests.post(
                BACKEND_URL,
                json=payload,
                timeout=10
            )
        except requests.exceptions.RequestException as e:
            st.error(f"Backend connection failed: {e}")
            st.stop()

    if response.status_code != 200:
        st.error(f"Backend error: {response.text}")
        st.stop()

    data = response.json()

    if data.get("status") != "success":
        st.error(data.get("message", "Unknown backend error"))
        st.stop()

    # --------------------------------------------------
    # Results
    # --------------------------------------------------
    st.header("Results")

    st.metric(
        label="Confidence Score",
        value=f"{data['confidence_score'] * 100:.1f}%"
    )

    # --------------------------------------------------
    # Explanations (VERBATIM from backend)
    # --------------------------------------------------
    st.subheader("Why this recommendation?")
    for _, explanation in data["decision_summary"].items():
        st.write("•", explanation)

    # --------------------------------------------------
    # Recommendations
    # --------------------------------------------------
    st.subheader("Recommended Materials")

    for rec in data["recommendations"]:
        with st.container():
            st.markdown(f"**{rec['material']}**")
            st.write(f"Confidence: {rec['confidence']}%")
            if "reason" in rec:
                st.write(rec["reason"])
            st.divider()
