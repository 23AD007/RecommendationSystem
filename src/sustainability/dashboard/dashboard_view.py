import sys
import os

# Add project root to PYTHONPATH
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)
sys.path.insert(0, PROJECT_ROOT)


import streamlit as st 
import pandas as pd
from src.sustainability.data_access.loader import load_data 
st.set_page_config(
    page_title="Sustainability Dashboard",
    layout="wide"
)
st.title("🌿 Sustainability Dashboard")
st.markdown("Welcome to the Sustainability Dashboard! This dashboard provides insights into the environmental impact of various packaging options based on key sustainability metrics.")
#load data
df= load_data()
 #check data
st.subheader("Data Overview")
st.dataframe(df.head())