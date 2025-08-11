import streamlit as st

st.set_page_config(page_title="Pediatric Age Predictor", layout="wide")

# Explicitly define pages with custom titles
home_page = st.Page("pages/0_home.py", title="Home")
data_info = st.Page("pages/1_Data_Info.py", title="Data Information")
data_sample = st.Page("pages/2_Data_Sampling.py", title="Data Sampling")
ml_page = st.Page("pages/3_Machine_Learning.py", title="Machine Learning")
predict_page = st.Page("pages/4_Predict_Bone_Age.py", title="Predict Bone Age")

# Create navigation
nav = st.navigation([home_page, data_info, data_sample, ml_page, predict_page])
nav.run()
