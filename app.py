import streamlit as st

st.set_page_config(page_title="Pediatric Age Predictor", layout="wide")

# Explicitly define pages with custom titles
home_page = st.Page("pages/Home.py", title="Home")
data_info = st.Page("pages/Data_Info.py", title="Data Information")
data_sample = st.Page("pages/Data_Sampling.py", title="Data Sampling")
ml_page = st.Page("pages/Machine_Learning.py", title="Machine Learning")
predict_page = st.Page("pages/Predict_Bone_Age.py", title="Predict Bone Age Test")

# Create navigation
nav = st.navigation([home_page, data_info, data_sample, ml_page, predict_page])
nav.run()
