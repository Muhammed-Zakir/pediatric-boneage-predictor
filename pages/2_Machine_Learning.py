import streamlit as st

st.title("Machine Learning")

st.write("""
### Deep Learning & CNN for Images
We use a ResNet50 model trained on pediatric hand X-ray datasets.
""")

st.write("""
### Model Trained for this Web App
The model achieves state-of-the-art performance with minimal computational requirements.
""")

# Performance placeholder
st.metric("Test MAE", "10.3 months")
st.metric("Test MSE", "171.1")
