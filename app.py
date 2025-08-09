import streamlit as st
from PIL import Image

st.set_page_config(page_title="Pediatric Age Predictor", layout="wide")


# ----- HEADER -----
st.markdown("<h1 style='text-align: center;'>A Web Based Pediatric Age Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Using Hand X-Ray Images</h3>", unsafe_allow_html=True)

st.write("**Description:** Pediatric hand X-rays can reveal bone growth and help predict age accurately. "
         "This system uses AI to provide fast, non-invasive age predictions, aiding medical professionals.")

# ----- ABOUT -----
st.subheader("About")
st.write("This project uses deep learning to predict a child's age from hand X-ray images. "
         "It is designed for medical research and educational purposes.")

# ----- TESTIMONIALS -----
st.subheader("Testimonials")
st.info("“This AI tool has streamlined age estimation in our pediatric department.” — Dr. Jane Doe")

# ----- CONTACT -----
st.subheader("Contact")
st.write("📧 Email: support@boneageapp.com")

# ----- FOOTER -----
st.markdown("<hr><p style='text-align:center;'>© 2025 Pediatric Age Predictor | All Rights Reserved</p>", unsafe_allow_html=True)
