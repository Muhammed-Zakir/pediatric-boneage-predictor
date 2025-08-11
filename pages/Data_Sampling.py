import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data():
    return pd.read_csv("boneage-training-dataset.csv")


df = load_data()

# --- Sampling Steps ----
# Stratification
st.title("📊 Data Sampling")
# Combination of up sampling and down sampling
st.header("Data Sampling")
st.write('''
In medical AI, preserving natural distributions while addressing imbalance is more valuable than forcing artificial balance. 
''')



# --- Preprocessing Steps ---
st.header("Preprocessing Steps")
st.markdown("""
- Resized all images to **256×256** pixels
- Normalized pixel values to [0, 1]
- Converted grayscale images to 3-channel format
- Applied augmentations: rotations, flips, scaling
- Removed corrupted or invalid images
""")

# --- Limitations & Notes ---
st.header("Limitations & Notes")
st.warning("""
- Some age groups have fewer samples, which may affect model accuracy.
- Possible annotation inconsistencies in bone age labels.
- Dataset mainly sourced from a single hospital, limiting generalizability.
""")
st.info("Future improvements will include more diverse and larger datasets.")