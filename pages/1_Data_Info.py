import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("Data – Information & Charts")

st.write("Dataset description goes here. Include gender, bone age distribution, and sampling methods.")

# Sample chart placeholder
df = pd.DataFrame({"Bone Age": [10, 20, 30], "Count": [100, 80, 50]})
fig, ax = plt.subplots()
ax.bar(df["Bone Age"], df["Count"])
ax.set_xlabel("Bone Age (months)")
ax.set_ylabel("Count")
st.pyplot(fig)
