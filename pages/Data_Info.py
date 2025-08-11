import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


@st.cache_data
def load_data():
    return pd.read_csv("boneage-training-dataset.csv")


df = load_data()

st.title("📊 Data Information")

# --- Dataset Overview ---
st.header("Dataset Overview")
st.write("""
The dataset consists of pediatric hand X-ray images used for training the bone age prediction model.

- **Source:** [RSNA Pediatric Bone Age Dataset](https://www.kaggle.com/kmader/rsna-bone-age)
- **Raw Images:** 12611w
- **Total Images:** ~12,000 (after preprocessing)
- **Labels:** Bone age in months, gender
- **Image Format:** PNG
""")

st.write('''
    The dataset is not homogenous even within the 0-19 range children, 
    there are vast differences between infants (0-1), toddlers (1-3), preschoolers (3-5), 
    primary school-aged children (6-11), early adolescents (12-14), and late adolescents/young adults (15-19). 
    Each of these sub-groups has distinct developmental, health, educational, and social needs. 
    The developmental factor is crucial for an accurate model in boneage prediction 
    ''')
# --- Sample Data for Charts ---

gender_filter = st.selectbox("Select Gender", options=["All", "Male", "Female"])

if gender_filter != "All":
    gender_map = {"Male": 1, "Female": 0}
    df = df[df["male"] == gender_map[gender_filter]]

# Distribution
st.header("Distributions")
st.write("""
Distribution
""")
# Histogram Plot
fig_hist, ax = plt.subplots(figsize=(4, 2.5))
sns.histplot(df["boneage"], bins=30, ax=ax)
ax.set_xlabel("Age (months)")
ax.set_ylabel("Count")
ax.set_title("Age Distribution")
st.pyplot(fig_hist)

# Age
st.header("Age Groups")
st.write("""
Different categories of the dataset population can be identified as infants, toddlers, child, adolescent.
Further depending on the development stages of children the dataset can be categorized as below by boneage which is global and Sri Lankan standard as well.
""")
# Age groups table
age_groups_data = {
    'Age Range (Months)': ['0-24 months','25-60 months','61-120 months','121-180 months','181-228 months'],
    'Age Range (Years)': ['0-2 years','2-5 years','5-10 years','10-15 years','15-19 years'],
    'Development Stage': ['Infants/Toddlers','Preschool','School Age','Pre-teen/Early Teen','Late Teen']
}
df_age_groups = pd.DataFrame(age_groups_data)
df_age_groups.index = df_age_groups.index + 1
st.table(df_age_groups)

# Age groups distribution
data = {
    "Age (Months)": ["0-24 months", "24-60 months", "60-120 months", "120-180 months", "180-228 months"],
    "Count": ['168', '911', '4201', '6650', '681'],
    "Percentage": ['1.33%', '7.22%', '33.31%', '52.73%', '5.40%']
}
df_age_groups_distribution = pd.DataFrame(data)
st.write("Age Groups Distribution")
df_age_groups_distribution = pd.DataFrame(df_age_groups_distribution)
st.dataframe(df_age_groups_distribution)

# Bar Plot
fig_bar, ax = plt.subplots()
labels = ['0-24mo', '24-60mo', '60-120mo', '120-180mo', '180-228mo']
sns.countplot(x='age_bin', data=df, order=labels, ax=ax, width=0.5)
ax.set_xlabel("Age Group")
ax.set_ylabel("Count")
ax.set_title("Distribution of Age Groups")
st.pyplot(fig_bar)

st.header("Genders")
st.write('''
Biological Differences, girls typically reach skeletal maturity earlier than boys (around 16-17 years vs 18-19 years). 
This means the same chronological age can correspond to different bone ages between genders. 
Growth Pattern Differences, puberty timing varies by gender, affecting bone development rates, especially in the 10–15-year 
range where the dataset is largely imbalanced by gender.
''')
# Gender Pie Plot
fig_pie, ax = plt.subplots()
gender_counts = df['male'].value_counts()
colors = ['skyblue', 'lightcoral']
ax.pie(gender_counts.values,
       labels=['Male', 'Female'],
       autopct='%1.1f%%',
       colors=colors,
       startangle=90)
ax.set_title("Gender Distribution")
ax.legend(['Male', 'Female'], title="Gender", loc="upper left", bbox_to_anchor=(1, 1))
st.pyplot(fig_pie)

fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='male', y='boneage', data=df, palette=['lightcoral', 'skyblue'], ax=ax)
ax.set_title("Age Distribution Pattern by Gender")
ax.set_xticklabels(['Female', 'Male'])
st.pyplot(fig)

# Age & Gender
# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='age_bin', hue='male', data=df, order=labels, ax=ax, palette=['lightcoral', 'skyblue'])
ax.set_title("Gender Distribution in Age Groups")
ax.set_xlabel("Age Group")
ax.set_ylabel("Count")
ax.legend(title='Gender', labels=['Female', 'Male'])
st.pyplot(fig)
