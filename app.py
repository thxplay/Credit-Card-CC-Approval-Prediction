import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Judul aplikasi
st.title("Exploratory Data Analysis - Credit Card Approval")

# Deskripsi
st.markdown("""
Aplikasi ini menampilkan hasil EDA dari dataset prediksi persetujuan kartu kredit.
""")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")

df = load_data()

# Tampilkan dataframe
st.subheader("Dataset")
st.dataframe(df.head())

# Statistik deskriptif
st.subheader("Descriptive Statistics")
st.write(df.describe())

# Visualisasi distribusi target
st.subheader("Distribusi Target")
fig, ax = plt.subplots()
sns.countplot(data=df, x='Approval', ax=ax)
st.pyplot(fig)

# Korelasi antar fitur
st.subheader("Correlation Matrix")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)