import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib
import logging

# Disable Plotly Narwhals Warning
os.environ["PLOTLY_DISABLE_NARWHALS"] = "true"

def eda():
    if not os.path.exists("saved/X_train.pkl"):
        st.warning("Data belum tersedia. Jalankan proses preprocessing dulu.")
        return

    X_train = joblib.load("saved/X_train.pkl")
    X_test = joblib.load("saved/X_test.pkl")
    y_train = joblib.load("saved/y_train.pkl")
    y_test = joblib.load("saved/y_test.pkl")

    eda_df = X_train.copy()
    eda_df['TARGET'] = y_train

    numerical_cols = eda_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = eda_df.select_dtypes(include=['object', 'category']).columns.tolist()

    st.header("1. 📌 Statistical Descriptive")
    st.write("Statistik deskriptif fitur numerik:")
    st.dataframe(eda_df[numerical_cols].describe())

    # ------------------------
    st.header("2. 🎯 Analisis Distribusi Target")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig, ax = plt.subplots()
        sns.countplot(x='TARGET', data=eda_df, ax=ax)
        ax.set_title("Distribusi Target (0 = Baik, 1 = Buruk)")
        st.pyplot(fig)

    with st.expander("💡 Analisa Distribusi Target"):
        st.markdown("📌 **Apakah distribusi target (label) dalam data ini seimbang?**")
        st.table(pd.DataFrame({
            "Aspek": ["🎯 Tujuan", "🧠 Insight", "✅ Kesimpulan"],
            "Penjelasan": [
                "Mengetahui apakah model akan menghadapi masalah class imbalance dalam memprediksi target (baik/buruknya kredit).",
                "Distribusi target sangat tidak seimbang — mayoritas nasabah berada dalam kategori TARGET = 0 (kredit baik). Kategori TARGET = 1 (kredit buruk) jauh lebih sedikit.",
                "Karena target tidak seimbang, maka diperlukan teknik oversampling untuk meningkatkan representasi dari kelas minoritas (TARGET = 1) sebelum melakukan training model."
            ]
        }))

    # ------------------------
    st.header("3. 🔍 Analisis Fitur Numerik")

    # a. Distribusi Umur
    st.subheader("a. Distribusi Umur")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(eda_df['AGE_YEARS'], kde=True, bins=30, ax=ax)
        ax.set_title("Distribusi Umur")
        st.pyplot(fig)

    with st.expander("💡 Analisa Distribusi Usia"):
        st.markdown("📌 **Bagaimana distribusi usia pemohon kartu kredit?**")
        st.table(pd.DataFrame({
            "Aspek": ["🎯 Tujuan", "🧠 Insight", "✅ Kesimpulan"],
            "Penjelasan": [
                "Memahami karakteristik usia nasabah agar bisa merancang strategi penawaran atau segmentasi yang sesuai.",
                "Usia pemohon berkisar antara 21 hingga 69 tahun. Distribusi cukup merata, dengan puncaknya antara usia 29–42 tahun. Tidak ada skew ekstrem atau outlier mencolok.",
                "Produk kredit menjangkau berbagai kelompok umur secara merata, terutama generasi usia kerja produktif."
            ]
        }))

    # b. Hubungan Umur dengan Target
    st.subheader("b. Hubungan Umur dengan Target")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x=eda_df['TARGET'], y=eda_df['AGE_YEARS'], ax=ax)
        ax.set_title("Hubungan Umur dengan Status Kredit (Target)")
        st.pyplot(fig)

    with st.expander("💡 Analisa Hubungan Umur-Target"):
        st.markdown("📌 **Apakah ada hubungan yang signifikan antara usia dan status kredit (TARGET)?**")
        st.table(pd.DataFrame({
            "Aspek": ["🎯 Tujuan", "🧠 Insight", "✅ Kesimpulan"],
            "Penjelasan": [
                "Mengetahui apakah umur bisa menjadi indikator awal dalam membedakan risiko kredit baik dan buruk.",
                "Median umur untuk TARGET 0 dan 1 hampir sama (43–45 tahun). Rentang umur dominan (IQR) juga mirip: 34–54 tahun. Rentang ekstrem dan outlier hampir identik untuk kedua kelas.",
                "Tidak ada perbedaan distribusi umur yang signifikan antara nasabah dengan kredit baik dan buruk. Maka, umur saja bukan penentu kuat untuk memprediksi risiko kredit."
            ]
        }))

    # c. Lama Bekerja
    st.subheader("c. Distribusi Lama Bekerja")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig, ax = plt.subplots()
        sns.histplot(eda_df['YEARS_EMPLOYED'], bins=30, ax=ax)
        ax.set_title("Distribusi Lama Bekerja")
        st.pyplot(fig)

    with st.expander("💡 Analisa Distribusi Lama Bekerja"):
        st.markdown("📌 **Seberapa lama rata-rata nasabah telah bekerja?**")
        st.table(pd.DataFrame({
            "Aspek": ["🎯 Tujuan", "🧠 Insight", "✅ Kesimpulan"],
            "Penjelasan": [
                "Mengetahui pola masa kerja dan potensi keterkaitannya dengan kebutuhan terhadap produk kredit.",
                "Sebagian besar pemohon bekerja di bawah 10 tahun. Nasabah dengan masa kerja lebih lama justru lebih sedikit mengajukan kredit.",
                "Produk kartu kredit mungkin lebih menarik bagi kelompok pekerja dengan masa kerja lebih pendek (mungkin karena kebutuhan finansial yang lebih tinggi atau belum stabil secara ekonomi)."
            ]
        }))

    # d. Distribusi Pendapatan
    st.subheader("d. Distribusi AMT_INCOME_TOTAL")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x=eda_df['AMT_INCOME_TOTAL'], ax=ax)
        ax.set_title("Distribusi Pendapatan Total")
        st.pyplot(fig)

    # e. Pendapatan vs Target
    st.subheader("e. Hubungan AMT_INCOME_TOTAL dengan Target")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig, ax = plt.subplots()
        sns.boxplot(x=eda_df['TARGET'], y=eda_df['AMT_INCOME_TOTAL'], ax=ax)
        ax.set_title("Pendapatan vs Target")
        st.pyplot(fig)

    with st.expander("💡 Analisa Distribusi dan hubungan Pendapatan"):
        st.markdown("📌 **Bagaimana distribusi pendapatan nasabah?**")
        st.markdown("📌 **Apakah ada pengaruh pendapatan terhadap status kredit?**")
        st.table(pd.DataFrame({
            "Aspek": ["🎯 Tujuan", "🧠 Insight", "✅ Kesimpulan"],
            "Penjelasan": [
                "Mengevaluasi apakah tingkat pendapatan berperan dalam menentukan kualitas kredit (baik atau buruk).",
                "Mayoritas nasabah berpendapatan di bawah 400.000. Banyak outlier pada sisi pendapatan tinggi, terutama pada TARGET 0. Namun, tidak ada perbedaan yang jelas dan konsisten antara TARGET 0 dan 1 berdasarkan distribusi pendapatan.",
                "Pendapatan tinggi tidak serta-merta menjamin kualitas kredit lebih baik. Distribusi pendapatan saja belum cukup untuk memisahkan nasabah berdasarkan risiko kredit."
            ]
        }))

    # ------------------------
    st.header("4. 🔗 Korelasi Antar Fitur Numerik")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        fig, ax = plt.subplots()
        sns.heatmap(eda_df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        ax.set_title('Correlation Matrix - Fitur Numerik')
        st.pyplot(fig)

    with st.expander("💡 Analisa Korelasi Antar Fitur Numerik"):
        st.markdown("📌 **Apakah fitur numerik dalam data saling berkorelasi satu sama lain?**")
        st.table(pd.DataFrame({
            "Aspek": ["🎯 Tujuan", "🧠 Insight", "✅ Kesimpulan"],
            "Penjelasan": [
                "Menilai apakah ada fitur numerik yang redundan atau saling mempengaruhi penting untuk menghindari multikolinearitas saat modeling.",
                "Korelasi paling tinggi antara CNT_CHILDREN dan CNT_FAM_MEMBERS (0.89). Korelasi sedang antara AGE_YEARS dan CNT_CHILDREN (0.33). Korelasi antar fitur lainnya sangat rendah (< 0.3).",
                "Sebagian besar fitur numerik tidak saling berkorelasi kuat, sehingga bisa dianggap independen dan aman digunakan secara bersamaan dalam pemodelan."
            ]
        }))

    # ------------------------
    st.header("5. 🧩 Analisis Fitur Kategorikal")
    cats_col = ['CNT_CHILDREN', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
                'OCCUPATION_TYPE', 'NAME_HOUSING_TYPE', 'NAME_FAMILY_STATUS']

    for col in cats_col:
        st.subheader(f"{col} vs TARGET")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig, ax = plt.subplots()
            sns.countplot(data=eda_df, x=col, hue=eda_df['TARGET'], ax=ax)
            ax.set_title(f"{col} vs TARGET")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)

        with st.expander(f"💡 Analisa  {col}"):
            if col == 'CNT_CHILDREN':
                st.markdown("📌 **Apakah jumlah anak berpengaruh terhadap risiko kredit?**")
                st.table(pd.DataFrame({
                    "Aspek": ["🎯 Tujuan", "🧠 Insight", "✅ Kesimpulan"],
                    "Penjelasan": [
                        "Menilai apakah jumlah anak memengaruhi kemampuan bayar kredit.",
                        "Risiko kredit buruk cenderung meningkat seiring bertambahnya jumlah anak. Mayoritas nasabah memiliki 0–2 anak.",
                        "Jumlah anak dapat dijadikan indikator risiko, meskipun perlu didukung fitur lain untuk prediksi yang lebih akurat."
                    ]
                }))
            elif col == 'NAME_INCOME_TYPE':
                st.markdown("📌 **Bagaimana hubungan jenis pekerjaan (NAME_INCOME_TYPE) dengan status kredit?**")
                st.table(pd.DataFrame({
                    "Aspek": ["🎯 Tujuan", "🧠 Insight", "✅ Kesimpulan"],
                    "Penjelasan": [
                        "Mengidentifikasi tipe sumber penghasilan yang terkait dengan risiko gagal bayar.",
                        "Kategori 'Working' mendominasi dan memiliki risiko gagal bayar lebih tinggi dibanding tipe lain seperti 'Pensioner' atau 'State servant'.",
                        "Jenis pekerjaan penting sebagai indikator risiko. Segmentasi berdasarkan penghasilan perlu diperhatikan dalam pemodelan."
                    ]
                }))
            elif col == 'NAME_EDUCATION_TYPE':
                st.markdown("📌 **Apakah tingkat pendidikan (NAME_EDUCATION_TYPE) memengaruhi kualitas kredit?**")
                st.table(pd.DataFrame({
                    "Aspek": ["🎯 Tujuan", "🧠 Insight", "✅ Kesimpulan"],
                    "Penjelasan": [
                        "Menilai hubungan antara tingkat pendidikan dengan kualitas kredit.",
                        "Tidak ada tren signifikan yang menunjukkan bahwa pendidikan lebih tinggi menjamin kualitas kredit lebih baik. Bahkan ada kredit buruk pada kelompok pendidikan tinggi.",
                        "Pendidikan bukan faktor dominan terhadap risiko kredit. Perlu dianalisis bersama fitur lain."
                    ]
                }))
            elif col == 'OCCUPATION_TYPE':
                st.markdown("📌 **Apakah jenis pekerjaan (OCCUPATION_TYPE) memengaruhi risiko gagal bayar?**")
                st.table(pd.DataFrame({
                    "Aspek": ["🎯 Tujuan", "🧠 Insight", "✅ Kesimpulan"],
                    "Penjelasan": [
                        "Mengevaluasi apakah jenis pekerjaan berpengaruh pada kualitas kredit.",
                        "Nasabah dengan pekerjaan sebagai 'Laborers' memiliki proporsi gagal bayar lebih tinggi. Sebaliknya, profesi tertentu menunjukkan stabilitas yang lebih baik.",
                        "Tipe pekerjaan bisa menjadi prediktor risiko yang kuat dan relevan untuk segmentasi nasabah."
                    ]
                }))
            elif col == 'NAME_HOUSING_TYPE':
                st.markdown("📌 **Apakah tipe tempat tinggal (NAME_HOUSING_TYPE) memengaruhi risiko gagal bayar?**")
                st.table(pd.DataFrame({
                    "Aspek": ["🎯 Tujuan", "🧠 Insight", "✅ Kesimpulan"],
                    "Penjelasan": [
                        "Menilai pengaruh jenis tempat tinggal terhadap stabilitas finansial nasabah.",
                        "Beberapa tipe tempat tinggal seperti 'With parents' dan 'Rented apartment' tampak memiliki proporsi kredit buruk yang sangat rendah.",
                        "Status tempat tinggal dapat menjadi indikator risiko sekunder, namun representasi datanya kecil dan perlu dikaji lebih lanjut."
                    ]
                }))
            elif col == 'NAME_FAMILY_STATUS':
                st.markdown("📌 **Apakah status pernikahan (NAME_FAMILY_STATUS) berkaitan dengan risiko gagal bayar?**")
                st.table(pd.DataFrame({
                    "Aspek": ["🎯 Tujuan", "🧠 Insight", "✅ Kesimpulan"],
                    "Penjelasan": [
                        "Menganalisis dampak status keluarga terhadap risiko gagal bayar.",
                        "Mayoritas nasabah berstatus 'Married' dan cenderung memiliki kualitas kredit baik. Kredit buruk hampir tidak ditemukan pada status 'Single' atau 'Widow'.",
                        "Status pernikahan mencerminkan stabilitas sosial-ekonomi dan berpotensi memengaruhi risiko kredit."
                    ]
                }))