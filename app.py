import streamlit as st
import pandas as pd
import joblib
import os
import datetime
import warnings

warnings.filterwarnings("ignore", message=".*use_column_width.*")

# ---------------------- Page Setup ----------------------
st.set_page_config(page_title="Credit Card Approval Prediction", page_icon="💳", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("model/lightgbm_pipeline.pkl")

model = load_model()

st.sidebar.title('⚙️ Menu Utama')
page = st.sidebar.radio('Pilih halaman:', ['Prediction', 'Data Understanding', 'Exploratory Data Analysis', 'About Me'])

# ---------------------- Prediction Page ----------------------
if page == 'Prediction':
    st.header("🧠 Input Data Nasabah")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["M", "F"])
            age = st.slider("Age (Years)", 18, 70, 30)
            own_car = st.selectbox("Own Car", ["Y", "N"])
            own_realty = st.selectbox("Own Realty", ["Y", "N"])
            children = st.number_input("Number of Children", 0, 10, 0)
            income_type = st.selectbox("Income Type", ['Working', 'Commercial associate', 'Pensioner', 'State servant', 'Student'])
            education = st.selectbox("Education Level", ['Secondary / secondary special', 'Higher education', 'Incomplete higher', 'Lower secondary', 'Academic degree'])
            family_status = st.selectbox("Family Status", ['Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'])

        with col2:
            housing_type = st.selectbox("Housing Type", ['House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment', 'Office apartment', 'Co-op apartment'])
            occupation = st.selectbox("Occupation Type", ['Laborers', 'Sales staff', 'Core staff', 'Managers', 'Drivers', 'High skill tech staff', 'Accountants', 'Medicine staff', 'Security staff', 'Cooking staff', 'Cleaning staff', 'Private service staff', 'Low-skill Laborers', 'Secretaries', 'Waiters/barmen staff', 'Realty agents', 'IT staff', 'HR staff'])
            income_total = st.number_input("Annual Income", 10000, 1000000, 135000, step=5000)
            years_employed = st.slider("Years Employed", 0, 50, 5)
            work_phone = st.radio("Has Work Phone?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            phone = st.radio("Has Phone?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            email = st.radio("Has Email?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            fam_members = st.slider("Family Members Count", 1, 10, 2)

        submitted = st.form_submit_button("🔍 Predict Approval")

    if submitted:
        X_input = pd.DataFrame([{
            'CODE_GENDER': gender,
            'AGE_YEARS': age,
            'FLAG_OWN_CAR': own_car,
            'FLAG_OWN_REALTY': own_realty,
            'CNT_CHILDREN': children,
            'NAME_INCOME_TYPE': income_type,
            'NAME_EDUCATION_TYPE': education,
            'NAME_FAMILY_STATUS': family_status,
            'NAME_HOUSING_TYPE': housing_type,
            'OCCUPATION_TYPE': occupation,
            'AMT_INCOME_TOTAL': income_total,
            'YEARS_EMPLOYED': years_employed,
            'FLAG_WORK_PHONE': work_phone,
            'FLAG_PHONE': phone,
            'FLAG_EMAIL': email,
            'CNT_FAM_MEMBERS': fam_members
        }])

        st.markdown("### 🧾 Preview Data Input")
        st.dataframe(X_input)

        try:
            with st.spinner("🔄 Processing prediction..."):
                pred = model.predict(X_input)[0]
                proba = model.predict_proba(X_input)[0][1]

            st.markdown("### 🎯 Hasil Prediksi")
            if pred == 0:
                st.success(f"✅ Kredit DISETUJUI — Probabilitas Disetujui: {1 - proba:.2%}")
            else:
                st.error(f"❌ Kredit DITOLAK — Probabilitas Ditolak: {proba:.2%}")

            # Simpan log hasil prediksi
            log_row = X_input.copy()
            log_row["Prediction"] = "Approved" if pred == 0 else "Rejected"
            log_row["Probability"] = round(proba if pred == 1 else 1 - proba, 4)
            log_row["Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            log_file = "prediction_log.csv"
            if os.path.exists(log_file):
                log_df = pd.read_csv(log_file)
                log_df = pd.concat([log_df, log_row], ignore_index=True)
            else:
                log_df = log_row

            log_df.to_csv(log_file, index=False)

            # Tombol download
            st.markdown("### 📥 Unduh Hasil Prediksi")
            csv = log_row.to_csv(index=False).encode("utf-8")
            st.download_button("📄 Download sebagai CSV", csv, "credit_prediction_result.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Gagal melakukan prediksi: {e}")

# ---------------------- Other Pages ----------------------
elif page == 'Data Understanding':
    from model import dataunderstanding
    dataunderstanding.dataunderstanding()

elif page == 'Exploratory Data Analysis':
    from model import eda
    eda.eda()

elif page == 'About Me':
    from model import aboutme
    aboutme.aboutme()