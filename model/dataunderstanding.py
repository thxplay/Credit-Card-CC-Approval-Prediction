import streamlit as st
import pandas as pd                                         
import numpy as np                                        
from sklearn.model_selection import train_test_split       
from sklearn.preprocessing import LabelEncoder              
from sklearn.ensemble import RandomForestClassifier         
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier 
import xgboost 

import matplotlib.pyplot as plt                             
import seaborn as sns                                       
import os   
import io
import joblib


# Scalling untuk MinMax Scaller
from sklearn.preprocessing import MinMaxScaler   

# untuk plot Q-Q
import scipy.stats as stats                                 

import warnings
warnings.filterwarnings("ignore", message=".*use_column_width.*")

def dataunderstanding():

    pd.set_option('display.max_rows', None)  # Menampilkan semua baris
    pd.set_option('display.max_columns', None)  # Jika ada banyak kolom

    st.markdown("<h2 style='text-align: left;'>Deskripsi:</h2>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: justify;'>Skor kartu kredit merupakan metode pengendalian risiko yang umum digunakan di industri keuangan. Metode ini memanfaatkan informasi pribadi dan data yang diajukan oleh pemohon kartu kredit untuk memprediksi kemungkinan terjadinya gagal bayar atau penggunaan kredit di masa depan. Berdasarkan hasil prediksi ini, bank dapat memutuskan apakah akan menyetujui permohonan kartu kredit. Skor kredit membantu mengkuantifikasi risiko secara objektif. Proyek ini bertujuan untuk membangun model prediksi persetujuan kartu kredit menggunakan data historis pemohon dengan pendekatan statistik klasik dan algoritma machine learning, serta mengevaluasi trade-off antara akurasi dan interpretabilitas model.</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: justify;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])  # kolom tengah lebih lebar
    with col2:
        st.image("images/cc.jpg", use_container_width=True)

    with st.expander("📋 About Dataset application_record.csv"):
        st.write('## Dataset : application_record.csv')
        st.info('438.557 baris dan 18 kolom')
        st.write('Berisi data pribadi dan keuangan dari setiap pemohon kartu kredit. Contohnya: Umur, jenis kelamin, penghasilan, jumlah anak, status pernikahan, pendidikan, jenis pekerjaan, dll.')
        st.markdown("""
        ### 📄 Deskripsi Dataset application_record.csv
                    
        | **Feature name** | **Explanation** | **Remarks** |
        |------------------|-----------------|-------------|
        | ID	| Client number | |
        | CODE_GENDER |	Gender | |
        | FLAG_OWN_CAR	| Is there a car | |
        | FLAG_OWN_REALTY	| Is there a property | |
        | CNT_CHILDREN	| Number of children | |
        | AMT_INCOME_TOTAL |	Annual income | |
        | NAME_INCOME_TYPE	| Income category	| |
        | NAME_EDUCATION_TYPE	| Education level | |
        | NAME_FAMILY_STATUS	| Marital status | |
        | NAME_HOUSING_TYPE	| Way of living | |
        | DAYS_BIRTH	| Birthday	| Count backwards from current day (0), -1 means yesterday |
        | DAYS_EMPLOYED	| Start date of employment	| Count backwards from current day(0). If positive, it means the person currently unemployed. |
        | FLAG_MOBIL |	Is there a mobile phone | |
        | FLAG_WORK_PHONE	| Is there a work phone | |
        | FLAG_PHONE	| Is there a phone | |
        | FLAG_EMAIL	| Is there an email	| |
        | OCCUPATION_TYPE	| Occupation | |
        | CNT_FAM_MEMBERS	| Family size | |
        """)
        app_df = pd.read_csv('dataset/application_record.csv')
        st.markdown("<h2 style='text-align: justify;'>Datasets</h2>", unsafe_allow_html=True)
        st.dataframe(app_df)
        if 'Unnamed: 0' in app_df.columns:
            app_df.drop(columns='Unnamed: 0', inplace=True)
        info_app_df = pd.DataFrame({
            "Kolom": app_df.columns,
            "Non-Null Count": app_df.notnull().sum().values,
            "Tipe Data": app_df.dtypes.astype(str).values
            })
        st.subheader("📋 Informasi Struktur Datasets")
        st.dataframe(info_app_df)

    with st.expander("📋 About Dataset application_record.csv"):
        st.write('## Dataset : application_record.csv')
        st.info('1.048.575 baris dan 3 kolom.')
        st.write('Berisi riwayat kredit masing-masing pemohon:')
        st.write('Status pembayaran kredit bulanan')
        st.write('STATUS bisa berupa:')
        st.write('0 (terlambat 1 - 29 hari),')
        st.write('1 (terlambat 30 - 59 hari),')
        st.write('2 (terlambat 60 - 89 hari),')
        st.write('3 (terlambat 90 - 119 hari),')
        st.write('4 (terlambat 120 - 149 hari),')
        st.write('5 (terlambat > 150 hari),')
        st.write('C (dibayar lunas),')
        st.write('X (tidak ada pinjaman aktif)')
        st.markdown("""           
        ### 📄 Deskripsi Dataset credit_record.csv
                    
        | **Feature name** | **Explanation** | **Remarks** |
        |------------------|-----------------|-------------|
        | ID	| Client number | |
        | MONTHS_BALANCE	| Record month	| The month of the extracted data is the starting point, backwards, 0 is the current month, -1 is the previous month, and so on |
        | STATUS |	Status | 0: 1-29 days past due 1: 30-59 days past due 2: 60-89 days overdue 3: 90-119 days overdue 4: 120-149 days overdue 5: Overdue or bad debts, write-offs for more than 150 days C: paid off that month X: No loan for the month |
        """)

        credit_df = pd.read_csv('dataset/credit_record.csv')
        st.markdown("<h2 style='text-align: justify;'>Datasets</h2>", unsafe_allow_html=True)
        st.dataframe(credit_df)
        if 'Unnamed: 0' in credit_df.columns:
            credit_df.drop(columns='Unnamed: 0', inplace=True)
        info_credit_df = pd.DataFrame({
            "Kolom": credit_df.columns,
            "Non-Null Count": credit_df.notnull().sum().values,
            "Tipe Data": credit_df.dtypes.astype(str).values
            })
        st.subheader("📋 Informasi Struktur Datasets")
        st.dataframe(info_credit_df)

    with st.expander("📓 Data Pre Processing (Python with Google Colab)"):

        # Step 1: Filtering
        st.subheader("📌 1. Filtering Data Kredit 12 Bulan Terakhir")
        st.markdown("<div style='text-align: justify;'>Ambil data riwayat kredit selama 12 bulan terakhir berdasarkan kolom <b>MONTHS_BALANCE</b></div>", unsafe_allow_html=True)
        recent_12 = credit_df[credit_df['MONTHS_BALANCE'] >= -12].copy()
        st.write("Jumlah data:", recent_12.shape, "(Baris, Kolom)",)
        st.dataframe(recent_12.head())

        # Step 2: Labeling
        st.subheader("🏷️ 2. Labeling - Menentukan Nasabah Kredit Baik dan Buruk")
        st.markdown("<div style='text-align: justify;'>Menandai pemohon dengan status kredit buruk berdasarkan nilai STATUS (2-5 dianggap buruk)</div>", unsafe_allow_html=True)
        bad_statuses_revised = ['2', '3', '4', '5']
        recent_12['bad_credit'] = recent_12['STATUS'].isin(bad_statuses_revised).astype(int)
        st.dataframe(recent_12[['ID', 'STATUS', 'bad_credit']].head())

        st.write("📌 Buat label TARGET berdasarkan maksimal nilai bad_credit tiap ID")
        labels_revised = recent_12.groupby('ID')['bad_credit'].max().reset_index()
        labels_revised.rename(columns={'bad_credit': 'TARGET'}, inplace=True)
        st.dataframe(labels_revised.head())

        # Step 3: Join
        st.subheader("🔗 3. Join - Menggabungkan Dataset Target ke Dataset Aplikasi")
        data = app_df.merge(labels_revised, on='ID', how='inner')
        st.write("Jumlah data hasil gabungan:", data.shape, "(Baris, Kolom)",)
        st.dataframe(data.head())

        # Step 4: Splitting
        st.subheader("🔄 4. Split Data (Train & Test)")
        st.markdown("<div style='text-align: justify;'>Memisahkan data menjadi variabel fitur (X) dan target (y), kemudian dilakukan split data training dan testing (80:20)</div>", unsafe_allow_html=True)
        X = data.drop(['TARGET'], axis=1)
        y = data['TARGET']

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

        st.write("Proporsi TARGET (y_train):")
        st.write(y_train.value_counts(normalize=True))

        st.write("Ukuran data:")
        st.write("X_train:", X_train.shape)
        st.write("X_test:", X_test.shape)

    # Data Cleaning & Manipulation
    with st.expander("🧹 Data Cleaning & Manipulation (Python with Google Colab)"):
        
        # Anomali replacement
        st.subheader("⚠️ 1. Pengecekan Anomali")
        st.markdown("<div style='text-align: justify;'>Ditemukan nilai anomali `365243` di kolom `DAYS_EMPLOYED`</div>", unsafe_allow_html=True)
        X_train['DAYS_EMPLOYED'] = X_train['DAYS_EMPLOYED'].replace(365243, np.nan)
        X_test['DAYS_EMPLOYED'] = X_test['DAYS_EMPLOYED'].replace(365243, np.nan)

        median_days_emp = X_train['DAYS_EMPLOYED'].median()
        X_train['DAYS_EMPLOYED'] = X_train['DAYS_EMPLOYED'].fillna(median_days_emp)
        X_test['DAYS_EMPLOYED'] = X_test['DAYS_EMPLOYED'].fillna(median_days_emp)
        pilih1 = st.radio("Tampilkan pengecekan?", ["Tidak", "Pengecekan"], key="cek_anomali1")
        if pilih1 == "Pengecekan":
            st.info(f"✅ Median dari DAYS_EMPLOYED (X_train): {median_days_emp}")

        st.markdown("<div style='text-align: justify;'>Ditemukan `DAYS_BIRTH` dan `DAYS_EMPLOYED` memiliki nilai negatif</div>", unsafe_allow_html=True)
        X_train['AGE_YEARS'] = (-X_train['DAYS_BIRTH']) // 365
        X_test['AGE_YEARS'] = (-X_test['DAYS_BIRTH']) // 365

        X_train['YEARS_EMPLOYED'] = (-X_train['DAYS_EMPLOYED']) // 365
        X_test['YEARS_EMPLOYED'] = (-X_test['DAYS_EMPLOYED']) // 365
        pilih2 = st.radio("Tampilkan pengecekan?", ["Tidak", "Pengecekan"], key="cek_anomali2")
        if pilih2 == "Pengecekan":
            st.write("Membuat menjadi satuannya Tahun dan posifit dengan membagi 365 pada fitur baru `AGE_YEARS` dan `YEARS_EMPLOYED`")
            st.info("✅ `DAYS_BIRTH` menjadi `AGE_YEARS` dan `DAYS_EMPLOYED` menjadi `YEARS_EMPLOYED` dengan nilai positif")

        st.markdown("---")
        
        # Cek tipe data
        st.subheader("✅ 2. Pengecekan tipe data")
        st.markdown("<div style='text-align: justify;'>Tidak ditemukan tipe data yang tidak sesuai</div>", unsafe_allow_html=True)
        pilih3 = st.radio("Tampilkan pengecekan?", ["Tidak", "Pengecekan"], key="cek_tipedata")
        if pilih3 == "Pengecekan":
            buffer = io.StringIO()
            X_train.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)
            
        st.markdown("---")

        # Cek duplikat
        st.subheader("✅ 3. Pengecekan Duplikat")
        st.markdown("<div style='text-align: justify;'>Tidak ditemukan duplikat.</div>", unsafe_allow_html=True)
        pilih4 = st.radio("Tampilkan pengecekan?", ["Tidak", "Pengecekan"], key="cek_duplikat")
        if pilih4 == "Pengecekan":
            duplikat_proporsi = len(X_train.drop_duplicates()) / len(X_train)
            st.write(f"Proporsi data unik: `{duplikat_proporsi:.4f}`")
            if duplikat_proporsi < 1:
                st.error("⚠️ Ditemukan duplikat dalam data.")
            else:
                st.success("✅ Tidak ditemukan duplikat dalam data.")

        st.markdown("---")

        # Cek missing value
        st.subheader("⚠️ 4. Pengecekan Missing Value")
        st.markdown("<div style='text-align: justify;'>Ditemukan Missing Value pada `OCCUPATION_TYPE`.</div>", unsafe_allow_html=True)
        pilih5 = st.radio("Tampilkan pengecekan?", ["Tidak", "Pengecekan"], key="cek_missing")
        if pilih5 == "Pengecekan":
            st.write("Missing value sebelum imputasi:")
            st.dataframe(X_train.isna().sum())
            
            X_train['OCCUPATION_TYPE'].fillna("Unknown", inplace=True)
            X_test['OCCUPATION_TYPE'].fillna("Unknown", inplace=True)
            st.info("✅ `OCCUPATION_TYPE` sudah diimputasi dengan Unknown")
            st.dataframe(X_train.isna().sum())
            
        st.markdown("---")

        # Outlier Visualisasi
        st.subheader("📊 5. Pengecekan Outlier (Numerikal)")

        # Tambahan :
        categorical_train = X_train.select_dtypes(include='object').columns.tolist()
        numerical_train = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

        drop_cols = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL', 'ID']
        numerical_train = [col for col in numerical_train if col not in drop_cols]

        # Drop kolom dari X_train dan X_test (inplace=False agar bisa dikontrol)
        X_train = X_train.drop(columns=drop_cols, errors='ignore')
        X_test = X_test.drop(columns=drop_cols, errors='ignore')
        print(X_train.head())

        st.info("✅ Fitur DAYS_BIRTH, DAYS_EMPLOYED, FLAG_MOBIL, ID di hapus dikarenakan tidak digunakan lagi untuk proses selanjutnya.")

        import matplotlib.pyplot as plt
        import seaborn as sns
        import scipy.stats as stats

        def check_plot(df_cs, column):
            plt.figure(figsize=(16, 4))

            plt.subplot(1, 3, 1)
            sns.histplot(df_cs[column], bins=30)
            plt.title(f'Histogram - {column}')

            plt.subplot(1, 3, 2)
            stats.probplot(df_cs[column], dist="norm", plot=plt)
            plt.ylabel('Variable quantiles')

            plt.subplot(1, 3, 3)
            sns.boxplot(y=df_cs[column])
            plt.title(f'Boxplot - {column}')

            st.pyplot(plt.gcf())
            plt.clf()

        pilih_var_outlier = st.selectbox("Pilih variabel numerik untuk melihat outlier:", numerical_train)
        check_plot(X_train, pilih_var_outlier)
        st.info("✅ Ditemukan Outlier pada CNT_CHILDREN (jumlah anak), AMT_INCOME_TOTAL (total penghasilan), dan CNT_FAM_MEMBERS (jumlah anggota keluarga) tapi idak dikonversikan / tidak dihapus karena sifatnya masih wajar.")
        
        # Feature Engineering
    with st.expander("⚙️ Feature Engineering (Python with Google Colab)"):
    
        # 1. Encoding & Scaling
        st.subheader("📊 1. Encoding dan Scaling")
        
        pilih6 = st.radio("📌 Tampilkan proses Encoding dan Scaling?", ["Tampilkan", "Sembunyikan"], index=0, key="fe6")
        if pilih6 == "Tampilkan":
            # Menentukan kolom numerik & kategorikal
            numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

            # Pipeline transformasi
            from sklearn.compose import ColumnTransformer
            from sklearn.preprocessing import StandardScaler, OneHotEncoder

            preprocessor = ColumnTransformer(transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

            st.success(f"✅ Ditemukan {len(numerical_features)} fitur numerik dan {len(categorical_features)} fitur kategorikal.")
            st.write("Fitur Numerik:", numerical_features)
            st.write("Fitur Kategorikal:", categorical_features)
            st.write(f"📐 X_train shape: {X_train.shape}")
            st.write(f"📐 X_test shape: {X_test.shape}")
            st.write("Distribusi label sebelum balancing:")
            st.dataframe(y_train.value_counts())

        st.markdown("---")

        # 2. Transform Data
        st.subheader("📊 2. Transformasi Data")
        
        pilih7 = st.radio("📌 Tampilkan transformasi data?", ["Tampilkan", "Sembunyikan"], index=0, key="fe7")
        if pilih7 == "Tampilkan":
            # Transformasi
            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            st.success("✅ Transformasi `fit_transform` untuk X_train dan `transform` untuk X_test telah dilakukan.")
            st.write(f"📐 Shape X_train_processed: {X_train_processed.shape}")
            st.write(f"📐 Shape X_test_processed : {X_test_processed.shape}")

        st.markdown("---")

        # 3. Oversampling
        st.subheader("📊 3. Oversampling (RandomOverSampler)")
        
        pilih8 = st.radio("📌 Tampilkan proses oversampling?", ["Tampilkan", "Sembunyikan"], index=0, key="fe8")
        if pilih8 == "Tampilkan":
            from imblearn.over_sampling import RandomOverSampler
            ros = RandomOverSampler(random_state=42)
            X_resampled, y_resampled = ros.fit_resample(X_train_processed, y_train)

            st.write("Jumlah label sebelum oversampling:")
            st.dataframe(y_train.value_counts())

            st.write("Jumlah label setelah oversampling:")
            st.dataframe(pd.Series(y_resampled).value_counts())

            st.success("✅ Oversampling dengan RandomOverSampler berhasil dilakukan.")

            # Kolom hasil encoding
            encoded_cols = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
            all_columns = list(encoded_cols) + numerical_features
            st.markdown("🧩 Kolom hasil encoding dan scaling:")
            st.text("\n".join(all_columns))

        st.markdown("---")

        # Simpan ke file
        if not os.path.exists("saved"):
            os.makedirs("saved")

        # Simpan hasil split jika belum ada
        if not os.path.exists("saved/X_train.pkl"):
            joblib.dump(X_train, "saved/X_train.pkl")
            joblib.dump(X_test, "saved/X_test.pkl")
            joblib.dump(y_train, "saved/y_train.pkl")
            joblib.dump(y_test, "saved/y_test.pkl")

        