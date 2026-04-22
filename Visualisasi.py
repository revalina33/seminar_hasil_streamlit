import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from wordcloud import WordCloud, STOPWORDS
import plotly.express as px

# ==============================
# 1️⃣ KONFIGURASI
# ==============================
st.set_page_config(page_title="Analisis Sentimen Shopee", layout="wide")

st.sidebar.title("🧭 Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["📊 Visualisasi Data & Tren", "⚙️ Perhitungan Algoritma"]
)

# ==============================
# 2️⃣ UPLOAD DATA
# ==============================
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload file ulasan (CSV/Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file:

    # ==============================
    # LOAD DATA
    # ==============================
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # ==============================
    # LABEL CLEANING
    # ==============================
    if 'Labeling' in df.columns:
        df['Labeling'] = df['Labeling'].astype(str).str.strip().str.capitalize()
        df['Labeling'] = df['Labeling'].apply(
            lambda x: x if x in ['Positif', 'Negatif'] else 'Positif'
        )
    else:
        st.error("Kolom Labeling tidak ditemukan")
        st.stop()

    # ==============================
    # TIME HANDLING
    # ==============================
    tgl_cols = [c for c in df.columns if c.lower() == 'tanggal']
    if tgl_cols:
        col_tgl = tgl_cols[0]
        df['Tanggal'] = pd.to_datetime(df[col_tgl], errors='coerce')
        df['Tanggal'] = df['Tanggal'].ffill().bfill()
        df['Tahun'] = df['Tanggal'].dt.year
        df['Bulan'] = df['Tanggal'].dt.month
    else:
        df['Tahun'] = 2024
        df['Bulan'] = 1

    nama_bulan = {
        1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun',
        7:'Jul',8:'Agu',9:'Sep',10:'Okt',11:'Nov',12:'Des'
    }

    # ==============================
    # 3️⃣ NAVIGASI
    # ==============================

    # =========================================================
    # 📊 VISUALISASI
    # =========================================================
    if page == "📊 Visualisasi Data & Tren":

        st.title("📊 Dashboard Analisis Sentimen Shopee")

        total = len(df)
        pos = len(df[df['Labeling']=='Positif'])
        neg = len(df[df['Labeling']=='Negatif'])

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("Positif", pos)
        c3.metric("Negatif", neg)

        st.divider()

        tab1, tab2 = st.tabs(["📈 Tren", "🔍 WordCloud"])

        # ================= TREND =================
        with tab1:
            trend = df.groupby(['Tahun','Labeling']).size().reset_index(name='Jumlah')

            fig = px.bar(
                trend,
                x='Tahun',
                y='Jumlah',
                color='Labeling',
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)

        # ================= WORDCLOUD =================
        with tab2:
            text = " ".join(df.iloc[:,0].astype(str))

            wc = WordCloud(
                stopwords=STOPWORDS,
                background_color='white'
            ).generate(text)

            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

    # =========================================================
    # ⚙️ MODELING 
    # =========================================================
    elif page == "⚙️ Perhitungan Algoritma":

        st.title("⚙️ Evaluasi Model ML")

        if st.button("🚀 Jalankan Komputasi"):

            st.info("Training model...")

            # ======================
            # SET SEED (FIX RANDOM)
            # ======================
            np.random.seed(42)

            # ======================
            # DATA
            # ======================
            col_text = 'stemming' if 'stemming' in df.columns else df.columns[0]

            X = df[col_text].fillna('')
            y = df['Labeling']

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            # ======================
            # TF-IDF 
            # ======================
            tfidf = TfidfVectorizer(
                max_features=5000,
                lowercase=True,
                norm='l2'
            )

            X_train = tfidf.fit_transform(X_train_raw)
            X_test = tfidf.transform(X_test_raw)

            # ======================
            # MODELS 
            # ======================
            models = {
                'Random Forest': RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                ),

                'SVM Linear': SVC(
                    kernel='linear',
                    probability=True,
                    random_state=42,
                    max_iter=5000
                ),

                'SVM RBF': SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=42,
                    max_iter=5000
                ),

                'SVM Poly': SVC(
                    kernel='poly',
                    probability=True,
                    random_state=42,
                    max_iter=5000
                ),

                'SVM Sigmoid': SVC(
                    kernel='sigmoid',
                    probability=True,
                    random_state=42,
                    max_iter=5000
                )
            }

            results = []

            # ======================
            # TRAINING LOOP
            # ======================
            for name, model in models.items():

                model.fit(X_train, y_train)
                pred = model.predict(X_test)

                acc = accuracy_score(y_test, pred)
                report = classification_report(y_test, pred, output_dict=True)

                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": report['Positif']['precision'],
                    "Recall": report['Positif']['recall'],
                    "F1": report['Positif']['f1-score']
                })

                st.write(f"### {name}")
                st.metric("Accuracy", f"{acc:.4f}")

            # ======================
            # RESULT TABLE
            # ======================
            res_df = pd.DataFrame(results)

            st.subheader("📊 Hasil Perbandingan")

            st.dataframe(
                res_df.set_index("Model").style.highlight_max(axis=0)
            )

            best = res_df.loc[res_df["F1"].idxmax()]
            st.success(f"🏆 Best Model: {best['Model']} (F1 Score)")
