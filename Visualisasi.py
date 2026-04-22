import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from wordcloud import WordCloud, STOPWORDS

import random
import os

# ==============================
# 🔥 FIX REPRODUCIBILITY TOTAL
# ==============================
seed = 42
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# ==============================
# 1️⃣ CONFIG
# ==============================
st.set_page_config(page_title="Analisis Sentimen Shopee", layout="wide")

st.sidebar.title("🧭 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", [
    "📊 Visualisasi Data & Tren",
    "⚙️ Perhitungan Algoritma"
])

# ==============================
# 2️⃣ UPLOAD DATA
# ==============================
uploaded_file = st.sidebar.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])

if uploaded_file:

    # ==============================
    # LOAD DATA
    # ==============================
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df = df.copy()

    # ==============================
    # FIX: SORT DATA (PENTING BANGET)
    # ==============================
    df = df.sort_values(by=df.columns[0]).reset_index(drop=True)

    # ==============================
    # LABEL CLEANING
    # ==============================
    if "Labeling" not in df.columns:
        st.error("Kolom Labeling tidak ditemukan")
        st.stop()

    df["Labeling"] = (
        df["Labeling"]
        .astype(str)
        .str.strip()
        .str.capitalize()
    )

    df["Labeling"] = df["Labeling"].apply(
        lambda x: x if x in ["Positif", "Negatif"] else "Positif"
    )

    # ==============================
    # TIME HANDLING
    # ==============================
    tgl_cols = [c for c in df.columns if c.lower() == "tanggal"]

    if tgl_cols:
        col_tgl = tgl_cols[0]
        df["Tanggal"] = pd.to_datetime(df[col_tgl], errors="coerce")
        df["Tanggal"] = df["Tanggal"].ffill().bfill()

        df["Tahun"] = df["Tanggal"].dt.year
        df["Bulan"] = df["Tanggal"].dt.month
    else:
        df["Tahun"] = 2024
        df["Bulan"] = 1

    nama_bulan = {
        1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun',
        7:'Jul',8:'Agu',9:'Sep',10:'Okt',11:'Nov',12:'Des'
    }

    # ==========================================================
    # 📊 VISUALISASI
    # ==========================================================
    if page == "📊 Visualisasi Data & Tren":

        st.title("Dashboard Sentimen Shopee")

        total = len(df)
        pos = len(df[df["Labeling"] == "Positif"])
        neg = len(df[df["Labeling"] == "Negatif"])

        c1, c2, c3 = st.columns(3)
        c1.metric("Total", total)
        c2.metric("Positif", pos)
        c3.metric("Negatif", neg)

        st.divider()

        tab1, tab2 = st.tabs(["Bulanan", "Tahunan"])

        with tab1:
            data = df.groupby(["Bulan", "Labeling"]).size().reset_index(name="Jumlah")

            fig = px.line(data, x="Bulan", y="Jumlah", color="Labeling", markers=True)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            data = df.groupby(["Tahun", "Labeling"]).size().reset_index(name="Jumlah")

            fig = px.bar(data, x="Tahun", y="Jumlah", color="Labeling", barmode="group")
            st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # ⚙️ MODEL TRAINING
    # ==========================================================
    elif page == "⚙️ Perhitungan Algoritma":

        st.title("Evaluasi Model ML")

        # ==============================
        # CACHE DATA (BIAR TIDAK RANDOM RE-RUN)
        # ==============================
        @st.cache_data
        def prepare_data(df):
            col_text = "stemming" if "stemming" in df.columns else df.columns[0]
            X = df[col_text].fillna("").astype(str)
            y = df["Labeling"]
            return X, y

        # ==============================
        # CACHE MODEL TRAINING
        # ==============================
        @st.cache_resource
        def train_models(X_train, y_train):

            models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=100,
                    random_state=42
                ),

                "SVM Linear": SVC(kernel="linear", random_state=42),

                "SVM RBF": SVC(kernel="rbf", random_state=42),

                "SVM Poly": SVC(kernel="poly", random_state=42),

                # 🔥 FIX IMPORTANT: sigmoid tanpa probability
                "SVM Sigmoid": SVC(kernel="sigmoid", random_state=42)
            }

            for m in models.values():
                m.fit(X_train, y_train)

            return models

        if st.button("🚀 Train Model"):

            X, y = prepare_data(df)

            # ==============================
            # FIX SPLIT STABIL
            # ==============================
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            # ==============================
            # TF-IDF FIXED
            # ==============================
            tfidf = TfidfVectorizer(max_features=5000)

            X_train = tfidf.fit_transform(X_train_raw)
            X_test = tfidf.transform(X_test_raw)

            models = train_models(X_train, y_train)

            results = []

            for name, model in models.items():

                pred = model.predict(X_test)

                acc = accuracy_score(y_test, pred)

                report = classification_report(
                    y_test,
                    pred,
                    output_dict=True
                )

                cm = confusion_matrix(y_test, pred)

                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": report["Positif"]["precision"],
                    "Recall": report["Positif"]["recall"],
                    "F1": report["Positif"]["f1-score"]
                })

                st.subheader(name)

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Acc", f"{acc:.4f}")
                col2.metric("Prec", f"{report['Positif']['precision']:.4f}")
                col3.metric("Rec", f"{report['Positif']['recall']:.4f}")
                col4.metric("F1", f"{report['Positif']['f1-score']:.4f}")

                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
                st.pyplot(fig)

                st.divider()

            res_df = pd.DataFrame(results)

            st.subheader("📊 Perbandingan Model")

            st.dataframe(res_df.style.highlight_max(axis=0))

            best = res_df.loc[res_df["F1"].idxmax()]
            st.success(f"Best Model: {best['Model']}")

else:
    st.info("Upload dataset dulu ya 🙂")
