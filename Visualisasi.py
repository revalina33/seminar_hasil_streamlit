import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from wordcloud import WordCloud, STOPWORDS
import plotly.express as px

# ==============================
# 1️⃣ CONFIG HALAMAN
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
st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload file CSV / Excel",
    type=["csv", "xlsx"]
)

df = None

if uploaded_file:
    try:
        if uploaded_file.name.endswith("csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        st.stop()

    # ==============================
    # CLEAN LABELING
    # ==============================
    if "Labeling" not in df.columns:
        st.error("Kolom 'Labeling' tidak ditemukan!")
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
    # HANDLE TANGGAL
    # ==============================
    if "Tanggal" in df.columns:
        df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
        df["Tanggal"] = df["Tanggal"].fillna(method="ffill")

        df["Tahun"] = df["Tanggal"].dt.year
        df["Bulan"] = df["Tanggal"].dt.month
    else:
        df["Tahun"] = 2024
        df["Bulan"] = 1

    bulan_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "Mei", 6: "Jun", 7: "Jul", 8: "Agu",
        9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"
    }

# ==============================
# JIKA BELUM UPLOAD
# ==============================
else:
    st.warning("📌 Silakan upload file terlebih dahulu")
    st.stop()

# ==============================
# 3️⃣ VISUALISASI
# ==============================
if page == "📊 Visualisasi Data & Tren":

    st.title("📊 Dashboard Sentimen Shopee")

    total = len(df)
    pos = len(df[df["Labeling"] == "Positif"])
    neg = len(df[df["Labeling"] == "Negatif"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Total", total)
    c2.metric("Positif", pos)
    c3.metric("Negatif", neg)

    st.divider()

    tab1, tab2, tab3 = st.tabs([
        "📅 Bulanan", "📈 Tahunan", "🔍 WordCloud"
    ])

    # ================= BULANAN =================
    with tab1:
        data = df.groupby(["Bulan", "Labeling"]).size().reset_index(name="Jumlah")
        data["Nama Bulan"] = data["Bulan"].map(bulan_map)

        fig = px.line(
            data,
            x="Nama Bulan",
            y="Jumlah",
            color="Labeling",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # ================= TAHUNAN =================
    with tab2:
        data = df.groupby(["Tahun", "Labeling"]).size().reset_index(name="Jumlah")

        fig = px.bar(
            data,
            x="Tahun",
            y="Jumlah",
            color="Labeling",
            barmode="group"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ================= WORDCLOUD =================
    with tab3:
        if "stemming" in df.columns:
            text = " ".join(df["stemming"].astype(str))
        else:
            text = " ".join(df.iloc[:, 0].astype(str))

        wc = WordCloud(
            width=800,
            height=400,
            stopwords=STOPWORDS,
            background_color="white"
        ).generate(text)

        fig, ax = plt.subplots()
        ax.imshow(wc)
        ax.axis("off")
        st.pyplot(fig)

# ==============================
# 4️⃣ MACHINE LEARNING
# ==============================
elif page == "⚙️ Perhitungan Algoritma":

    st.title("⚙️ Model Training (RF & SVM)")

    if st.button("🚀 Jalankan Model"):

        text_col = "stemming" if "stemming" in df.columns else df.columns[0]

        X = df[text_col].fillna("")
        y = df["Labeling"]

        # ==============================
        # FIX ERROR STRATIFY
        # ==============================
        if len(y.unique()) < 2:
            st.error("Label harus ada Positif & Negatif")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        tfidf = TfidfVectorizer(max_features=5000)
        X_train = tfidf.fit_transform(X_train)
        X_test = tfidf.transform(X_test)

        models = {
            "Random Forest": RandomForestClassifier(),
            "SVM Linear": SVC(kernel="linear"),
            "SVM RBF": SVC(kernel="rbf")
        }

        hasil = []

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)
            report = classification_report(y_test, pred, output_dict=True)

            hasil.append({
                "Model": name,
                "Accuracy": acc,
                "Precision": report["Positif"]["precision"],
                "Recall": report["Positif"]["recall"],
                "F1": report["Positif"]["f1-score"]
            })

            st.subheader(name)
            st.write("Accuracy:", acc)

            cm = confusion_matrix(y_test, pred, labels=["Negatif", "Positif"])

            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            st.pyplot(fig)

        # ==============================
        # RESULT TABLE
        # ==============================
        res = pd.DataFrame(hasil)

        st.subheader("📊 Perbandingan Model")
        st.dataframe(res)

        best = res.loc[res["F1"].idxmax()]
        st.success(f"🏆 Model terbaik: {best['Model']}")
