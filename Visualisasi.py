import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Tambahan untuk load model
import os      # Tambahan untuk cek file
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px

# ==============================
# 1️⃣ Konfigurasi Halaman
# ==============================
st.set_page_config(page_title="Analisis Sentimen Shopee", layout="wide")

st.sidebar.title("🧭 Navigasi")
page = st.sidebar.radio("Pilih Halaman:", ["📊 Visualisasi Data & Tren", "⚙️ Perhitungan Algoritma"])

# ==============================
# 2️⃣ Upload Data
# ==============================
st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("📁 Upload file ulasan (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Clean data awal
    if 'Labeling' in df.columns:
        df['Labeling'] = df['Labeling'].astype(str).str.strip().str.capitalize()
        valid_labels = ['Positif', 'Negatif']
        df['Labeling'] = df['Labeling'].apply(lambda x: x if x in valid_labels else 'Positif')
    else:
        st.error("Kolom 'Labeling' tidak ditemukan!")
        st.stop()

    # Penanganan Waktu
    tgl_cols = [c for c in df.columns if c.lower() == 'tanggal']
    if tgl_cols:
        col_tgl = tgl_cols[0]
        df['Tanggal_Converted'] = pd.to_datetime(df[col_tgl].astype(str).str.strip(), errors='coerce')
        df['Tanggal_Converted'] = df['Tanggal_Converted'].ffill().bfill()
        df['Tahun'] = df['Tanggal_Converted'].dt.year
        df['Bulan'] = df['Tanggal_Converted'].dt.month
    else:
        df['Tahun'] = df.get('Tahun', 2024)
        df['Bulan'] = df.get('Bulan', 1)

    nama_bulan = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'Mei',6:'Jun',
                  7:'Jul',8:'Agu',9:'Sep',10:'Okt',11:'Nov',12:'Des'}

    # ==============================
    # 3️⃣ Halaman Visualisasi
    # ==============================
    if page == "📊 Visualisasi Data & Tren":
        st.title("📊 Dashboard Analisis Sentimen Shopee")
        
        total_n = len(df) 
        pos_n = len(df[df['Labeling']=='Positif'])
        neg_n = len(df[df['Labeling']=='Negatif'])
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Ulasan", f"{total_n} 📄")
        m2.metric("Sentimen Positif", f"{pos_n}", f"{(pos_n/total_n*100 if total_n>0 else 0):.1f}%")
        m3.metric("Sentimen Negatif", f"{neg_n}", f"-{(neg_n/total_n*100 if total_n>0 else 0):.1f}%", delta_color="inverse")
        st.divider()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📅 Tren Bulanan", "📈 Tren Tahunan", "🔍 WordCloud", "🔎 Explore Data", "🎯 Validasi Rating"
        ])

        with tab1:
            st.subheader("Grafik Garis Sentimen Bulanan")
            years = sorted(df['Tahun'].dropna().unique())
            if years:
                filter_thn = st.selectbox("Pilih Tahun:", years, key='sb_thn')
                df_thn = df[df['Tahun'] == filter_thn]
                if not df_thn.empty:
                    monthly_data = df_thn.groupby(['Bulan', 'Labeling']).size().reset_index(name='Jumlah')
                    monthly_data['Nama Bulan'] = monthly_data['Bulan'].map(nama_bulan)
                    fig_mon = px.line(monthly_data, x='Nama Bulan', y='Jumlah', color='Labeling', markers=True,
                                      color_discrete_map={'Negatif':'#ff4b4b','Positif':'#00cc96'},
                                      category_orders={"Nama Bulan": list(nama_bulan.values())})
                    st.plotly_chart(fig_mon, use_container_width=True)

        with tab2:
            st.subheader("Tren Sentimen Tahunan")
            chart_type = st.radio("Tipe grafik:", ["Bar Chart", "Pie Chart"], horizontal=True)
            trend_data = df.groupby(['Tahun','Labeling']).size().reset_index(name='Jumlah')
            if chart_type == "Bar Chart":
                fig_trend = px.bar(trend_data, x='Tahun', y='Jumlah', color='Labeling', barmode='group',
                                   color_discrete_map={'Negatif':'#ff4b4b','Positif':'#00cc96'})
                st.plotly_chart(fig_trend, use_container_width=True)
            else:
                sel_thn_pie = st.selectbox("Pilih Tahun:", ["Semua"] + sorted(df['Tahun'].unique().tolist()))
                df_pie = trend_data if sel_thn_pie == "Semua" else trend_data[trend_data['Tahun'] == sel_thn_pie]
                fig_pie = px.pie(df_pie, names='Labeling', values='Jumlah', color='Labeling',
                                 color_discrete_map={'Negatif':'#ff4b4b','Positif':'#00cc96'})
                st.plotly_chart(fig_pie, use_container_width=True)

        with tab3:
            st.subheader("🔍 Analisis Keluhan & Topik Utama")
            f1, f2, f3 = st.columns(3)
            with f1: wc_thn = st.selectbox("Pilih Tahun:", sorted(df['Tahun'].unique()), key='wc_thn_new')
            with f2: wc_bln = st.selectbox("Pilih Bulan:", list(nama_bulan.keys()), format_func=lambda x: nama_bulan[x], key='wc_bln_new')
            with f3: wc_lbl = st.radio("Sentimen:", ["Negatif", "Positif"], horizontal=True, key='wc_lbl_new')

            df_wc = df[(df['Tahun'] == wc_thn) & (df['Bulan'] == wc_bln) & (df['Labeling'] == wc_lbl)]
            if not df_wc.empty:
                col_text = 'stemming' if 'stemming' in df.columns else df.columns[0]
                text_data = " ".join(df_wc[col_text].astype(str).fillna(''))
                custom_stop = set(["shopee", "yang", "dan", "nya", "itu", "saya", "jadi", "untuk", "ada", "ini", "kalau", "gak", "aja", "ke", "di"])
                all_stop = STOPWORDS.union(custom_stop)

                col_viz1, col_viz2 = st.columns([2, 1])
                with col_viz1:
                    wc = WordCloud(width=800, height=450, background_color='white', stopwords=all_stop,
                                  colormap='Reds' if wc_lbl=='Negatif' else 'Greens').generate(text_data)
                    fig_wc, ax = plt.subplots()
                    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
                    st.pyplot(fig_wc)
                with col_viz2:
                    words = [w for w in text_data.split() if w.lower() not in all_stop and len(w) > 2]
                    if words:
                        word_freq = pd.Series(words).value_counts().head(10).reset_index()
                        word_freq.columns = ['Kata', 'Jumlah']
                        fig_bar = px.bar(word_freq, x='Jumlah', y='Kata', orientation='h',
                                         color_discrete_sequence=['#ff4b4b' if wc_lbl=='Negatif' else '#00cc96'])
                        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, height=400)
                        st.plotly_chart(fig_bar, use_container_width=True)

        with tab4:
            st.subheader("🔎 Penelusuran Detail Ulasan")
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1: f_thn = st.selectbox("Filter Tahun:", ["Semua"] + sorted(df['Tahun'].unique().tolist()))
            with col_f2: f_bln = st.selectbox("Filter Bulan:", ["Semua"] + list(nama_bulan.values()))
            with col_f3: f_lbl = st.selectbox("Filter Sentimen:", ["Semua", "Positif", "Negatif"])
            
            query = st.text_input("🔍 Cari kata kunci (Contoh: kurir, lemot, bug):")
            df_filtered = df.copy()
            if f_thn != "Semua": df_filtered = df_filtered[df_filtered['Tahun'] == f_thn]
            if f_bln != "Semua": 
                inv_nama_bulan = {v: k for k, v in nama_bulan.items()}
                df_filtered = df_filtered[df_filtered['Bulan'] == inv_nama_bulan[f_bln]]
            if f_lbl != "Semua": df_filtered = df_filtered[df_filtered['Labeling'] == f_lbl]
            
            if query:
                search_target = 'stemming' if 'stemming' in df.columns else df.columns[0]
                df_filtered = df_filtered[df_filtered[search_target].astype(str).str.contains(query, case=False, na=False)]
            st.dataframe(df_filtered, use_container_width=True)

        with tab5:
            st.subheader("🎯 Analisis Kesesuaian Rating vs Labeling")
            rating_cols = [c for c in df.columns if 'rating' in c.lower() or 'star' in c.lower()]
            if rating_cols:
                col_rating = rating_cols[0]
                def cek_kesesuaian(row):
                    r, l = row[col_rating], row['Labeling']
                    if (r <= 2 and l == 'Negatif') or (r >= 4 and l == 'Positif'): return "Sesuai"
                    elif r == 3: return f"Rating 3 ({l})"
                    else: return "Tidak Sesuai (Anomali)"

                df['Status_Validasi'] = df.apply(cek_kesesuaian, axis=1)
                v_col1, v_col2 = st.columns([1, 2])
                with v_col1:
                    val_counts = df['Status_Validasi'].value_counts().reset_index()
                    fig_val = px.pie(val_counts, names='Status_Validasi', values='count', color='Status_Validasi',
                                    color_discrete_map={'Sesuai':'#00cc96', 'Tidak Sesuai (Anomali)':'#ff4b4b'})
                    st.plotly_chart(fig_val, use_container_width=True)
                with v_col2:
                    dist_rating = df.groupby([col_rating, 'Labeling']).size().reset_index(name='Jumlah')
                    fig_dist = px.bar(dist_rating, x=col_rating, y='Jumlah', color='Labeling', barmode='group', text_auto=True,
                                     color_discrete_map={'Negatif':'#ff4b4b','Positif':'#00cc96'})
                    st.plotly_chart(fig_dist, use_container_width=True)

    # ==============================
    # 4️⃣ Halaman Algoritma
    # ==============================
    elif page == "⚙️ Perhitungan Algoritma":
        st.title("⚙️ Evaluasi Performa Model")
        tab_eval, tab_tfidf = st.tabs(["🚀 Pelatihan & Skor", "🔢 Preview Bobot TF-IDF"])

        with tab_eval:
            st.write(f"Dataset: **{len(df)}** ulasan.")
            
            # --- TIPS UNTUK USER ---
            st.info("💡 Agar hasil VS Code & GitHub sama persis: Pastikan file `requirements.txt` berisi `scikit-learn==1.7.2`.")

            if st.button("🚀 Jalankan Komputasi"):
                with st.spinner("Menghitung..."):
                    col_text = 'stemming' if 'stemming' in df.columns else df.columns[0]
                    X = df[col_text].fillna('')
                    y = df['Labeling']

                    # Stratify=y memastikan distribusi label di Train & Test selalu sama
                    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )

                    tfidf = TfidfVectorizer(max_features=5000)
                    X_train = tfidf.fit_transform(X_train_raw)
                    X_test = tfidf.transform(X_test_raw)

                    st.session_state['tfidf_matrix'] = X_train
                    st.session_state['tfidf_features'] = tfidf.get_feature_names_out()

                    # Kunci random_state di semua model
                    models = {
                        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                        'SVM Linear': SVC(kernel='linear', probability=True, random_state=42),
                        'SVM RBF': SVC(kernel='rbf', probability=True, random_state=42),
                        'SVM Polynomial': SVC(kernel='poly', probability=True, random_state=42),
                        'SVM Sigmoid': SVC(kernel='sigmoid', probability=True, random_state=42)
                    }

                    all_metrics = []
                    for name, model in models.items():
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        acc = accuracy_score(y_test, y_pred)
                        rep = classification_report(y_test, y_pred, output_dict=True)
                        cm = confusion_matrix(y_test, y_pred, labels=['Negatif', 'Positif'])

                        all_metrics.append({
                            'Model': name, 'Akurasi': acc, 
                            'Precision': rep['Positif']['precision'], 
                            'Recall': rep['Positif']['recall'], 
                            'F1-Score': rep['Positif']['f1-score']
                        })
                        
                        # Tampilkan hasil per model
                        with st.expander(f"📍 Lihat Detail: {name}"):
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Accuracy", f"{acc:.4f}")
                            m2.metric("Precision", f"{rep['Positif']['precision']:.4f}")
                            m3.metric("Recall", f"{rep['Positif']['recall']:.4f}")
                            m4.metric("F1-Score", f"{rep['Positif']['f1-score']:.4f}")
                            
                            fig_cm, ax = plt.subplots(figsize=(4,3))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                        xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
                            st.pyplot(fig_cm)

                    # Tabel Perbandingan
                    res_df = pd.DataFrame(all_metrics)
                    st.subheader("📊 Perbandingan Akhir")
                    st.dataframe(res_df.set_index('Model').style.highlight_max(axis=0, color='lightgreen').format("{:.4f}"), use_container_width=True)
                    
                    best_m = res_df.loc[res_df['F1-Score'].idxmax()]
                    st.success(f"🏆 Terbaik: **{best_m['Model']}** (F1: {best_m['F1-Score']:.4f})")

        with tab_tfidf:
            if 'tfidf_matrix' in st.session_state:
                weights = np.asarray(st.session_state['tfidf_matrix'].mean(axis=0)).ravel()
                df_tfidf = pd.DataFrame({'Kata': st.session_state['tfidf_features'], 'Bobot': weights})
                df_tfidf = df_tfidf.sort_values('Bobot', ascending=False).head(20)
                
                fig_t = px.bar(df_tfidf, x='Bobot', y='Kata', orientation='h', color='Bobot', color_continuous_scale='Viridis')
                fig_t.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_t, use_container_width=True)
            else:
                st.warning("Klik 'Jalankan Komputasi' di tab pertama.")
