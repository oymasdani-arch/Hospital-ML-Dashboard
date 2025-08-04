# dashboard.py

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import io
import base64
from datetime import datetime
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from fpdf import FPDF
import warnings
warnings.filterwarnings('ignore')

# ================================
# 📱 Konfigurasi Halaman
# ================================
st.set_page_config(
    page_title="Dashboard Prediksi Lama Rawat Inap",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# 📌 Header Dashboard
# ================================
st.markdown("""
<div class="main-header">
    <h1 style='color: white; text-align: center; margin: 0;'>
        🏥 Dashboard Prediksi Lama Rawat Inap Pasien
    </h1>
    <p style='color: white; text-align: center; margin: 0; font-size: 1.2em;'>
        Sistem Prediksi Berbasis Machine Learning menggunakan Data RME
    </p>
</div>
""", unsafe_allow_html=True)

# ================================
# 🔧 Fungsi Utilitas
# ================================
@st.cache_data
def load_data():
    """Load dan cache data untuk performa yang lebih baik"""
    try:
        df = pd.read_csv("data/data_rme_dummy_baru.csv")
        return df
    except FileNotFoundError:
        st.error("❌ File data tidak ditemukan! Pastikan file 'data/data_rme_dummy_baru.csv' tersedia.")
        st.stop()

@st.cache_data
def prepare_model_data(df):
    """Prepare data untuk model ML"""
    try:
        model_columns = joblib.load("model/model_columns.pkl")
        df_model = df.drop(columns=["Lama_Rawat", "Cepat_Sembuh", "Inisial_Nama"])
        df_model_encoded = pd.get_dummies(df_model)
        df_model_encoded = df_model_encoded.reindex(columns=model_columns, fill_value=0)
        return df_model_encoded
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan! Pastikan model sudah dilatih.")
        st.stop()

@st.cache_data
def load_all_models():
    """Load semua model ML"""
    model_options = ["LinearRegression", "DecisionTree", "RandomForest", "GradientBoosting", "XGBoost"]
    models = {}
    
    for name in model_options:
        try:
            models[name] = joblib.load(f"model/model_{name}.pkl")
        except FileNotFoundError:
            st.warning(f"⚠️ Model {name} tidak ditemukan!")
    
    return models

def calculate_metrics(y_true, y_pred):
    """Hitung metrik evaluasi"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape
    }

# ================================
# 📊 Load Data dan Model
# ================================
df = load_data()
df_model_encoded = prepare_model_data(df)
models = load_all_models()

# ================================
# 🔧 Sidebar - Kontrol Dashboard
# ================================
st.sidebar.markdown("## 🎛️ Kontrol Dashboard")

# Pilih model
model_options = list(models.keys())
selected_model = st.sidebar.selectbox(
    "🤖 Pilih Model Prediksi:",
    model_options,
    help="Pilih model machine learning untuk prediksi"
)

# Filter data
st.sidebar.markdown("### 🔍 Filter Data")

# Filter berdasarkan diagnosa
unique_diagnosa = sorted(df["Diagnosa_ICD"].unique())
selected_diagnosa = st.sidebar.multiselect(
    "Diagnosa ICD:",
    unique_diagnosa,
    default=[],
    help="Pilih satu atau lebih diagnosa untuk filter"
)

# Filter berdasarkan rentang usia
min_age, max_age = int(df["Usia"].min()), int(df["Usia"].max())
age_range = st.sidebar.slider(
    "Rentang Usia:",
    min_age, max_age, (min_age, max_age),
    help="Filter pasien berdasarkan rentang usia"
)

# Filter berdasarkan asuransi
unique_asuransi = sorted(df["Asuransi"].unique())
selected_asuransi = st.sidebar.multiselect(
    "Jenis Asuransi:",
    unique_asuransi,
    default=[],
    help="Filter berdasarkan jenis asuransi"
)

# Apply filters
df_filtered = df.copy()

if selected_diagnosa:
    df_filtered = df_filtered[df_filtered["Diagnosa_ICD"].isin(selected_diagnosa)]

df_filtered = df_filtered[
    (df_filtered["Usia"] >= age_range[0]) & 
    (df_filtered["Usia"] <= age_range[1])
]

if selected_asuransi:
    df_filtered = df_filtered[df_filtered["Asuransi"].isin(selected_asuransi)]

# ================================
# 📈 Overview Statistik
# ================================
st.markdown("## 📊 Overview Data Pasien")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "👥 Total Pasien",
        f"{len(df_filtered):,}",
        delta=f"{len(df_filtered) - len(df):,}" if len(df_filtered) != len(df) else None
    )

with col2:
    avg_rawat = df_filtered["Lama_Rawat"].mean()
    st.metric(
        "⏱️ Rata-rata Rawat Inap",
        f"{avg_rawat:.1f} hari",
        delta=f"{avg_rawat - df['Lama_Rawat'].mean():.1f}" if len(df_filtered) != len(df) else None
    )

with col3:
    cepat_sembuh_pct = (df_filtered["Cepat_Sembuh"].sum() / len(df_filtered)) * 100
    st.metric(
        "🚀 Cepat Sembuh",
        f"{cepat_sembuh_pct:.1f}%",
        delta=f"{cepat_sembuh_pct - (df['Cepat_Sembuh'].sum() / len(df)) * 100:.1f}%" if len(df_filtered) != len(df) else None
    )

with col4:
    unique_diagnosa_count = df_filtered["Diagnosa_ICD"].nunique()
    st.metric(
        "🏥 Jenis Diagnosa",
        f"{unique_diagnosa_count}",
        delta=f"{unique_diagnosa_count - df['Diagnosa_ICD'].nunique()}" if len(df_filtered) != len(df) else None
    )

with col5:
    avg_age = df_filtered["Usia"].mean()
    st.metric(
        "👤 Rata-rata Usia",
        f"{avg_age:.1f} tahun",
        delta=f"{avg_age - df['Usia'].mean():.1f}" if len(df_filtered) != len(df) else None
    )

# ================================
# 🤖 Prediksi dengan Model Terpilih
# ================================
if selected_model in models:
    model = models[selected_model]
    
    # Filter encoded data sesuai dengan filter yang dipilih
    df_filtered_indices = df_filtered.index
    df_model_filtered = df_model_encoded.loc[df_filtered_indices]
    
    # Lakukan prediksi
    predictions = model.predict(df_model_filtered)
    df_filtered = df_filtered.copy()
    df_filtered["Prediksi_Lama_Rawat"] = predictions
    
    # ================================
    # 📊 Evaluasi Model
    # ================================
    st.markdown("## 🧠 Evaluasi Semua Model")
    
    # Evaluasi semua model
    eval_results = []
    for name, model_obj in models.items():
        preds = model_obj.predict(df_model_filtered)
        metrics = calculate_metrics(df_filtered["Lama_Rawat"], preds)
        metrics["Model"] = name
        eval_results.append(metrics)
    
    eval_df = pd.DataFrame(eval_results).set_index("Model")
    
    # Tampilkan tabel evaluasi
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.dataframe(
            eval_df.round(3),
            use_container_width=True
        )
    
    with col2:
        best_model_r2 = eval_df["R²"].idxmax()
        best_r2_score = eval_df.loc[best_model_r2, "R²"]
        
        st.markdown(f"""
        <div class="success-box">
            <h4>🏆 Model Terbaik</h4>
            <p><strong>{best_model_r2}</strong></p>
            <p>R² Score: <strong>{best_r2_score:.3f}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Info metrik
        st.markdown("""
        <div class="info-box">
            <h5>📚 Penjelasan Metrik:</h5>
            <ul>
                <li><strong>MAE</strong>: Mean Absolute Error</li>
                <li><strong>MSE</strong>: Mean Squared Error</li>
                <li><strong>RMSE</strong>: Root Mean Squared Error</li>
                <li><strong>R²</strong>: Coefficient of Determination</li>
                <li><strong>MAPE</strong>: Mean Absolute Percentage Error</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualisasi perbandingan model
    fig_comparison = go.Figure()
    
    metrics_to_plot = ['MAE', 'RMSE', 'MAPE']
    for metric in metrics_to_plot:
        fig_comparison.add_trace(go.Bar(
            name=metric,
            x=eval_df.index,
            y=eval_df[metric],
            text=eval_df[metric].round(2),
            textposition='auto',
        ))
    
    fig_comparison.update_layout(
        title="📊 Perbandingan Metrik Error Model",
        xaxis_title="Model",
        yaxis_title="Nilai Error",
        barmode='group',
        height=400
    )
    
    st.plotly_chart(fig_comparison, use_container_width=True)
    
    # ================================
    # 📈 Analisis Prediksi
    # ================================
    st.markdown("## 🔮 Analisis Hasil Prediksi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot: Aktual vs Prediksi
        fig_scatter = px.scatter(
            df_filtered,
            x="Lama_Rawat",
            y="Prediksi_Lama_Rawat",
            hover_data=["Inisial_Nama", "Diagnosa_ICD", "Usia"],
            title="🎯 Aktual vs Prediksi Lama Rawat Inap",
            labels={
                "Lama_Rawat": "Lama Rawat Aktual (hari)",
                "Prediksi_Lama_Rawat": "Prediksi Lama Rawat (hari)"
            }
        )
        
        # Tambahkan garis diagonal untuk referensi prediksi sempurna
        min_val = min(df_filtered["Lama_Rawat"].min(), df_filtered["Prediksi_Lama_Rawat"].min())
        max_val = max(df_filtered["Lama_Rawat"].max(), df_filtered["Prediksi_Lama_Rawat"].max())
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Prediksi Sempurna',
            line=dict(dash='dash', color='red')
        ))
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Distribusi error
        errors = df_filtered["Lama_Rawat"] - df_filtered["Prediksi_Lama_Rawat"]
        fig_hist = px.histogram(
            x=errors,
            nbins=30,
            title="📊 Distribusi Error Prediksi",
            labels={"x": "Error (Aktual - Prediksi)", "y": "Frekuensi"}
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="red", 
                          annotation_text="Error = 0")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # ================================
    # 📋 Tabel Detail Pasien
    # ================================
    st.markdown("## 📋 Detail Data Pasien dan Prediksi")
    
    # Kontrol tampilan tabel
    col1, col2, col3 = st.columns(3)
    with col1:
        show_rows = st.selectbox("Tampilkan baris:", [10, 25, 50, 100, "Semua"])
    with col2:
        sort_by = st.selectbox("Urutkan berdasarkan:", 
                              ["Lama_Rawat", "Prediksi_Lama_Rawat", "Usia", "Diagnosa_ICD"])
    with col3:
        sort_order = st.selectbox("Urutan:", ["Descending", "Ascending"])
    
    # Prepare data untuk tabel
    display_columns = ["Inisial_Nama", "Usia", "Jenis_Kelamin", "Asuransi", 
                      "Diagnosa_ICD", "Tindakan", "Lama_Rawat", "Prediksi_Lama_Rawat", "Cepat_Sembuh"]
    
    df_display = df_filtered[display_columns].copy()
    df_display["Error"] = df_display["Lama_Rawat"] - df_display["Prediksi_Lama_Rawat"]
    df_display["Error_Absolut"] = abs(df_display["Error"])
    
    # Sorting
    ascending = sort_order == "Ascending"
    df_display = df_display.sort_values(sort_by, ascending=ascending)
    
    # Limit rows
    if show_rows != "Semua":
        df_display = df_display.head(show_rows)
    
    # Format untuk tampilan yang lebih baik
    df_display["Lama_Rawat"] = df_display["Lama_Rawat"].apply(lambda x: f"{x:.0f} hari")
    df_display["Prediksi_Lama_Rawat"] = df_display["Prediksi_Lama_Rawat"].apply(lambda x: f"{x:.1f} hari")
    df_display["Error"] = df_display["Error"].apply(lambda x: f"{x:+.1f}")
    df_display["Cepat_Sembuh"] = df_display["Cepat_Sembuh"].apply(lambda x: "✅ Ya" if x == 1 else "❌ Tidak")
    
    st.dataframe(
        df_display.drop(columns=["Error_Absolut"]),
        use_container_width=True,
        hide_index=True
    )
    
    # ================================
    # 📊 Analisis Mendalam
    # ================================
    st.markdown("## 📊 Analisis Mendalam")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏥 Diagnosa", "💊 Tindakan", "👥 Demografi", "💰 Asuransi"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top diagnosa
            top_diagnosa = df_filtered["Diagnosa_ICD"].value_counts().head(10)
            fig_diagnosa = px.bar(
                x=top_diagnosa.values,
                y=top_diagnosa.index,
                orientation='h',
                title="🔝 Top 10 Diagnosa Terbanyak",
                labels={"x": "Jumlah Pasien", "y": "Diagnosa ICD"}
            )
            st.plotly_chart(fig_diagnosa, use_container_width=True)
        
        with col2:
            # Rata-rata lama rawat per diagnosa
            avg_by_diagnosa = df_filtered.groupby("Diagnosa_ICD")["Lama_Rawat"].mean().sort_values(ascending=False).head(10)
            fig_avg_diagnosa = px.bar(
                x=avg_by_diagnosa.values,
                y=avg_by_diagnosa.index,
                orientation='h',
                title="⏱️ Rata-rata Lama Rawat per Diagnosa",
                labels={"x": "Rata-rata Hari", "y": "Diagnosa ICD"}
            )
            st.plotly_chart(fig_avg_diagnosa, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Tindakan terbanyak
            top_tindakan = df_filtered["Tindakan"].value_counts().head(10)
            fig_tindakan = px.bar(
                x=top_tindakan.values,
                y=top_tindakan.index,
                orientation='h',
                title="🔝 Top 10 Tindakan Terbanyak",
                labels={"x": "Jumlah Pasien", "y": "Tindakan"}
            )
            st.plotly_chart(fig_tindakan, use_container_width=True)
        
        with col2:
            # Rata-rata lama rawat per tindakan
            avg_by_tindakan = df_filtered.groupby("Tindakan")["Lama_Rawat"].mean().sort_values(ascending=False).head(10)
            fig_avg_tindakan = px.bar(
                x=avg_by_tindakan.values,
                y=avg_by_tindakan.index,
                orientation='h',
                title="⏱️ Rata-rata Lama Rawat per Tindakan",
                labels={"x": "Rata-rata Hari", "y": "Tindakan"}
            )
            st.plotly_chart(fig_avg_tindakan, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribusi usia
            fig_age = px.histogram(
                df_filtered,
                x="Usia",
                nbins=20,
                title="📊 Distribusi Usia Pasien",
                labels={"x": "Usia (tahun)", "y": "Jumlah Pasien"}
            )
            st.plotly_chart(fig_age, use_container_width=True)
        
        with col2:
            # Lama rawat berdasarkan jenis kelamin
            gender_stats = df_filtered.groupby("Jenis_Kelamin").agg({
                "Lama_Rawat": ["mean", "count"]
            }).round(2)
            gender_stats.columns = ["Rata-rata Lama Rawat", "Jumlah Pasien"]
            
            fig_gender = px.bar(
                gender_stats,
                x=gender_stats.index,
                y="Rata-rata Lama Rawat",
                title="👫 Rata-rata Lama Rawat berdasasrkan Jenis Kelamin",
                labels={"x": "Jenis Kelamin", "y": "Rata-rata Hari"}
            )
            st.plotly_chart(fig_gender, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        
        with col1:
            # Komposisi asuransi
            asuransi_count = df_filtered["Asuransi"].value_counts()
            fig_asuransi = px.pie(
                values=asuransi_count.values,
                names=asuransi_count.index,
                title="💰 Komposisi Jenis Asuransi"
            )
            st.plotly_chart(fig_asuransi, use_container_width=True)
        
        with col2:
            # Rata-rata lama rawat per asuransi
            avg_by_asuransi = df_filtered.groupby("Asuransi")["Lama_Rawat"].mean().sort_values(ascending=False)
            fig_avg_asuransi = px.bar(
                x=avg_by_asuransi.values,
                y=avg_by_asuransi.index,
                orientation='h',
                title="⏱️ Rata-rata Lama Rawat per Asuransi",
                labels={"x": "Rata-rata Hari", "y": "Jenis Asuransi"}
            )
            st.plotly_chart(fig_avg_asuransi, use_container_width=True)
    
    # ================================
    # 📊 Insights dan Rekomendasi
    # ================================
    st.markdown("## 💡 Insights dan Rekomendasi")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pasien dengan prediksi error tertinggi
        df_high_error = df_display.copy()
        df_high_error["Error_Absolut"] = df_filtered["Lama_Rawat"] - df_filtered["Prediksi_Lama_Rawat"]
        df_high_error["Error_Absolut"] = abs(df_high_error["Error_Absolut"])
        top_errors = df_high_error.nlargest(5, "Error_Absolut")
        
        st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Kasus dengan Error Prediksi Tertinggi</h4>
            <p>Perlu perhatian khusus untuk kasus-kasus berikut:</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.dataframe(
            top_errors[["Inisial_Nama", "Diagnosa_ICD", "Lama_Rawat", "Prediksi_Lama_Rawat", "Error"]],
            hide_index=True,
            use_container_width=True
        )
    
    with col2:
        # Statistik cepat sembuh
        cepat_sembuh_stats = df_filtered.groupby("Cepat_Sembuh").agg({
            "Lama_Rawat": ["count", "mean", "std"]
        }).round(2)
        
        st.markdown("""
        <div class="info-box">
            <h4>🚀 Analisis Cepat Sembuh</h4>
        </div>
        """, unsafe_allow_html=True)
        
        cepat_sembuh_pct = (df_filtered["Cepat_Sembuh"].sum() / len(df_filtered)) * 100
        st.write(f"**Persentase Pasien Cepat Sembuh:** {cepat_sembuh_pct:.1f}%")
        
        avg_cepat = df_filtered[df_filtered["Cepat_Sembuh"] == 1]["Lama_Rawat"].mean()
        avg_normal = df_filtered[df_filtered["Cepat_Sembuh"] == 0]["Lama_Rawat"].mean()
        
        st.write(f"**Rata-rata Lama Rawat (Cepat Sembuh):** {avg_cepat:.1f} hari")
        st.write(f"**Rata-rata Lama Rawat (Normal):** {avg_normal:.1f} hari")
    
    # ================================
    # 📤 Export dan Download
    # ================================
    st.markdown("## 📤 Export Data dan Laporan")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Download Data Prediksi (Excel)", use_container_width=True):
            # Prepare data untuk export
            export_data = df_filtered.select_dtypes(include=[np.number, object]).copy()
            
            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                export_data.to_excel(writer, sheet_name='Data_Prediksi', index=False)
                eval_df.to_excel(writer, sheet_name='Evaluasi_Model')
            
            output.seek(0)
            
            st.download_button(
                label="📥 Download Excel",
                data=output.getvalue(),
                file_name=f"prediksi_rawat_inap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        if st.button("📈 Download Laporan Model (CSV)", use_container_width=True):
            csv = eval_df.to_csv()
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"evaluasi_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.button("📄 Generate Laporan PDF", use_container_width=True):
            st.info("🔄 Fitur generate PDF akan segera tersedia!")
    
    # ================================
    # 📚 Informasi Tambahan
    # ================================
    with st.expander("📚 Informasi Model dan Metodologi"):
        st.markdown("""
        ### 🎯 Tujuan Penelitian
        Dashboard ini dikembangkan untuk memprediksi lama rawat inap pasien menggunakan berbagai algoritma machine learning 
        berdasarkan data Rekam Medis Elektronik (RME).
        
        ### 🤖 Model yang Digunakan
        - **Linear Regression**: Model regresi linear sederhana
        - **Decision Tree**: Pohon keputusan untuk prediksi
        - **Random Forest**: Ensemble dari multiple decision trees
        - **Gradient Boosting**: Boosting algorithm untuk meningkatkan akurasi
        - **XGBoost**: Extreme Gradient Boosting dengan optimisasi tinggi
        
        ### 📊 Metrik Evaluasi
        - **MAE (Mean Absolute Error)**: Rata-rata error absolut
        - **MSE (Mean Squared Error)**: Rata-rata kuadrat error
        - **RMSE (Root Mean Squared Error)**: Akar dari MSE
        - **R² (Coefficient of Determination)**: Proporsi varians yang dijelaskan model
        - **MAPE (Mean Absolute Percentage Error)**: Rata-rata persentase error absolut
        
        ### 🔍 Fitur Input
        Model menggunakan fitur-fitur berikut untuk prediksi:
        - Usia pasien
        - Jenis kelamin
        - Diagnosa ICD (International Classification of Diseases)
        - Jenis tindakan medis
        - Jenis asuransi
        - Data demografi lainnya
        
        ### 📈 Interpretasi Hasil
        - **R² > 0.8**: Model sangat baik
        - **R² 0.6-0.8**: Model baik
        - **R² 0.4-0.6**: Model cukup
        - **R² < 0.4**: Model perlu perbaikan
        
        ### ⚠️ Catatan Penting
        - Hasil prediksi hanya untuk keperluan penelitian dan analisis
        - Keputusan medis tetap harus melibatkan tenaga medis profesional
        - Model perlu divalidasi lebih lanjut sebelum implementasi klinis
        """)
    
    # ================================
    # 🔍 Fitur Prediksi Individual
    # ================================
    st.markdown("## 🔮 Prediksi Individual Pasien Baru")
    
    with st.expander("➕ Tambah Prediksi Pasien Baru", expanded=False):
        st.markdown("### 📝 Input Data Pasien Baru")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            new_age = st.number_input("Usia", min_value=0, max_value=120, value=30)
            new_gender = st.selectbox("Jenis Kelamin", df["Jenis_Kelamin"].unique())
            new_insurance = st.selectbox("Asuransi", df["Asuransi"].unique())
        
        with col2:
            new_diagnosis = st.selectbox("Diagnosa ICD", df["Diagnosa_ICD"].unique())
            new_treatment = st.selectbox("Tindakan", df["Tindakan"].unique())
        
        with col3:
            st.markdown("### 🎯 Hasil Prediksi")
            
            if st.button("🔮 Lakukan Prediksi", use_container_width=True):
                # Buat dataframe untuk pasien baru
                new_patient_data = pd.DataFrame({
                    'Usia': [new_age],
                    'Jenis_Kelamin': [new_gender],
                    'Asuransi': [new_insurance],
                    'Diagnosa_ICD': [new_diagnosis],
                    'Tindakan': [new_treatment]
                })
                
                # Encode data baru
                new_patient_encoded = pd.get_dummies(new_patient_data)
                new_patient_encoded = new_patient_encoded.reindex(columns=df_model_encoded.columns, fill_value=0)
                
                # Prediksi dengan semua model
                predictions_all = {}
                for name, model_obj in models.items():
                    pred = model_obj.predict(new_patient_encoded)[0]
                    predictions_all[name] = pred
                
                # Tampilkan hasil
                st.success(f"✅ Prediksi berhasil dibuat!")
                
                for name, pred in predictions_all.items():
                    confidence = "Tinggi" if name == best_model_r2 else "Sedang"
                    st.metric(
                        f"{name}",
                        f"{pred:.1f} hari",
                        delta=f"Confidence: {confidence}"
                    )
                
                # Rekomendasi
                avg_pred = np.mean(list(predictions_all.values()))
                if avg_pred <= 3:
                    st.success("🟢 **Prediksi: Rawat Inap Singkat** - Kemungkinan cepat sembuh tinggi")
                elif avg_pred <= 7:
                    st.warning("🟡 **Prediksi: Rawat Inap Sedang** - Perlu monitoring rutin")
                else:
                    st.error("🔴 **Prediksi: Rawat Inap Panjang** - Perlu perhatian khusus")
    
    # ================================
    # 📋 Ringkasan Executive
    # ================================
    st.markdown("## 📋 Ringkasan Executive")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"""
        ### 🎯 Hasil Utama Analisis
        
        **Dataset yang Dianalisis:**
        - Total pasien: {len(df_filtered):,} dari {len(df):,} pasien
        - Periode data: Data RME historis
        - Jenis diagnosa: {df_filtered['Diagnosa_ICD'].nunique()} kategori
        
        **Performa Model Terbaik ({best_model_r2}):**
        - R² Score: {eval_df.loc[best_model_r2, 'R²']:.3f}
        - MAE: {eval_df.loc[best_model_r2, 'MAE']:.2f} hari
        - RMSE: {eval_df.loc[best_model_r2, 'RMSE']:.2f} hari
        
        **Insights Kunci:**
        - Rata-rata lama rawat inap: {df_filtered['Lama_Rawat'].mean():.1f} hari
        - Persentase pasien cepat sembuh: {(df_filtered['Cepat_Sembuh'].sum() / len(df_filtered)) * 100:.1f}%
        - Diagnosa tersering: {df_filtered['Diagnosa_ICD'].mode().iloc[0]}
        """)
    
    with col2:
        st.markdown("""
        <div class="success-box">
            <h4>✅ Manfaat Sistem</h4>
            <ul>
                <li>Perencanaan kapasitas RS</li>
                <li>Optimasi sumber daya</li>
                <li>Prediksi biaya perawatan</li>
                <li>Monitoring kualitas layanan</li>
                <li>Deteksi dini kasus kompleks</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # ================================
    # 📞 Footer dan Kontak
    # ================================
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;'>
        <h3>🎓 Dashboard Prediksi Lama Rawat Inap Pasien</h3>
        <p><strong>Sistem Prediksi Berbasis Machine Learning untuk Data RME</strong></p>
        <p><em>Dashboard ini dikembangkan untuk keperluan penelitian skripsi</em></p>
        <p>📊 Dashboard dibuat dengan Streamlit | 🤖 Powered by Scikit-learn & XGBoost</p>
        <p><small>⚠️ Disclaimer: Hasil prediksi hanya untuk keperluan penelitian dan tidak menggantikan penilaian medis profesional</small></p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.error(f"❌ Model {selected_model} tidak tersedia!")
    st.info("💡 Pastikan semua model sudah dilatih dengan menjalankan script `model_rme.py`")