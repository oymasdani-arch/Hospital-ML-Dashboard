# README.md

# 🏥 Dashboard Prediksi Lama Rawat Inap Pasien

## 🎯 Deskripsi Proyek

Dashboard ini dikembangkan untuk memprediksi lama rawat inap pasien menggunakan berbagai algoritma machine learning berdasarkan data Rekam Medis Elektronik (RME). Sistem ini dirancang untuk membantu rumah sakit dalam:

- 📊 Perencanaan kapasitas dan sumber daya
- 💰 Estimasi biaya perawatan
- 🎯 Prediksi kebutuhan perawatan pasien
- 📈 Analisis pola rawat inap

## 🚀 Fitur Utama

### 📊 Dashboard Interaktif

- **Multi-Model Prediction**: Mendukung 10+ algoritma ML
- **Real-time Filtering**: Filter berdasarkan diagnosa, usia, asuransi
- **Interactive Visualizations**: Chart dan grafik interaktif dengan Plotly
- **Responsive Design**: Tampilan responsif untuk berbagai ukuran layar

### 🤖 Machine Learning Models

- Linear Regression
- Ridge & Lasso Regression
- Decision Tree
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- K-Nearest Neighbors
- Support Vector Regression

### 📈 Analisis Mendalam

- **Model Comparison**: Perbandingan performa semua model
- **Feature Importance**: Analisis fitur yang paling berpengaruh
- **Error Analysis**: Distribusi dan analisis error prediksi
- **Patient Insights**: Analisis berdasarkan demografi dan diagnosa

### 🔮 Prediksi Individual

- Input data pasien baru
- Prediksi dengan semua model
- Confidence level untuk setiap prediksi
- Rekomendasi berdasarkan hasil prediksi

## 📁 Struktur Proyek

```
project/
├── dashboard.py              # Dashboard utama Streamlit
├── model_rme.py             # Script training model
├── requirements.txt         # Dependencies
├── .streamlit/
│   └── config.toml         # Konfigurasi Streamlit
├── data/
│   └── data_rme_dummy_baru.csv  # Dataset RME
├── model/                   # Model ML yang sudah dilatih
│   ├── model_*.pkl         # File model
│   └── model_columns.pkl   # Nama kolom fitur
├── output/                  # Hasil export
│   ├── *.xlsx             # Export Excel
│   ├── *.csv              # Export CSV
│   └── *.json             # Metadata
└── plots/                   # Visualisasi
    ├── model_comparison.png
    └── cv_scores.png
```

## 🛠️ Instalasi dan Setup

### 1. Clone atau Download Project

```bash
git clone <repository-url>
cd dashboard-prediksi-rawat-inap
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Persiapkan Data

Pastikan file `data/data_rme_dummy_baru.csv` tersedia dengan kolom:

- `Inisial_Nama`: Inisial nama pasien
- `Usia`: Usia pasien (tahun)
- `Jenis_Kelamin`: L/P
- `Asuransi`: Jenis asuransi
- `Diagnosa_ICD`: Kode diagnosa ICD-10
- `Tindakan`: Jenis tindakan medis
- `Lama_Rawat`: Target variable (hari)
- `Cepat_Sembuh`: Label binary (0/1)

### 4. Training Model

```bash
python model_rme.py
```

### 5. Jalankan Dashboard

```bash
streamlit run dashboard.py
```

## 🎛️ Cara Penggunaan

### 1. **Pilih Model**

- Gunakan sidebar untuk memilih model prediksi
- Bandingkan performa berbagai model

### 2. **Filter Data**

- **Diagnosa ICD**: Filter berdasarkan jenis penyakit
- **Rentang Usia**: Slider untuk filter usia
- **Jenis Asuransi**: Filter berdasarkan asuransi

### 3. **Analisis Hasil**

- **Overview**: Lihat statistik umum dataset
- **Model Evaluation**: Bandingkan metrik semua model
- **Predictions**: Analisis hasil prediksi vs aktual
- **Deep Analysis**: Eksplorasi mendalam per kategori

### 4. **Prediksi Individual**

- Input data pasien baru
- Dapatkan prediksi dari semua model
- Lihat confidence level dan rekomendasi

### 5. **Export Data**

- Download hasil prediksi dalam format Excel
- Export evaluasi model dalam CSV
- Generate laporan visualisasi PDF

## 📊 Metrik Evaluasi

### Metrik yang Digunakan:

- **MAE (Mean Absolute Error)**: Rata-rata error absolut
- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar dari MSE
- **R² (Coefficient of Determination)**: Proporsi varians yang dijelaskan
- **MAPE (Mean Absolute Percentage Error)**: Rata-rata persentase error

### Interpretasi R² Score:

- **R² > 0.8**: Model sangat baik
- **R² 0.6-0.8**: Model baik
- **R² 0.4-0.6**: Model cukup
- **R² < 0.4**: Model perlu perbaikan

## 🔧 Kustomisasi

### Menambah Model Baru

1. Edit `model_rme.py` di fungsi `get_models()`
2. Tambahkan model baru ke dictionary
3. Jalankan training ulang

### Menambah Fitur Analisis

1. Edit `dashboard.py`
2. Tambahkan tab atau section baru
3. Implementasikan visualisasi sesuai kebutuhan

### Mengubah Tema

1. Edit `.streamlit/config.toml`
2. Ubah warna dan styling sesuai preferensi

## ⚠️ Catatan Penting

### Disclaimer

- Hasil prediksi hanya untuk keperluan penelitian dan analisis
- Keputusan medis tetap harus melibatkan tenaga medis profesional
- Model perlu divalidasi lebih lanjut sebelum implementasi klinis

### Data Privacy

- Pastikan data pasien sudah di-anonymize
- Tidak menyimpan data sensitif dalam sistem
- Ikuti regulasi privasi data yang berlaku

### Performance

- Untuk dataset besar (>100k records), pertimbangkan sampling
- Monitor penggunaan memory saat menjalankan dashboard
- Gunakan caching Streamlit untuk performa optimal

## 🐛 Troubleshooting

### Error Common:

1. **File tidak ditemukan**: Pastikan struktur folder sesuai
2. **Memory error**: Reduce dataset size atau upgrade RAM
3. **Model loading error**: Jalankan training ulang
4. **Streamlit error**: Update ke versi terbaru

### Tips Optimasi:

- Gunakan `@st.cache_data` untuk fungsi yang memakan waktu
- Limit jumlah data yang ditampilkan di tabel
- Compress image untuk visualisasi

## 👥 Kontribusi

Untuk berkontribusi pada proyek ini:

1. Fork repository
2. Buat feature branch
3. Commit changes
4. Submit pull request

## 📞 Support

Jika mengalami masalah atau memiliki pertanyaan:

1. Cek dokumentasi ini
2. Review error logs
3. Buka issue di repository

## 📜 License

Proyek ini dibuat untuk keperluan akademik/penelitian. Silakan disesuaikan dengan kebutuhan institusi.

---

**🎓 Dashboard Prediksi Lama Rawat Inap Pasien**  
_Sistem Prediksi Berbasis Machine Learning untuk Data RME_  
_Dikembangkan untuk keperluan penelitian skripsi_
