# 🚀 Panduan Deployment Dashboard Prediksi Lama Rawat Inap

## 📋 Daftar Isi

1. [Quick Start](#quick-start)
2. [Setup Development](#setup-development)
3. [Deployment Production](#deployment-production)
4. [Troubleshooting](#troubleshooting)
5. [API Documentation](#api-documentation)
6. [Kontribusi](#kontribusi)

---

## 🚀 Quick Start

### Cara Tercepat Menjalankan Sistem

```bash
# 1. Clone/download project
git clone <repository-url>
cd dashboard-prediksi-rawat-inap

# 2. Jalankan setup otomatis
python run_system.py

# 3. Ikuti instruksi di terminal
```

Script `run_system.py` akan otomatis:

- ✅ Cek dependencies
- ✅ Install requirements
- ✅ Setup direktori
- ✅ Training model (jika diperlukan)
- ✅ Menjalankan dashboard

---

## 🛠️ Setup Development

### Prerequisites

- Python 3.8+
- RAM minimal 4GB
- Storage minimal 2GB

### Manual Setup

1. **Setup Environment**

```bash
# Buat virtual environment (opsional tapi direkomendasikan)
python -m venv venv

# Aktivasi environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Setup Data**

```bash
# Jika belum punya data, generate dummy data
python generate_dummy_data.py

# Atau letakkan file data di: data/data_rme_dummy_baru.csv
```

4. **Training Model**

```bash
python model_rme.py
```

5. **Jalankan Dashboard**

```bash
streamlit run dashboard.py
```

### Struktur Project Lengkap

```
dashboard-prediksi-rawat-inap/
├── 📄 dashboard.py              # Dashboard utama
├── 🤖 model_rme.py             # Script training model
├── 🎮 run_system.py            # Setup otomatis
├── 📊 generate_dummy_data.py   # Generator data dummy
├── 📋 requirements.txt         # Dependencies
├── 📚 README.md               # Dokumentasi utama
├── 📁 .streamlit/
│   └── config.toml            # Konfigurasi Streamlit
├── 📁 data/
│   └── data_rme_dummy_baru.csv # Dataset
├── 📁 model/                   # Model ML tersimpan
│   ├── model_*.pkl           # File model
│   └── model_columns.pkl     # Metadata kolom
├── 📁 output/                 # Hasil export
│   ├── *.xlsx               # File Excel
│   ├── *.csv                # File CSV
│   └── *.json               # Metadata
└── 📁 plots/                  # Visualisasi
    ├── model_comparison.png
    └── cv_scores.png
```

---

## 🌐 Deployment Production

### Option 1: Streamlit Cloud (Gratis)

1. **Persiapan Repository**

```bash
# Push code ke GitHub
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy ke Streamlit Cloud**

- Kunjungi [share.streamlit.io](https://share.streamlit.io)
- Connect GitHub repository
- Set main file: `dashboard.py`
- Deploy!

3. **Environment Variables** (jika diperlukan)

```toml
# Di Streamlit Cloud, tambahkan secrets.toml:
[general]
data_path = "data/data_rme_dummy_baru.csv"
model_path = "model/"
```

### Option 2: Heroku

1. **Buat Procfile**

```bash
echo "web: streamlit run dashboard.py --server.port=$PORT --server.address=0.0.0.0" > Procfile
```

2. **Deploy ke Heroku**

```bash
# Install Heroku CLI, lalu:
heroku login
heroku create your-app-name
git push heroku main
```

### Option 3: Docker

1. **Buat Dockerfile**

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.address", "0.0.0.0"]
```

2. **Build dan Run**

```bash
docker build -t dashboard-rme .
docker run -p 8501:8501 dashboard-rme
```

### Option 4: VPS/Server

1. **Setup Server** (Ubuntu/CentOS)

```bash
# Update sistem
sudo apt update && sudo apt upgrade -y

# Install Python dan pip
sudo apt install python3 python3-pip -y

# Clone project
git clone <repository-url>
cd dashboard-prediksi-rawat-inap

# Install dependencies
pip3 install -r requirements.txt

# Setup sebagai service systemd (opsional)
sudo nano /etc/systemd/system/dashboard-rme.service
```

2. **Service Configuration**

```ini
[Unit]
Description=Dashboard RME
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/dashboard-prediksi-rawat-inap
ExecStart=/usr/bin/python3 -m streamlit run dashboard.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

3. **Aktivasi Service**

```bash
sudo systemctl enable dashboard-rme
sudo systemctl start dashboard-rme
sudo systemctl status dashboard-rme
```

---

## 🔧 Troubleshooting

### Error Common dan Solusi

#### 1. **"ModuleNotFoundError"**

```bash
# Solusi: Install dependencies yang missing
pip install -r requirements.txt

# Atau install manual:
pip install streamlit pandas scikit-learn plotly
```

#### 2. **"File not found: data/data_rme_dummy_baru.csv"**

```bash
# Solusi: Generate dummy data
python generate_dummy_data.py

# Atau pastikan file data ada di lokasi yang benar
```

#### 3. **"Model file not found"**

```bash
# Solusi: Training model terlebih dahulu
python model_rme.py
```

#### 4. **Memory Error saat Training**

```python
# Solusi: Reduce dataset size atau upgrade RAM
# Edit di model_rme.py:
df_sample = df.sample(n=5000)  # Ambil 5000 sample saja
```

#### 5. **Streamlit Port Already in Use**

```bash
# Solusi: Gunakan port berbeda
streamlit run dashboard.py --server.port 8502

# Atau kill process yang menggunakan port 8501:
# Windows:
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8501 | xargs kill -9
```

#### 6. **Dashboard Lambat/Hang**

```python
# Optimasi di dashboard.py:
# 1. Tambahkan @st.cache_data
@st.cache_data
def load_data():
    return pd.read_csv("data/data_rme_dummy_baru.csv")

# 2. Limit data yang ditampilkan
df_display = df.head(1000)  # Tampilkan 1000 baris saja

# 3. Optimize plots
fig.update_layout(showlegend=False)  # Kurangi legend jika tidak perlu
```

### Performance Optimization

#### 1. **Caching Strategy**

```python
# Cache data loading
@st.cache_data(ttl=3600)  # Cache 1 jam
def load_data():
    return pd.read_csv("data/data_rme_dummy_baru.csv")

# Cache model loading
@st.cache_resource
def load_models():
    return joblib.load("model/model_RandomForest.pkl")
```

#### 2. **Memory Management**

```python
# Untuk dataset besar
def optimize_memory(df):
    # Convert ke kategori untuk menghemat memory
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype('category')

    # Convert int64 ke int32 jika memungkinkan
    for col in df.select_dtypes(include=['int64']):
        if df[col].max() < 2147483647:
            df[col] = df[col].astype('int32')

    return df
```

#### 3. **Database Integration** (untuk data besar)

```python
import sqlite3
import sqlalchemy

# Setup database connection
@st.cache_resource
def get_database_connection():
    return sqlite3.connect('rme_data.db')

# Load data dari database
@st.cache_data
def load_data_from_db(query):
    conn = get_database_connection()
    return pd.read_sql(query, conn)
```

---

## 📡 API Documentation

### Model Prediction API

Jika ingin membuat API endpoint untuk prediksi:

```python
# api.py - FastAPI implementation
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="RME Prediction API")

class PatientData(BaseModel):
    usia: int
    jenis_kelamin: str
    asuransi: str
    diagnosa_icd: str
    tindakan: str

@app.post("/predict")
async def predict_los(patient: PatientData):
    # Load model
    model = joblib.load("model/model_RandomForest.pkl")
    columns = joblib.load("model/model_columns.pkl")

    # Prepare data
    data = pd.DataFrame([patient.dict()])
    data_encoded = pd.get_dummies(data)
    data_encoded = data_encoded.reindex(columns=columns, fill_value=0)

    # Predict
    prediction = model.predict(data_encoded)[0]

    return {
        "predicted_los": round(prediction, 2),
        "recommendation": "Normal" if prediction <= 7 else "Extended stay"
    }

# Jalankan dengan: uvicorn api:app --reload
```

### Webhook Integration

```python
# webhook.py - Untuk integrasi dengan sistem RS
import requests

def send_prediction_to_his(patient_id, prediction):
    """Send prediction result to Hospital Information System"""

    webhook_url = "https://your-his-system.com/api/predictions"

    payload = {
        "patient_id": patient_id,
        "predicted_los": prediction,
        "timestamp": datetime.now().isoformat(),
        "model_version": "v1.0"
    }

    headers = {
        "Authorization": "Bearer YOUR_API_TOKEN",
        "Content-Type": "application/json"
    }

    response = requests.post(webhook_url, json=payload, headers=headers)
    return response.status_code == 200
```

---

## 🔒 Security Considerations

### 1. **Data Privacy**

```python
# Anonymize data
def anonymize_patient_data(df):
    # Hash patient identifiers
    df['patient_hash'] = df['Inisial_Nama'].apply(
        lambda x: hashlib.sha256(x.encode()).hexdigest()[:8]
    )
    df.drop('Inisial_Nama', axis=1, inplace=True)
    return df
```

### 2. **Access Control**

```python
# Simple authentication
def check_password():
    def password_entered():
        if st.session_state["password"] == "your_secure_password":
            st.session_state["password_correct"] = True
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input("Password", type="password",
                     on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        st.text_input("Password", type="password",
                     on_change=password_entered, key="password")
        st.error("Password incorrect")
        return False
    else:
        return True

# Di dashboard.py, tambahkan di awal:
if not check_password():
    st.stop()
```

### 3. **Environment Variables**

```python
import os
from dotenv import load_dotenv

load_dotenv()

# Gunakan environment variables untuk config sensitif
DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")
```

---

## 📊 Monitoring dan Logging

### 1. **Application Logging**

```python
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Di fungsi-fungsi penting:
logger.info(f"Model {model_name} loaded successfully")
logger.warning(f"Data quality issue detected: {issue}")
logger.error(f"Prediction failed: {error}")
```

### 2. **Performance Monitoring**

```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()

        logger.info(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@monitor_performance
def train_model():
    # Training code here
    pass
```

---

## 🤝 Kontribusi

### Development Workflow

1. **Fork Repository**
2. **Create Feature Branch**

```bash
git checkout -b feature/new-feature
```

3. **Make Changes**
4. **Test Changes**

```bash
python -m pytest tests/  # Jika ada unit tests
streamlit run dashboard.py  # Manual testing
```

5. **Submit Pull Request**

### Coding Standards

- **Python**: Follow PEP 8
- **Documentation**: Docstrings untuk semua fungsi
- **Comments**: Gunakan bahasa Indonesia untuk comment
- **Naming**: Variable dan function names dalam bahasa Inggris

### Testing

```python
# tests/test_model.py
import pytest
import pandas as pd
from model_rme import generate_dummy_rme_data

def test_data_generation():
    df = generate_dummy_rme_data(100)
    assert len(df) == 100
    assert 'Lama_Rawat' in df.columns
    assert df['Lama_Rawat'].min() >= 1

def test_model_prediction():
    # Test model prediction logic
    pass
```

---

## 📞 Support

### Getting Help

1. **Check Documentation**: README.md dan file ini
2. **Search Issues**: Cek existing issues di repository
3. **Create New Issue**: Jika belum ada solusi

### Issue Templates

**Bug Report:**

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce:

1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment:**

- OS: [e.g. Windows 10]
- Python version: [e.g. 3.9]
- Browser: [e.g. Chrome 90]
```

**Feature Request:**

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Additional context**
Add any other context or screenshots about the feature request.
```

---

## 📜 License

Project ini dikembangkan untuk keperluan akademik/penelitian. Silakan disesuaikan dengan kebutuhan institusi masing-masing.

---

**🎓 Dashboard Prediksi Lama Rawat Inap Pasien**  
_Sistem Prediksi Berbasis Machine Learning untuk Data RME_  
_Versi: 1.0.0_  
_Last Updated: 5-Agustus-2025_
_Disusun oleh Dani - IT RSUD Kebayoran Baru_
