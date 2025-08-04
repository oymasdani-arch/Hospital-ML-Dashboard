# run_system.py
# ==========================================
# Script untuk Setup dan Menjalankan Sistem
# Dashboard Prediksi Lama Rawat Inap Pasien
# ==========================================

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header():
    """Print header aplikasi"""
    print("=" * 70)
    print("🏥 DASHBOARD PREDIKSI LAMA RAWAT INAP PASIEN")
    print("📊 Sistem Prediksi Berbasis Machine Learning")
    print("=" * 70)
    print()

def check_python_version():
    """Cek versi Python"""
    print("🐍 Mengecek versi Python...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8+ diperlukan!")
        print(f"   Versi saat ini: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def check_file_structure():
    """Cek struktur file yang diperlukan"""
    print("📁 Mengecek struktur file...")
    
    required_files = [
        "dashboard.py",
        "model_rme.py", 
        "requirements.txt"
    ]
    
    required_dirs = [
        "data",
        ".streamlit"
    ]
    
    # Cek file
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    # Cek direktori
    missing_dirs = []
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            missing_dirs.append(dir_name)
    
    if missing_files or missing_dirs:
        print("❌ File/direktori berikut tidak ditemukan:")
        for file in missing_files:
            print(f"   - {file}")
        for dir_name in missing_dirs:
            print(f"   - {dir_name}/")
        return False
    
    print("✅ Struktur file - OK")
    return True

def check_data_file():
    """Cek keberadaan file data"""
    print("📊 Mengecek file data...")
    
    data_file = "data/data_rme_dummy_baru.csv"
    if not Path(data_file).exists():
        print(f"❌ File data tidak ditemukan: {data_file}")
        print("💡 Pastikan file data sudah ditempatkan di lokasi yang benar")
        return False
    
    print("✅ File data ditemukan - OK")
    return True

def install_requirements():
    """Install dependencies"""
    print("📦 Menginstall dependencies...")
    
    try:
        # Cek apakah requirements.txt ada
        if not Path("requirements.txt").exists():
            print("❌ File requirements.txt tidak ditemukan!")
            return False
        
        # Install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Error saat install dependencies:")
            print(result.stderr)
            return False
        
        print("✅ Dependencies berhasil diinstall - OK")
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def create_directories():
    """Buat direktori yang diperlukan"""
    print("📁 Membuat direktori...")
    
    directories = [
        "model",
        "output", 
        "plots",
        ".streamlit"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Direktori berhasil dibuat - OK")

def create_streamlit_config():
    """Buat file konfigurasi Streamlit"""
    print("⚙️ Membuat konfigurasi Streamlit...")
    
    config_content = """[global]
developmentMode = false
logLevel = "info"

[server]
headless = true
enableCORS = false
enableXsrfProtection = true
maxUploadSize = 200
maxMessageSize = 200

[browser]
gatherUsageStats = false
serverAddress = "localhost"
serverPort = 8501

[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
"""
    
    config_path = Path(".streamlit/config.toml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, "w") as f:
        f.write(config_content)
    
    print("✅ Konfigurasi Streamlit dibuat - OK")

def train_models():
    """Training model ML"""
    print("🤖 Memulai training model...")
    print("⏳ Proses ini mungkin memakan waktu beberapa menit...")
    
    try:
        result = subprocess.run([sys.executable, "model_rme.py"], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Error saat training model:")
            print(result.stderr)
            return False
        
        print("✅ Training model selesai - OK")
        print("📊 Output training:")
        print(result.stdout[-500:])  # Print 500 karakter terakhir
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

def check_models():
    """Cek apakah model sudah ada"""
    print("🔍 Mengecek model yang tersedia...")
    
    model_dir = Path("model")
    if not model_dir.exists():
        return False
    
    # Cek file model yang diperlukan
    required_models = [
        "model_LinearRegression.pkl",
        "model_RandomForest.pkl", 
        "model_XGBoost.pkl",
        "model_columns.pkl"
    ]
    
    missing_models = []
    for model_file in required_models:
        if not (model_dir / model_file).exists():
            missing_models.append(model_file)
    
    if missing_models:
        print(f"❌ Model berikut belum ada: {missing_models}")
        return False
    
    print("✅ Model ditemukan - OK")
    return True

def run_dashboard():
    """Jalankan dashboard Streamlit"""
    print("🚀 Menjalankan dashboard...")
    print("🌐 Dashboard akan terbuka di: http://localhost:8501")
    print("⏹️  Tekan Ctrl+C untuk menghentikan dashboard")
    print()
    
    try:
        # Jalankan streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard dihentikan oleh user")
    except Exception as e:
        print(f"❌ Error menjalankan dashboard: {str(e)}")

def main():
    """Fungsi utama"""
    print_header()
    
    # Step 1: Cek Python version
    if not check_python_version():
        return
    
    # Step 2: Cek struktur file
    if not check_file_structure():
        print("💡 Pastikan semua file yang diperlukan sudah ada")
        return
    
    # Step 3: Cek file data
    if not check_data_file():
        return
    
    # Step 4: Buat direktori
    create_directories()
    
    # Step 5: Buat konfigurasi Streamlit
    create_streamlit_config()
    
    # Step 6: Install dependencies
    print("\n" + "="*50)
    user_input = input("📦 Install dependencies? (y/n): ").lower()
    if user_input in ['y', 'yes']:
        if not install_requirements():
            return
    
    # Step 7: Training model (jika belum ada)
    print("\n" + "="*50)
    if not check_models():
        user_input = input("🤖 Model belum ada. Lakukan training? (y/n): ").lower()
        if user_input in ['y', 'yes']:
            if not train_models():
                print("❌ Training gagal. Cek error di atas.")
                return
        else:
            print("⚠️ Dashboard membutuhkan model untuk berjalan!")
            return
    
    # Step 8: Jalankan dashboard
    print("\n" + "="*50)
    user_input = input("🚀 Jalankan dashboard sekarang? (y/n): ").lower()
    if user_input in ['y', 'yes']:
        print("\n🎉 Semua persiapan selesai!")
        print("🚀 Memulai dashboard...")
        time.sleep(2)
        run_dashboard()
    else:
        print("\n✅ Setup selesai!")
        print("💡 Untuk menjalankan dashboard secara manual:")
        print("   streamlit run dashboard.py")

if __name__ == "__main__":
    main()