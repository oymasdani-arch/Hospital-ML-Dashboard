# generate_dummy_data.py
# ==========================================
# Generator Data Dummy untuk RME
# Untuk keperluan testing dan demo
# ==========================================

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seed untuk reproducibility
np.random.seed(42)
random.seed(42)

def generate_dummy_rme_data(n_samples=1000):
    """
    Generate data dummy RME untuk testing
    
    Args:
        n_samples (int): Jumlah sample yang akan dibuat
        
    Returns:
        pd.DataFrame: DataFrame berisi data dummy RME
    """
    print(f"🔄 Generating {n_samples} data dummy RME...")
    
    # Data referensi yang realistis
    diagnosa_icd = [
        'J44.1', 'I25.9', 'E11.9', 'N18.6', 'I10', 'J18.9', 'K59.0', 'M79.9',
        'R06.0', 'Z51.1', 'I50.9', 'N39.0', 'K21.9', 'E78.5', 'M25.9',
        'J45.9', 'I63.9', 'E11.7', 'J44.0', 'I20.9', 'K92.2', 'N20.0',
        'R10.4', 'Z51.8', 'I48', 'J20.9', 'K76.9', 'E87.6', 'I35.0',
        'J06.9', 'R50.9', 'N28.9', 'I25.1', 'E11.6', 'J98.9'
    ]
    
    diagnosa_names = [
        'Penyakit Paru Obstruktif Kronik', 'Penyakit Jantung Iskemik', 
        'Diabetes Melitus', 'Gagal Ginjal Kronik', 'Hipertensi',
        'Pneumonia', 'Konstipasi', 'Nyeri Otot', 'Sesak Nafas',
        'Kemoterapi', 'Gagal Jantung', 'Infeksi Saluran Kemih',
        'Gastroesophageal Reflux', 'Dislipidemia', 'Arthritis',
        'Asma', 'Stroke', 'Diabetes dengan Komplikasi', 'PPOK Akut',
        'Angina Pektoris', 'Perdarahan GI', 'Batu Ginjal',
        'Nyeri Perut', 'Radioterapi', 'Atrial Fibrillation',
        'Bronkitis', 'Penyakit Hati', 'Gangguan Elektrolit',
        'Stenosis Aorta', 'Infeksi Saluran Nafas', 'Demam',
        'Penyakit Ginjal', 'Infark Miokard', 'Diabetes Neuropati',
        'Penyakit Paru Lain'
    ]
    
    tindakan_medis = [
        'Observasi', 'Medikamentosa', 'Oksigen Terapi', 'Infus',
        'Kateter Urin', 'NGT', 'Fisioterapi', 'Hemodialisis',
        'Transfusi Darah', 'Nebulizer', 'ECG', 'Foto Thorax',
        'CT Scan', 'MRI', 'Endoskopi', 'Biopsi', 'Operasi Minor',
        'Operasi Mayor', 'Kemoterapi', 'Radioterapi', 'Bronkoskopi',
        'Ekokardiografi', 'Angiografi', 'Pemasangan Stent',
        'Drainage', 'Ventilator', 'CPAP', 'BiPAP'
    ]
    
    asuransi_types = ['BPJS', 'Asuransi Swasta', 'Umum', 'Karyawan', 'Askes']
    jenis_kelamin = ['L', 'P']
    
    # Generate data
    data = []
    
    for i in range(n_samples):
        # Generate inisial nama
        inisial = f"P{i+1:04d}"
        
        # Generate usia dengan distribusi realistis
        # Lebih banyak pasien dewasa dan lansia
        usia = np.random.choice(
            range(1, 90), 
            p=generate_age_distribution()
        )
        
        # Jenis kelamin
        gender = random.choice(jenis_kelamin)
        
        # Asuransi dengan probabilitas berbeda
        asuransi = np.random.choice(
            asuransi_types, 
            p=[0.6, 0.15, 0.15, 0.07, 0.03]  # BPJS dominan
        )
        
        # Diagnosa ICD
        diagnosa_idx = random.randint(0, len(diagnosa_icd)-1)
        diagnosa = diagnosa_icd[diagnosa_idx]
        
        # Tindakan (beberapa diagnosa memiliki tindakan yang lebih spesifik)
        if 'J' in diagnosa:  # Penyakit pernafasan
            tindakan_pool = ['Oksigen Terapi', 'Nebulizer', 'Foto Thorax', 'Observasi', 'Medikamentosa']
        elif 'I' in diagnosa:  # Penyakit kardiovaskular
            tindakan_pool = ['ECG', 'Ekokardiografi', 'Observasi', 'Medikamentosa', 'Infus']
        elif 'E11' in diagnosa:  # Diabetes
            tindakan_pool = ['Medikamentosa', 'Observasi', 'Infus', 'ECG']
        elif 'N' in diagnosa:  # Penyakit ginjal
            tindakan_pool = ['Hemodialisis', 'Medikamentosa', 'Kateter Urin', 'Infus']
        else:
            tindakan_pool = tindakan_medis[:10]  # Tindakan umum
        
        tindakan = random.choice(tindakan_pool)
        
        # Generate lama rawat berdasarkan diagnosa dan usia
        lama_rawat = generate_realistic_los(diagnosa, usia, tindakan)
        
        # Tentukan apakah cepat sembuh (<=3 hari)
        cepat_sembuh = 1 if lama_rawat <= 3 else 0
        
        # Tambahkan noise realistis
        if random.random() < 0.1:  # 10% kasus outlier
            lama_rawat = max(1, lama_rawat + random.randint(5, 15))
        
        data.append({
            'Inisial_Nama': inisial,
            'Usia': usia,
            'Jenis_Kelamin': gender,
            'Asuransi': asuransi,
            'Diagnosa_ICD': diagnosa,
            'Tindakan': tindakan,
            'Lama_Rawat': lama_rawat,
            'Cepat_Sembuh': cepat_sembuh
        })
    
    df = pd.DataFrame(data)
    
    print(f"✅ Data dummy berhasil dibuat:")
    print(f"   - Total pasien: {len(df):,}")
    print(f"   - Rata-rata lama rawat: {df['Lama_Rawat'].mean():.2f} hari")
    print(f"   - Median lama rawat: {df['Lama_Rawat'].median():.2f} hari")
    print(f"   - Range lama rawat: {df['Lama_Rawat'].min()} - {df['Lama_Rawat'].max()} hari")
    print(f"   - Pasien cepat sembuh: {df['Cepat_Sembuh'].sum()} ({df['Cepat_Sembuh'].mean()*100:.1f}%)")
    print(f"   - Jenis diagnosa: {df['Diagnosa_ICD'].nunique()}")
    print(f"   - Jenis tindakan: {df['Tindakan'].nunique()}")
    
    return df

def generate_age_distribution():
    """Generate distribusi usia yang realistis untuk pasien RS"""
    # Distribusi usia pasien RS (lebih banyak dewasa dan lansia)
    ages = list(range(1, 90))
    probs = []
    
    for age in ages:
        if age < 18:  # Anak-anak
            prob = 0.008
        elif age < 30:  # Dewasa muda
            prob = 0.012
        elif age < 50:  # Dewasa
            prob = 0.015
        elif age < 65:  # Dewasa tua
            prob = 0.020
        else:  # Lansia
            prob = 0.025
    
        probs.append(prob)
    
    # Normalize probabilities
    probs = np.array(probs)
    probs = probs / probs.sum()
    
    return probs

def generate_realistic_los(diagnosa, usia, tindakan):
    """Generate lama rawat yang realistis berdasarkan diagnosa, usia, dan tindakan"""
    
    # Base LOS berdasarkan diagnosa
    base_los = {
        'J': 5,   # Respiratory
        'I': 6,   # Cardiovascular  
        'E': 4,   # Endocrine
        'N': 7,   # Renal
        'K': 4,   # Digestive
        'M': 3,   # Musculoskeletal
        'R': 3,   # Symptoms
        'Z': 2    # Factors influencing health
    }
    
    # Ambil karakter pertama diagnosa
    diagnosa_category = diagnosa[0] if diagnosa else 'R'
    base = base_los.get(diagnosa_category, 4)
    
    # Adjustment berdasarkan usia
    if usia < 18:
        age_factor = 0.8  # Anak cenderung lebih cepat sembuh
    elif usia < 65:
        age_factor = 1.0  # Dewasa normal
    else:
        age_factor = 1.3  # Lansia cenderung lebih lama
    
    # Adjustment berdasarkan tindakan
    tindakan_factor = {
        'Observasi': 0.7,
        'Medikamentosa': 0.8,
        'Operasi Mayor': 2.0,
        'Operasi Minor': 1.5,
        'Hemodialisis': 1.4,
        'Kemoterapi': 1.6,
        'Radioterapi': 1.3,
        'Ventilator': 2.5,
        'ICU': 2.2,
        'Transfusi Darah': 1.2,
        'Fisioterapi': 0.9
    }
    
    factor = tindakan_factor.get(tindakan, 1.0)
    
    # Hitung LOS final
    los = base * age_factor * factor
    
    # Tambahkan variabilitas random
    los = los * np.random.normal(1.0, 0.3)
    
    # Pastikan minimal 1 hari, maksimal 30 hari
    los = max(1, min(30, round(los)))
    
    return int(los)

def add_realistic_correlations(df):
    """Tambahkan korelasi realistis antar variabel"""
    
    # Korelasi usia dengan beberapa diagnosa
    for idx, row in df.iterrows():
        if row['Usia'] > 65:
            # Lansia lebih rentan penyakit tertentu
            if random.random() < 0.3:
                cardiovascular_codes = ['I25.9', 'I10', 'I50.9', 'I48', 'I63.9']
                df.at[idx, 'Diagnosa_ICD'] = random.choice(cardiovascular_codes)
        
        if row['Usia'] > 50:
            # Dewasa tua lebih rentan diabetes
            if random.random() < 0.15:
                diabetes_codes = ['E11.9', 'E11.7', 'E11.6']
                df.at[idx, 'Diagnosa_ICD'] = random.choice(diabetes_codes)
    
    # Update lama rawat berdasarkan diagnosa yang diupdate
    for idx, row in df.iterrows():
        new_los = generate_realistic_los(row['Diagnosa_ICD'], row['Usia'], row['Tindakan'])
        df.at[idx, 'Lama_Rawat'] = new_los
        df.at[idx, 'Cepat_Sembuh'] = 1 if new_los <= 3 else 0
    
    return df

def generate_additional_features(df):
    """Generate fitur tambahan yang mungkin berguna untuk ML"""
    
    # Kategori usia
    df['Kategori_Usia'] = df['Usia'].apply(lambda x: 
        'Anak' if x < 18 else
        'Dewasa_Muda' if x < 30 else
        'Dewasa' if x < 50 else
        'Dewasa_Tua' if x < 65 else
        'Lansia'
    )
    
    # Kategori diagnosa berdasarkan sistem organ
    df['Sistem_Organ'] = df['Diagnosa_ICD'].apply(lambda x:
        'Respirasi' if x.startswith('J') else
        'Kardiovaskular' if x.startswith('I') else
        'Endokrin' if x.startswith('E') else
        'Ginjal' if x.startswith('N') else
        'Pencernaan' if x.startswith('K') else
        'Muskuloskeletal' if x.startswith('M') else
        'Gejala' if x.startswith('R') else
        'Lainnya'
    )
    
    # Tingkat kompleksitas tindakan
    kompleksitas_tindakan = {
        'Observasi': 'Rendah',
        'Medikamentosa': 'Rendah', 
        'Oksigen Terapi': 'Sedang',
        'Infus': 'Rendah',
        'Fisioterapi': 'Rendah',
        'Operasi Minor': 'Sedang',
        'Operasi Mayor': 'Tinggi',
        'Hemodialisis': 'Tinggi',
        'Kemoterapi': 'Tinggi',
        'Ventilator': 'Tinggi'
    }
    
    df['Kompleksitas_Tindakan'] = df['Tindakan'].map(kompleksitas_tindakan).fillna('Sedang')
    
    # Risk score sederhana berdasarkan usia dan kompleksitas
    df['Risk_Score'] = (
        (df['Usia'] / 100) + 
        df['Kompleksitas_Tindakan'].map({'Rendah': 0.1, 'Sedang': 0.3, 'Tinggi': 0.5})
    ).round(2)
    
    return df

def save_data(df, filename="data/data_rme_dummy_baru.csv"):
    """Simpan data ke file CSV"""
    
    # Buat direktori jika belum ada
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Simpan ke CSV
    df.to_csv(filename, index=False)
    
    print(f"💾 Data berhasil disimpan ke: {filename}")
    print(f"📊 Dimensi data: {df.shape}")
    print(f"📋 Kolom: {list(df.columns)}")

def generate_summary_report(df):
    """Generate laporan ringkasan data"""
    
    print("\n" + "="*60)
    print("📊 LAPORAN RINGKASAN DATA DUMMY RME")
    print("="*60)
    
    print(f"\n📈 STATISTIK UMUM:")
    print(f"   - Total Records: {len(df):,}")
    print(f"   - Jumlah Kolom: {len(df.columns)}")
    print(f"   - Rentang Tanggal: Data Dummy Generated on {datetime.now().strftime('%Y-%m-%d')}")
    
    print(f"\n👥 DEMOGRAFI PASIEN:")
    print(f"   - Rata-rata Usia: {df['Usia'].mean():.1f} tahun")
    print(f"   - Median Usia: {df['Usia'].median():.1f} tahun")
    print(f"   - Range Usia: {df['Usia'].min()} - {df['Usia'].max()} tahun")
    
    gender_dist = df['Jenis_Kelamin'].value_counts()
    print(f"   - Jenis Kelamin:")
    for gender, count in gender_dist.items():
        pct = (count/len(df))*100
        print(f"     * {gender}: {count:,} ({pct:.1f}%)")
    
    print(f"\n🏥 INFORMASI KLINIS:")
    print(f"   - Rata-rata Lama Rawat: {df['Lama_Rawat'].mean():.2f} hari")
    print(f"   - Median Lama Rawat: {df['Lama_Rawat'].median():.2f} hari")
    print(f"   - Range Lama Rawat: {df['Lama_Rawat'].min()} - {df['Lama_Rawat'].max()} hari")
    
    cepat_sembuh = df['Cepat_Sembuh'].sum()
    pct_cepat = (cepat_sembuh/len(df))*100
    print(f"   - Pasien Cepat Sembuh: {cepat_sembuh:,} ({pct_cepat:.1f}%)")
    
    print(f"   - Jumlah Diagnosa Unik: {df['Diagnosa_ICD'].nunique()}")
    print(f"   - Jumlah Tindakan Unik: {df['Tindakan'].nunique()}")
    
    print(f"\n💰 DISTRIBUSI ASURANSI:")
    asuransi_dist = df['Asuransi'].value_counts()
    for asuransi, count in asuransi_dist.items():
        pct = (count/len(df))*100
        print(f"   - {asuransi}: {count:,} ({pct:.1f}%)")
    
    print(f"\n🔝 TOP 5 DIAGNOSA:")
    top_diagnosa = df['Diagnosa_ICD'].value_counts().head()
    for diagnosa, count in top_diagnosa.items():
        pct = (count/len(df))*100
        print(f"   - {diagnosa}: {count:,} ({pct:.1f}%)")
    
    print(f"\n🔝 TOP 5 TINDAKAN:")
    top_tindakan = df['Tindakan'].value_counts().head()
    for tindakan, count in top_tindakan.items():
        pct = (count/len(df))*100
        print(f"   - {tindakan}: {count:,} ({pct:.1f}%)")
    
    if 'Sistem_Organ' in df.columns:
        print(f"\n🏥 DISTRIBUSI SISTEM ORGAN:")
        sistem_dist = df['Sistem_Organ'].value_counts()
        for sistem, count in sistem_dist.items():
            pct = (count/len(df))*100
            print(f"   - {sistem}: {count:,} ({pct:.1f}%)")
    
    print("\n" + "="*60)

def main():
    """Fungsi utama untuk generate data dummy"""
    
    print("🏥 GENERATOR DATA DUMMY RME")
    print("=" * 50)
    
    # Input jumlah data
    try:
        n_samples = int(input("📊 Masukkan jumlah data yang ingin dibuat (default: 1000): ") or "1000")
        if n_samples <= 0:
            print("❌ Jumlah data harus lebih dari 0!")
            return
    except ValueError:
        print("❌ Input tidak valid! Menggunakan default 1000 data.")
        n_samples = 1000
    
    # Generate basic data
    print(f"\n🔄 Membuat {n_samples:,} data dummy...")
    df = generate_dummy_rme_data(n_samples)
    
    # Tambahkan korelasi realistis
    print("🔗 Menambahkan korelasi realistis...")
    df = add_realistic_correlations(df)
    
    # Generate fitur tambahan
    add_features = input("\n📈 Tambahkan fitur tambahan untuk ML? (y/n): ").lower()
    if add_features in ['y', 'yes']:
        print("➕ Menambahkan fitur tambahan...")
        df = generate_additional_features(df)
    
    # Simpan data
    filename = input(f"\n💾 Nama file output (default: data/data_rme_dummy_baru.csv): ").strip()
    if not filename:
        filename = "data/data_rme_dummy_baru.csv"
    
    save_data(df, filename)
    
    # Generate summary report
    show_report = input("\n📋 Tampilkan laporan ringkasan? (y/n): ").lower()
    if show_report in ['y', 'yes']:
        generate_summary_report(df)
    
    print(f"\n✅ Proses selesai!")
    print(f"📁 File data tersimpan di: {filename}")
    print(f"🎯 Data siap digunakan untuk training model ML")

if __name__ == "__main__":
    main()
        