# Mini Bootcamp Artificial Intelligence — Machine Learning Classification

Repository ini dirancang sebagai materi bootcamp dan modul praktikum Machine Learning untuk membimbing peserta memahami alur kerja klasifikasi (end-to-end Machine Learning pipeline) dalam Bahasa Indonesia.

---

## Materi Utama Bootcamp

Materi inti yang akan digunakan selama sesi bootcamp adalah notebook berikut:

### [`notebooks/crop_buah_model.ipynb`](notebooks/crop_buah_model.ipynb) — Klasifikasi Buah Tropis dengan KNN

Notebook ini menjadi panduan utama bootcamp. Peserta akan dibimbing langkah demi langkah membangun model klasifikasi **K-Nearest Neighbors (KNN)** untuk memprediksi jenis buah tropis (Pisang, Mangga, Pepaya, Semangka) berdasarkan 7 parameter kondisi lingkungan dan tanah.

**Yang akan dipelajari:**
- Import library dan data loading (`pandas`, `numpy`, `sklearn`)
- Exploratory Data Analysis (EDA): Histogram, Boxplot, Correlation Heatmap, Scatter Plot
- Penanganan Outlier dengan metode IQR Capping
- Standardisasi fitur menggunakan `StandardScaler` (Z-Score)
- Label Encoding untuk konversi label teks ke numerik
- Training model KNN dan evaluasi performa (Accuracy, Precision, Recall, F1-Score)
- Analisis Confusion Matrix dan Error Analysis
- Save & Load model menggunakan `joblib`

**Dataset:** [`dataset/crop_buah_tropis.csv`](dataset/crop_buah_tropis.csv) (2.400 sampel, 4 kelas)

---

## Latihan dan Referensi Tambahan

Setelah menyelesaikan materi utama, peserta dapat mengasah kemampuan lebih lanjut melalui notebook latihan berikut:

### [`notebooks/iris_clasification_model.ipynb`](notebooks/iris_clasification_model.ipynb) — Klasifikasi Bunga Iris
- **Dataset:** [`dataset/Iris.csv`](dataset/Iris.csv) (150 sampel, 3 kelas)
- **Tingkat:** Dasar / Pemula
- **Fokus:** Pengenalan konsep dasar klasifikasi multiclass, normalisasi `MinMaxScaler`, dan perbandingan 5 algoritma (KNN, Decision Tree, Random Forest, SVM, Logistic Regression)

### [`notebooks/crop_recommendation_model.ipynb`](notebooks/crop_recommendation_model.ipynb) — Rekomendasi Tanaman
- **Dataset:** [`dataset/crop_recommendation.csv`](dataset/crop_recommendation.csv) (2.200 sampel, 22 kelas)
- **Tingkat:** Lanjutan
- **Fokus:** Klasifikasi skala besar dengan 22 jenis tanaman, standardisasi `StandardScaler`, dan perbandingan 5 algoritma (KNN, Decision Tree, Random Forest, SVM, Naive Bayes)

---

## Konsep yang Dipelajari

Seluruh notebook disusun mengajarkan tahapan standar industri Data Science:

1. **Persiapan Library dan Data Loading** — Manipulasi data menggunakan `pandas` dan komputasi numerik `numpy`.
2. **Exploratory Data Analysis (EDA)** — Visualisasi distribusi fitur (Histogram, Boxplot, Pairplot, Violin Plot), analisis korelasi Pearson, dan ekstraksi domain knowledge melalui rata-rata karakteristik lingkungan per kelas.
3. **Data Preprocessing** — Penanganan outlier (IQR Capping), Feature Scaling (`MinMaxScaler` / `StandardScaler`), Label Encoding, dan Stratified Train/Test Split.
4. **Model Training** — Pelatihan algoritma klasifikasi KNN (materi utama) serta perbandingan multi-algoritma (latihan tambahan).
5. **Model Evaluation** — Evaluasi metrik (Accuracy, Precision, Recall, F1-Score), Confusion Matrix, dan Error Analysis untuk mengidentifikasi kelemahan prediksi model.
6. **Model Serialization (Save & Load)** — Penyimpanan model terlatih, Scaler, dan LabelEncoder menggunakan `joblib` untuk simulasi deployment.

---

## Struktur Direktori

```text
mini-bootcamp-artificial-intelligence/
├── dataset/                            # Dataset (.csv)
│   ├── Iris.csv                        # Dataset Bunga Iris (150 sampel, 3 kelas)
│   ├── crop_buah_tropis.csv            # Dataset Buah Tropis (2.400 sampel, 4 kelas)
│   └── crop_recommendation.csv         # Dataset Rekomendasi Tanaman (2.200 sampel, 22 kelas)
├── docs/                               # Dokumentasi penjelasan teori lengkap (.md)
│   ├── dataset.md                      # Spesifikasi detail ketiga dataset
│   ├── iris_clasification_model.md     # Panduan modul Iris
│   ├── crop_buah_model.md              # Panduan modul Buah Tropis (Materi Utama)
│   └── crop_recommendation_model.md    # Panduan modul Rekomendasi Tanaman
├── model/                              # Model terlatih (.joblib)
│   ├── best_buah_model.joblib
│   ├── best_crop_model.joblib
│   └── best_iris_model.joblib
├── notebooks/                          # Jupyter Notebook (.ipynb)
│   ├── crop_buah_model.ipynb           # Materi Utama Bootcamp
│   ├── iris_clasification_model.ipynb  # Latihan Tambahan
│   └── crop_recommendation_model.ipynb # Latihan Tambahan
└── README.md
```

---

## Cara Menggunakan (Quick Start)

### 1. Clone Repositori
```bash
git clone https://github.com/Fiyanz/mini-bootcamp-artificial-intelligence.git
cd mini-bootcamp-artificial-intelligence
```

### 2. Instalasi Dependensi (menggunakan uv)

Sangat disarankan menggunakan [`uv`](https://github.com/astral-sh/uv) untuk manajemen virtual environment dan package yang jauh lebih cepat.

Jika Anda belum menginstal `uv`, silakan instal terlebih dahulu sesuai OS Anda (atau lihat dokumentasi resminya).

**Membuat Virtual Environment dan Instalasi Package:**
```bash
# Membuat virtual environment
uv venv

# Aktivasi virtual environment
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Instalasi dependensi dengan uv pip (sangat cepat)
uv pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

*(Alternatif jika Anda menggunakan `pip` bawaan Python, gunakan `python3 -m venv .venv` lalu `pip install ...`)*

### 3. Jalankan Jupyter Notebook
```bash
jupyter notebook
```
Buka folder `notebooks/` dan mulai dari **`crop_buah_model.ipynb`** sebagai materi utama bootcamp.

---

## Dokumentasi Referensi

Penjelasan teori mendalam untuk setiap modul tersedia di folder [`docs/`](docs/):
- [Spesifikasi Dataset](docs/dataset.md)
- [Panduan Modul Buah Tropis — Materi Utama](docs/crop_buah_model.md)
- [Panduan Modul Iris](docs/iris_clasification_model.md)
- [Panduan Modul Rekomendasi Tanaman](docs/crop_recommendation_model.md)
