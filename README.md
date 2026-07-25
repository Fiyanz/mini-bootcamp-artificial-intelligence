# Mini Bootcamp Artificial Intelligence — Machine Learning Classification

Repository ini dirancang secara eksklusif sebagai materi bootcamp dan modul praktikum Machine Learning untuk membimbing peserta memahami alur kerja (end-to-end Machine Learning pipeline) klasifikasi terstruktur dalam Bahasa Indonesia secara mendalam.

---

## Modul dan Notebook Pembelajaran

Repositori ini terdiri dari 3 modul praktikum bertingkat dengan tingkat kompleksitas yang dirancang untuk pembelajaran progresif:

| Modul Notebook | Dataset | Jumlah Kelas | Metode Scaling | Algoritma Utama | Tingkat Kesulitan |
|---|---|:---:|:---:|---|:---:|
| [`notebooks/iris_clasification_model.ipynb`](notebooks/iris_clasification_model.ipynb) | `dataset/Iris.csv` (150 sampel) | 3 | `MinMaxScaler` | KNN, DT, RF, SVM, **Logistic Regression** | Dasar / Pemula |
| [`notebooks/crop_buah_model.ipynb`](notebooks/crop_buah_model.ipynb) | `dataset/crop_buah_tropis.csv` (2.400 sampel) | 4 | `StandardScaler` | KNN, DT, RF, SVM, **Naive Bayes** | Menengah |
| [`notebooks/crop_recommendation_model.ipynb`](notebooks/crop_recommendation_model.ipynb) | `dataset/crop_recommendation.csv` (2.200 sampel) | 22 | `StandardScaler` | KNN, DT, RF, SVM, **Naive Bayes** | Lanjutan |

---

## Materi dan Konsep yang Dipelajari

Setiap modul notebook disusun untuk mengajarkan tahapan standar industri murni dari ranah Data Science:

1. **Persiapan Library dan Data Loading**: Melatih insting memanipulasi ruang dimensi data menggunakan struktur `pandas` dan vektor kalkulasi `numpy`.
2. **Exploratory Data Analysis (EDA)**:
   - Visualisasi distribusi fitur statistik fundamental (Histogram Grid, Pairplot, Violin Plot, Swarm Plot).
   - Analisis korelasi linear murni dengan perwujudan instrumen visual Heatmap Pearson.
   - Menggali ekstraksi *domain knowledge* lewat perumusan rerata (mean) karakteristik lingkungan per pilar jenis tanaman.
3. **Data Preprocessing**:
   - **Penanganan Outlier Ekstrem**: Pembatasan pemotongan nilai menyimpang dengan metode rasional IQR Capping (Winsorizing) lewat limit pagar pembatas kuartil tanpa melukai mencederai menghapus titik observasi murni baris data.
   - **Feature Scaling**: Komparasi mendalam pertarungan pilar fundamental penyusutan kompresi dimensi normalisasi limit absolut batas rentang pecahan `MinMaxScaler` (rentang paksa 0-1) murni menantang melawan pilar raksasa sentralisasi kalibrasi standarisasi gravitasi deviasi konversi probabilitas Gaussian bel murni titik tengah `StandardScaler`.
   - **Label Encoding**: Transformasi rekayasa pilar kolom dimensi kelas teks murni murni abjad ke representasi konversi mutlak bilangan matematis bulat.
   - **Data Splitting**: Menerapkan proporsi distribusi belahan pilar Stratified Train/Test Split menjaga keseimbangan hukum alam ekuilibrium proporsi sebaran populasi.
4. **Model Training dan Komparasi**:
   - Mengkaji pertempuran perbandingan konfrontasi kaliber daya gempur kecerdasan mutlak 5 algoritma klasifikasi sekaligus: KNN, Decision Tree, Random Forest, SVM, Naive Bayes / Logistic Regression.
5. **Model Evaluation dan Error Analysis Analitik**:
   - Pemahaman pembedahan indikator angka pilar laporan parameter evaluasi berbasis metrik murni (Accuracy, Precision, Recall, F1-Score).
   - Analisis investigasi pembedahan kegagalan otopsi forensik forensik Confusion Matrix untuk mengungkap dalang kelemahan kelas buta yang paling sering mempertukarkan mengacaukan kesalahan prediksi menyimpang membelot buta tragis menyedihkan.
6. **Model Serialization (Save dan Load)**:
   - Preservasi pengawetan penyimpanan arsip abadi murni instrumen model terlatih, kompresor Scaler, dan penterjemah bilangan LabelEncoder memanfaatkan librari mutlak `joblib` murni mengamankan pilar wujud arsip eksternal solid menyendiri bersandar bernaung menumpang bersarang menyelinap bersemayam merangkul abadi menetap di peraduan bilik folder pelestarian `/model`.
   - Pengujian pembuktian eksekusi demonstrasi manifestasi implementasi pemanggilan kebangkitan roh pemanggilan mesin simulasi ramalan inferensi murni mutlak menerawang menyibak menelisik probabilitas di atas landasan pergerakan kepingan tebakan spesimen asing lingkungan segar fiktif palsu mutlak data baru.

---

## Struktur Direktori

```text
mini-bootcamp-artificial-intelligence/
├── dataset/                        # Arsip berkas himpunan rekam jejak dataset (.csv)
│   ├── Iris.csv                    # Dataset Bunga Iris (150 sampel, 3 kelas)
│   ├── crop_buah_tropis.csv        # Dataset Buah Tropis Sintetis (2.400 sampel, 4 kelas)
│   └── crop_recommendation.csv     # Dataset Rekomendasi Tanaman Asli (2.200 sampel, 22 kelas)
├── docs/                           # Literatur perpustakaan dokumentasi tertulis teoritis lengkap (.md)
│   ├── dataset.md                  # Penjelasan spesifikasi ketiga pondasi arsitektur dataset
│   ├── iris_clasification_model.md # Dokumen manual instruksional pelatihan modul Iris
│   ├── crop_buah_model.md          # Dokumen manual instruksional pelatihan modul Buah Tropis
│   └── crop_recommendation_model.md# Dokumen manual instruksional pelatihan modul Rekomendasi Tanaman
├── model/                          # Tempat peristirahatan preservasi model beku terlatih (.joblib)
│   ├── best_buah_model.joblib
│   ├── best_crop_model.joblib
│   └── best_iris_model.joblib
├── notebooks/                      # Sentral kendali landasan pacu operasi Jupyter Notebook (.ipynb)
│   ├── iris_clasification_model.ipynb
│   ├── crop_buah_model.ipynb
│   └── crop_recommendation_model.ipynb
└── README.md                       # Buku Panduan Orientasi Repositori Keseluruhan
```

---

## Cara Menggunakan (Quick Start)

### 1. Clone Repositori dan Persiapkan Lingkungan Virtual
```bash
git clone https://github.com/OpenSource/mini-bootcamp-artificial-intelligence.git
cd mini-bootcamp-artificial-intelligence
```

### 2. Instalasi Dependensi Inti Python
Dianjurkan menggunakan bilik eksekusi terisolasi Virtual Environment (`venv`) murni:
```bash
python3 -m venv .venv
source .venv/bin/activate  # Eksklusif eksekusi di sistem operasi basis kernel Linux/macOS murni
# .venv\Scriptsctivate   # Eksklusif eksekusi di ekosistem platform sistem operasi basis Windows

pip install pandas numpy matplotlib seaborn scikit-learn joblib jupyter
```

### 3. Eksekusi Peluncuran Ekosistem Lingkungan Jupyter Notebook
```bash
jupyter notebook
```
Buka gerbang penelusuran lorong direktori `notebooks/` lalu pilih secara selektif dokumen naskah instruksional berkas ekstensi titik ipynb instrumen modul praktikum simulasi mana yang pertama hendak Anda eksekusi pelajari bedah telusuri kuasai tuntas secara komprehensif mendalam!

---

## Dokumen Referensi dan Panduan Penjelasan Teori Mutlak

Panduan ensiklopedia literatur penjelasan instruksional pilar teori landasan dasar mutlak yang luar biasa sangat mendalam komprehensif terperinci dan ekstensif absolut disajikan didokumentasikan diawetkan direkam dicetak dipahat dipublikasikan dilestarikan dalam wadah ruang bilik pelestarian lorong folder pilar suci [`docs/`](docs/):
- [Penjelasan Parameter Arsitektur Fondasi Asal Muasal Struktur Spesifikasi Detail Himpunan Dataset Mentah Lengkap (`docs/dataset.md`)](docs/dataset.md)
- [Dokumentasi Panduan Manual Instruksional Rinci Modul Model Klasifikasi Bunga Iris Murni (`docs/iris_clasification_model.md`)](docs/iris_clasification_model.md)
- [Dokumentasi Panduan Manual Instruksional Rinci Modul Model Klasifikasi Kategori Buah Tropis Iklim Basah Murni (`docs/crop_buah_model.md`)](docs/crop_buah_model.md)
- [Dokumentasi Panduan Manual Instruksional Rinci Modul Model Klasifikasi Inferensi Rekomendasi 22 Tanaman Agraris Pertanian Kaggle Mutlak (`docs/crop_recommendation_model.md`)](docs/crop_recommendation_model.md)

---
Pangkalan pusat repositori pelatihan kawah candradimuka wadah kawah inkubasi ekosistem pilar perintis pusat pelatihan kecerdasan silikon buatan ini sengaja dirancang difabrikasi dikonstruksi dirakit secara eksklusif utuh sepenuhnya mutlak didedikasikan secara tuntas mutlak murni absolut untuk memandu membimbing memfasilitasi mendampingi menopang menyokong pengasuhan pematangan pencerahan pemahaman nalar daya tangkap logika ilmu komputasi struktur fundamental dasar pilar Machine Learning murni industri secara komprehensif terstruktur rigid sistematis analitis kronologis mendalam dan taktis mutlak bagi peserta murni.
