# Model Rekomendasi Tanaman - Crop Recommendation

**Tujuan Pembelajaran:** 
Modul ini bertujuan untuk membimbing peserta bootcamp dalam membangun model klasifikasi multiclass (*banyak kelas*) end-to-end. Fokus utamanya adalah merekomendasikan salah satu dari 22 jenis tanaman yang paling cocok ditanam berdasarkan 7 kondisi lingkungan (kandungan Nitrogen, Fosfor, Kalium, suhu udara, kelembaban udara, pH tanah, dan curah hujan). Peserta akan mengeksplorasi data, menangani pencilan (outliers), menstandarisasi fitur, dan melatih 5 algoritma Machine Learning berbeda untuk mencari model dengan akurasi tertinggi.

**Dataset:** Kaggle Crop Recommendation Dataset (`crop_recommendation.csv`)

---

## Table of Contents

1. [Import Library](#1-import-library)
2. [Data Loading & Initial Exploration](#2-data-loading--initial-exploration)
3. [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
4. [Data Processing](#4-data-processing)
5. [Data Splitting](#5-data-splitting)
6. [Model Training](#6-model-training)
7. [Model Evaluation - Classification Report & Confusion Matrix](#7-model-evaluation---classification-report--confusion-matrix)
8. [Model Comparison (KP, CM, TP, FP, FN, TN)](#8-model-comparison-kp-cm-tp-fp-fn-tn)
9. [Visualizing Model Performance](#9-visualizing-model-performance)
10. [Best Model & Error Analysis](#10-best-model--error-analysis)
11. [Menyimpan Model (Save Model)](#11-menyimpan-model-save-model)
12. [Memuat Model dan Pengujian (Load Model & Testing)](#12-memuat-model-dan-pengujian-load-model--testing)

---

## 1. Import Library

Tahap pertama dalam setiap proyek Data Science adalah mengimpor pustaka (library) perangkat lunak yang akan menyediakan fungsi-fungsi khusus untuk pengolahan data, visualisasi, dan pemodelan matematis.

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, f1_score, recall_score,
    classification_report
)
```

**Konsep Dasar Library:**
- `pandas` (Python Data Analysis Library): Digunakan untuk memanipulasi dan menganalisis data terstruktur. Struktur data utamanya adalah *DataFrame* (mirip tabel Excel atau SQL) dan *Series* (array satu dimensi).
- `matplotlib.pyplot` & `seaborn`: Library visualisasi data. `matplotlib` memberikan kontrol tingkat rendah terhadap elemen grafik, sedangkan `seaborn` dibangun di atas matplotlib untuk membuat grafik statistik yang lebih estetis dengan sintaks yang lebih ringkas.
- `numpy` (Numerical Python): Library fundamental untuk komputasi numerik yang sangat cepat menggunakan struktur array N-dimensi. Operasi matematis di balik layar Machine Learning sangat bergantung pada array Numpy.
- `joblib`: Digunakan untuk *serialization* dan *deserialization* (menyimpan objek Python yang ada di memori RAM ke dalam file fisik di hard disk, dan sebaliknya). Sangat efisien untuk menyimpan model Machine Learning yang besar.
- `sklearn` (Scikit-Learn): Library standar emas di Python untuk algoritma Machine Learning klasik. Modul ini menyediakan berbagai fungsi untuk pra-pemrosesan data (`preprocessing`), pembagian data (`model_selection`), berbagai algoritma model klasifikasi, dan fungsi perhitungan evaluasi performa (`metrics`).

---

## 2. Data Loading & Initial Exploration

Langkah ini bertujuan untuk memuat data mentah ke dalam memori komputer dan melakukan inspeksi awal untuk memahami struktur, kelengkapan, dan kebersihan data sebelum diolah lebih lanjut.

### Load Dataset

```python
data = pd.read_csv("../dataset/crop_recommendation.csv")
data.head()
```

### Cek Informasi Dataset

```python
data.info()
```

### Describe Statistik

```python
data.describe(include='all')
```

### Cek Missing Values

```python
data_null = data.isnull().sum()
data_null[data_null > 0]
```

### Cek Duplikat

```python
duplicate = data.duplicated()
print(data[duplicate])
```

### Cek Distribusi Label

```python
jumlah_unik = data['label'].nunique()
print(f"Jumlah label unik: {jumlah_unik}")
print(data['label'].value_counts())
```

**Penjelasan Konsep & Metode:**
- `read_csv()`: Membaca file Comma Separated Values (CSV) dan mengubahnya menjadi objek DataFrame pandas.
- `head()`: Menampilkan 5 baris pertama dari dataset. Ini berguna untuk mendapatkan gambaran visual sekilas tentang bentuk data dan nilai yang ada di tiap kolom.
- `info()`: Memberikan ringkasan struktural dataset, mencakup jumlah total baris, nama-nama kolom, jumlah nilai non-null (tidak kosong) per kolom, dan tipe data (seperti `float64` untuk angka desimal atau `object` untuk teks).
- `describe()`: Secara otomatis menghitung statistik deskriptif dasar untuk fitur numerik, seperti rata-rata (mean), standar deviasi (std), nilai minimum, maksimum, serta kuartil (25%, 50% atau median, 75%). Ini sangat membantu mengidentifikasi rentang nilai data.
- `isnull().sum()`: Mengevaluasi setiap sel di DataFrame, apakah bernilai kosong (NaN/Null) atau tidak, lalu menjumlahkannya per kolom. Dataset yang memiliki *missing values* memerlukan teknik penanganan khusus seperti imputasi (pengisian nilai) atau penghapusan baris.
- `duplicated()`: Mencari baris data yang informasinya sama persis 100% dengan baris lain. Data duplikat sering kali dihapus karena dapat menyebabkan model menjadi bias (menghafal data yang berulang).
- `value_counts()`: Menghitung frekuensi kemunculan tiap kelas/label unik. Jika satu kelas memiliki 1000 sampel sementara kelas lain hanya 10 sampel, dataset disebut *imbalanced* (tidak seimbang) dan memerlukan penanganan khusus. Pada kasus kita, tiap kelas memiliki tepat 100 sampel (sangat seimbang).

---

## 3. Exploratory Data Analysis (EDA)

EDA adalah tahap analisis investigasi kritis menggunakan statistik ringkasan dan alat visualisasi grafik. Tujuannya adalah untuk menemukan pola, anomali, tren, serta memeriksa asumsi awal mengenai data. Pemahaman yang dalam tentang data akan memandu keputusan pada tahap *Data Processing* dan *Modeling*.

### 3.1 Distribusi Fitur Numerik (Histogram Grid)

```python
num_features = data.select_dtypes(include=[np.number])
num_cols = num_features.columns

plt.figure(figsize=(14, 10))
for i, column in enumerate(num_cols, 1):
    plt.subplot(3, 4, i)
    sns.histplot(data[column], bins=30, kde=True, color='teal')
    plt.title(f'Distribusi {column}')
plt.tight_layout()
plt.show()
```

**Penjelasan Konsep:**
- **Histogram** berfungsi untuk menampilkan sebaran frekuensi sekumpulan data berkelanjutan. Lebar batang (bins) merepresentasikan rentang nilai (misal: suhu 20-25 derajat), dan tinggi batang merepresentasikan seberapa banyak data (sampel) yang jatuh pada rentang tersebut.
- **KDE (Kernel Density Estimate)** adalah garis lengkung mulus yang mengestimasi bentuk probabilitas distribusi data secara keseluruhan.
- Melalui histogram, kita bisa mengetahui apakah data berdistribusi normal (seperti lonceng simetris), *skewed* (menceng ke kiri atau kanan), atau *bimodal* (memiliki dua puncak).

### 3.2 Deteksi Outlier (Box Plot)

```python
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.boxplot(x=data[col], color='skyblue', ax=axes[i])
    axes[i].set_title(f'Box Plot — {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('')

for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Deteksi Outlier per Fitur Numerik', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
```

**Penjelasan Konsep Box Plot (Diagram Kotak Garis):**
- **Kotak (Box)**: Mewakili *Interquartile Range* (IQR), yaitu area di mana 50% data pertengahan berkumpul. Batas kiri kotak adalah Kuartil 1 (Q1, persentil ke-25) dan batas kanan adalah Kuartil 3 (Q3, persentil ke-75).
- **Garis Tengah di dalam kotak**: Adalah nilai Median (Kuartil 2, persentil ke-50).
- **Kumis (Whiskers)**: Garis yang memanjang dari kotak. Mengindikasikan sebaran batas bawah normal dan batas atas normal dari data.
- **Titik-titik individu di luar kumis**: Itulah yang disebut sebagai **Outliers** (pencilan). Ini adalah nilai-nilai ekstrem yang sangat jauh berbeda dari pola umum data lainnya (sangat tinggi atau sangat rendah). Outliers dapat merusak proses kalkulasi mean dan merusak akurasi algoritma Machine Learning jika tidak ditangani.

### 3.3 Matriks Korelasi Pearson

```python
plt.figure(figsize=[12, 10])
correlation_matrix = num_features.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
plt.title("Matriks Korelasi - Crop Dataset", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Penjelasan Konsep Korelasi:**
- **Correlation Matrix** menghitung dan menampilkan nilai *Koefisien Korelasi Pearson* antar setiap pasangan fitur numerik. Nilai rentang dari -1 hingga +1.
- Nilai mendekati **+1**: Menunjukkan korelasi linear positif yang kuat. Jika fitur A naik, fitur B ikut naik secara konsisten. (Warna merah gelap di heatmap).
- Nilai mendekati **-1**: Menunjukkan korelasi linear negatif yang kuat. Jika fitur A naik, fitur B justru turun secara konsisten. (Warna biru gelap).
- Nilai mendekati **0**: Tidak ada hubungan linear yang nyata antar kedua fitur.
- Mengapa ini penting? Jika ada dua fitur independen yang saling berkorelasi sangat kuat (mendekati 1), terjadi fenomena **Multikolinearitas**. Dalam beberapa model statistik, fitur berlebihan ini bisa menimbulkan masalah. Ini juga membantu kita mengidentifikasi fitur mana yang mungkin berhubungan langsung dengan target klasifikasi.

### 3.4 Distribusi Label (Count Plot)

```python
col = 'label'

plt.figure(figsize=(14, 8))
ax = sns.countplot(
    y=col,
    data=data,
    palette='viridis',
    order=data['label'].value_counts().index
)

for container in ax.containers:
    ax.bar_label(container, padding=3, fontsize=10)

plt.title(f'Distribusi {col}', fontsize=14, fontweight='bold')
plt.xlabel('Jumlah Sampel')
plt.ylabel('Jenis Tanaman (Label)')
plt.tight_layout()
plt.show()
```

**Penjelasan Konsep:**
- **Count Plot** menampilkan hitungan frekuensi kemunculan tiap kategori label (jenis tanaman).
- Visualisasi ini adalah langkah validasi krusial untuk memastikan tidak adanya ketidakseimbangan (imbalance) antar kelas. Model algoritma akan kesulitan mempelajari kelas minoritas secara adil jika datanya terlalu sedikit dibandingkan kelas mayoritas.

### 3.5 Rata-Rata Karakteristik Lingkungan per Tanaman (Heatmap Groupby)

```python
crop_summary = data.groupby('label')[num_cols].mean()

plt.figure(figsize=(12, 10))
sns.heatmap(crop_summary, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=0.5)
plt.title('Rata-Rata Karakteristik Lingkungan per Jenis Tanaman', fontsize=14, fontweight='bold')
plt.xlabel('Fitur Lingkungan', fontsize=11)
plt.ylabel('Jenis Tanaman', fontsize=11)
plt.tight_layout()
plt.show()
```

**Penjelasan Konsep:**
- Fungsi `groupby('label').mean()` mengelompokkan data berdasarkan jenis tanaman, lalu menghitung rata-rata matematika dari N, P, K, suhu, curah hujan, dst, khusus untuk baris milik tanaman tersebut.
- Visualisasi menggunakan heatmap mempermudah kita "membaca" pola khas kebutuhan tiap tanaman. Misalnya, kita dapat langsung melihat baris tanaman 'rice' (padi) membutuhkan curah hujan rata-rata yang sangat tinggi dibandingkan 'muskmelon'. Ini memberikan *domain knowledge* intuitif sebelum melatih model.

### 3.6 Analisis Hubungan Nutrisi (N vs P Scatter Plot)

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='N', y='P', hue='label', data=data, palette='tab20', alpha=0.7, s=50)
plt.title('Scatter Plot — Kadar Nitrogen (N) vs Fosfor (P)', fontsize=14, fontweight='bold')
plt.xlabel('Nitrogen (N)', fontsize=11)
plt.ylabel('Fosfor (P)', fontsize=11)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Penjelasan Konsep:**
- **Scatter Plot (Diagram Pencaran)** memetakan dua variabel numerik pada sumbu X dan Y. Warna titik (hue) merepresentasikan label tanaman.
- Titik-titik yang memiliki warna serupa cenderung berkumpul (mengelompok/klaster). Dengan plot N vs P, kita bisa mengidentifikasi apakah kombinasi kedua nutrisi ini cukup sebagai pembeda mutlak antar tanaman. Jika banyak titik berbeda warna saling menumpuk (overlapping), berarti fitur N dan P saja tidak cukup untuk model membuat klasifikasi yang akurat, dan fitur lain harus diikutsertakan.

### 3.7 Analisis Iklim (Suhu vs Curah Hujan Scatter Plot)

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(x='temperature', y='rainfall', hue='label', data=data, palette='tab20', alpha=0.7, s=50)
plt.title('Scatter Plot — Suhu vs Curah Hujan (Rainfall)', fontsize=14, fontweight='bold')
plt.xlabel('Suhu (Derajat C)', fontsize=11)
plt.ylabel('Curah Hujan (mm)', fontsize=11)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Penjelasan Konsep:**
- Analisis kombinasi lingkungan iklim murni (Suhu vs Hujan) tanpa memandang kualitas tanah. Ini penting untuk memahami profil habitat mikro tiap spesies (contoh: tropis kering vs subtropis basah).

### 3.8 Sebaran pH Tanah per Tanaman (Box Plot pH)

```python
plt.figure(figsize=(14, 6))
sns.boxplot(x='label', y='ph', data=data, palette='Set3')
plt.title('Sebaran pH Tanah per Jenis Tanaman', fontsize=14, fontweight='bold')
plt.xlabel('')
plt.ylabel('pH Tanah', fontsize=11)
plt.xticks(rotation=90)
plt.axhline(y=7.0, color='red', linestyle='--', alpha=0.7, label='pH Netral (7.0)')
plt.legend()
plt.tight_layout()
plt.show()
```

**Penjelasan Konsep:**
- Boxplot per kategori fitur sangat ampuh untuk menganalisis varians (lebar pita/spread) batas toleransi masing-masing tanaman terhadap kondisi spesifik seperti asiditas tanah (pH). Tanaman dengan rentang boxplot sempit berarti sangat sensitif terhadap perubahan pH, sementara yang rentangnya lebar lebih toleran/tangguh. Garis merah referensi (pH 7.0) memberikan patokan universal asam-netral-basa.

---

## 4. Data Processing

Tahap *Data Processing* (Pra-pemrosesan Data) adalah transformasi struktural dan perlakuan pembersihan pada dataset mentah sebelum diumpankan (fed) ke dalam algoritma model. "Garbage In, Garbage Out"—sebaik apa pun model ML, akan memberikan hasil buruk jika datanya kotor.

### 4.1 Pisahkan Fitur dan Label

```python
X = data.drop(columns='label').copy()   # fitur: N, P, K, temperature, humidity, ph, rainfall
y = data['label']                        # label: jenis tanaman yang direkomendasikan
```

**Penjelasan Konsep:**
- Di dalam dunia klasifikasi yang diawasi (Supervised Learning), kita harus membedakan secara tegas antara input variabel independen yang disebut **Fitur Prediktif (dinotasikan sebagai `X` kapital)**, dengan target keluaran variabel dependen yang disebut **Label (dinotasikan sebagai `y` kecil)**.

### 4.2 Penanganan Outlier dengan Metode Kuartil (IQR)

```python
num_cols = X.columns

for col in num_cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 - 1.5 * IQR
    
    # Capping nilai outlier (Winsorizing)
    X[col] = np.clip(X[col], lower_bound, upper_bound)
```

#### Visualisasi Box Plot Setelah Penanganan Outlier

```python
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(X.columns):
    sns.boxplot(x=X[col], color='skyblue', ax=axes[i])
    axes[i].set_title(f'Box Plot — {col} (Setelah IQR)', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('')

for j in range(len(X.columns), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Box Plot Setelah Penanganan Outlier (IQR Capping)', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
```

**Penjelasan Konsep Matematis IQR Capping:**
- Rentang Antar-Kuartil (Interquartile Range / IQR) adalah rentang data di mana distribusi data 50% pertengahan berada, yakni selisih nilai persentil ke-75 ($Q_3$) dan persentil ke-25 ($Q_1$).
- Rumus Matematis:
  - $IQR = Q_3 - Q_1$
  - $\text{Lower Bound} = Q_1 - 1.5 \times IQR$
  - $\text{Upper Bound} = Q_3 + 1.5 \times IQR$
- Setiap nilai angka yang jatuh lebih kecil dari `Lower Bound` atau lebih besar dari `Upper Bound` secara statistik dianggap outlier atau pencilan.
- **Teknik Capping (Winsorizing)**: Menggunakan fungsi `numpy.clip()`, semua data outlier tersebut tidak dihapus (karena menghapus baris dapat merusak keseimbangan klasifikasi). Sebagai gantinya, nilainya dipotong dan dibatasi secara paksa sehingga maksimal senilai `Upper Bound` atau minimal senilai `Lower Bound`.

### 4.3 Scaling Fitur (StandardScaler)

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

**Penjelasan Konsep:**
- Dataset kita memiliki skala (satuan ukur) yang berbeda-beda. Kandungan Nitrogen mungkin bernilai antara 0-140, sedangkan Curah Hujan bisa bernilai hingga 200-300 milimeter, dan nilai pH umumnya hanya di rentang kecil 3-9.
- Algoritma Machine Learning (khususnya berbasis perhitungan jarak metrik seperti K-Nearest Neighbors dan Support Vector Machine) sangat rentan bias terhadap kolom yang secara inheren memiliki angka lebih besar secara mutlak (mereka mengira curah hujan "lebih penting" dari pH murni hanya karena 300 lebih besar dari 9).
- **Standardisasi (StandardScaler)** mengubah keseluruhan sebaran distribusi data pada masing-masing kolom, sehingga nilai rata-rata sampel keseluruhan menjadi 0 ($\mu = 0$) dan deviasi standar merapat menjadi 1 ($\sigma = 1$). Data baru ini dinamakan "Z-score".
- Rumus Matematis Standardisasi:
  $$X_{scaled} = \frac{X - \mu}{\sigma}$$
- `fit_transform()`: Dua proses menjadi satu. `fit` mempelajari, menghitung, dan menyimpan $\mu$ dan $\sigma$ dari dataset; sementara `transform` menerapkan formula Z-score menggunakan parameter yang dipelajari tersebut.

### 4.4 Label Encoding

```python
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print(y)
```

**Penjelasan Konsep:**
- Algoritma Machine Learning tidak dapat mencerna kalimat string seperti "apple" atau "banana". Semua representasi matematika harus dalam bentuk numerik skalar.
- `LabelEncoder` menelusuri kamus unik tipe label target (dalam urutan alfabetik), lalu menugaskan angka unik. Contoh: "apple" menjadi angka 0, "banana" menjadi angka 1, "coconut" menjadi angka 2, dst.

---

## 5. Data Splitting (80:20)

Tahapan pemisahan ini mensimulasikan situasi dunia nyata, mencegah fenomena "Overfitting" (kondisi ketika model sekadar "menghafal buta" data yang diajarkan namun gagal saat dihadapkan dengan variasi data yang benar-benar baru di masa depan).

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=0.2,
    stratify=y
)

print(f"Train set shape: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Test set shape:  X_test  = {X_test.shape}, y_test  = {y_test.shape}")
```

**Penjelasan Konsep & Detail Parameter:**
- `test_size=0.2`: Kita memotong dataset menjadi 2 potongan, di mana 80% (dari 2200 = 1760 baris) dialokasikan sebagai data `Train` untuk melatih otak mesin, sementara porsi 20% yang disisakan secara tersembunyi sebagai `Test` atau data ujian akhir untuk mengevaluasi apakah mesin benar-benar paham secara logis, bukan sekadar menghafal.
- `random_state=42`: Ini adalah penanda (seed) pseudo-random number generator. Agar data yang displit teracak, tetapi pengacakannya "terkunci". Jika script dijalankan ribuan kali di komputer berbeda, baris-baris yang masuk subset train dan subset test akan selalu konsisten sama. Angka bebas, lazim diatur ke nilai acak klasik "42".
- `stratify=y`: Parameter sangat esensial. Parameter ini memerintahkan komputer melakukan pembagian sedemikian rupa dengan mempertahankan dan mendistribusikan proporsi fraksional yang tepat dari masing-masing kelas dari himpunan `y` asli. Tanpa `stratify`, pengacakan nasib buruk bisa saja menaruh semua sampel "mangga" ke dalam data latih, dan tidak menyisakan sama sekali "mangga" di data uji, sehingga performa model menjadi cacat secara evaluasi.

---

## 6. Model Training

Inilah jantung utama dari Machine Learning—sebuah tahapan "Training" di mana program komputer merumuskan fungsi generalisasi matematis dengan mengekstraksi dan menyerap pola dari relasi `X_train` menuju `y_train`. Untuk komparasi, modul ini melatih 5 variasi keluarga algoritma utama.

```python
knn = KNeighborsClassifier().fit(X_train, y_train)
dt  = DecisionTreeClassifier().fit(X_train, y_train)
rf  = RandomForestClassifier().fit(X_train, y_train)
svm = SVC().fit(X_train, y_train)
nb  = GaussianNB().fit(X_train, y_train)

print("Training selesai untuk seluruh algoritma.")
```

**Konsep Analitis Cara Kerja Setiap Algoritma:**
1. **K-Nearest Neighbors (KNN)**: Adalah algoritma probabilistik non-parametrik yang menyimpan seluruh data train. Saat dipaksa memprediksi titik data *Test* baru, KNN menghitung jarak geometris spasial berdimensi N (misal jarak Euclidean) terhadap seluruh titik, mencari $K$ (misalnya 5) tetangga terdekat, dan menentukan kelas titik tersebut berdasarkan voting jumlah mayoritas tetangganya.
2. **Decision Tree (Pohon Keputusan)**: Membangun logika *If-Else* secara hierarkis seperti permainan 20 pertanyaan. Model meninjau seluruh fitur pada setiap iterasi simpul (node), mengevaluasi menggunakan rumusan pemisahan metrik keheterogenan informasi murni (*Information Gain* atau *Gini Impurity*), untuk meretas data set menjadi cabang murni yang menghasilkan kepastian kelas prediksi tertinggi di daun bawah (leaf node).
3. **Random Forest**: Algoritma ansambel tingkat lanjut (*Bagging Ensembles*). Alih-alih membuat satu pohon tunggal yang mudah bias (overfit), Random Forest secara membabi buta membangkitkan ratusan variasi spesimen *Decision Trees* paralel dengan sampling baris data latih yang diacak dan fitur yang dirandom. Hasil prediksi didapat dari kalkulasi musyawarah voting mayoritas seluruh ratusan pohon, membuat prediksi menjadi luar biasa robust/tangguh.
4. **Support Vector Machine (SVM)**: Pendekatan komputasional matematis kelas berat. Berusaha menggambar hiperbidang geometri (Hyperplane) ruang batas pemisah dengan orientasi yang menyajikan celah pelebaran (margin) maksimal terjauh terhadap titik sampel perbatasan antar-kelas (*support vectors*). Untuk dataset saling tertumpuk (*non-linear*), digunakanlah manuver *Kernel Trick* yang memproyeksikan data ke dimensi imaginer jauh lebih tinggi untuk menemukan sudut tebas ideal.
5. **Naive Bayes (GaussianNB)**: Mengandalkan formula probabilitas murni peninggalan pendeta Inggris kuno Thomas Bayes (Teorema Bayes). Pendekatannya terbalik—bukan memisahkan batas, tetapi menghitung profil probabilitas kemunculan statistik dari asumsi terdistribusi (Gaussian / kurva lonceng normal) yang "naif", mengasumsikan bahwa eksistensi fitur Ph tanah sepenuhnya independen lepas dari eksistensi fitur Nitrogen. Secara mengejutkan, meskipun "naif" sering salah secara realitas fisik, kinerja prediktifnya menakjubkan dan beroperasi dengan kecepatan kilat tiada banding.

---

## 7. Model Evaluation - Classification Report & Confusion Matrix

Mengevaluasi secara saintifik dan empiris, apakah model sudah siap dideploy ke dunia nyata, atau sekadar mesin tebakan serampangan (*random guessing*). Kita menguji para kandidat model lulusan proses Training melawan instrumen ujian suci: data tersembunyi (`X_test`) dan membandingkan hasil outputnya dengan `y_test`.

### Fungsi Evaluasi Kustom

```python
def evaluate_model(name, model, X_test, y_test, label_names):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print(f"==== {name} ====")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1-Score  : {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("
Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    plt.figure(figsize=(14, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return {
        'Model': name,
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall':    recall_score(y_test, y_pred, average='weighted'),
        'F1-Score':  f1_score(y_test, y_pred, average='weighted'),
    }

label_names = label_encoder.classes_

results = [
    evaluate_model("KNN",            knn, X_test, y_test, label_names),
    evaluate_model("Decision Tree",  dt,  X_test, y_test, label_names),
    evaluate_model("Random Forest",  rf,  X_test, y_test, label_names),
    evaluate_model("SVM",            svm, X_test, y_test, label_names),
    evaluate_model("Naive Bayes",    nb,  X_test, y_test, label_names),
]
```

**Penjelasan Konsep & Logika Evaluasi Mutlak:**
- Fase pengujian `predict(X_test)` memaksa model berimprovisasi melempar serangkaian tebakan target berdasar fitur mentah tak dikenal.
- Pendekatan multi-kelas mensyaratkan parameter *averaging scheme*. Menggunakan `average='weighted'` memastikan angka metrik keseluruhan dieksekusi dengan perhitungan matematis yang memihak bobot (*weighted*) dari rasio dukungan (*support level*) kuantitas unik pada masing-masing sub-kelas, ketimbang sekadar dibagi rata biasa (`macro`), yang dapat menyesatkan jika kelak menjumpai sub-kelas dataset tak imbang.
- Library `classification_report` adalah instrumen brilian yang meringkas paparan rinci setiap kelas secara parsial dari sudut pandang metrik Presisi, *Recall*, dan harmonisasi *F1-Score*.
- Visualisasi Heatmap Confusion Matrix menguraikan setiap langkah model. Sumbu Vertikal adalah kenyataan kebenaran abadi (Actual Label), Sumbu Horizontal adalah fatwa sang mesin algoritma (Predicted).

### Metrik Evaluasi Kinerja Industri Klasifikasi yang Digunakan

| Metrik Standar | Rumus Matematis Murni | Deskripsi & Interpretasi Konseptual |
|--------|-------|--------------|
| **Accuracy (Akurasi)** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Indikator kasat mata paling lazim; merupakan persentase proporsi tebakan presisi valid mutlak yang benar dari keseluruhan populasi pengujian. Sangat andal untuk dataset dengan ekuilibrium kelas tinggi. |
| **Precision (Presisi)** | $\frac{TP}{TP + FP}$ | Skema reliabilitas tebakan konfirmasi murni. Ketika sang mesin berteori bahwa kelas sampel ini adalah "Positif", seberapa persen peluang teorinya tersebut secara faktual terbukti valid? Krusial bila kesalahan *False Positive* menghadirkan komplikasi kerugian besar. |
| **Recall (Sensitivitas)** | $\frac{TP}{TP + FN}$ | Skema sapu bersih jangkauan pemindaian. Mengkalkulasi berapa probabilitas entitas kelas "Positif" nyata yang tersisir selamat oleh mesin tanpa bersembunyi tenggelam lari sebagai kegagalan defisit negatif ganda (*False Negative*). |
| **F1-Score** | $2 \times \frac{Precision \times Recall}{Precision + Recall}$ | Rata-rata silang aritmatika harmonik hibrida penyeimbang antara ekstremitas *Precision* dan kecondongan longgar *Recall*. Mutlak diutamakan bagi perbandingan komparatif model multi-kelas yang rentan ketidakseimbangan klasifikasi asimetris. |
| **Confusion Matrix** | Struktur Matriks Komprehensif Berdimensi $N \times N$ | Papan catur rekapitulasi analitik komutatif per-kelas prediksi vis a vis realita murni. Matriks persegi ini melucuti persis di titik kelemahan komparatif manakah algoritma kebingungan mendistribusikan salah tebak. |

---

## 8. Model Comparison (KP, CM, TP, FP, FN, TN)

Sub-seksi terkhusus ini menyelenggarakan kalkulasi bedah murni membongkar konstruksi parsial Confusion Matrix menjadi fondasi unit partikel terurai absolut elemen: True Positive (TP), False Positive (FP), False Negative (FN), dan True Negative (TN) untuk arsitektur multi-kelas bertumpuk yang sangat rumit.

```python
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Mengekstraksi partikel fondasi konseptual
    tp = cm.diagonal().sum()                        
    fp = cm.sum(axis=0) - cm.diagonal()            
    fn = cm.sum(axis=1) - cm.diagonal()            
    tn = cm.sum() - (tp + fp.sum() + fn.sum())     

    return {
        "Confusion Matrix": cm,
        "True Positive":    tp,
        "False Positive":   fp.sum(),
        "False Negative":   fn.sum(),
        "True Negative":    tn,
        "Accuracy":         accuracy_score(y_test, y_pred),
        "Precision":        precision_score(y_test, y_pred, average='weighted'),
        "Recall":           recall_score(y_test, y_pred, average='weighted'),
        "F1-Score":         f1_score(y_test, y_pred, average='weighted'),
    }

result = {
    'K-Nearest Neighbors (KNN)': evaluate_model(knn, X_test, y_test),
    'Decision Tree (DT)':        evaluate_model(dt,  X_test, y_test),
    'Random Forest (RF)':        evaluate_model(rf,  X_test, y_test),
    'Support Vector Machine (SVM)': evaluate_model(svm, X_test, y_test),
    'Naive Bayes (NB)':          evaluate_model(nb,  X_test, y_test)
}

rows = []
for model_name, metrics in result.items():
    rows.append({
        "Model":     model_name,
        "Accuracy":  metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall":    metrics["Recall"],
        "F1-Score":  metrics["F1-Score"],
    })

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
```

**Penjelasan Konseptual Fundamental Dekonstruksi Confusion Matrix Multi-Kelas:**
Dalam terminologi paradigma One-vs-Rest, untuk sembarang titik ekuivalen kelas C tertentu, berlaku definisi:
- **True Positive (TP) - Prediksi Positif Akurat:** Sang model menyatakan ini milik Kelas C; dan kenyataan sejati memang membuktikan sampel ini milik Kelas C. (Terletak persis melintang menapak di sepanjang titik garis diagonal matriks).
- **False Positive (FP) - Alarm Positif Palsu / Prediksi Prematur / Kesalahan Tipe I:** Sang model gegabah melontarkan alarm mengklaim bahwa suatu sampel merupakan bagian dari keluarga Kelas C; tetapi kenyataan membuktikan sampel itu milik kerabat Kelas Asing lain tak dikenal. (Mengkalkulasi agregat akumulatif penjumlah seluruh angka penyusup dalam lajur kolom vertikal spesifik C, tanpa mencampuradukkan sang angka murni sel diagonal C sendiri).
- **False Negative (FN) - Insiden Kebobolan / Luput Sapuan / Kesalahan Tipe II:** Terdapat sebuah individu sampel yang dalam kenyataan absolut merupakan bagian dari Kelas C; tetapi sang mesin dengan malangnya rabun mengklasifikasikannya bergeser tercecer menjadi milik Kelas Asing lain. (Dihitung dengan mensumasi segenap nilai menyimpang pada retakan pita baris horizontal spesifik Kelas C secara menyamping komprehensif, selain dari si diagonal TP sendiri).
- **True Negative (TN) - Penolakan Negatif Valid Mutlak:** Sampel murni pada esensinya memang bukan Kelas C; dan mesin cerdas memalingkan muka merestui penolakan rasional untuk menuduhnya sebagai Kelas C. Kalkulasi komputasi ekuasi aljabarnya diraih dari pengurangan total populasi raya sampel menyisihkan sumbangsih kumulatif angka absolut TP, akumulasi angka absolut vektor FP utuh, dan selongsong akumulasi angka absolut vektor FN.

---

## 9. Visualizing Model Performance

### 9.1 Bar Chart - Perbandingan Akurasi

```python
plt.figure(figsize=(10, 5))
bars = plt.bar(summary_df['Model'], summary_df['Accuracy'],
               color=['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0'])
plt.title('Perbandingan Akurasi Model Klasifikasi Mutlak', fontsize=14, fontweight='bold')
plt.ylabel('Tingkat Metrik Accuracy')
plt.ylim(0.8, 1.02)
plt.xticks(rotation=15, ha='right')

for bar, val in zip(bars, summary_df['Accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.2%}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()
```

### 9.2 Heatmap - Perbandingan Seluruh Dimensi Metrik Terpadu

```python
metrics_df = summary_df.set_index('Model')[['Accuracy','Precision','Recall','F1-Score']]

plt.figure(figsize=(8, 4))
sns.heatmap(metrics_df, annot=True, fmt='.2%', cmap='YlGn',
            vmin=0.8, vmax=1.0, linewidths=0.5)
plt.title('Pemetaan Komparasi Spektrum Semua Metrik', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Penjelasan Konsep Pemahaman Visualisasi Komparasi:**
- Grafik *Bar Chart* menerjemahkan komparasi angka numerik murni membosankan ke presentasi visual dramatis yang langsung dipahami audiens publik bisnis sekilas lirikan, menegaskan kasta pemeringkatan arsitektur peraih tahta Accuracy tertinggi yang dominan.
- *Heatmap Metrik* meredam distorsi jebakan "accuracy paradox". Menyatukan barisan 4 pilarnya secara serentak melenyapkan penipuan model yang hanya seolah sempurna di accuracy, namun sesungguhnya hancur terperosok bobrok di dimensi rasio *Precision* ataupun merayap tertatih pada dimensi jangkauan *Recall*. Transisi warna cerah solid hijau memastikan reliabilitas yang seragam dan tangguh merata tanpa cacat lubang dimensi metrik di titik celah mana pun.

---

## 10. Best Model & Error Analysis

Memilih sang pemenang mahkota takhta tidak sekadar berhenti pada publikasi proklamasi pencapaian skor semata; investigasi wajib berlanjut membedah mikroskopis otopsi membedah rentetan kegagalan sang model sang juara agar teridentifikasi benang merah misteri di mana kelemahannya bersemayam.

### Pemilihan Kasta Model Terbaik Berdasarkan Skala Accuracy

```python
best_model_name = summary_df.loc[summary_df['Accuracy'].idxmax(), 'Model']
print(f"Model Klasifikasi Penakluk Terbaik: {best_model_name}")

model_map = {
    'K-Nearest Neighbors (KNN)': knn,
    'Decision Tree (DT)': dt,
    'Random Forest (RF)': rf,
    'Support Vector Machine (SVM)': svm,
    'Naive Bayes (NB)': nb
}
best_model = model_map[best_model_name]
y_pred_best = best_model.predict(X_test)
```

### Confusing Matrix Otopsi Kegagalan (Error Analysis Top 5 Misklasifikasi Fatal)

```python
cm_best = confusion_matrix(y_test, y_pred_best)

# Kalkulasi isolasi ekstraksi murni cacat prediksi tiap himpunan kelas
errors_per_class = cm_best.sum(axis=1) - cm_best.diagonal()
top5_idx = errors_per_class.argsort()[-5:][::-1]
top5_labels = label_encoder.classes_[top5_idx]

# Sub-matriks pembedahan pengirisan eksklusif khusus 5 kelas paling menderita kebingungan
cm_top5 = cm_best[np.ix_(top5_idx, top5_idx)]

plt.figure(figsize=(7, 5))
sns.heatmap(cm_top5, annot=True, fmt='d', cmap='Reds',
            xticklabels=top5_labels, yticklabels=top5_labels)
plt.title(f'Pembedahan Kesalahan Fatal: Top 5 Kelas Saling Tertukar
({best_model_name})', fontweight='bold')
plt.xlabel('Diagnosa Prediksi Sang Model')
plt.ylabel('Kenyataan Kebenaran Aktual Realitas')
plt.tight_layout()
plt.show()
```

**Penjelasan Konseptual Fundamental Error Analysis Saintifik:**
Analisis kegagalan (Error Analysis) mendedahkan ruang sempit iterasi peningkatan pengembangan kecerdasan di fase kelak mendatang:
1. `cm_best.diagonal()` menyarikan serpihan kristalisasi kemenangan absolut prediksi kebenaran murni tiap kelas (elemen garis melintang diagonal utuh matriks cm_best).
2. `cm_best.sum(axis=1)` menghimpun kompilasi komprehensif sumasi total mutlak deret horizontal per matriks (menghitung absolut total murni himpunan sampel yang sesungguhnya secara biologis sejati bernaung pada label spesies keluarga tersebut di data tes).
3. Ekuasi aritmatika logis dari operasi kalkulus pengurangan `cm_best.sum(axis=1) - cm_best.diagonal()` mengekspos deret kalkulasi selisih yang tak terbantahkan mendefinisikan total kerugian kegagalan prediksi (rasio mutlak error) yang menimpa per pilar kelas.
4. Fungsi manuver peririsan cerdas multidimensional indeksing tensor murni `np.ix_()` meretas, menduplikasi khusus sekelompok subset himpunan titik iris silang ruang baris dan rentang dimensi lajur kolom terbatas, menjebol sub-matriks spesifik sempit, khusus menjaring sekawanan 5 pilar kelas paling membingungkan (Top-5 confusion rank) untuk ditelisik keterkaitan pertukaran takdir kegagalannya di atas hamparan panggung peta visualisasi matriks panas spesifik benderang (Red Heatmap khusus area persimpangan krisis).

---

## 11. Menyimpan Model (Save Model)

Sebuah iterasi model Machine Learning yang telah diajarkan mengolah ribuan sampel mengonsumsi daya intelektual waktu dan CPU komputer komputasi yang tak tergantikan. Maka fase ini bermaksud membekukan sel otak pengetahuan yang telah dikerahkan model agar tersimpan rapi secara abadi dalam arsip solid dan dapat diluncurkan terbang kembali esok lusa seketika menyentuh dimensi deployment ranah produksi industri.

```python
import joblib
import os

os.makedirs('../model', exist_ok=True)

# Membekukan serangkaian trinitas otak intelektual komputasi eksekusi model
joblib.dump(best_model, '../model/best_crop_model.joblib')
joblib.dump(scaler, '../model/crop_scaler.joblib')
joblib.dump(label_encoder, '../model/crop_label_encoder.joblib')

print("Operasi Serialisasi Intelektual Berhasil: Model, Scaler, dan Label Encoder tersimpan rapi abadi ke pilar folder model/")
```

**Penjelasan Konsep Esensi Penyimpanan Model Ekspor Industri Mutlak:**
- Metodologi librari purbakala Python murni Pickle maupun Joblib modern menunaikan perannya merekam DNA susunan genetik state objek di memori operasional ke berkas padat biner file ber-ekstensi formasi khusus `.joblib`.
- **Trinitas Penyimpanan Krusial:** Ekspor penyalinan tidak sah dilakukan eksklusif menyendiri untuk sekadar si arsitektur pilar algoritma otak utamanya saja (`best_model`). Ia harus diselaraskan abadi tergandeng selamanya bersama para asisten pilar preprocessing mutlak sang pelayan setianya (`scaler` penyeragam dimensi z-score dan sang penterjemah numerik sandi `label_encoder`). Gagal membawanya serentak mengakibatkan kebingungan distorsi tak tertolong manakala menerjemahkan angka inferensi yang menyimpang dimensi standarnya kelak di siklus perputaran masa uji data segar mendatang.

---

## 12. Memuat Model dan Pengujian (Load Model & Testing)

Manifestasi simulasi utuh siklus daur ulang pemulihan kebangkitan roh otak model terbekukan, untuk ditugaskan beroperasi mengais interpretasi wahyu atas serpihan sekelumit set data sampel baru (inferensi), murni layaknya sebuah rintisan perwujudan sistem *Backend Server Machine Learning API Deployment* di ranah korporat sesungguhnya.

```python
import joblib

# Menerawang merestorasi membangkitkan kepingan kecerdasan roh abadi Trinitas dari peristirahatan persinggahan folder
loaded_model = joblib.load('../model/best_crop_model.joblib')
loaded_scaler = joblib.load('../model/crop_scaler.joblib')
loaded_encoder = joblib.load('../model/crop_label_encoder.joblib')

# Inisialisasi simulasi set kepingan serbuk data pengujian asimilasi murni mentah sampel segar lingkungan ladang baru yang tak pernah terkuak di mata algoritma model
df_baru = pd.DataFrame({
    'N': [90],
    'P': [42],
    'K': [43],
    'temperature': [20.87],
    'humidity': [82.00],
    'ph': [6.50],
    'rainfall': [202.93]
})

# PERINGATAN KRITIS: Mutlak eksklusif menggunakan titah perintah komando .transform() BUKAN .fit_transform() di fase eksekusi suci pengujian!
data_scaled = loaded_scaler.transform(df_baru)

# Mengerahkan titah sabda peluncuran ramalan suci komando kalkulasi prediksi model
prediksi = loaded_model.predict(data_scaled)

# Mentranslasi sandi dekoding murni angka sakral algoritma mesin algoritma ke lidah wujud manifestasi bahasa manusia yang dipahami
prediksi_label = loaded_encoder.inverse_transform(prediksi)

print(f"Rekomendasi Tanaman Mutlak Terpilih Berdasarkan Analisis Lingkungan Ladang Adalah: {prediksi_label[0]}")
```

**Penjelasan Konsep Kritis Simulasi Deploy Inferensi Ujian Baru:**
- Fungsi instrumen utilitas `joblib.load()` melakukan manuver bedah *deserialization* (kebalikan dari *serialization*) untuk merestorasi menghidupkan merekronstruksi utuh pilar objek instansiasi memori instan di atas puing-puing serbuk bongkahan bit-bit biner hardisk.
- Hukum Haram Inferensi Transformasi: Praktisi murni mutlak wajib melayangkan seruan komando eksklusif sintaksis penjelmaan fungsi `loaded_scaler.transform()`, karena ia patuh buta mutlak bergantung menggunakan pijakan pilar titik pusat memori rata-rata metrik (Mean parameter $\mu$) lama dan pilar bentangan memori simpangan baku (Deviasi Standar $\sigma$) lampau warisan bawaan data pembelajaran training murni purba (yang merajai 1760 sampel kuno). Segala tindak ceroboh keliru memanggil ritual perintah `fit_transform()` pada data sekerat pengujian akan fatal menelan memporak-porandakan titik acuan orientasi kalibrasi gravitasi pusat asal (meriset sebaran hanya untuk 1 sampel ini, merusak koherensi semesta Z-score).
- Keajaiban operasi penyadapan invers dekoding penterjemahan `loaded_encoder.inverse_transform()` menuntaskan paripurna penutupan mata rantai rentetan siklus siklus menterjemahkan mengonversi bahasa sandi matematika skalar integral kode numerik (contohnya angka absolut 20) mengarungi balik menerobos batasan mesin, menjelma bertransformasi kembali bermetamorfosis sempurna wujud eksistensial bahasa kalbu string murni manusiawi layaknya kalimat teks ("rice").

---

## Ringkasan Project

| Tahapan Operasional | Objektif Misi Peran Esensial | Output |
|--------|--------|--------|
| Import Library | Persiapan pengadaan instrumen tools fondasi saintifik ML | Eksekusi kelancaran ketersediaan library tanpa hambatan sintaks error |
| Data Loading | Memompa masuk transfer bongkahan rekam jejak dataset mentah historis 22 faksi kelas | Himpunan baris kolom DataFrame pandas utuh siap diolah |
| EDA (Eksplorasi) | Membongkar pencerahan pandangan mendalam menginvestigasi memahami hakikat sebaran variabilitas relasi takdir pola pilar distribusi data dan profil matriks iklim | Deretan Galeri Subplot Grid, Anatomi Rangkaian Matriks Panas Rata-rata Lingkungan Iklim |
| Data Processing | Ritual sterilisasi amputasi pemotongan IQR Outlier Capping, standarisasi seragam penyelarasan StandardScaler, & konversi dekoding LabelEncoder | X matriks pilar mulus seragam suci, array label vektor y matriks angka integer |
| Data Splitting | Isolasi karantina perpecahan Train/Test split pakem (80:20) | 4 Kepingan Subset Fragmen Terpecah Belah: X_train, X_test, y_train, y_test |
| Model Training | Eksekusi kawah candradimuka penempaan pelatihan murni 5 ksatria sakti perwakilan aliran algoritma penguasa (KNN, DT, RF, SVM, NB) | Kelahiran penciptaan 5 entitas mesin model siap tempur |
| Evaluation | Eksekusi peluncuran instrumen interogasi pengujian pengukuran penyeimbangan performa kaliber kalkulasi rapor saintifik | Penjabaran laporan purna bakti Classification Report & hamparan rentang pilar Confusion Matrix |
| Comparison | Konfrontasi penyandingan komparasi matriks adu ketangkasan pilar klasemen tabel kompetisi turnamen akbar antar pahlawan arsitektur mesin model | Ekstrak saripati Summary DataFrame laporan perbandingan paripurna final klasemen |
| Visualization | Transformasi metamorfosis penciptaan keajaiban translasi presentasi keindahan desain estetis grafis artistik visual wujud perbandingan hierarki kemapanan visual kinerja pertempuran metrik ksatria model arsitektur | Grafis Batang Tinggi Akurasi Menjulang & Permukaan Matriks Panas Spektrum Intensitas Gradasi Visual Warna Metrik Keunggulan |
| Best Model | Penobatan seremonial penyaringan seleksi elit paripurna penobatan satu kampiun tertinggi pemuncak klasemen jawara agung takhta mutlak & sayatan analisa otopsi diagnosis bedah misteri kesalahan keterpurukan tragis fatal error class confusion anomaly | Titah deklarasi Best model + CM khusus persembahan panggung diagnosis isolasi error |
| Save & Load Model | Ekspor konservasi pelestarian abadi keabadian warisan & pembuktian replikasi kebangkitan simulasi inkarnasi penyelesaian problem data segar misterius asing | Warisan artefak relikui File `.joblib` suci abadi di ruang kuil perpustakaan `model/` |

---

## Tips Belajar
1. Kewajiban fundamental mutlak, jalankan proses rutinitas pemicuan tombol setiap blok sel baris secara urut sistematis komprehensif tak henti (Cell -> Run All). Hindari loncatan tak konsisten yang merobek logika komputasional mesin Python memori alokasi lokal state mesin variabel.
2. Selidiki hayati dalam-dalam resapi hakikat lukisan hasil mahakarya plot EDA untuk melihat dan menyimak bisikan pesan tersirat mengenai spesifikasi profil parameter syarat fundamental ciri khas eksistensi masing-masing data tiap individu varietas spesies keanekaragaman flora biologis spesies tanaman botani.
3. Bereksperimenlah membangkang berinovasi liar memodifikasi merubah tatanan arsitektur fraksi pengaturan pecahan pembagian parameter `test_size` pemisahan belahan dataset (misal geser rasio kuota parameter menjadi rentang radikal ekstrem 0.1, 0.25, 0.3) lantas selidiki secara saksama amati investigasi tajam segala rentetan konsekuensi domino dampaknya berimbas mendistorsi kestabilan nilai fluktuasi akurasi metrik unjuk gigi performa instrumen penilaian kekuatan model teratas juara.
4. Lakukan penemuan terobosan manuver agresif modifikasi berani bermain mempermainkan otak-atik suntikan bumbu suplemen *hyperparameter* di luar nalar pada rahim jantung algoritma model tak bernyawa (misal merevisi merombak pergeseran modifikasi rentang rentetan pergerakan jangkauan parameter jumlah bilangan rasio `n_neighbors` pada KNN) untuk memantau merangkum menelaah pergeseran kebangkitan guncangan perubahan gelombang riak pasang surut fluktuasi grafik kenaikan akurasi kompetisi pertempuran sengit klasemen model algoritma kejam algoritma arena gladiatur.
5. Pantau cermati selami renungkan merenung dengan saksama konsentrasi meresapi peta kekacauan pusaran pusara medan area zona arena petak kuadrat confusion matrix, memeras konsentrasi investigasi taktis kejam pelacakan pemburuan penelusuran identifikasi misteri enigma detektif forensik kriminal pengungkapan rahasia pilar kelas subspesies keluarga botani mana yang senantiasa saling bersitegang bertarung silang tertukar dan galilah pelajari secara forensik selidikilah temukan deduksi penarikan konklusi akar musabab penyebab utamanya langsung dari sumber fondasi data mentah kemiripan parameter karakteristik wujud pertautan sebaran kelembapan nutrisi iklim pertautan variasi curah fiturnya genetik asal di rahim pusara rahim data asal mulanya murni hulu.
