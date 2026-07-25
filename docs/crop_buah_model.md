# Model Klasifikasi Buah Tropis - Crop Buah Tropis

**Tujuan Pembelajaran:** 
Memberikan wadah pelatihan pengolahan dataset klasifikasi multiclass kelas menengah yang mendemonstrasikan kompleksitas dataset riil di dunia industri. Berbeda dengan dataset statis pada modul pemula, dataset ini memiliki variabilitas alami (*noise*) kondisi iklim dan sifat kimiawi tanah untuk memproyeksikan presisi kecerdasan mutlak. Peserta akan ditantang menentukan 1 dari 4 pilihan kandidat varietas taksonomi famili botani buah-buahan endemik daerah tropis basah (Pisang, Mangga, Pepaya, Semangka) dengan tingkat akurasi model yang dipaksa mentok di 92-97%, memaksa peserta untuk membedah *Error Analysis* yang mendalam.

**Dataset:** Dataset Buah Tropis Sintetis Realistis (`crop_buah_tropis.csv`)

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

Tahap inisialisasi modul Python yang mendasari manipulasi data tabular, komputasi numerik, pembuatan visualisasi matematis, klasifikasi kecerdasan buatan, serta instrumen ekspor-impor objek persisten.

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

**Penjelasan Konsep Kritis Analitik:**
- `pandas` dan `numpy`: Fondasi operasi aljabar matriks dan manipulasi *DataFrame* berkinerja tinggi.
- `matplotlib.pyplot` dan `seaborn`: Mesin penderet grafik visual untuk mentransformasikan ribuan angka ke dalam geometri pola persepsi visual.
- `joblib`: Menangani serialisasi (konversi objek python ke susunan *byte* memori disk) agar model algoritma bisa disimpan dan didistribusikan.
- `sklearn` (Scikit-Learn): Penyedia instrumen Machine Learning. Modul buah tropis ini memperkenalkan fungsi standardisasi distribusi kurva lonceng normal `StandardScaler` dan memfokuskan pilar matematis probabilistik murni `GaussianNB` (Naive Bayes).

---

## 2. Data Loading & Initial Exploration

Langkah pemuatan arsitektur matriks tabel ke dalam Random Access Memory (RAM) untuk ditelaah secara statistik deskriptif sebelum diumpankan ke permesinan algoritma ML.

```python
data = pd.read_csv("../dataset/crop_buah_tropis.csv")
data.head()
```

### Cek Informasi Dataset

```python
print(f"Dataset shape: {data.shape}")
data.info()
```

### Describe Statistik

```python
data.describe(include='all')
```

### Cek Missing Values & Duplikat

```python
print("Missing values:", data.isnull().sum().sum())
print("Duplikat:", data.duplicated().sum())
```

### Cek Distribusi Label

```python
print(data['label'].value_counts())
```

**Penjelasan Analisa Tahapan Konsep Murni:**
- Matriks dataset berdimensi $2400 \times 8$ (2400 baris sampel observasi independen dan 8 kolom matriks).
- Proporsi sebaran data diatur agar terdistribusi merata sempurna (*uniform distribution*), yakni 600 sampel per ras varietas (Banana, Mango, Papaya, Watermelon).
- **Variansi dan Noise Alami:** Dataset buatan ini secara matematis diinjeksi dengan deviasi *noise*. Rentang kelembaban, pH tanah, atau curah hujan sengaja direkayasa agar titik persebaran batas kelasnya saling tumpang tindih (*overlapping bounds*). Kondisi empiris ini mencegah model mencapai akurasi $100\%$ dan menciptakan *blind spot* (titik buta) alami yang dapat memicu *False Positives* dan *False Negatives*.

---

## 3. Exploratory Data Analysis (EDA)

Analisis geometri data menggunakan perpaduan *Descriptive Statistics* dan perwujudan plot visual multivariat.

### 3.1 Distribusi Fitur Numerik (Histogram Grid 3x3)

```python
num_features = data.select_dtypes(include=[np.number])
num_cols = num_features.columns

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(data[col], bins=30, kde=True, color='teal', ax=axes[i])
    axes[i].set_title(f'Distribusi {col}', fontsize=12, fontweight='bold')
    axes[i].set_xlabel('')

for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Distribusi Fitur Numerik Dataset Buah Tropis', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
```

### 3.2 Deteksi Outlier (Boxplot Grid 3x3)

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

### 3.3 Matriks Korelasi Pearson

```python
plt.figure(figsize=(10, 8))
correlation_matrix = num_features.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
            fmt='.2f', linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Matriks Korelasi Pearson — Dataset Buah Tropis', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

### 3.4 Rata-Rata Karakteristik Lingkungan per Buah (Heatmap Groupby)

```python
crop_summary = data.groupby('label')[num_cols].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(crop_summary, annot=True, fmt='.1f', cmap='YlGnBu', linewidths=0.5)
plt.title('Rata-Rata Karakteristik Lingkungan per Jenis Buah Tropis', fontsize=14, fontweight='bold')
plt.xlabel('Fitur Lingkungan', fontsize=11)
plt.ylabel('Jenis Buah', fontsize=11)
plt.tight_layout()
plt.show()
```

**Konsep Domain Knowledge Berbasis Statistik Rata-rata ($\mu$):**
Tabel rekapitulasi agregasi `groupby().mean()` memberikan cetak biru *domain knowledge* industri pertanian murni.
- **Korelasi Iklim:** Kita bisa melihat perbedaan mencolok kebutuhan curah hujan ($\mu_{Rainfall}$). Mangga (Mango) adalah vegetasi iklim lebih kering (~50% kelembapan) sementara Pepaya (Papaya) beradaptasi murni di kelembapan tinggi ekstrim (>80%).
- Keunggulan EDA ini menuntun Data Scientist menentukan jika model nanti melakukan kesalahan tebakan antara Mangga dan Semangka, analisis awal ini akan memvalidasi apakah curah hujan sampel tersebut berada pada "persimpangan irisan" distribusi kedua kelas tersebut.

---

## 4. Data Processing

Fase sterilisasi kualitas data yang krusial sebelum ditelan oleh permesinan optimisasi gradien kalkulus.

### 4.1 Pisahkan Fitur dan Label

```python
X = data.drop(columns='label').copy()
y = data['label']
```

### 4.2 Penanganan Outlier dengan Metode Kuartil (IQR)

```python
num_cols = X.columns

for col in num_cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Capping nilai outlier membatasi ekstremitas
    X[col] = np.clip(X[col], lower_bound, upper_bound)
```

**Matematika Capping Outlier (Winsorizing):**
Nilai-nilai ekstrim secara brutal menggeser posisi *Mean* (rata-rata) pada saat standardisasi. Alih-alih memenggal dan membuang observasi $X > Upper Bound$ yang dapat merusak keseimbangan ukuran jumlah sampel $N$, operasi pembatasan aljabar `np.clip` memancangkan pagar limit paksa, membuat sampel penyimpangan maksimum tertahan tepat setara di angka batas $Upper Bound$.

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

### 4.3 Standardisasi Fitur (StandardScaler)

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

**Matematika Transformasi Z-Score:**
Berbeda dari *MinMaxScaler* yang "memenjarakan" rentang nilai dalam interval absolut $[0, 1]$, *StandardScaler* mentransformasi bentuk sebaran kurva populasi data itu sendiri menjadi sebuah fungsi Distribusi Normal Standar.
Rumus matematika:
$$z = \frac{X - \mu}{\sigma}$$
Setiap nilai fitur numerik asli ($X$) akan dikurangi oleh rata-rata fiturnya ($\mu$) lalu dibagi dengan Standar Deviasi/Simpangan Bakunya ($\sigma$).
Hasil akhirnya: Vektor kolom tersebut akan memiliki titik berat keseimbangan nilai *Mean* baru persis tepat di titik nol ($\mu = 0$) dengan fluktuasi varians satu satuan mutlak ($\sigma = 1$). Hal ini mengamankan konvergensi jarak spasial bagi model SVM dan KNN.

### 4.4 Label Encoding

```python
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
```

### 4.5 Bagi Data Latih dan Uji (80:20)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=0.2,
    stratify=y
)
```

**Matematika Pemisahan Stratifikasi:**
Parameter `stratify=y` menjamin prinsip *Proportional Representation*. Apabila proporsi data populasi asli untuk keempat buah tropis adalah masing-masing tepat $25\%$, maka subset *Training* (1920 baris) dan subset *Testing* (480 baris) akan diiris dan diaduk sedemikian rupa agar masing-masing memiliki fraksi matematis representasi persis tepat $25\%$ pada kelas komponen pembentuknya secara berimbang murni absolut.

---

## 5. Model Training

```python
knn = KNeighborsClassifier().fit(X_train, y_train)
dt  = DecisionTreeClassifier().fit(X_train, y_train)
rf  = RandomForestClassifier().fit(X_train, y_train)
svm = SVC().fit(X_train, y_train)
nb  = GaussianNB().fit(X_train, y_train)

print("Training selesai untuk 5 model klasifikasi...")
```

**Pilar Matematika Algoritma Gaussian Naive Bayes (Fokus Modul):**
*Naive Bayes* bekerja dengan filosofi murni peninggalan pendeta Thomas Bayes tentang Probabilitas Bersyarat (*Conditional Probability*).
Persamaan matematis mutlak Teorema Bayes dalam ranah klasifikasi probabilitas fitur (X) terhadap kelas (y):
$$P(y \mid X) = \frac{P(X \mid y) \cdot P(y)}{P(X)}$$
- $P(y \mid X)$ adalah *Posterior Probability*: Seberapa besar peluang buah tersebut adalah Mangga jika diketahui nilai $N, P, K$ tertentu.
- $P(X \mid y)$ adalah *Likelihood*: Probabilitas ditemukannya kadar rentang $N, P, K$ khusus pada himpunan data yang terbukti berlabel Mangga.
- Mengapa disebut "Naive"? Karena fungsi ini secara polos berasumsi keras (naif) bahwa eksistensi fitur $N$ benar-benar saling terisolasi mutlak (independen) tak ada kaitannya sama sekali dengan eksistensi fitur $P$ dan $K$, suatu kejadian matematis probabilitas yang mustahil sempurna di dunia alam bebas, namun secara praktek performa algoritmanya mengejutkan tangguh dan luar biasa efisien komputasinya.

---

## 6. Model Evaluation & Comparison

```python
def evaluate_model(name, model, X_test, y_test, label_names):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    print(f"==== {name} ====")
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred, average='weighted'):.4f}")
    print(f"F1-Score  : {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_names))

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix — {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
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

---

## 7. Model Comparison

```python
rows = []
for res in results:
    rows.append({
        "Model":     res["Model"],
        "Accuracy":  res["Accuracy"],
        "Precision": res["Precision"],
        "Recall":    res["Recall"],
        "F1-Score":  res["F1-Score"],
    })

summary_df = pd.DataFrame(rows)
print(summary_df.to_string(index=False))
```

---

## 8. Best Model & Error Analysis

Memilih sang pemenang mahkota takhta, dan langsung melangkah mengeksekusi otopsi membedah mikroskopis rentetan celah kegagalan sang model sang juara.

```python
best_model_name = summary_df.loc[summary_df['Accuracy'].idxmax(), 'Model']
print(f"Model terbaik: {best_model_name}")

model_map = {
    'KNN': knn,
    'Decision Tree': dt,
    'Random Forest': rf,
    'SVM': svm,
    'Naive Bayes': nb
}
best_model = model_map[best_model_name]
y_pred_best = best_model.predict(X_test)

cm_best = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Reds',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix — {best_model_name}\n(Model Terbaik)', fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
```

**Penjelasan Dasar Anatomi Prediksi Kegagalan Confusion Matrix:**
Dataset ini sengaja disintesis agar terjadi gesekan tumpang tindih fitur. Di sinilah letak kritikal Confusion Matrix.
Jika Akurasi berhenti di angka rasio empiris ~95%, di manakah tepatnya 5% letak kebodohan algoritma model?
- Titik koordinat diagonal merah gelap (sisi kiri atas ke sisi kanan bawah matriks) adalah wujud murni indikator kemenangan **True Positives (TP)**. Makin besar angkanya, makin akurat tebakan model.
- Elemen di luar lintasan diagonal utama adalah indikator cacat noda kegagalan, wujud inkarnasi pergesekan matriks *False Positives* (Tebakan Halu) dan *False Negatives* (Tebakan Meleset/Luput). Peserta harus mencari letak angka anomali terbesar di luar rel lintasan diagonal. Jika angka kesilapan terbanyak terpusat pada irisan sumbu Aktual Pepaya menyeberang diprediksi sebagai Semangka, kita bisa menarik konklusi deduksi saintifik bahwa proporsi rentang kelembaban dan nutrisi antara Pepaya dan Semangka memiliki kesamaan ekstrem.

---

## 9. Menyimpan Model (Save Model)

```python
import joblib
import os

os.makedirs('../model', exist_ok=True)

joblib.dump(best_model, '../model/best_buah_model.joblib')
joblib.dump(scaler, '../model/buah_scaler.joblib')
joblib.dump(label_encoder, '../model/buah_label_encoder.joblib')

print("Model, Scaler, dan Label Encoder berhasil disimpan ke folder model/")
```

---

## 10. Memuat Model dan Pengujian (Load Model & Testing)

Eksekusi siklus penutupan murni inferensi model untuk menguji mempraktekkan tebakan langsung layaknya model dijalankan pada operasi *Application Programming Interface (API)* di lini belakang ranah perangkat lunak industri sejati.

```python
import joblib

loaded_model = joblib.load('../model/best_buah_model.joblib')
loaded_scaler = joblib.load('../model/buah_scaler.joblib')
loaded_encoder = joblib.load('../model/buah_label_encoder.joblib')

data_baru = pd.DataFrame({
    'N': [20],
    'P': [27],
    'K': [30],
    'temperature': [31.50],
    'humidity': [50.20],
    'ph': [5.80],
    'rainfall': [95.00]
})

# Operasi Konversi Penskalaan Terstandar menggunakan Titik Pusat Historis Murni
data_scaled = loaded_scaler.transform(data_baru)

# Pelaksanaan Sabda Penentuan Kalkulasi Eksekusi Inferensi Model Matematis
prediksi = loaded_model.predict(data_scaled)

# Penafsiran Terjemahan Dekripsi Sandi Mesin Algoritma Merujuk Kembali Ke Wujud Teks Label Manusia
prediksi_label = loaded_encoder.inverse_transform(prediksi)

print(f"Buah yang direkomendasikan: {prediksi_label[0]}")
```

---

## Ringkasan Project

| Tahapan Operasional | Objektif Misi Peran Esensial | Output |
|--------|--------|--------|
| Import Library | Persiapan pengadaan instrumen tools fondasi saintifik ML | Eksekusi kelancaran ketersediaan library tanpa hambatan sintaks error |
| Data Loading | Memompa masuk transfer bongkahan rekam jejak dataset mentah historis 4 faksi kelas | Himpunan baris kolom DataFrame pandas utuh siap diolah |
| EDA (Eksplorasi) | Membongkar pencerahan pandangan mendalam menginvestigasi memahami hakikat sebaran variabilitas | Deretan Galeri Subplot Grid, Anatomi Rangkaian Matriks Panas Rata-rata Lingkungan Iklim |
| Data Processing | Ritual sterilisasi amputasi batas IQR Outlier Capping, standarisasi seragam Z-Score StandardScaler, & konversi dekoding LabelEncoder | X matriks pilar mulus seragam suci, array label vektor y matriks angka integer |
| Data Splitting | Isolasi karantina perpecahan Train/Test split rasio pakem standar (80:20) proporsional seimbang | 4 Kepingan Subset Fragmen Terpecah Belah: X_train, X_test, y_train, y_test |
| Model Training | Eksekusi kawah candradimuka penempaan pelatihan komparasi murni 5 ksatria pilar arsitektur algoritma | Kelahiran penciptaan 5 entitas mesin model siap tempur |
| Evaluation | Eksekusi peluncuran instrumen interogasi pengujian pengukuran penyeimbangan performa | Penjabaran laporan purna bakti Classification Report & hamparan rentang Confusion Matrix |
| Save & Load | Ekspor konservasi pelestarian abadi keabadian warisan simulasi rekayasa deployment sistem perangkat lunak | File `.joblib` suci abadi di ruang kuil perpustakaan `model/` |

---

## Tips Belajar
1. Kewajiban fundamental mutlak, jalankan proses rutinitas eksekusi setiap blok sel baris secara urut tak henti (Cell -> Run All). 
2. Amati dengan panca indra kritis analitis bagaimana dampak distorsi penambahan riak gelombang variansi alami bumbu guncangan pada dataset sintetis membatasi rentang skor akurasi mutlak secara masuk akal empiris, mencetak menciptakan ruang kesalahan *misclassification* murni.
3. Selidiki pantau perhatikan area matriks kekacauan *Confusion Matrix* khusus secara eksklusif absolut hanya untuk membedah memata-matai varietas klaster buah apa yang saling mempertukarkan dan merancukan tebakan.
4. Bereksperimenlah memodifikasi suntikan hyperparameter algoritma Naive Bayes atau KNN dan lihat apakah penyesuaian matematika model sanggup mendobrak menembus batasan dinding 96% akurasi.
5. Manfaatkan berkas abadi `.joblib` di sesi akhir penutupan kelulusan modul untuk mensimulasikan integrasi murni mutlak eksekusi ke ranah operasional server atau aplikasi berbasis Python (*Flask/Streamlit*).
