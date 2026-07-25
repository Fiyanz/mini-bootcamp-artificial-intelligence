# Model Klasifikasi Bunga Iris - Iris Classification

**Tujuan Pembelajaran:** 
Modul ini adalah gerbang pertama untuk memasuki dunia Machine Learning. Melalui dataset Iris, peserta akan belajar konsep dasar klasifikasi. Fokus utamanya adalah memahami fondasi matematika dan logika dari berbagai algoritma algoritma (K-Nearest Neighbors, Decision Tree, Random Forest, SVM, dan Logistic Regression). Peserta juga akan mempelajari mengapa normalisasi data itu penting dan bagaimana mengukur kinerja model secara sistematis.

**Dataset:** Iris Dataset (UCI Machine Learning Repository) (`Iris.csv`)

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

Modul Python esensial untuk manipulasi data, visualisasi, dan pemodelan matematis.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, f1_score, recall_score,
    classification_report
)
```

**Penjelasan Konsep Dasar:**
- `pandas`: Library utama untuk membaca data (seperti file CSV) ke dalam bentuk tabel yang disebut **DataFrame**. Bayangkan DataFrame seperti lembar kerja Excel di dalam Python.
- `numpy`: Library untuk komputasi numerik. Dalam Machine Learning, data diubah menjadi bentuk matriks matematika. Numpy memungkinkan operasi matriks ini berjalan dengan sangat cepat.
- `sklearn` (Scikit-Learn): Ini adalah "kotak perkakas" utama Machine Learning. Semua rumus matematika rumit dari algoritma klasifikasi sudah diprogram di sini.

---

## 2. Data Loading & Initial Exploration

Langkah pertama adalah memuat data dan melihat "bentuk fisik" data tersebut.

### Load Dataset

```python
data = pd.read_csv("../dataset/Iris.csv")
data.head()
```

### Pembersihan dan Inspeksi Kolom

```python
data = data.drop(columns='Id')
data.rename(columns={'Species': 'label'}, inplace=True)

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

**Konsep Matematika & Data:**
- **Dimensi Matriks (Shape):** Dataset ini adalah matriks berukuran $150 \times 5$. Artinya terdapat 150 baris observasi (vektor baris) dan 5 kolom fitur (vektor kolom).
- **Kolom 'Id':** Kolom ini dihapus karena secara matematis, nomor urut tidak memiliki korelasi linear maupun non-linear dengan bentuk fisik bunga. Jika dimasukkan, model bisa menjadi bias.
- **Keseimbangan Kelas:** Terdapat 3 kelas (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`) dan masing-masing memiliki 50 sampel. Dalam matematika statistik probabilitas, peluang prior ($P(C_i)$) masing-masing kelas adalah seragam, yaitu $\frac{50}{150} = 0.333$. Ini berarti dataset bersifat **balanced** (seimbang).

---

## 3. Exploratory Data Analysis (EDA)

EDA digunakan untuk melihat sebaran data secara visual. Otak manusia lebih cepat memahami grafik dibandingkan kumpulan angka dalam matriks.

### 3.1 Pairplot (Scatter Matrix)

```python
sns.pairplot(data, hue='label', palette='Set2', diag_kind='kde',
             plot_kws={'alpha': 0.7, 's': 40, 'edgecolor': 'white'})
plt.suptitle('Pairplot - Iris Dataset', y=1.02, fontsize=16)
plt.show()
```

**Penjelasan Dasar:**
Pairplot adalah matriks plot dari seluruh kemungkinan kombinasi dua fitur. Tujuannya adalah untuk mencari garis pemisah linear (linear separability). Jika Anda bisa menarik garis lurus secara visual untuk memisahkan satu spesies dari spesies lain, maka model Machine Learning juga akan sangat mudah mempelajarinya.

### 3.2 Violin Plot per Fitur

```python
num_features = data.select_dtypes(include=[np.number])
num_cols = num_features.columns

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.violinplot(x='label', y=col, data=data, palette='Set2',
                   inner='box', ax=axes[i])
    axes[i].set_title(f'Distribusi {col} per Spesies', fontsize=12)
    axes[i].set_xlabel('')

plt.suptitle('Violin Plot - Distribusi Fitur per Spesies', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
```

**Konsep Matematika Distribusi Densitas:**
Lebar dari biola (violin) pada sumbu horizontal mewakili estimasi kepadatan probabilitas (Kernel Density Estimation / KDE). Semakin lebar area pada sumbu Y tertentu, semakin tinggi frekuensi kumulatif atau peluang matematis ditemukannya ukuran kelopak di nilai tersebut.

### 3.3 Swarm Plot (Deteksi Outlier & Sebaran Individual)

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.swarmplot(x='label', y=col, data=data, palette='Set2',
                  size=5, ax=axes[i])
    axes[i].set_title(f'Swarm Plot - {col}', fontsize=12)
    axes[i].set_xlabel('')

plt.suptitle('Swarm Plot - Sebaran Individual per Spesies', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
```

### 3.4 Correlation Matrix

```python
plt.figure(figsize=[8, 6])
correlation_matrix = num_features.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
            fmt='.2f', linewidths=0.5)
plt.title('Matriks Korelasi - Iris Dataset', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

**Matematika Koefisien Korelasi Pearson:**
Nilai korelasi $r$ antara variabel $X$ dan $Y$ dihitung dengan rumus turunan kovarians:
$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
- Nilai $r = +1$ berarti pergerakan garis lurus sempurna searah.
- Nilai $r = -1$ berarti pergerakan garis lurus sempurna berlawanan arah.
- Nilai $r = 0$ berarti tidak ada hubungan garis lurus antar dua variabel.

### 3.5 Scatter Plot - Fitur Petal (Panjang vs Lebar)

```python
plt.figure(figsize=(10, 7))
for species in data['label'].unique():
    subset = data[data['label'] == species]
    plt.scatter(subset['PetalLengthCm'], subset['PetalWidthCm'],
                label=species, alpha=0.7, s=60, edgecolors='white')

plt.xlabel('Petal Length (cm)', fontsize=12)
plt.ylabel('Petal Width (cm)', fontsize=12)
plt.title('Scatter Plot - Petal Length vs Petal Width', fontsize=14, fontweight='bold')
plt.legend(title='Spesies', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 4. Data Processing

### 4.1 Pisahkan Fitur dan Target

```python
X = data.drop(columns='label').copy()
y = data['label']
```

Secara matematika, $X$ adalah matriks berdimensi $m \times n$ (150 sampel, 4 dimensi fitur), sedangkan $y$ adalah vektor label berdimensi $m \times 1$.

### 4.2 Penanganan Outlier dengan Metode Kuartil (IQR)

```python
num_cols = X.columns

for col in num_cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Capping nilai outlier
    X[col] = np.clip(X[col], lower_bound, upper_bound)
```

**Matematika Kuartil:**
- $Q_1$ adalah titik data di posisi ke-25%.
- $Q_3$ adalah titik data di posisi ke-75%.
- $IQR$ (Interquartile Range) adalah jarak di antara keduanya ($Q_3 - Q_1$).
Data yang melampaui $1.5 \times IQR$ dari kuartil luar secara statistik dianggap terlalu jauh dari pusat massa data.

### 4.3 Normalisasi Fitur (MinMaxScaler)

```python
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
```

**Konsep Matematika MinMaxScaler:**
Algoritma seperti KNN dan SVM mengukur jarak geometris (misalnya Euclidean distance). Jika fitur `A` berkisar dari 0 - 1000 dan fitur `B` berkisar dari 0 - 1, maka jarak akan didominasi sepenuhnya oleh fitur `A`.
Untuk mencegahnya, kita mengompres seluruh rentang nilai fitur menjadi berada pada skala sempit $0$ hingga $1$.

Rumus Normalisasi Mutlak:
$$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$
- Jika nilai $X$ adalah nilai terendah ($X_{min}$), maka pembilang bernilai 0, sehingga hasil akhir $0$.
- Jika nilai $X$ adalah nilai tertinggi ($X_{max}$), maka hasil akhir $\frac{X_{max} - X_{min}}{X_{max} - X_{min}} = 1$.

### 4.4 Label Encoding

```python
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
```

**Konsep:** Algoritma hanya memahami angka. String teks "Iris-setosa" dikonversi secara matematis menjadi skalar integer `0`, "Iris-versicolor" menjadi `1`, dan "Iris-virginica" menjadi `2`.

---

## 5. Data Splitting (70:30)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=0.3,
    stratify=y
)

print(f"Train set shape: X_train = {X_train.shape}, y_train = {y_train.shape}")
print(f"Test set shape:  X_test  = {X_test.shape}, y_test  = {y_test.shape}")
```

**Konsep Validasi Data:**
Kita memotong 30% matriks observasi ($0.30 \times 150 = 45$ baris data) dan mengisolasinya menjadi himpunan matriks `X_test`. Mesin tidak boleh melihat data ini selama proses belajar (training), agar kita bisa menguji apakah mesin tersebut benar-benar cerdas menyimpulkan pola, atau hanya menghafal kunci jawaban `X_train`.

---

## 6. Model Training (Penjelasan Matematika Algoritma)

Kita akan melatih lima algoritma cerdas yang berbeda secara fundamental:

```python
knn = KNeighborsClassifier().fit(X_train, y_train)
dt  = DecisionTreeClassifier().fit(X_train, y_train)
rf  = RandomForestClassifier().fit(X_train, y_train)
svm = SVC().fit(X_train, y_train)
lr  = LogisticRegression(max_iter=200).fit(X_train, y_train)

print("Training selesai untuk 5 model...")
```

**Konsep Matematika Algoritma:**

1. **K-Nearest Neighbors (KNN)**:
   KNN beroperasi sepenuhnya pada bidang geometri spasial. Untuk setiap data pengujian, jarak geometris diukur ke seluruh titik data latih menggunakan formula **Jarak Euclidean**:
   $$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$
   Jika nilai $K=5$, algoritma mengambil 5 data latih dengan nilai $d(x, y)$ terkecil. Label prediksi adalah hasil voting terbanyak dari 5 tetangga tersebut.

2. **Logistic Regression (LR)**:
   Ini adalah perluasan dari regresi linear (persamaan garis lurus $z = w_1 x_1 + w_2 x_2 + b$). Karena ini masalah klasifikasi (bukan tebak angka), persamaan linear disuntikkan ke dalam fungsi aktivasi logistik **Sigmoid Curve**, yang membatasi nilai tebakan keluarannya menjadi rentang probabilitas $0$ hingga $1$.
   Rumus Fungsi Sigmoid:
   $$\sigma(z) = \frac{1}{1 + e^{-z}}$$

3. **Support Vector Machine (SVM)**:
   SVM adalah pembedah matriks yang tangguh. Algoritma mencari persamaan sebuah bidang batas pemisah (disebut *Hyperplane*: $w \cdot x + b = 0$) yang membentang tepat di tengah kelompok kelas. SVM menggunakan optimisasi matematika murni mencari Margin terbesar (jarak terjauh) antara hiperbidang tersebut dengan sampel terluar masing-masing kelas (yang disebut *Support Vectors*).

4. **Decision Tree (Pohon Keputusan)**:
   Murni bekerja berdasarkan teori informasi berantai hierarkis bersyarat IF-ELSE. Ia mencari fitur pemisah yang dapat menurunkan nilai ketidakmurnian populasi kelompok (*Impurity*) sekecil mungkin pada setiap percabangan rantingnya.
   Rumus Gini Impurity (Ketidakmurnian Kelas):
   $$Gini = 1 - \sum_{i=1}^{c} (P_i)^2$$
   (Dimana $P_i$ adalah probabilitas titik data acak jatuh ke dalam kelas tertentu).

5. **Random Forest**:
   Menerapkan paradigma probabilitas matematis dari *Teorema The Wisdom of Crowds*. Algoritma mensimulasikan ribuan sub-sampel acak untuk merakit seratus model Decision Tree secara paralel, lalu menjumlahkan suara akhir mereka untuk prediksi yang jauh lebih kuat dan anti-bias.

---

## 7. Model Evaluation - Classification Report & Confusion Matrix

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
    evaluate_model("KNN",                 knn, X_test, y_test, label_names),
    evaluate_model("Decision Tree",       dt,  X_test, y_test, label_names),
    evaluate_model("Random Forest",       rf,  X_test, y_test, label_names),
    evaluate_model("SVM",                 svm, X_test, y_test, label_names),
    evaluate_model("Logistic Regression", lr,  X_test, y_test, label_names),
]
```

### Konsep Matematika Metrik Evaluasi

| Nama Metrik | Perhitungan Matematika | Tujuan Pemahaman |
|---|---|---|
| **Accuracy (Akurasi)** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Mengukur dari total seluruh ujian (pembagi di bawah), seberapa banyak soal yang dijawab dengan benar mutlak (pembilang di atas). |
| **Precision (Presisi)** | $\frac{TP}{TP + FP}$ | Presisi menjawab pertanyaan: "Dari semua bunga yang diklaim mesin sebagai kelas A, seberapa banyak persen rasio bunga yang *benar-benar* A?". Sangat kritis jika tebakan salah menyebabkan kerugian finansial. |
| **Recall (Sensitivitas)** | $\frac{TP}{TP + FN}$ | Recall menjawab pertanyaan kebalikan: "Dari total seluruh populasi bunga A yang sungguhan ada, berapa persen yang berhasil dipindai dan dikenali mesin tanpa terlewatkan?". |
| **F1-Score** | $2 \times \frac{Precision \times Recall}{Precision + Recall}$ | Mean harmonik dari metrik Precision dan Recall. Jika salah satu nilai memburuk hancur mendekati 0, rumusan perkalian penyebut akan membanting tajam nilai rata-rata keseluruhan sebagai bentuk penalti adil. |

---

## 8. Model Comparison (KP, CM, TP, FP, FN, TN)

```python
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

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
    'Logistic Regression (LR)':  evaluate_model(lr,  X_test, y_test)
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

**Matematika Komposisi Confusion Matrix Lanjut (Multi-Kelas):**
Pada studi kasus ini kita memiliki matriks $3 \times 3$.
- `cm.diagonal()` mengekstrak vektor blok kebenaran $(c_{0,0}, c_{1,1}, c_{2,2})$ di mana indeks tebakan sumbu X sama dengan indeks realitas sumbu Y.
- **TP (True Positive)** adalah penjumlahan metrik kebenaran seluruh sumbu elemen silang diagonal.
- `cm.sum(axis=0)` merangkum semua deretan vertikal agregat kolom tebakan model (termasuk FP).
- `cm.sum(axis=1)` menghimpun kumulasi total seluruh barisan lintasan rentang horizontal data biologis absolut sebenarnya di lapangan (termasuk FN).

---

## 9. Visualizing Model Performance

### 9.1 Bar Chart - Perbandingan Akurasi

```python
plt.figure(figsize=(10, 5))
bars = plt.bar(summary_df['Model'], summary_df['Accuracy'],
               color=['#2196F3','#4CAF50','#FF9800','#E91E63','#9C27B0'])
plt.title('Perbandingan Akurasi Model', fontsize=14, fontweight='bold')
plt.ylabel('Accuracy')
plt.ylim(0.8, 1.02)
plt.xticks(rotation=15, ha='right')

for bar, val in zip(bars, summary_df['Accuracy']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.2%}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()
```

### 9.2 Heatmap - Semua Metrik

```python
metrics_df = summary_df.set_index('Model')[['Accuracy','Precision','Recall','F1-Score']]

plt.figure(figsize=(8, 4))
sns.heatmap(metrics_df, annot=True, fmt='.2%', cmap='YlGn',
            vmin=0.8, vmax=1.0, linewidths=0.5)
plt.title('Perbandingan Semua Metrik', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## 10. Best Model & Error Analysis

```python
best_model_name = summary_df.loc[summary_df['Accuracy'].idxmax(), 'Model']
print(f"Model terbaik: {best_model_name}")

model_map = {
    'K-Nearest Neighbors (KNN)': knn,
    'Decision Tree (DT)': dt,
    'Random Forest (RF)': rf,
    'Support Vector Machine (SVM)': svm,
    'Logistic Regression (LR)': lr
}
best_model = model_map[best_model_name]
y_pred_best = best_model.predict(X_test)

cm_best = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(7, 5))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Reds',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - {best_model_name}\n(Model Terbaik)', fontweight='bold')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()
```

**Analisis Ilmiah Otopsi Iris:**
Jika ditelusuri dari Pairplot EDA di atas, letak spasial penyebaran `Iris-versicolor` saling memotong merapat tumpang tindih tipis berhimpitan dan bertabrakan dengan ras `Iris-virginica`. Otomatis, secara probabilitas murni, satu-satunya kebingungan dan misklasifikasi (error matrix) mesin yang niscaya seringkali meletup muncul selalu berpusar berada di persimpangan celah dua spesies rumit ini.

---

## 11. Menyimpan Model (Save Model)

```python
import joblib
import os

os.makedirs('../model', exist_ok=True)

joblib.dump(best_model, '../model/best_iris_model.joblib')
joblib.dump(scaler, '../model/iris_scaler.joblib')
joblib.dump(label_encoder, '../model/iris_label_encoder.joblib')

print("Model, Scaler, dan Label Encoder berhasil disimpan ke folder model/")
```

**Landasan Proses Serialisasi Byte:**
Fungsi `joblib` mencetak seluruh pohon hierarkis struktur memori Python, termasuk angka bobot gradien regresi, bobot koefisien SVM margin batas, rentang parameter MinMaxScaler $X_{min}$ dan $X_{max}$ murni terdalam, memuatnya mengonversinya melilitnya dalam serangkaian rantai heksadesimal kode biner berantai yang diletakkan padat dalam ruang media simpan memori flashdisk atau hard drive. 

---

## 12. Memuat Model dan Pengujian (Load Model & Testing)

```python
import joblib

loaded_model = joblib.load('../model/best_iris_model.joblib')
loaded_scaler = joblib.load('../model/iris_scaler.joblib')
loaded_encoder = joblib.load('../model/iris_label_encoder.joblib')

# Inisialisasi simulasi data baru murni dalam ranah operasi inferensi matematis mutlak
data_baru = pd.DataFrame({
    'SepalLengthCm': [5.1],
    'SepalWidthCm': [3.5],
    'PetalLengthCm': [1.4],
    'PetalWidthCm': [0.2]
})

# Peringatan Krusial Matematika: Eksekusi wajib menggunakan mutlak instruksi fungsi transform()!
data_scaled = loaded_scaler.transform(data_baru)

# Menunaikan eksekusi perintah perhitungan probabilistik dan proyeksi geometri 
prediksi = loaded_model.predict(data_scaled)

# Dekripsi dari bilangan matriks skalar numerik kelas kembali merujuk ke rupa pilar untaian kata kalimat string alfabet spesies
prediksi_label = loaded_encoder.inverse_transform(prediksi)

print(f"Spesies Iris yang diprediksi: {prediksi_label[0]}")
```

**Peringatan Kritis Matematika Eksekusi:**
- Instruksi pemanggilan skrip `loaded_scaler.transform()` mutlak disyaratkan alih-alih `fit_transform()`. Pasalnya jika `fit_transform()` secara tak sengaja dikerahkan di ranah ini, fungsi itu dengan ceroboh buta mutlak akan merancang memetakan ulang titik referensi gravitasi batas rentang normalitas absolut interval dimensi interval data $X_{min}$ hingga $X_{max}$ yang sepenuhnya hanya berlandaskan pada observasi sebongkah titik semata (yaitu ukuran kelopak tunggal $5.1 cm$), menindih nilai kalibrasi dasar ratusan riwayat pilar titik terdahulu, mengakibatkan perhitungan ekuasi probabilitas prediksi yang gagal parah dan sangat cacat logika hitungannya.
- Modul ini ditutup mengakhiri mata rantai penuh kehidupan sebuah rancang bangun sistem mesin kecerdasan prediksi matematis terintegrasi paripurna dari baris hulu ke batas muara deployment secara murni analitik.
