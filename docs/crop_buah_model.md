# Panduan Lengkap: Klasifikasi Buah Tropis dengan K-Nearest Neighbors (KNN)

Halo! Selamat datang di modul **Klasifikasi Buah Tropis**. Di sini, kita akan belajar Machine Learning dari nol, langkah demi langkah. Anggap saja dokumen ini sebagai "buku catatan" atau "papan tulis" kita selama sesi bootcamp. Bahasanya santai saja, tapi materi di dalamnya 100% menggunakan standar industri profesional.

**Apa tujuan kita di sini?**
Kita akan mengajari komputer (mesin) agar bisa membedakan 4 jenis buah tropis: **Pisang, Mangga, Pepaya, dan Semangka**. Mesin akan menebak buah apa yang cocok ditanam berdasarkan 7 kondisi tanah dan lingkungan (seperti jumlah Nitrogen, suhu, curah hujan, dll).

Karena ini kelas pemula, kita hanya akan fokus pada satu algoritma yang paling intuitif dan mudah dipahami, yaitu **K-Nearest Neighbors (KNN)**.

Yuk, kita mulai perjalanannya!

---

## 1. Import Library: Menyiapkan Alat Tempur

Sebelum membangun rumah, kita butuh alat-alat seperti palu dan gergaji. Di Python, "alat-alat" ini disebut *library*.

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
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, f1_score, recall_score,
    classification_report
)
```

**Penjelasan Santai:**
- **`pandas`**: Ini ibarat Microsoft Excel versi coding. Fungsinya untuk membaca data kita yang berbentuk tabel baris dan kolom.
- **`numpy`**: Alat kalkulator super cepat untuk menghitung rumus-rumus matematika di balik layar.
- **`matplotlib` & `seaborn`**: Kuas dan kanvas kita. Kita pakai ini untuk menggambar grafik berwarna-warni biar datanya enak dilihat (visualisasi).
- **`sklearn` (Scikit-Learn)**: Ini kotak ajaib yang berisi rumus-rumus Machine Learning. Kita ambil model `KNeighborsClassifier` dari sini.
- **`joblib`**: Alat pembeku. Kalau model kita sudah pintar, kita bekukan pakai joblib agar bisa dipakai lagi besok-besok tanpa harus mengajari dari awal.

---

## 2. Kenalan dengan Data (Data Loading)

Tahap pertama adalah memanggil data kita ke dalam program. Datanya berbentuk file Excel/CSV bernama `crop_buah_tropis.csv`.

```python
data = pd.read_csv("../dataset/crop_buah_tropis.csv")
data.head()
```

Dataset ini punya **2.400 baris** (sampel tanah) dan **8 kolom** (7 kolom lingkungan + 1 kolom nama buah). 

Penting untuk mengecek dua hal di awal:
1. **Missing Values**: Ada sel yang kosong tidak? (Kalau kosong, harus kita isi atau buang).
2. **Keseimbangan Kelas**: Apakah jumlah sampel Mangga dan Semangka sama rata? (Penting! Kalau mesin cuma sering lihat Mangga, nanti dia jadi bias dan menebak semuanya Mangga). Di dataset ini, setiap buah punya tepat 600 sampel. Sangat seimbang!

---

## 3. Exploratory Data Analysis (EDA): Melihat "Wajah" Data

EDA itu ibarat detektif yang sedang mengamati tempat kejadian perkara. Kita lihat data dari berbagai sudut pakai grafik. Tujuannya? Biar kita "kenal" sifat asli data kita sebelum disuruh masuk ke mesin ML.

### A. Histogram: Melihat Sebaran Data
Histogram melihat kemana data paling banyak berkumpul. Misalnya fitur `suhu`. Apakah kebanyakan buah suka suhu 25 derajat atau 30 derajat?

```python
# Contoh membuat histogram untuk melihat sebaran fitur 'suhu'
plt.figure(figsize=(8, 5))
sns.histplot(data['suhu'], kde=True, color='blue')
plt.title('Sebaran Data Suhu')
plt.xlabel('Suhu')
plt.ylabel('Frekuensi')
plt.show()
```

### B. Box Plot: Mencari Si "Penyusup" (Outlier)
Box plot adalah kotak yang menunjukkan rentang "wajar" dari sebuah data. Titik-titik yang berada jauh di luar kotak disebut **Outlier** (Pencilan). 
*Kenapa outlier berbahaya?* Bayangkan rata-rata curah hujan adalah 100 mm. Tiba-tiba ada 1 data yang curah hujannya 10.000 mm (mungkin salah ketik saat input data). Kalau data ini dibiarkan, rata-rata curah hujan kita bakal kacau dan mesin bakal kebingungan.

```python
# Contoh membuat boxplot untuk mencari outlier di fitur 'curah_hujan'
plt.figure(figsize=(8, 5))
sns.boxplot(x=data['curah_hujan'], color='orange')
plt.title('Box Plot Curah Hujan')
plt.xlabel('Curah Hujan (mm)')
plt.show()
```

### C. Heatmap Korelasi: Siapa Teman Siapa?
Ini adalah peta warna-warni yang menunjukkan hubungan (korelasi) antar fitur.
- **Nilai mendekati 1 (Merah)**: Hubungan positif kuat. Kalau fitur A naik, fitur B ikut naik.
- **Nilai mendekati -1 (Biru)**: Hubungan negatif kuat. Kalau A naik, B malah turun.
- **Nilai 0**: Nggak ada hubungannya sama sekali.

*Kenapa penting?* Kalau ada dua fitur yang korelasinya merah pekat (mendekati 1), itu artinya mereka memberikan informasi yang sama (mubazir). Kadang kita bisa membuang salah satunya biar mesin tidak kerja terlalu berat.

```python
# Membuat heatmap untuk melihat korelasi antar fitur numerik
plt.figure(figsize=(10, 8))
# Menghapus kolom 'nama_buah' karena teks tidak bisa dihitung korelasinya
korelasi = data.drop('nama_buah', axis=1).corr()
sns.heatmap(korelasi, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Heatmap Korelasi Fitur Lingkungan')
plt.show()
```

---

## 4. Data Processing: Memandikan Data Sebelum Latihan

Data mentah itu ibarat sayuran yang baru dicabut dari tanah, masih banyak lumpurnya. Kita harus bersihkan (preprocessing) dulu sebelum dimasak.

### A. Memisahkan Fitur ($X$) dan Target ($y$)
Kita harus membelah data menjadi dua.
- **$X$ (Fitur/Soal)**: Kondisi N, P, K, suhu, curah hujan, dll.
- **$y$ (Target/Kunci Jawaban)**: Nama buahnya (Mangga, Pepaya, dll).

```python
# X berisi semua kolom sebagai fitur, KECUALI target
X = data.drop('nama_buah', axis=1)

# y hanya berisi kolom target ('nama_buah')
y = data['nama_buah']
```

### B. Menjinakkan Outlier (Metode IQR)
Ingat si penyusup di tahap Box Plot tadi? Kita atasi pakai matematika sederhana: **Interquartile Range (IQR)**.
1. Kita cari batas kuartil bawah ($Q1$) dan batas kuartil atas ($Q3$).
2. Cari jaraknya: $IQR = Q3 - Q1$.
3. Tentukan "Pagar Bawah": $Q1 - (1.5 \times IQR)$
4. Tentukan "Pagar Atas": $Q3 + (1.5 \times IQR)$

Setiap angka curah hujan yang melompat melewati pagar atas, nggak kita buang, tapi kita **paksa turun** (capping) agar nilainya sama persis dengan tinggi pagar atas. Ini menyelamatkan jumlah data kita agar tidak berkurang.

```python
# Contoh menerapkan pembatasan nilai (capping) untuk menangani outlier
for kolom in X.columns:
    Q1 = X[kolom].quantile(0.25)
    Q3 = X[kolom].quantile(0.75)
    IQR = Q3 - Q1
    
    pagar_bawah = Q1 - (1.5 * IQR)
    pagar_atas = Q3 + (1.5 * IQR)
    
    # Pangkas nilai yang melebihi batas
    X[kolom] = np.where(X[kolom] > pagar_atas, pagar_atas, X[kolom])
    X[kolom] = np.where(X[kolom] < pagar_bawah, pagar_bawah, X[kolom])
```

### C. Standardisasi (Z-Score)
Ini bagian yang sangat, sangat penting untuk algoritma KNN.
Bayangkan fitur **pH tanah** angkanya cuma 5 sampai 7. Tapi fitur **Curah Hujan** angkanya bisa 100 sampai 300.
Mesin itu nggak ngerti konteks. Dia cuma lihat angka. Karena 300 jauh lebih besar dari 7, mesin bakal menganggap "Curah Hujan itu fitur yang paling penting, pH tanah abaikan saja!"

Agar mesin adil, semua angka kita kompres ke dalam skala yang sama pakai `StandardScaler`.
Rumus matematikanya (Z-Score):
$$Z = \frac{X - \mu}{\sigma}$$
*Keterangan:*
- $X$ = Angka aslinya
- $\mu$ (mu) = Nilai rata-rata dari seluruh data
- $\sigma$ (sigma) = Standar deviasi (sebaran datanya)

Hasilnya? Semua kolom akan punya rata-rata 0. Adil, kan?

```python
# Menyiapkan 'timbangan' Z-Score
scaler = StandardScaler()

# Mengubah data asli X menjadi data yang skalanya adil (Z-Score)
X_scaled = scaler.fit_transform(X)
```

### D. Label Encoding
Mesin cuma bisa baca angka, nggak bisa baca huruf teks. Jadi "Mangga", "Pisang", "Pepaya" kita ubah jadi angka 0, 1, 2, 3 pakai `LabelEncoder`.

```python
# Menyiapkan 'kamus' untuk mengubah teks menjadi angka
label_encoder = LabelEncoder()

# Mengubah target ('nama_buah') menjadi angka (0, 1, 2, 3)
y_encoded = label_encoder.fit_transform(y)
```

### E. Data Splitting (80:20)
Kita pisah data jadi dua: **Data Latih (80%)** dan **Data Uji (20%)**.
Ibarat anak sekolah, 80% soal dipakai buat latihan di kelas, 20% sisanya disembunyikan sama guru buat soal Ujian Nasional. Tujuannya biar kita tahu apakah mesinnya beneran pintar, atau cuma jago menghafal soal latihan.

```python
# Memecah data dengan proporsi Train 80% dan Test 20%
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
```

---

## 5. Inti Materi: Algoritma K-Nearest Neighbors (KNN)

Sekarang kita masuk ke otaknya! Kita pakai algoritma **KNN**. Bagaimana cara kerja KNN? Sangat simpel.

**Prinsip Kerja KNN:** "Katakan padaku siapa teman-teman terdekatmu, maka aku akan tahu siapa dirimu."

Bayangkan ada titik misterius baru masuk (data ujian). KNN akan melihat tetangga-tetangga terdekatnya. Kalau $K = 5$, mesin akan mencari 5 titik data latihan yang jaraknya paling dekat dengan si titik misterius ini.
Kalau dari 5 titik itu, 3 titik adalah "Mangga" dan 2 titik adalah "Pepaya", maka secara voting mayoritas, titik misterius itu akan ditebak sebagai **"Mangga"**.

### Perhitungan Matematikanya (Jarak Euclidean)
Bagaimana mesin tahu mana yang "dekat"? Dia menghitung jarak lurus (seperti ditarik pakai penggaris) menggunakan rumus *Euclidean Distance*, yang sebenarnya adalah pengembangan dari rumus Pythagoras peninggalan zaman Yunani kuno!

Rumusnya:
$$d(p, q) = \sqrt{(p_1 - q_1)^2 + (p_2 - q_2)^2 + \dots + (p_n - q_n)^2}$$
*(Intinya, jarak adalah akar dari total selisih angka setiap fitur yang dikuadratkan).*

Inilah kenapa tahap **Standardisasi (Z-Score)** di atas sangat wajib. Kalau angkanya nggak disamakan skalanya, perhitungan jarak Pythagoras ini bakal hancur lebur ditarik oleh fitur yang angkanya ribuan.

```python
# Cukup 2 baris untuk melatih model!
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
```

---

## 6. Ujian Nasional: Evaluasi Performa Model

Mesin sudah belajar, sekarang kita tes pakai Data Uji (20% tadi).

```python
# Meminta model menebak jawaban dari soal Ujian Nasional (X_test)
y_pred = knn.predict(X_test)
```

### A. Mengenal Confusion Matrix
Confusion Matrix itu rapornya si mesin. Matriks ini blak-blakan ngasih tau di mana mesin ini pinter, dan di mana dia oon (suka tertukar).

Ada 4 komponen dasar dari tebakan mesin:
1. **True Positive (TP)**: Mesin nebak "Mangga", dan aslinya emang "Mangga". (Benar!).
2. **True Negative (TN)**: Mesin nebak "Ini bukan Mangga", dan aslinya emang bukan Mangga. (Benar!).
3. **False Positive (FP)**: Mesin teriak "Ini Mangga!", padahal aslinya Semangka. (Alarm palsu / Sok tahu).
4. **False Negative (FN)**: Aslinya Mangga, tapi mesin diam aja dan bilang "Bukan Mangga". (Kebobolan / Kurang peka).

Di grafik Heatmap Confusion Matrix, lihatlah kotak-kotak yang berbaris menyilang (diagonal). Itu adalah total tebakan yang **Benar (TP)**. Makin gede angkanya, makin bagus. Kotak-kotak di luar garis itu adalah kesalahan tebakan. Dari situ kita bisa tahu, "Wah, modelnya suka kebingungan bedain Pepaya sama Semangka nih!"

```python
# Membuat Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Menampilkan dengan visualisasi yang cantik (Heatmap)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.xlabel('Tebakan Mesin (Predicted)')
plt.ylabel('Kenyataan Asli (Actual)')
plt.title('Confusion Matrix Tebakan KNN')
plt.show()
```

### B. Metrik Evaluasi: Rapor Angka
1. **Accuracy (Akurasi)**: Dari 100 soal ujian, berapa yang dijawab benar? 
   $$\frac{TP + TN}{TP + TN + FP + FN}$$
2. **Precision (Ketelitian)**: Kalau mesin ngotot bilang ini "Mangga", seberapa bisa kita percaya omongannya? 
   $$\frac{TP}{TP + FP}$$
3. **Recall (Daya Ingat)**: Dari semua Mangga yang ada di dunia nyata, berapa banyak yang berhasil dikenali sama mesin (nggak ada yang lolos)?
   $$\frac{TP}{TP + FN}$$
4. **F1-Score**: Nilai keseimbangan (rata-rata harmonik) antara Precision dan Recall. Kalau salah satu jelek, F1-Score bakal anjlok buat ngasih hukuman.

```python
# Melihat hasil rapor angka dengan metrik klasifikasi
akurasi = accuracy_score(y_test, y_pred)
print(f"Akurasi   : {akurasi * 100:.2f}%")

# Menampilkan laporan komprehensif (Precision, Recall, F1-Score tiap buah)
print("\nLaporan Lengkap (Classification Report):\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
```

---

## 7. Save & Load Model: Bawa Pulang Ilmu!

Bayangkan butuh waktu 3 hari buat melatih AI. Masa tiap kali kita matikan laptop, kita harus melatih ulang dari awal? Kan capek!

Oleh karena itu, otak AI yang sudah cerdas ini kita "bekukan" ke dalam sebuah file berekstensi `.joblib`.
Proses ini disebut *Serialization*.

```python
import joblib

# Menyimpan otak KNN
joblib.dump(knn, '../model/best_buah_model.joblib')

# JANGAN LUPA simpan juga Scaler dan Encoder-nya!
joblib.dump(scaler, '../model/buah_scaler.joblib')
joblib.dump(label_encoder, '../model/buah_label_encoder.joblib')
```

**Kenapa Scaler dan Encoder wajib ikut disimpan?**
Karena otak model ini sudah terbiasa dengan "makanan" berupa Z-Score. Kalau besok-besok ada orang masukin data tanah dari kebun asli (angka ratusan), mesinnya bakal kaget dan salah tebak. Data asli itu harus di-*transform* dulu pakai timbangan `Scaler` yang sama persis seperti saat dia latihan, baru disuapin ke otak `KNN`.

---

### Selesai!
Sekarang Anda sudah paham bukan bagaimana K-Nearest Neighbors bekerja dari hulu ke hilir. Silakan kembali ke [Jupyter Notebook `crop_buah_model.ipynb`](../notebooks/crop_buah_model.ipynb) dan mainkan sendiri kodenya (klik **Run All**). 

Jangan takut untuk bereksperimen, misalnya: Coba ubah nilai parameter $K$ di KNN menjadi 3 atau 10 (`KNeighborsClassifier(n_neighbors=3)`), lalu lihat apakah rapornya jadi makin bagus atau malah hancur? Selamat mencoba!
