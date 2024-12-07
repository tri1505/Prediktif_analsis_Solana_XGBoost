# Laporan Proyek Machine Learning - Tri Ramdhany
## Domain Proyek
### Latar Belakang
Pasar cryptocurrency telah mengalami pertumbuhan pesat dalam beberapa tahun terakhir, dengan semakin banyaknya jenis aset digital yang diperkenalkan dan diperdagangkan. Salah satu aset yang menonjol adalah Solana (SOL), sebuah cryptocurrency yang dikenal dengan kemampuan skalabilitas tinggi dan kecepatan transaksi yang unggul. Sebagai salah satu dari 10 besar cryptocurrency berdasarkan kapitalisasi pasar, Solana menarik perhatian baik dari investor ritel maupun institusional.

Namun, volatilitas tinggi yang menjadi ciri khas cryptocurrency, termasuk Solana, membuat proses pengambilan keputusan investasi menjadi tantangan. Fluktuasi harga yang signifikan sering kali dipengaruhi oleh berbagai faktor, seperti sentimen pasar, adopsi teknologi, kebijakan pemerintah, dan dinamika pasar global. Dalam konteks ini, kemampuan untuk memprediksi harga Solana secara akurat menjadi sangat berharga bagi investor, pedagang, dan analis pasar.

Teknologi kecerdasan buatan dan pembelajaran mesin (machine learning) telah terbukti menjadi alat yang efektif dalam menganalisis data kompleks dan memprediksi tren masa depan. Dengan memanfaatkan algoritma prediktif, model dapat dilatih untuk mengidentifikasi pola-pola yang tersembunyi dalam data historis dan memberikan estimasi harga di masa depan. Pendekatan ini tidak hanya membantu mengurangi risiko dalam perdagangan, tetapi juga memberikan wawasan berharga tentang faktor-faktor utama yang memengaruhi pergerakan harga Solana.

Proyek ini bertujuan untuk mengembangkan model prediktif harga Solana menggunakan data historis pasar, termasuk harga, volume perdagangan, serta indikator teknis dan fundamental. Hasil penelitian ini diharapkan dapat memberikan kontribusi signifikan dalam pemahaman dan pengelolaan risiko dalam investasi cryptocurrency, khususnya Solana, di tengah dinamika pasar yang terus berkembang.

## Business Understanding
Investor cryptocurrency, khususnya Solana, adalah individu yang berusaha mendapatkan keuntungan dari fluktuasi harga aset digital. Dalam dunia investasi, kemampuan untuk menganalisis data historis seperti harga pembukaan, harga penutupan, harga tertinggi, harga terendah merupakan keahlian penting. Melalui analisis ini, investor berupaya memprediksi pergerakan harga Solana di masa depan berdasarkan pola historis yang ada.

Solana terkenal dengan volatilitasnya yang tinggi, di mana harga dapat berfluktuasi secara signifikan dalam waktu singkat. Prediksi yang terlalu optimis dapat menyebabkan kerugian besar, sementara sikap terlalu konservatif bisa mengakibatkan kehilangan peluang investasi yang berpotensi menguntungkan. Oleh karena itu, diperlukan sebuah metode yang mampu mengelola ketidakpastian ini dengan lebih akurat.

Salah satu solusi yang dapat diterapkan adalah dengan memanfaatkan teknik machine learning. Dengan memodelkan data historis, teknik ini dapat membantu investor mendapatkan prediksi yang lebih akurat mengenai pergerakan harga Solana. Prediksi yang akurat tidak hanya membantu investor meminimalkan risiko kerugian, tetapi juga memaksimalkan peluang keuntungan di pasar yang dinamis dan sulit diprediksi ini.

### Problem Statement
Berdasarkan kondisi yang telah diuraikan sebelumnya, proyek ini akan mengembangkan sebuah sistem prediksi harga Solana berdasarkan data historis yaitu harga tertinggi, harga terendah, harga pembukaan, harga penutupan dimana masing masing faktor memiliki peran penting dalam pergerakan harga Solana seperti:
- Harga Tertinggi: Mencerminkan titik harga tertinggi yang dicapai Solana dalam periode tertentu. Faktor ini penting karena menunjukkan batas atas kekuatan beli di pasar dan sering kali menjadi patokan untuk menentukan apakah harga sedang mendekati titik resistensi.
- Harga Terendah: Menunjukkan titik terendah harga Solana dalam periode tertentu. Ini membantu mengidentifikasi titik dukungan dimana tekanan jual mungkin telah memudar, memberikan gambaran mengenai batas bawah dari volatilitas pasar.
- Harga Pembukaan: Adalah harga awal dari Solana di awal periode perdagangan. Perbandingan antara harga pembukaan dan harga penutupan bisa memberikan indikasi tentang tren pasar yang sedang terjadi, apakah tren naik (bullish) atau turun (bearish).
- Harga Penutupan: Harga pada akhir periode perdagangan merupakan salah satu indikator kunci yang sering digunakan untuk melihat kecenderungan harga secara keseluruhan. Perubahan harga penutupan dari waktu ke waktu membantu dalam memahami pola tren harga di masa depan.

Dengan menggunakan teknologi machine learning algoritma XGBoost dan hyperparameter tuning diharapkan dapat menjawab permasalahan berikut: 
- Bagaimana cara memanfaatkan data historis harga Solana seperti harga pembukaan, harga penutupan, harga tertinggi, harga terendah untuk memprediksi harganya di masa depan?
- Bagaimana cara mengoptimalkan kinerja algoritma XGBoost untuk memprediksi harga Solana melalui parameter-parameter penting seperti learning rate, max_depth, subsample, dan n_estimators?

### Goals
Untuk menjawab problem statement tersebut, akan dibuat predictive modelling dengan tujuan atau goals sebagai berikut:
- Membuat model machine learning yang dapat memprediksi harga Solana untuk 10 hari ke depan berdasarkan parameter yang ditetapkan.
- Mencari nilai optimal untuk learning rate, max_depth, subsample, dan n_estimators pada algoritma XGBoost melalui proses hyperparameter tuning, dengan tujuan memaksimalkan akurasi prediksi harga Solana.

### Solution Statement
Untuk mencapai goals tersebut, ada 2 pendekatan yang akan digunakan yaitu:
- Menggunakan Algoritma XGBoost: XGBoost merupakan algoritma ensemble yang sangat cocok untuk tugas prediksi, terutama pada data yang kompleks dan nonlinear seperti data harga Solana. XGBoost bekerja dengan membangun banyak pohon keputusan (decision tree) secara berurutan, dengan setiap pohon belajar dari kesalahan pohon sebelumnya. Struktur ensemble ini memungkinkan XGBoost menangkap pola yang kompleks dalam data dan menghasilkan prediksi yang lebih akurat.
- Menentukan Nilai Optimal untuk Hyperparameter XGBoost: Untuk mengoptimalkan kinerja model XGBoost, kita akan melakukan hyperparameter tuning. Hyperparameter adalah parameter yang tidak dipelajari oleh model selama pelatihan, melainkan diatur sebelum pelatihan dimulai. Beberapa hyperparameter penting pada XGBoost antara lain:
  - learning_rate: Mengontrol tingkat di mana model belajar dari setiap iterasi. Nilai yang terlalu besar dapat menyebabkan overfitting, sedangkan nilai yang terlalu kecil dapat memperlambat konvergensi.
  - max_depth: Mengontrol kedalaman maksimum pohon keputusan. Nilai yang terlalu besar dapat menyebabkan overfitting, sedangkan nilai yang terlalu kecil dapat menghambat kemampuan model untuk menangkap pola yang kompleks.
  - subsample: Mengontrol proporsi data yang digunakan untuk membangun setiap pohon keputusan. Subsampling dapat membantu mengurangi overfitting.
  - n_estimators: Menentukan jumlah pohon keputusan yang akan dibangun. Jumlah pohon yang terlalu sedikit dapat menyebabkan underfitting, sedangkan jumlah pohon yang terlalu banyak dapat menyebabkan overfitting.

## Data Understanding
Data yang akan digunakan pada proyek kali ini adalah [Solana Historical Data](https://www.investing.com/indices/investing.com-sol-usd-historical-data) yang diunduh dariInvesting.com 

Dataset ini memiliki rentang waktu dari 13 Juli 2020 sampai 7 Desember 2024, Nantinya dari dataset tersebut model dapat belajar dari berbagai kondisi pasar dan menangkap perubahan pola yang terjadi seiring waktu. Hal ini memungkinkan model untuk memberikan prediksi yang lebih akurat dan adaptif terhadap dinamika pasar kripto yang kompleks.

Dataset berisi 1602 records dan 7 kolom yang memiliki karakteristik sebagai berikut:

- Date: Tanggal data harga.
- High: Harga tertinggi yang dicapai Solana pada hari tersebut.
- Low: Harga terendah yang dicapai Solana pada hari tersebut.
- Open: Harga pembukaan Solana pada hari tersebut.
- Price: Harga penutupan Solana pada hari tersebut.
- Vol.: Volume perdagangan Solana pada hari tersebut.
- Change%: perubahan persentase dalam harga Solana (SOL) dalam periode waktu tertentu.

Untuk memahami data, selanjutnya akan dilakukan proses berikut:
### 1. Data Loading
Supaya isi dataset lebih mudah dipahami, kita perlu melakukan proses loading data terlebih dahulu dengan import library pandas untuk dapat membaca file datanya.
### 2. Exploratory Data Analysis
#### Informasi Dataset
Mengecek informasi pada dataset dengan fungsi info() berikut.
<br>

<!-- ![image](https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/df_info.jpg) -->

<img src="https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/df_info.jpg" alt="image" width="300"/>

<br>

Berdasarkan informasi di atas dataset pertama memiliki beberapa kriteria antara lain:
- 4 Kolom dengan tipe float64 yaitu High, Low, Open, Price
- 3 Kolom dengan tipe object yaitu VOl., Change%

<br>



#### Cek Missing Value
Jika data terdiri dari ratusan bahkan ribuan baris tentu akan susah dalam menemukan nilai field yang kosong. Oleh karena itu, Pandas memungkinkan kita dapat menemukan missing value secara cepat dengan fungsi isna() dan sum().
<br>

<p align="left">
  <img src="https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/miss_na.jpg" alt="dataset 1" width="100"/>
</p>

<!-- ![image](https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/miss_na.jpg) -->

Pada dataset ditemukan missing value sebanyak 398 pada kolom Vol.


#### Analisis Tren Harga
Pada bagian ini, kita akan menggali tren harga Solana dengan tujuan untuk memahami pola pergerakannya dari waktu ke waktu. Dengan menganalisis data historis, kita dapat mengidentifikasi faktor-faktor yang mempengaruhi fluktuasi harga dan memberikan konteks yang lebih mendalam untuk prediksi yang dihasilkan oleh model.
<!-- ![image](https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/pergerakan_harga.jpg) -->

<img src="https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/pergerakan_harga.jpg" alt="image" width="680"/>

Dari visualisasi di atas nampak beberapa informasi di antaranya:
- Tren Keseluruhan: Terlihat adanya tren peningkatan harga Solana secara umum selama periode tersebut.
- Volatilitas: Terdapat periode fluktuasi harga yang signifikan, menunjukkan volatilitas yang tinggi. Misalnya, peningkatan dan penurunan harga yang tajam dapat diamati pada rentang waktu tertentu, seperti di seperti di tahun 2022-07 sampai 2023-07 dan terlihat pula mengalami ETH di tahun 2024-01.
- Hubungan antar harga: Harga pembukaan, penutupan, tertinggi, dan terendah cenderung bergerak bersama-sama, yang mengindikasikan adanya korelasi antara berbagai aspek aktivitas harga harian Solana.

#### Rata-Rata Pergerakan Harga
Rata-rata Pergerakan Sederhana (Simple Moving Average - SMA) adalah metode perataan data harga dalam periode waktu tertentu untuk membantu mengidentifikasi tren dalam harga aset, seperti Solana.

Perhitungan SMA dilakukan dengan cara mengambil rata-rata harga penutupan aset selama sejumlah hari tertentu, kemudian menggeser (rolling) jendela waktu tersebut setiap harinya untuk mendapatkan nilai SMA terbaru. Dengan cara ini, fluktuasi harga harian dapat dihaluskan sehingga tren harga lebih mudah dianalisis.

```math
\text{SMA}n = \frac{C_t + C{t-1} + C_{t-2} + \dots + C_{t-n+1}}{n} 
```

Di mana:
```math
\begin{aligned}
\text{SMA}_n & = \text{Simple Moving Average untuk periode } n \text{ hari.} \\[10pt] 
C_t & = \text{Harga penutupan pada hari ke-} t.\\[10pt] 
n & = \text{Jumlah hari dalam periode perhitungan SMA.}
\end{aligned}
```

<br>

Untuk SMA jangka pendek - menengah akan digunakan periode 50 hari, sedangkan untuk jangka panjang akan digunakan periode 200 hari.

***Dengan Alasan Sebagai Berikut***

- SMA-50 sering digunakan untuk mengukur tren jangka pendek hingga menengah. Periode 50 hari dianggap cukup untuk menunjukkan fluktuasi harga terbaru tanpa terlalu banyak "noise" dari pergerakan harga harian
- SMA-200 adalah indikator yang lebih umum digunakan untuk melihat tren jangka panjang. Periode ini cukup panjang untuk memberikan gambaran stabil tentang pergerakan harga dan menghilangkan fluktuasi harian yang tidak signifikan

<!-- ![image](https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/harga_SMA.jpg4) -->
<br>

<img src="https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/harga_SMA.jpg" alt="image" width="680"/>

Dari visualasi di atas, didapatkan informasi sebagai berikut:

SMA-200 bertindak sebagai support (level harga di mana tren penurunan cenderung berhenti) dalam jangka panjang, sedangkan SMA-50 bertindak sebagai support dan resistance (level harga di mana tren kenaikan cenderung berhenti) dalam jangka pendek dan menengah.

- Golden Cross: Ketika SMA-50 memotong SMA-200 dari bawah ke atas, ini menandakan potensi dimulainya tren kenaikan harga (bullish)
- Death Cross: Ketika SMA-50 memotong SMA-200 dari atas ke bawah, ini menandakan potensi dimulainya tren penurunan harga (bearish)

## Data Preparation
Pada bagian ini akan dilakukan beberapa persiapan data yaitu:

### 1. Membagi Feature dan Target serta Shifting Data
Dalam rangka memprediksi harga Solana untuk 10 hari ke depan, langkah pertama adalah membagi feature dan target. Numerical feature yang digunakan terdiri dari kolom high, low, open, Price. Target merupakan prediksi harga penutupan (closing price) Solana pada hari ke-10 setelah data pada hari yang sedang diproses. Untuk menetapkan kolom target ini, digunakan teknik shifting, di mana data harga penutupan bergeser sebanyak 10 hari ke depan dengan kode berikut:
```python
df['Prediction_5D'] = df['Price'].shift(-10)
```

Setelah itu, fitur numerik perlu dimundurkan 10 hari agar selaras dengan target prediksi. Hal ini dilakukan untuk memastikan bahwa model menggunakan data dari 5 hari sebelumnya sebagai dasar prediksi harga di masa depan. Kode yang digunakan untuk proses shifting adalah sebagai berikut:
```python
df['High_shifted'] = df['High'].shift(10)
df['Low_shifted'] = df['Low'].shift(10)
df['Open_shifted'] = df['Open'].shift(10)
df['Price_shifted'] = df['Price'].shift(10)
```

Pemunduran fitur ini penting untuk mencegah kebocoran data, di mana model dapat "melihat" informasi dari masa depan yang seharusnya belum tersedia. Pendekatan ini sesuai dengan prinsip dalam analisis time series, di mana variabel input (fitur) harus mencerminkan periode waktu sebelum variabel target. Setelah dilakukan pemunduran data dan menghapus nilai null, kini terdapat total 1582 baris data yang siap untuk digunakan dalam model.

### 2. Split Dataset
Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model. proporsi pembagian data latih dan uji adalah 80:20. Proporsi tersebut cukup ideal untuk model dengan jumlah data 1582. Namun, jika memiliki dataset berukuran besar, kita perlu memikirkan strategi pembagian dataset lain agar proporsi data uji tidak terlalu banyak.  Pembagian ini menggunakan fungsi train_test_split dari sklearn hasil yang diperoleh berikut:

<!-- ![image](https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/data_train.jpg) -->

<img src="https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/data_train.jpg" alt="image" width="260"/>

## Modeling
Pada tahap ini menggunakan algoritma machine learning XGBoost dengan menerapkan hyperparameter tuning untuk mencari nilai learning rate, max_depth, subsample, dan n_estimators terbaik. Model yang sudah dilatih akan dievaluasi dengan metrik MAE dan MSE, penjelasan lebih detail terkait metrik evaluasi akan dibahas saat evaluasi model.

### 1. Konsep Dasar Algoritma XGBoost
XGBoost merupakan algoritma ensemble yang sangat cocok untuk tugas prediksi, terutama pada data yang kompleks dan nonlinear seperti data harga Solana. XGBoost membangun banyak pohon keputusan (decision tree) secara berurutan, di mana setiap pohon belajar dari kesalahan pohon sebelumnya. Dengan adanya regularisasi, algoritma ini tidak hanya berusaha untuk mengurangi kesalahan prediksi, tetapi juga untuk menghindari overfitting. Struktur ensemble ini memungkinkan XGBoost menangkap pola yang kompleks dalam data dan menghasilkan prediksi yang lebih akurat.

### 2. Melatih Model Baseline
Pada tahap ini, model XGBoost dilatih tanpa melakukan penyesuaian parameter, menggunakan konfigurasi default yang disediakan oleh library. Tujuan dari langkah ini adalah untuk mendapatkan baseline performance yang akan menjadi acuan bagi evaluasi model selanjutnya. Model ini dilatih menggunakan fitur-fitur yang telah disiapkan sebelumnya, termasuk nilai-nilai high, low, open, Price, yang telah dimundurkan 10 hari.

Dari pelatihan model baseline, didapatka hasil sebagai berikut:
<!-- ![image](https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/before_tuning.jpg) -->

<img src="https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/before_tuning.jpg" alt="image" width="300"/>


Dari informasi di atas, didapatkan Mean Squared Error (MSE) sebesar 702.99 dan Mean Absolute Error (MAE) sebesar 16.10, yang menggambarkan tingkat kesalahan prediksi dalam model ini. Dengan MAE sebesar 16.10, ini berarti rata-rata prediksi harga Solana meleset sekitar 16.10 USD dari harga sebenarnya. Mengingat volatilitas harga Solana yang tinggi.

Namun, MSE yang lebih tinggi menunjukkan adanya outlier atau prediksi dengan kesalahan yang lebih besar. Untuk meningkatkan akurasi model dan mengurangi kesalahan prediksi, langkah selanjutnya adalah melakukan hyperparameter tuning.

### 3. Hyperparameter Tuning
Hyperparameter tuning adalah sebuah proses untuk melakukan optimalisasi parameter pada sebuah model. Dalam KNN, terdapat beberapa parameter yang menjadi pembangun model.
Untuk proyek ini Parameter yang di gunakan ada 4 yaitu:
- learning_rate: Mengontrol tingkat di mana model belajar dari setiap iterasi. Nilai yang terlalu besar dapat menyebabkan overfitting, sedangkan nilai yang terlalu kecil dapat memperlambat konvergensi.
- max_depth: Mengontrol kedalaman maksimum pohon keputusan. Nilai yang terlalu besar dapat menyebabkan overfitting, sedangkan nilai yang terlalu kecil dapat menghambat kemampuan model untuk menangkap pola yang kompleks.
- subsample: Mengontrol proporsi data yang digunakan untuk membangun setiap pohon keputusan. Subsampling dapat membantu mengurangi overfitting.
- n_estimators: Menentukan jumlah pohon keputusan yang akan dibangun. Jumlah pohon yang terlalu sedikit dapat menyebabkan underfitting, sedangkan jumlah pohon yang terlalu banyak dapat menyebabkan overfitting.

Selanjutnya ditentukkan kandidat untuk memilih parameter terbaik dengan ketentuan berikut:
- learning_rate: 0.01, 0.1, 0.2
- max_depth: 3, 5, 7
- n_estimators: 100, 200, 300
- subsample: 0.8, 1.0

Kemudian dengan menggunakan GridSearchCV Scikit-Learn untuk mencari parameter yang dilakukan secara brute force dan melaporkan mana parameter yang memiliki akurasi paling baik. Setelah dilakukan proses pencarian parameter yang optimal menggunakan GridSearch diperoleh parameter nilai learning_rate = 0.01, max_depth = 3, n_estimators = 300, dan subsample = 0.8 yang akan digunakan untuk melakukan fit model yang diperoleh hasil berikut :
<!-- ![image](https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/after_tuning.jpg) -->

<img src="https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/after_tuning.jpg" alt="image" width="350"/>

Nilai akurasi model meningkat setelah diterapkan hyperparameter tuning dengan perolehan nilai Mean Squared Error: 478.63 dan Mean Absolute Error: 14.11. Tentunya performa model lebih baik jika dibandingkan dengan akurasi sebelum dilakukan tuning.

## Evaluation
Seperti yang telah dijelaskan sebelumnya, metrik evaluasi yang digunakan adalah Mean Absolute Error (MAE) dan Mean Square Error (MSE).

MAE dan MSE adalah dua metrik yang umum digunakan untuk mengukur kinerja model regresi, termasuk model XGBoost. Keduanya mengukur seberapa jauh prediksi model dari nilai sebenarnya, namun dengan cara yang sedikit berbeda.

- Mean Absolute Error (MAE)
  
  MAE adalah metrik yang mengukur rata-rata selisih absolut antara nilai prediksi model dengan nilai aktual. Dengan kata lain, MAE menghitung rata-rata dari nilai absolut perbedaan antara nilai yang diprediksi oleh model dengan nilai sebenarnya. Nilai MAE yang lebih kecil menunjukkan bahwa model semakin akurat dalam membuat prediksi.
  
  MAE memiliki keunggulan karena lebih robust terhadap outlier dibandingkan dengan MSE. Artinya, nilai outlier tidak akan terlalu memengaruhi nilai MAE secara signifikan.
  
  MAE sering digunakan ketika kita ingin mendapatkan gambaran umum tentang seberapa besar kesalahan model secara rata-rata.
  
    ```math
    \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
    ```
  
    Di mana:
    ```math
    \begin{aligned}
    \text{MAE} & = \text{Mean Absolute Error} \\
    n & = \text{Jumlah data observasi} \\
    y_i & = \text{Nilai aktual ke-} i \\
    \hat{y}_i & = \text{Nilai prediksi ke-} i
    \end{aligned}
    ```

- Mean Squared Error (MSE)

  MSE adalah metrik lain yang populer untuk mengukur kinerja model regresi. MSE menghitung rata-rata kuadrat dari selisih antara nilai prediksi dan nilai aktual.
  
  Dengan mengkuadratkan selisih, MSE memberikan bobot yang lebih besar pada kesalahan yang besar. Ini berarti bahwa model akan lebih "dihukum" jika membuat prediksi yang jauh dari nilai sebenarnya.
  
  MSE sering digunakan ketika kita ingin memberikan penalti yang lebih besar pada kesalahan yang besar, karena kesalahan yang besar dapat memiliki konsekuensi yang lebih signifikan.
  ```math
  \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  ```
  
  Di mana:
  ```math
  \begin{aligned}
  \text{MSE} & = \text{Mean Squared Error} \\
  n & = \text{Jumlah data observasi} \\
  y_i & = \text{Nilai aktual ke-} i \\
  \hat{y}_i & = \text{Nilai prediksi ke-} i
  \end{aligned}
  ```

### Harga Aktual vs Prediksi
Dalam analisis ini, kita akan membandingkan harga aktual Solana dengan harga yang diprediksi oleh model XGBoost melalui visualisasi scatterplot. Visualisasi ini tidak hanya memungkinkan kita untuk melihat sejauh mana prediksi model sejalan dengan data aktual, tetapi juga membantu kita mengidentifikasi pola dan outlier yang mungkin ada.

<!-- ![image](https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/aktual_prediksi.jpg) -->

<img src="https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/aktual_prediksi.jpg" alt="image" width="680"/>

Dari visualiasi di atas didapatkan informasi sebagai berikut:
- Meskipun ada beberapa titik data yang melenceng dari garis ideal, tetapi secara keseluruhan sebaran titik-titik pada grafik harga aktual vs harga prediksi menunjukkan bahwa prediksi model XGBoost cukup akurat dan mengikuti tren harga aktual dengan baik.
- Namun, perlu diingat bahwa model tetap memiliki keterbatasan dan prediksi harga Solana di dunia nyata dipengaruhi oleh banyak faktor kompleks yang mungkin tidak sepenuhnya tercakup dalam model, seperti sentimen pasar, regulasi pemerintah, dan berita terkait Solana.

### Prediksi Harga 10 Hari ke Depan
Selanjutnya, akan diprediksi harga Solana untuk lima hari ke depan, suatu hal yang menjadi salah satu tujuan utama dalam pembuatan model ini. Karena baris data terakhir berada di tanggal 19 Oktober 2024, maka hasil prediksi akan berada di tanggal 20-24 Oktober 2024.
<!-- ![image](https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/10_hari.jpg) -->

<img src="https://github.com/tri1505/Prediktif_analsis_Solana_XGBoost/blob/main/10_hari.jpg" alt="image" width="300"/>

Berdasarkan model XGBoost yang telah dituning, prediksi harga Solana untuk periode 8 hingga 17 Desember 2024 menunjukkan fluktuasi harga yang bervariasi. dilihat dari harga prediksi tersebut harga solana masih dalam posisi sideways karena tidak ada lonjakan harga yang tinggi maupun rendah, pada fase ini koin solana ini di dukung naratif yang positif yaitu solana bisa menjadi alternatif network pengganti ETH yang lebih cepat dengan gas fee yang lebih murah maka posisi sideways ini bisa digunakan para investor untuk melakukan akumulasi dan Menunggu momentum untuk harga naik

Secara keseluruhan, hasil prediksi ini memberikan insight penting bagi investor dan trader untuk merencanakan strategi perdagangan mereka berdasarkan proyeksi harga Solana dalam jangka pendek.

## References
1. Sihombing S, Rizky Nasution M, Sadalia I. Analisis Fundamental Cryptocurrency terhadap Fluktuasi Harga: Studi Kasus Tahun 2019-2020. Jurnal Akuntansi, Keuangan, dan Manajemen. 2021 Jun 20;2(3):213–24.


