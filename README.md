# Laporan Proyek Machine Learning – Intan Charolina

## 1. Domain Proyek
### Latar Belakang:
Dalam beberapa tahun terakhir, platform berbagi properti seperti Airbnb telah mengalami pertumbuhan yang pesat, terutama di kota-kota besar seperti New York City. Salah satu tantangan utama yang dihadapi oleh tuan rumah (host) di Airbnb adalah menentukan harga optimal untuk properti mereka. Harga yang terlalu tinggi dapat menyebabkan berkurangnya penyewaan, sementara harga yang terlalu rendah dapat mengurangi potensi pendapatan. Situasi ini diperparah oleh faktor-faktor eksternal seperti musim, lokasi, fasilitas properti, dan permintaan pasar.

Model predictive analytics dapat digunakan untuk memprediksi harga optimal berdasarkan data historis dan fitur-fitur terkait. Dengan prediksi yang akurat, host dapat menyesuaikan harga properti mereka secara dinamis untuk memaksimalkan pendapatan sambil tetap kompetitif di pasar.

#### Mengapa Masalah Ini Harus Diselesaikan:
Mengoptimalkan harga properti di platform Airbnb sangat penting bagi tuan rumah untuk mencapai keseimbangan antara tingkat hunian yang tinggi dan pendapatan maksimal. Keputusan penetapan harga yang tidak tepat dapat menyebabkan hilangnya peluang pendapatan atau bahkan kerugian finansial.

#### Hasil Riset Terkait:
Referensi:
Kalehbasti, P. R., Nikolenko, L., & Rezae, H. (2021). *Airbnb price prediction using machine learning and sentiment analysis*. In A. Holzinger et al. Vol. LNCS 12844, pp. 173–184.

## 2. Business Understanding
### Problem Statements:
- **Pernyataan Masalah 1**: 
  Bagaimana menentukan harga optimal untuk properti Airbnb di New York City berdasarkan data historis dan fitur properti?
- **Pernyataan Masalah 2**: 
  Bagaimana pengaruh fitur-fitur seperti lokasi, jumlah kamar, dan fasilitas terhadap harga properti?

### Goals:
- Mengembangkan model machine learning yang dapat memprediksi harga properti berdasarkan data historis.
- Mengidentifikasi fitur-fitur yang paling berpengaruh dalam penetapan harga properti Airbnb.

### Solution Statements:
- Menggunakan tiga algoritma berbeda: K-Nearest Neighbor (KNN), Random Forest, dan AdaBoost untuk memprediksi harga properti.
- Melakukan hyperparameter tuning (GridSearch) pada model yang terpilih untuk meningkatkan akurasi prediksi.

## 3. Data Understanding
### Informasi Data:
Dataset yang digunakan dalam proyek ini adalah "New York City Airbnb Open Data" yang diunduh dari Kaggle. Dataset ini mencakup informasi mengenai listing properti di Airbnb di New York City, termasuk harga, lokasi, jumlah kamar, dan fasilitas.

#### Informasi Mengenai Dataset:
- **Nama Dataset**: New York City Airbnb Open Data
- **Sumber**: [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
- **Ukuran Dataset**: 49.5 MB (CSV format)
- **Jumlah Baris (Data Points)**: 48,895 baris
- **Jumlah Kolom (Fitur)**: 16 kolom

#### Variabel dalam Dataset:
- `id`: Identifikasi unik untuk setiap listing.
- `name`: Nama dari listing yang diposting oleh pemilik.
- `host_id`: Identifikasi unik untuk setiap pemilik (host).
- `host_name`: Nama dari pemilik listing.
- `neighbourhood_group`: Wilayah besar atau borough di New York City (misalnya: Manhattan, Brooklyn).
- `neighbourhood`: Nama spesifik dari lingkungan atau area.
- `latitude`: Latitude dari lokasi listing.
- `longitude`: Longitude dari lokasi listing.
- `room_type`: Jenis ruangan yang disewakan (misalnya: entire home/apt, private room).
- `price`: Harga per malam untuk listing (dalam USD).
- `minimum_nights`: Jumlah minimum malam yang harus dipesan oleh tamu.
- `number_of_reviews`: Jumlah ulasan yang diterima oleh listing.
- `last_review`: Tanggal ulasan terakhir yang diterima oleh listing.
- `reviews_per_month`: Rata-rata ulasan per bulan.
- `calculated_host_listings_count`: Jumlah listing aktif yang dimiliki oleh host.
- `availability_365`: Jumlah hari listing tersedia dalam setahun.

#### Kondisi Data:
- **Missing Values**: Ada beberapa kolom dengan nilai yang hilang, seperti `last_review` dan `reviews_per_month`.
- **Outliers**: Terdapat kemungkinan adanya outliers terutama dalam kolom `price` dan `minimum_nights`.
- **Data Format**: Sebagian besar kolom berisi data kategorikal atau numerik sederhana, namun beberapa kolom seperti `last_review` berisi data tanggal yang memerlukan pemrosesan tambahan (Transformasi Fitur).

#### Visualisasi Data:
Sebagai bagian dari tahap exploratory data analysis (EDA), dilakukan visualisasi data untuk memahami distribusi harga, jumlah ulasan, dan keterkaitan antara fitur-fitur lainnya.

### Penjelasan Proses EDA (Explanatory Data Analysis):
#### a. Univariate Analysis:
- **Categorical Fitur**:
  - Visualisasi Distribusi `neighbourhood_group`:
  ![ss6](https://raw.githubusercontent.com/intancharolina079/Latproyeks1/main/Screenshot%202024-08-14%20113302.png)
    - Proses: Membuat plot untuk melihat bagaimana data didistribusikan di berbagai `neighbourhood_group`.
    - Alasan: Untuk memahami sebaran properti di berbagai wilayah utama New York City.
    - Hasil: Memberikan gambaran mengenai wilayah mana yang paling banyak atau paling sedikit digunakan dalam dataset.    

  - Visualisasi Distribusi `room_type`:
    ![ss5](https://raw.githubusercontent.com/intancharolina079/Latproyeks1/main/Screenshot%202024-08-14%20113417.png)
    - Proses: Membuat plot untuk melihat distribusi berbagai tipe kamar (`room_type`).
    - Alasan: Untuk mengetahui preferensi tipe kamar yang ditawarkan di platform Airbnb.
    - Hasil: Memahami dominasi tipe kamar tertentu dalam dataset, yang bisa mempengaruhi harga.

- **Numerical Fitur**:
  - Visualisasi Distribusi `price`:
    ![ss3](https://raw.githubusercontent.com/intancharolina079/Latproyeks1/main/Screenshot%202024-08-14%20113433.png)
    - Proses: Membuat histogram atau boxplot untuk melihat distribusi harga (`price`).
    - Alasan: Untuk memahami rentang harga dan adanya outliers pada fitur `price`.
    - Hasil: Memberikan informasi mengenai sebaran harga, apakah tersebar normal atau ada anomali.

  - Visualisasi Distribusi `minimum_nights`:
    ![ss2](https://raw.githubusercontent.com/intancharolina079/Latproyeks1/main/Screenshot%202024-08-14%20113453.png)
    - Proses: Membuat histogram untuk melihat sebaran `minimum_nights`.
    - Alasan: Untuk mengetahui seberapa banyak malam minimum yang biasanya disyaratkan oleh host.
    - Hasil: Memahami pola umum dalam persyaratan minimum nights, yang mungkin berdampak pada harga.

  - Visualisasi Distribusi `number_of_reviews`:
    ![ss1](https://raw.githubusercontent.com/intancharolina079/Latproyeks1/main/Screenshot%202024-08-14%20113500.png)
    - Proses: Membuat histogram untuk melihat distribusi `number_of_reviews`.
    - Alasan: Untuk mengetahui pola ulasan yang diterima properti.
    - Hasil: Memberikan insight mengenai popularitas properti berdasarkan jumlah ulasan.

  - Visualisasi Distribusi `reviews_per_month`:
    ![ss](https://raw.githubusercontent.com/intancharolina079/Latproyeks1/main/Screenshot%202024-08-14%20113508.png)
    - Proses: Membuat histogram untuk melihat sebaran `reviews_per_month`.
    - Alasan: Untuk memahami frekuensi ulasan yang diterima setiap bulannya.
    - Hasil: Mengetahui bagaimana keterlibatan pengguna dan popularitas properti dari waktu ke waktu.

#### b. Multivariate Analysis:
- **Categorical Fitur**:
  ![Screenshot](https://raw.githubusercontent.com/intancharolina079/Latproyeks1/main/Screenshot%202024-08-14%20172032.png)
  - Analisis Hubungan antara `neighbourhood_group` dan `room_type`:
    - Proses: Membuat crosstab atau plot untuk melihat hubungan antara wilayah (`neighbourhood_group`) dan tipe kamar (`room_type`).
    - Alasan: Untuk memahami apakah ada preferensi tipe kamar yang dominan di wilayah tertentu.
    - Hasil: Memberikan informasi mengenai kombinasi wilayah dan tipe kamar yang paling populer, yang bisa mempengaruhi harga.

- **Numerical Fitur**:
  ![ss8](https://raw.githubusercontent.com/intancharolina079/Latproyeks1/main/111111111111111111111.png)
  - Pairplot untuk beberapa fitur (`price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`):
    - Proses: Membuat pairplot untuk melihat hubungan antara beberapa fitur numerik.
    - Alasan: Untuk memahami korelasi dan pola antara berbagai fitur numerik.
    - Hasil: Mengetahui fitur mana yang berkorelasi kuat atau lemah dengan fitur lainnya, yang penting untuk modeling.

  - Correlation Matrix untuk Fitur Numerical:
    ![ss7](https://raw.githubusercontent.com/intancharolina079/Latproyeks1/main/222222222222.png)
    - Proses: Membuat matriks korelasi untuk mengevaluasi hubungan antar fitur numerik.
    - Alasan: Untuk memahami seberapa kuat hubungan antar fitur, yang bisa membantu dalam seleksi fitur.
    - Hasil: Menentukan fitur mana yang memiliki hubungan linier kuat, yang bisa berdampak pada performa model.

### 4. Data Preparation
#### Teknik Data Preparation:
1. **Encoding Fitur**: One-hot encoding dilakukan untuk mengubah fitur kategorikal seperti `neighbourhood_group`, `neighbourhood`, dan `room_type` menjadi format numerik yang dapat digunakan oleh algoritma machine learning. Ini penting karena sebagian besar algoritma tidak dapat bekerja dengan data kategorikal secara langsung.

2. **Splitting**: Melakukan pembagian data menjadi 90:10, yang mana 90% untuk train dan 10% untuk test. Dataset ini tergolong dalam kategori dataset yang berukuran besar, maka pembagian menggunakan perbandingan 90:10.

3. **Standarisasi**: Membantu membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Dalam Standarisasi menggunakan model `StandardScaler` yang ditemukan dari library `sklearn`. Untuk menghindari kebocoran informasi pada data uji, kita hanya akan menerapkan fitur standarisasi pada data latih. Kemudian, pada tahap evaluasi, kita akan melakukan standarisasi pada data uji.

### Standarisasi
```
scaler = StandardScaler()
numerical_features = X_train.select_dtypes(include=['float64', 'int64']).columns
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_train[numerical_features].head()
```
### Evaluation Model
```
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
```
4. **Penanganan Missing Value**: Kolom `reviews_per_month` diisi dengan nilai 0 karena properti tanpa ulasan dianggap tidak memiliki ulasan per bulan. Untuk `name` dan `host_name` mengandung beberapa missing values yang tidak signifikan, namun informasi ini dianggap tidak esensial untuk analisis prediktif.
5. **Penanganan Outliers**: Outliers diidentifikasi menggunakan visualisasi boxplot dan distribusi histogram. Data yang jauh dari nilai rata-rata atau berada di luar boxplot dianggap sebagai outliers. Outliers ini dapat memengaruhi kinerja model prediktif, terutama dalam algoritma yang sensitif terhadap nilai ekstrem, sehingga perlu ditangani.
6. **Transformasi Fitur**: Beberapa fitur mungkin mengalami transformasi untuk memperbaiki distribusi atau untuk membuatnya lebih sesuai dengan asumsi model yang akan digunakan.

#### Alasan Dilakukannya Data Preparation:
   Proses data preparation diperlukan untuk memastikan bahwa model machine learning yang dibangun dapat bekerja dengan baik tanpa adanya bias.

## 5. Modeling
Informasi fitur yang dibutuhkan model untuk membuat prediksi:
- Semua fitur yang berada pada numerik yang ada di dalam `numerical_features`, bersama dengan fitur kategorikal yang sudah di-encode yaitu `neighbourhood_group`, `neighbourhood`, dan `room_type` digunakan sebagai input untuk model KNN. 

```
X = df_encoded.drop('price', axis=1)
```
Artinya, semua kolom selain kolom price digunakan sebagai input (fitur) untuk model. Jadi, kolom-kolom seperti `latitude`, `longitude`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`, `last_review_year`, dan `last_review_month`, serta hasil encoding dari `neighbourhood_group`, `neighbourhood`, dan `room_type` digunakan sebagai fitur dalam model.

```
y = df_encoded['price']
```
Artinya, kolom price adalah target yang akan diprediksi oleh model.

### Modeling dalam proyek ini menggunakan tiga algoritma yang berbeda:
1. **K-Nearest Neighbor (KNN)**:
Algoritma ini dipilih karena kemampuannya dalam menangani data non-linear dan memberikan prediksi yang baik pada dataset kecil hingga menengah.
   - **Proses Kerja KNN pada Data**: 
     - Menghitung jarak antara satu listing properti dengan listing lainnya berdasarkan fitur numerik dan fitur kategorikal yang sudah di-encode.
     - Menentukan sejumlah k tetangga terdekat (dalam parameter k yang ditentukan) yang memiliki harga yang sudah diketahui.
     - Mengambil rata-rata harga dari k tetangga terdekat ini dan menggunakannya sebagai prediksi harga untuk listing properti baru.
   - **Parameters**:
     - n_neighbors: Jumlah tetangga terdekat yang digunakan untuk prediksi `[5, 10, 15, 20].`
     - weights: Metode pembobotan jarak, apakah akan menggunakan uniform (setiap tetangga memiliki bobot yang sama) atau distance (tetangga yang lebih dekat memiliki bobot lebih besar) `['uniform', 'distance']`
     - p: Menentukan jenis jarak yang digunakan, `p=1` untuk jarak Manhattan dan `p=2` untuk jarak Euclidean.
   - **GridSearchCV**:
   Digunakan untuk menemukan kombinasi parameter terbaik berdasarkan Mean Squared Error (MSE) yang negatif sebagai metrik evaluasi. Setelah melakukan pencarian, model terbaik disimpan sebagai best_knn

2. **Random Forest**:
Algoritma ensemble yang terdiri dari banyak pohon keputusan, mampu menangani data dengan kompleksitas tinggi dan variabel penting.
   - **Proses Kerja Random Forest pada Data**: Ketika model membuat prediksi, hasil dari setiap pohon dikombinasikan melalui rata-rata. Dengan demikian, prediksi harga dari Random Forest adalah agregasi dari banyak pohon keputusan yang meminimalkan varians dan risiko overfitting.
   - **Parameters**:
     - n_estimators: Jumlah pohon yang digunakan dalam ensemble `[100, 200, 300]`.
     - max_depth: Kedalaman maksimum setiap pohon, yang mempengaruhi seberapa detail pohon tersebut dalam membagi data `[10, 20, 30]`.
     - min_samples_split: Jumlah minimum sampel yang diperlukan untuk membagi node.
     - min_samples_leaf: Jumlah minimum sampel yang harus dimiliki oleh leaf node.
     - max_features: Jumlah maksimum fitur yang dipertimbangkan untuk split di setiap node.
     - bootstrap: Menentukan apakah sampel bootstrap akan digunakan saat membangun pohon.
     
   - **GridSearchCV**:
     Digunakan untuk mengoptimalkan parameter dan memilih model terbaik (best_rf).

3. **AdaBoost**:
Algoritma boosting yang bertujuan untuk meningkatkan kinerja model dengan fokus pada kesalahan yang dibuat oleh model sebelumnya.
   - **Proses Kerja AdaBoost pada Data**: Model pertama yang sangat sederhana (seperti pohon keputusan tunggal atau stump) dilatih pada dataset. Pada iterasi berikutnya, model baru dilatih dengan menitikberatkan pada kesalahan dari model sebelumnya. Proses ini terus berulang, dengan setiap model berikutnya semakin fokus pada data yang sulit diprediksi oleh model sebelumnya.
   - **Parameters**:
     - n_estimators: Jumlah estimator yang digunakan dalam boosting `[50, 100, 150]`.
     - learning_rate: Mengontrol kontribusi setiap estimator terhadap model akhir `[0.01, 0.1, 1.0]`.
   - **GridSearchCV**:
     Digunakan untuk menemukan parameter optimal dan menentukan model terbaik (best_boost).

   ### Kelebihan dan Kekurangan:
   * KNN: Mudah diimplementasikan namun memiliki kekurangan dalam hal efisiensi komputasi pada dataset besar.
   * Random Forest: Memberikan hasil prediksi yang lebih stabil namun memerlukan tuning yang baik untuk mencegah overfitting.
   * AdaBoost: Dapat meningkatkan kinerja model yang lemah namun rentan terhadap noise dalam data.

   ### Improvement:
   Dilakukan hyperparameter tuning pada model KNN, Random Forest, dan AdaBoost untuk meningkatkan hasil prediksi.

## 6. Evaluation
Metrik evaluasi yang digunakan untuk mengukur kinerja model adalah Mean Squared Error (MSE). MSE adalah metrik yang menghitung rata-rata dari kuadrat selisih antara nilai prediksi dan nilai aktual. MSE digunakan untuk mengukur seberapa baik model regresi dalam memprediksi nilai numerik, di mana semakin kecil nilai MSE, semakin baik model dalam memprediksi data.

<img src="https://raw.githubusercontent.com/intancharolina079/Latproyeks1/main/Screenshot%202024-08-14%20173253.png" alt="Deskripsi Gambar" width="300"/>

Keterangan:
- N = jumlah dataset
- yi = nilai sebenarnya
- y_pred = nilai prediksi

### Hasil Evaluasi:
Pada evaluasi model ini, tiga algoritma yaitu KNN, Random Forest (RF), dan Boosting digunakan untuk memprediksi data. Hasil perhitungan MSE pada data train dan test untuk masing-masing algoritma adalah sebagai berikut:

| Model | MSE (Training) | MSE (Testing) | 
|-------|----------------|---------------|
| KNN   | 0.000          | 2.341269      |
| RF    | 0..67473       | 1.942955      | 
| Ada   | 2.226237       | 2.240535      |

#### Analisis Hasil:
   * KNN: Model KNN memiliki MSE yang sangat rendah pada data pelatihan, yang berarti model hampir sempurna dalam memprediksi data yang dilatihkan. Namun, MSE pada data pengujian cukup tinggi (2.341), menunjukkan bahwa model ini mungkin mengalami overfitting. KNN tampaknya sangat cocok untuk data pelatihan, tetapi kurang mampu menggeneralisasi pada data baru.

   * Random Forest: Random Forest menunjukkan keseimbangan yang baik antara MSE pada data pelatihan dan pengujian. MSE pada data pelatihan (0.675) dan pengujian (1.943) keduanya relatif rendah, menunjukkan bahwa model ini mampu menangkap pola dalam data tanpa overfitting. Random Forest terbukti sebagai model yang lebih stabil dan dapat diandalkan dibandingkan KNN dan Boosting dalam konteks ini.

   * Boosting: Model Boosting memiliki MSE yang cukup tinggi pada data pelatihan (2.226) dan pengujian (2.240). Ini menunjukkan bahwa model Boosting tidak sebaik Random Forest dalam mempelajari pola dari data, dan kemampuannya untuk memprediksi pada data baru tidak seoptimal model Random Forest. Model ini mungkin terlalu sederhana untuk menangkap kompleksitas dalam data atau membutuhkan tuning lebih lanjut.

### Analisis prediksi   

| y_true | pred_KNN | pred_RF | pred_Boosting |
|--------|----------|---------|---------------|
| 100    | 107.8    | 113.2   |  147.6        |

**Prediksi KNN: 107.8**
- Model ini memberikan prediksi yang paling mendekati nilai aktual (107.8), meskipun MSE pada data pengujian lebih tinggi dibandingkan Random Forest. Hal ini menunjukkan bahwa meskipun KNN tampaknya memiliki hasil yang baik dalam kasus ini, performanya secara keseluruhan kurang konsisten.

**Prediksi RF: 113.2**
- Prediksi Random Forest adalah 113.2, yang sedikit lebih tinggi dari prediksi KNN tetapi masih dalam kisaran yang wajar. Dengan MSE yang lebih rendah pada data pengujian, Random Forest tampaknya lebih dapat diandalkan dalam memprediksi harga properti secara umum.

**Prediksi Boosting: 147.6**
- Model Boosting memberikan prediksi yang jauh lebih tinggi dari nilai aktual (147.6). Hal ini menunjukkan bahwa model ini mungkin kurang cocok untuk dataset ini, dan memiliki kesalahan prediksi yang lebih besar dibandingkan kedua model lainnya.

#### Kesimpulan Analisis:
   Random Forest sebagai model yang paling handal dalam memprediksi harga properti Airbnb. Meskipun prediksinya sedikit lebih tinggi dari nilai aktual, performa keseluruhan pada data pengujian menunjukkan bahwa model ini lebih stabil dan akurat dibandingkan KNN dan Boosting. KNN, meskipun memberikan prediksi yang paling dekat dalam kasus ini, menunjukkan tanda-tanda overfitting, sementara Boosting memerlukan peningkatan lebih lanjut agar dapat memberikan hasil yang lebih akurat.

   ### Kesimpulan Akhir
   ### Apakah sudah menjawab problem statement?

* Pernyataan Masalah 1: 
Model telah dikembangkan untuk memprediksi harga optimal properti Airbnb berdasarkan data historis dan fitur properti. Berdasarkan evaluasi, model ini memberikan nilai prediksi yang dapat digunakan untuk menentukan harga properti.
* Pernyataan Masalah 2: 
Random Forest sebagai salah satu model telah mengidentifikasi fitur-fitur yang berpengaruh terhadap harga, seperti `neighbourhood_group` dan `room_type`.

### Apakah berhasil mencapai goals yang diharapkan?

* Goal 1: 
Model yang dikembangkan berhasil memprediksi harga properti, dengan Random Forest menunjukkan performa terbaik pada data test (MSE terendah: 1.94).
* Goal 2: 
Fitur penting telah diidentifikasi, membantu pemilik properti dalam menentukan harga berdasarkan karakteristik properti mereka.

### Apakah solusi yang direncanakan berdampak?

Pemilihan model terbaik melalui GridSearchCV dan tuning hyperparameter memberikan model yang lebih akurat, sehingga membantu dalam pengambilan keputusan yang lebih baik dalam konteks bisnis.

## 7. Referensi
- Kalehbasti, P. R., Nikolenko, L., & Rezae, H. (2021). *Airbnb price prediction using machine learning and sentiment analysis*. In A. Holzinger et al. Vol. LNCS 12844, pp. 173–184.
- [Kaggle](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data)
