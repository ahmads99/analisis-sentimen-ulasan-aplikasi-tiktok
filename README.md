# Analisis Sentimen Ulasan TikTok

## Tentang Proyek
Selamat datang di proyek **Analisis Sentimen Ulasan TikTok**! Proyek ini lahir dari sesi ngoding maraton di malam Minggu yang penuh kopi dan semangat. Kami menganalisis ulasan pengguna TikTok untuk menentukan sentimennya (positif, negatif, atau netral) menggunakan pendekatan berbasis kamus (lexicon-based) dan model deep learning. Plus, kami bikin aplikasi web interaktif dengan Streamlit biar kamu bisa coba prediksi sentimen langsung. Keren, kan?

## Fitur Utama
- **Text Cleaning**: Hapus mentions, hashtag, link, angka, tanda baca, ubah huruf kecil, perbaiki kata slang, tokenisasi, dan buang kata-kata umum (stopwords).
- **Dictionary-Based Analysis**: Hitung skor sentimen pake kamus kata positif dan negatif, plus visualisasi distribusi sentimen.
- **Deep Learning Models**: Empat model canggih (LSTM, RNN, CNN, GRU) dilatih untuk klasifikasi sentimen, dengan GRU sebagai bintangnya.
- **Cool Visualizations**: Word cloud, distribusi panjang teks, dan barplot kata paling sering.
- **Streamlit App**: Antarmuka web modern untuk input ulasan dan lihat prediksi sentimen dengan style yang eye-catching.

## Dataset
- **Sumber**: `data_scraping_tiktok.csv` (ulasan pengguna TikTok)
- **Kolom Digunakan**: `content` (teks ulasan), `score` (rating), `at` (waktu)
- **Preprocessing**: Filter ulasan 2024-2025, hapus kolom ga relevan.

## Teknologi yang Digunakan
- **Bahasa**: Python
- **Library**: Pandas, NumPy, NLTK, Sastrawi, TensorFlow, Keras, Scikit-learn, Matplotlib, Seaborn, WordCloud, Streamlit
- **Model**: LSTM, RNN, CNN, GRU
- **Deployment**: Streamlit untuk aplikasi web

## Cara Install
1. Clone repo ini:
   ```bash
   git clone <url-repo>
   ```
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Download resource NLTK:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('punkt_tab')
   ```
4. Pastikan file dataset (`data_scraping_tiktok.csv`) dan model (`model_gru.h5`) ada di folder yang tepat.

## Cara Pakai
1. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
2. Buka di browser, masukkan ulasan TikTok, klik "Prediksi Sentimen", dan lihat hasilnya dengan style kece!