# Dashboard Prediksi Harga Saham & Kripto Real-Time

Repositori ini menyediakan dashboard interaktif berbasis [Streamlit](https://streamlit.io) untuk memprediksi harga saham dan kripto secara real-time menggunakan model **Long Short-Term Memory (LSTM)**. Data harga diambil langsung dari Yahoo Finance sehingga dapat diperbarui kapan pun pengguna menekan tombol *Perbarui Data*.

## Fitur Utama

- Pemilihan jenis aset (saham atau kripto) dengan penyesuaian ticker otomatis.
- Konfigurasi parameter model LSTM secara interaktif (look back, epochs, batch size, horizon prediksi).
- Visualisasi harga historis, perbandingan harga aktual vs prediksi, dan proyeksi harga ke depan.
- Metrik evaluasi (RMSE, MAE, MAPE) untuk menilai performa prediksi.
- Penyimpanan cache untuk mempercepat proses pengambilan data dan pelatihan ulang model.

## Cara Menjalankan

1. **Klon repositori** (jika belum).
   ```bash
   git clone https://github.com/firstindra/firstindra.git
   cd firstindra
   ```

2. **Buat virtual environment (opsional tetapi direkomendasikan)** dan aktifkan.
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   .venv\\Scripts\\activate   # Windows
   ```

3. **Instal dependensi**.
   ```bash
   pip install -r requirements.txt
   ```

4. **Jalankan aplikasi Streamlit**.
   ```bash
   streamlit run app.py
   ```

5. Buka alamat yang ditampilkan oleh Streamlit (biasanya `http://localhost:8501`) di browser Anda dan gunakan dashboard.

## Catatan

- Model LSTM dilatih ulang setiap kali parameter diubah, jadi gunakan kombinasi parameter yang sesuai dengan spesifikasi perangkat.
- Untuk data intraday (1m-30m), Yahoo Finance hanya menyediakan historis sekitar 7 hari.
- Prediksi yang dihasilkan bersifat eksperimental dan tidak boleh dijadikan acuan utama untuk keputusan investasi.

## Informasi Kontak

- 👋 Hi, I’m @firstindra
- 👀 I’m interested in cryptocurency
- 🌱 I’m currently learning all above crypto
- 💞️ I’m looking to collaborate on the project of crypto
- 📫 How to reach me indramaulana43@gmail.com
