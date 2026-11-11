# Prediksi Segmen Harga Rumah Jabodetabek (Streamlit)

Aplikasi memperkirakan segmen harga rumah (Murah / Menengah / Mahal) memakai model Random Forest.

## Jalankan di komputer sendiri
```
pip install -r requirements.txt
streamlit run app.py
```
Gunakan Python 3.11 atau lebih baru. Versi scikit-learn harus 1.8.0 karena model disimpan dengan versi tersebut.

## Deploy ke Streamlit Community Cloud
1. Buat repositori GitHub publik, lalu unggah seluruh isi folder ini (app.py harus berada di direktori utama).
2. Buka https://share.streamlit.io dan masuk dengan akun GitHub.
3. Pilih "Create app", tentukan repositori, cabang (main), dan berkas utama (app.py).
4. Pada "Advanced settings", pilih Python 3.11 atau lebih baru, lalu klik Deploy.
5. Salin tautan aplikasi yang diberikan dan tempelkan pada laporan (Bab V, subbab 5.6).

## Berkas
- app.py: antarmuka Streamlit
- utils_fitur.py: rekayasa fitur (sama persis dengan saat pelatihan)
- model_terbaik.pkl: pipeline Random Forest hasil joblib
- meta_model.json: opsi kategori, batas segmen, dan koordinat median per kecamatan
- requirements.txt: daftar pustaka
