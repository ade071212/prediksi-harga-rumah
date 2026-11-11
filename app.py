# app.py : Aplikasi prediksi segmen harga rumah Jabodetabek
# Jalankan dengan: streamlit run app.py
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from utils_fitur import buat_fitur

BASE = Path(__file__).parent

st.set_page_config(page_title="Prediksi Segmen Harga Rumah Jabodetabek", page_icon="🏠", layout="centered")


@st.cache_resource
def muat_model():
    """Memuat pipeline (preprocessing + Random Forest) yang sudah dilatih."""
    return joblib.load(BASE / "model_terbaik.pkl")


@st.cache_data
def muat_meta():
    with open(BASE / "meta_model.json", encoding="utf-8") as f:
        return json.load(f)


model = muat_model()
meta = muat_meta()
lookup = meta["lookup"]
batas1, batas2 = meta["batas"]
nama_kelas = {int(k): v for k, v in meta["label"].items()}


def rupiah_miliar(x):
    return f"Rp{x / 1e9:.1f} miliar".replace(".", ",")


# ------------------------------------------------------------------ Tampilan
st.title("🏠 Prediksi Segmen Harga Rumah Jabodetabek")
st.write(
    "Masukkan karakteristik rumah, lalu aplikasi akan memperkirakan **segmen harga**-nya "
    f"berdasarkan model Random Forest yang dilatih dari data listing properti di Jabodetabek."
)

with st.expander("Apa arti ketiga segmen harga?"):
    st.markdown(
        f"- **Murah**: di bawah {rupiah_miliar(batas1)}\n"
        f"- **Menengah**: {rupiah_miliar(batas1)} sampai di bawah {rupiah_miliar(batas2)}\n"
        f"- **Mahal**: {rupiah_miliar(batas2)} ke atas"
    )

st.subheader("1. Lokasi")
c1, c2 = st.columns(2)
kota = c1.selectbox("Kota / Kabupaten", sorted(lookup.keys()), index=sorted(lookup.keys()).index("Bekasi"))
daftar_kec = sorted(lookup[kota].keys(), key=lambda k: -lookup[kota][k]["n"])
kecamatan = c2.selectbox("Kecamatan / Kawasan", daftar_kec, help="Diurutkan dari kawasan dengan data listing terbanyak.")

st.subheader("2. Ukuran dan ruangan")
c1, c2 = st.columns(2)
luas_tanah = c1.number_input("Luas tanah (m²)", min_value=10, max_value=10000, value=100, step=5)
luas_bangunan = c2.number_input("Luas bangunan (m²)", min_value=10, max_value=10000, value=90, step=5)
c1, c2, c3 = st.columns(3)
kamar_tidur = c1.number_input("Kamar tidur", min_value=1, max_value=15, value=3)
kamar_mandi = c2.number_input("Kamar mandi", min_value=1, max_value=15, value=2)
lantai = c3.number_input("Jumlah lantai", min_value=1, max_value=5, value=2)
c1, c2, c3, c4 = st.columns(4)
carport = c1.number_input("Carport", min_value=0, max_value=15, value=1)
garasi = c2.number_input("Garasi", min_value=0, max_value=15, value=0)
kt_art = c3.number_input("K. tidur ART", min_value=0, max_value=7, value=0)
km_art = c4.number_input("K. mandi ART", min_value=0, max_value=5, value=0)

st.subheader("3. Legalitas, kondisi, dan fasilitas")
c1, c2 = st.columns(2)
sertifikat = c1.selectbox("Sertifikat", meta["opsi"]["certificate"])
kondisi = c2.selectbox("Kondisi properti", meta["opsi"]["property_condition"], index=1)
perabot = c1.selectbox("Perabotan", meta["opsi"]["furnishing"])
listrik = c2.selectbox("Daya listrik", meta["opsi"]["electricity"], index=1)
n_fasilitas = st.slider("Jumlah fasilitas lingkungan (taman, kolam renang, masjid, dst.)", 0, 30, 5)

# ------------------------------------------------------------------ Prediksi
if st.button("Prediksi segmen harga", type="primary", use_container_width=True):
    if luas_bangunan > 5 * luas_tanah:
        st.warning("Luas bangunan lebih dari lima kali luas tanah. Pastikan angkanya sudah benar.")

    koordinat = lookup[kota][kecamatan]
    data_baru = pd.DataFrame([{
        "city": kota, "lat": koordinat["lat"], "long": koordinat["long"],
        "land_size_m2": luas_tanah, "building_size_m2": luas_bangunan,
        "bedrooms": kamar_tidur, "bathrooms": kamar_mandi, "floors": lantai,
        "carports": carport, "garages": garasi, "maid_bedrooms": kt_art, "maid_bathrooms": km_art,
        "jumlah_fasilitas": n_fasilitas, "certificate": sertifikat,
        "property_condition": kondisi, "furnishing": perabot, "electricity": listrik,
    }])
    fitur = buat_fitur(data_baru)          # rekayasa fitur yang sama persis dengan saat pelatihan
    prediksi = int(model.predict(fitur)[0])
    proba = model.predict_proba(fitur)[0]

    st.divider()
    st.subheader("Hasil prediksi")
    teks = f"Segmen harga: **{nama_kelas[prediksi].upper()}** (keyakinan model {proba[prediksi]:.1%})"
    if prediksi == 0:
        st.success(teks)
    elif prediksi == 1:
        st.info(teks)
    else:
        st.error(teks)

    st.write("Probabilitas untuk setiap segmen:")
    for k in sorted(nama_kelas):
        a, b = st.columns([1, 4])
        a.write(nama_kelas[k])
        b.progress(float(proba[k]), text=f"{proba[k]:.1%}")

    if proba.max() < 0.6:
        st.caption("Keyakinan model relatif rendah, artinya rumah ini berada di dekat batas antarsegmen.")

st.divider()
m = meta["metrik"]
st.caption(
    f"Model: Random Forest (F1-Macro data uji {m['f1_test']:.3f}, akurasi {m['acc_test']:.1%}). "
    "Prediksi bersifat estimasi berdasarkan data listing (harga penawaran, bukan harga transaksi) "
    "dan tidak menggantikan penilaian resmi dari penilai properti."
)
