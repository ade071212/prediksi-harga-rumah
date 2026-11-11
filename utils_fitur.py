"""Fungsi rekayasa fitur yang dipakai bersama oleh skrip pelatihan dan aplikasi Streamlit.

Memakai satu fungsi yang sama di kedua tempat menjamin input pengguna diproses
dengan cara yang persis sama seperti data latih (konsistensi pipeline, Bab 4).
"""
import numpy as np
import pandas as pd

# Titik acuan pusat kota Jakarta (Monumen Nasional)
MONAS_LAT, MONAS_LON = -6.1754, 106.8272

FITUR_NUMERIK = [
    "bedrooms", "bathrooms", "floors", "carports", "garages",
    "maid_bedrooms", "maid_bathrooms", "jumlah_fasilitas",
    "log_luas_tanah", "log_luas_bangunan", "rasio_bangunan_tanah",
    "lat", "long", "jarak_monas_km",
]
FITUR_KATEGORIK = ["city", "certificate", "property_condition", "furnishing", "electricity"]
SEMUA_FITUR = FITUR_NUMERIK + FITUR_KATEGORIK

LABEL_KELAS = {0: "Murah", 1: "Menengah", 2: "Mahal"}
BATAS_HARGA = (1_000_000_000, 2_500_000_000)  # Rp1 miliar dan Rp2,5 miliar


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def buat_fitur(d: pd.DataFrame) -> pd.DataFrame:
    """Menerima kolom mentah (land_size_m2, building_size_m2, lat, long, dst.)
    dan mengembalikan DataFrame dengan kolom SEMUA_FITUR."""
    d = d.copy()
    d["log_luas_tanah"] = np.log1p(d["land_size_m2"])
    d["log_luas_bangunan"] = np.log1p(d["building_size_m2"])
    d["rasio_bangunan_tanah"] = (d["building_size_m2"] / d["land_size_m2"]).clip(upper=5)
    d["jarak_monas_km"] = haversine_km(d["lat"], d["long"], MONAS_LAT, MONAS_LON)
    return d[SEMUA_FITUR]
