"""
WISTARA — Bali Tourism Recommendation System
=============================================
Streamlit app yang menghubungkan frontend HTML/CSS/JS dengan
model Content-Based Filtering (TF-IDF + Cosine Similarity).

Cara jalankan:
    streamlit run app.py

Struktur file yang dibutuhkan di folder yang sama:
    app.py
    bali_rec_model.pkl
"""

import json
import pickle
import math
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

# ══════════════════════════════════════════════
# 1. KONFIGURASI HALAMAN
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="WISTARA — Explore Bali",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Hapus padding default Streamlit agar full-width
st.markdown(
    """
    <style>
        html, body {
              overflow-x: hidden;
          overflow-y: auto;
          margin: 0;
          padding: 0;
        }
        /* Hilangkan semua padding Streamlit default */
        .block-container { padding: 0 !important; max-width: 100% !important; }
        [data-testid="stAppViewContainer"] { padding: 0; }
        [data-testid="stHeader"] { display: none; }
        [data-testid="stToolbar"] { display: none; }
        footer { display: none; }
        #MainMenu { display: none; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════
# 2. LOAD MODEL DAN DATA
# ══════════════════════════════════════════════
@st.cache_resource
def load_model():
    """
    Load model dari file pickle.
    @st.cache_resource memastikan model hanya dimuat SEKALI
    walau user refresh — ini penting untuk performa.
    """
    try:
        with open("bali_rec_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        return model_data["dataframe"], model_data["similarity_df"]
    except FileNotFoundError:
        st.error("❌ File 'bali_rec_model.pkl' tidak ditemukan. Pastikan file ada di folder yang sama dengan app.py")
        st.stop()

df, similarity_df = load_model()


# ══════════════════════════════════════════════
# 3. FUNGSI-FUNGSI REKOMENDASI (BACKEND PYTHON)
# ══════════════════════════════════════════════

def get_recommendations_by_category(
    kategori: str,
    kabupaten: str = "Semua",
    min_rating: float = 3.0,
    top_k: int = 20,
) -> list[dict]:
    """
    Fungsi rekomendasi berbasis KATEGORI.
    Menggunakan filtering langsung dari dataframe (bukan similarity).

    Alur:
    1. Filter baris yang kategorinya cocok
    2. Filter berdasarkan kabupaten (opsional)
    3. Filter berdasarkan minimum rating
    4. Urutkan rating descending
    5. Ambil top_k baris
    6. Konversi ke list of dict untuk dikirim ke JS
    """
    result = df.copy()

    # Filter kategori (case-insensitive)
    if kategori and kategori.lower() != "semua":
        result = result[result["kategori"].str.lower() == kategori.lower()]

    # Filter kabupaten
    if kabupaten and kabupaten.lower() not in ("semua", "all"):
        result = result[result["kabupaten_kota"] == kabupaten]

    # Filter rating minimum
    result = result[result["rating"] >= min_rating]

    # Urutkan: rating tinggi dulu, lalu jumlah_rating (popularitas)
    result = result.sort_values(
        by=["rating", "jumlah_rating"], ascending=[False, False]
    ).head(top_k)

    return _df_to_list(result)


def get_similar_destinations(
    nama_tempat: str,
    top_k: int = 8,
) -> list[dict]:
    """
    Fungsi rekomendasi berbasis KEMIRIPAN KONTEN (Content-Based Filtering).
    Menggunakan cosine similarity matrix yang sudah dihitung dari TF-IDF.

    Alur:
    1. Cek apakah nama_tempat ada di similarity_df
    2. Ambil baris similarity untuk tempat tersebut
    3. Urutkan descending (paling mirip di atas)
    4. Skip tempat itu sendiri (similarity = 1.0)
    5. Ambil top_k nama
    6. Ambil data lengkapnya dari df
    """
    if nama_tempat not in similarity_df.index:
        return []

    # Ambil skor similarity untuk tempat ini terhadap semua tempat lain
    sim_scores = similarity_df[nama_tempat].sort_values(ascending=False)

    # Hapus diri sendiri
    sim_scores = sim_scores.drop(index=nama_tempat, errors="ignore")

    # Ambil nama top_k tempat paling mirip
    top_names = sim_scores.head(top_k).index.tolist()

    # Ambil data lengkap dari df
    result = df[df["nama_tempat_wisata"].isin(top_names)].copy()

    # Preserve urutan similarity
    result["_sim_rank"] = result["nama_tempat_wisata"].apply(
        lambda x: top_names.index(x) if x in top_names else 999
    )
    result = result.sort_values("_sim_rank").drop(columns=["_sim_rank"])

    return _df_to_list(result)


def _df_to_list(result_df: pd.DataFrame) -> list[dict]:
    """
    Helper: konversi DataFrame ke list of dict yang JSON-serializable.
    Semua nilai NaN dibersihkan supaya tidak error saat json.dumps().
    """
    records = []
    for _, row in result_df.iterrows():
        img = str(row["link_gambar"]) if str(row["link_gambar"]) != "nan" else ""
        gmaps = str(row["link_google_maps"]) if str(row["link_google_maps"]) != "nan" else ""
        lat = float(row["latitude"]) if not math.isnan(float(row["latitude"])) else -8.4
        lon = float(row["longitude"]) if not math.isnan(float(row["longitude"])) else 115.2

        records.append({
            "id": str(row["id_tempat"]),
            "name": str(row["nama_tempat_wisata"]),
            "kategori": str(row["kategori"]),
            "kecamatan": str(row["kecamatan"]),
            "kabupaten": str(row["kabupaten_kota"]),
            "rating": round(float(row["rating"]), 1),
            "jumlah_rating": int(row["jumlah_rating"]),
            "img": img,
            "lat": lat,
            "lon": lon,
            "gmaps": gmaps,
        })
    return records


# ══════════════════════════════════════════════
# 4. AMBIL PARAMETER FILTER DARI URL / QUERY STRING
#    (Streamlit menggunakan st.query_params)
# ══════════════════════════════════════════════

# Ambil filter dari query params (dikirim dari JS via URL)
params = st.query_params

def get_param(name, default):
    value = params.get(name, default)

    if isinstance(value, list):
        return value[0]

    return value

selected_kategori = get_param("kategori", "Semua")
selected_kabupaten = get_param("kabupaten", "Semua")
selected_min_rating = float(get_param("min_rating", "3.0"))
selected_similar_to = get_param("similar_to", "")

# Siapkan data awal berdasarkan filter
if selected_similar_to:
    # Mode: tampilkan tempat serupa dengan yang dipilih
    cards_data = get_similar_destinations(selected_similar_to, top_k=12)
    page_mode = "similar"
else:
    cards_data = get_recommendations_by_category(
        kategori=selected_kategori,
        kabupaten=selected_kabupaten,
        min_rating=selected_min_rating,
        top_k=30,
    )
    page_mode = "explore"

# Jika tidak ada hasil, fallback ke semua tanpa filter
if not cards_data:
    cards_data = get_recommendations_by_category(
        kategori="Semua", kabupaten="Semua", min_rating=3.0, top_k=30
    )

# Siapkan pilihan filter dari data asli
all_kategori = ["Semua"] + sorted(df["kategori"].unique().tolist())
all_kabupaten = ["Semua"] + sorted(df["kabupaten_kota"].unique().tolist())

# Statistik untuk hero stats bar
total_destinations = len(df)
avg_rating = round(df["rating"].mean(), 1)
total_kabupaten = df["kabupaten_kota"].nunique()

# Serialisasi data ke JSON (akan diembed ke dalam JS)
cards_json = json.dumps(cards_data, ensure_ascii=False)
kategori_json = json.dumps(all_kategori, ensure_ascii=False)
kabupaten_json = json.dumps(all_kabupaten, ensure_ascii=False)


# ══════════════════════════════════════════════
# 5. HTML TEMPLATE
#    Tips: Gunakan f-string dengan hati-hati.
#    Semua kurung kurawal JS harus di-escape: { → {{, } → }}
#    KECUALI variabel Python yang ingin diinjeksikan.
# ══════════════════════════════════════════════

html_code = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Playfair+Display:ital,wght@0,500;0,700;1,500&display=swap" rel="stylesheet"/>
<style>
/* ─── RESET & ROOT ─── */
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
:root{{
  --green:#2D6A4F;
  --green-mid:#40916C;
  --green-light:#74C69D;
  --green-pale:#D8F3DC;
  --sage:#7A8C6E;
  --sage-pale:#EEF2EB;
  --forest:#1B4332;
  --earth:#9E8B72;
  --cream:#F7F5F0;
  --off-white:#FAFAF7;
  --text-dark:#1C1C18;
  --text-mid:#5C5C52;
  --text-light:#9C9C8E;
  --border:rgba(122,140,110,0.18);
  --shadow:0 8px 40px rgba(27,67,50,0.12);
  --shadow-sm:0 4px 20px rgba(27,67,50,0.08);
  --r-sm:8px;
  --r-md:14px;
  --r-lg:20px;
  --r-xl:28px;
  --r-pill:100px;
}}

/* ─── BASE ─── */
html{{
  scroll-behavior:smooth;
  font-size:16px;
}}
body{{
  font-family:'Poppins',sans-serif;
  background:var(--off-white);
  color:var(--text-dark);
  overflow-x:hidden;
  min-height:100vh;
  /* Penting: hilangkan margin body default */
  margin:0;
  padding:0;
  width:100%;
}}

/* ══════════════════════════════════════
   NAVBAR
══════════════════════════════════════ */
nav{{
  position:fixed;
  top:0;left:0;right:0;
  z-index:999;
  display:flex;
  align-items:center;
  justify-content:space-between;
  padding:16px 40px;
  background:rgba(5,18,12,0.35);
  backdrop-filter:blur(20px);
  -webkit-backdrop-filter:blur(20px);
  border-bottom:1px solid rgba(255,255,255,0.06);
  transition:all 0.35s ease;
}}
nav.scrolled{{
  background:rgba(250,250,247,0.97);
  border-bottom:1px solid var(--border);
  padding:12px 40px;
  box-shadow:0 2px 20px rgba(27,67,50,0.08);
}}
.nav-logo{{
  font-size:18px;
  font-weight:800;
  letter-spacing:3px;
  text-transform:uppercase;
  color:#fff;
  transition:color 0.3s;
  cursor:pointer;
}}
nav.scrolled .nav-logo{{color:var(--forest);}}
.nav-links{{
  display:flex;
  align-items:center;
  gap:36px;
  list-style:none;
}}
.nav-links li a{{
  font-size:13px;
  font-weight:400;
  letter-spacing:0.5px;
  color:rgba(255,255,255,0.85);
  text-decoration:none;
  cursor:pointer;
  position:relative;
  padding-bottom:2px;
  transition:color 0.2s;
}}
.nav-links li a::after{{
  content:'';
  position:absolute;
  bottom:0;left:0;
  width:0;height:1px;
  background:#fff;
  transition:width 0.25s;
}}
.nav-links li a:hover::after{{width:100%;}}
.nav-links li a:hover{{color:#fff;}}
nav.scrolled .nav-links li a{{color:var(--text-mid);}}
nav.scrolled .nav-links li a::after{{background:var(--green);}}
nav.scrolled .nav-links li a:hover{{color:var(--green);}}

/* ══════════════════════════════════════
   HERO SECTION
   Fix: height 100svh, tidak overflow, teks di posisi tengah-bawah
══════════════════════════════════════ */
#home{{
  position:relative;
  width:100%;
  /* svh = small viewport height — lebih stabil di mobile */
  height:100svh;
  min-height:520px;
  max-height:860px;
  display:flex;
  flex-direction:column;
  justify-content:flex-end;
  overflow:hidden;
}}
.hero-bg{{
  position:absolute;
  inset:0;
  background:
    url('https://i.pinimg.com/1200x/0b/40/7f/0b407f324f3948b4b5878e834d4839a2.jpg')
    center 40% / cover no-repeat;
  transform:scale(1.0);
  will-change:transform;
}}
.hero-overlay{{
  position:absolute;
  inset:0;
  background:linear-gradient(
    180deg,
    rgba(5,18,12,0.30) 0%,
    rgba(5,18,12,0.10) 40%,
    rgba(5,18,12,0.70) 100%
  );
}}
.hero-content{{
  position:relative;
  z-index:2;
  /* Padding bawah lebih besar beri ruang dari stats bar */
  padding:0 52px 120px;
  max-width:860px;
}}
.hero-eyebrow{{
  font-size:12px;
  font-weight:400;
  letter-spacing:2px;
  text-transform:uppercase;
  color:rgba(255,255,255,0.75);
  margin-bottom:14px;
}}
.hero-headline{{
  font-family:'Poppins',sans-serif;
  font-size:clamp(38px,5.5vw,78px);
  font-weight:800;
  line-height:0.92;
  color:#fff;
  letter-spacing:-1px;
  text-transform:uppercase;
  text-shadow:0 4px 40px rgba(0,0,0,0.25);
  margin-bottom:28px;
}}
.hero-headline-row{{
  display:flex;
  align-items:center;
  gap:20px;
  flex-wrap:wrap;
  margin-bottom:6px;
}}
.hero-cta-inline{{
  display:inline-flex;
  align-items:center;
  gap:8px;
  background:#fff;
  color:var(--green);
  font-family:'Poppins',sans-serif;
  font-size:13px;
  font-weight:700;
  letter-spacing:0.3px;
  padding:12px 26px;
  border-radius:var(--r-pill);
  border:none;
  cursor:pointer;
  text-decoration:none;
  box-shadow:0 4px 24px rgba(0,0,0,0.2);
  transition:transform 0.2s,box-shadow 0.2s;
  white-space:nowrap;
  align-self:center;
}}
.hero-cta-inline:hover{{
  transform:translateY(-2px);
  box-shadow:0 8px 32px rgba(0,0,0,0.25);
}}
.hero-sub{{
  font-size:14px;
  font-weight:300;
  color:rgba(255,255,255,0.72);
  line-height:1.7;
  max-width:420px;
}}

/* Stats bar — menempel di bawah hero, di atas content */
.hero-stats-bar{{
  position:absolute;
  bottom:0;left:0;right:0;
  z-index:3;
  display:flex;
  background:rgba(255,255,255,0.10);
  backdrop-filter:blur(16px);
  -webkit-backdrop-filter:blur(16px);
  border-top:1px solid rgba(255,255,255,0.12);
}}
.hero-stat{{
  flex:1;
  padding:18px 28px;
  display:flex;
  flex-direction:column;
  gap:3px;
  border-right:1px solid rgba(255,255,255,0.10);
}}
.hero-stat:last-child{{border-right:none;}}
.hstat-num{{font-size:20px;font-weight:700;color:#fff;letter-spacing:-0.5px;}}
.hstat-label{{font-size:10px;font-weight:400;letter-spacing:1.2px;text-transform:uppercase;color:rgba(255,255,255,0.52);}}

/* ══════════════════════════════════════
   GAP ANTARA HERO DAN EXPLORE
   (Section transition dengan padding yang lega)
══════════════════════════════════════ */
#explore{{
  background:var(--cream);
  /* Padding atas yang lega sebagai gap dari hero */
  padding:72px 0 100px;
}}
.explore-header{{
  text-align:center;
  padding:0 40px 52px;
}}
.section-tag{{
  display:inline-flex;
  align-items:center;
  gap:12px;
  color:var(--green-mid);
  font-size:11px;
  font-weight:600;
  letter-spacing:2px;
  text-transform:uppercase;
  margin-bottom:16px;
}}
.section-tag-line{{width:36px;height:1px;background:var(--green-light);}}
.section-h2{{
  font-family:'Playfair Display',serif;
  font-size:clamp(30px,3.2vw,48px);
  font-weight:500;
  color:var(--forest);
  line-height:1.15;
  margin-bottom:14px;
}}
.section-h2 em{{font-style:italic;color:var(--green-mid);}}
.section-sub{{
  font-size:14px;
  font-weight:300;
  color:var(--text-mid);
  max-width:460px;
  margin:0 auto;
  line-height:1.8;
}}

/* ══════════════════════════════════════
   MODE BANNER (tampil jika mode similar)
══════════════════════════════════════ */
.mode-banner{{
  display:none;
  align-items:center;
  gap:12px;
  background:var(--green-pale);
  border:1px solid var(--green-light);
  border-radius:var(--r-md);
  padding:14px 20px;
  margin:0 32px 28px;
  font-size:13px;
  color:var(--forest);
}}
.mode-banner.visible{{display:flex;}}
.mode-banner strong{{font-weight:700;}}
.mode-banner-close{{
  margin-left:auto;
  background:none;border:none;
  font-size:18px;cursor:pointer;
  color:var(--green-mid);line-height:1;
}}

/* ══════════════════════════════════════
   LAYOUT: SIDEBAR + MAIN
══════════════════════════════════════ */
.explore-layout{{
  display:grid;
  grid-template-columns:270px 1fr;
  gap:28px;
  padding:0 32px;
  max-width:1400px;
  margin:0 auto;
  align-items:start;
}}

/* ─── SIDEBAR ─── */
.sidebar-card{{
  position:sticky;
  top:88px;
  background:#fff;
  border:1px solid var(--border);
  border-radius:var(--r-xl);
  padding:28px 24px;
  box-shadow:var(--shadow-sm);
}}
.sb-title{{
  font-family:'Playfair Display',serif;
  font-size:16px;
  font-weight:500;
  color:var(--forest);
  margin-bottom:24px;
  padding-bottom:16px;
  border-bottom:1px solid var(--border);
  display:flex;
  align-items:center;
  gap:8px;
}}
.filter-group{{margin-bottom:22px;}}
.filter-label{{
  font-size:10px;
  font-weight:700;
  letter-spacing:1.5px;
  text-transform:uppercase;
  color:var(--text-light);
  display:block;
  margin-bottom:10px;
}}
.chip-wrap{{display:flex;flex-wrap:wrap;gap:7px;}}
.chip{{
  background:var(--sage-pale);
  border:1px solid rgba(122,140,110,0.2);
  color:var(--sage);
  font-size:12px;
  font-weight:500;
  padding:6px 12px;
  border-radius:var(--r-pill);
  cursor:pointer;
  transition:all 0.18s;
  user-select:none;
}}
.chip:hover{{background:var(--green-pale);border-color:var(--green-light);color:var(--green);}}
.chip.active{{background:var(--green);border-color:var(--green);color:#fff;}}
.f-select{{
  width:100%;
  padding:10px 14px;
  border:1px solid var(--border);
  border-radius:var(--r-md);
  background:var(--sage-pale);
  color:var(--text-dark);
  font-family:'Poppins',sans-serif;
  font-size:13px;
  outline:none;
  appearance:none;
  cursor:pointer;
  transition:border-color 0.2s;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='11' height='7'%3E%3Cpath d='M1 1l4.5 4.5L10 1' stroke='%237A8C6E' stroke-width='1.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
  background-repeat:no-repeat;
  background-position:right 12px center;
  padding-right:32px;
}}
.f-select:focus{{border-color:var(--green-mid);}}
.rating-row{{display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;}}
.rating-display-val{{font-size:18px;font-weight:700;color:var(--forest);}}
.rating-stars{{color:#C8A96E;font-size:13px;letter-spacing:2px;}}
input[type=range]{{
  width:100%;
  -webkit-appearance:none;
  appearance:none;
  height:4px;
  border-radius:2px;
  background:var(--border);
  outline:none;
  cursor:pointer;
}}
input[type=range]::-webkit-slider-thumb{{
  -webkit-appearance:none;
  width:18px;height:18px;
  border-radius:50%;
  background:#fff;
  border:2px solid var(--green-mid);
  box-shadow:0 2px 8px rgba(64,145,108,0.3);
  cursor:pointer;
}}
.sb-apply{{
  width:100%;
  margin-top:24px;
  padding:13px;
  background:var(--green);
  color:#fff;
  border:none;
  border-radius:var(--r-md);
  font-family:'Poppins',sans-serif;
  font-size:13px;
  font-weight:600;
  letter-spacing:0.5px;
  cursor:pointer;
  transition:background 0.2s;
}}
.sb-apply:hover{{background:var(--forest);}}
.sb-reset{{
  width:100%;
  margin-top:8px;
  padding:11px;
  background:transparent;
  color:var(--text-light);
  border:1px solid var(--border);
  border-radius:var(--r-md);
  font-family:'Poppins',sans-serif;
  font-size:12px;
  font-weight:500;
  cursor:pointer;
  transition:all 0.2s;
}}
.sb-reset:hover{{color:var(--green-mid);border-color:var(--green-light);}}

/* ─── MAIN CONTENT ─── */
.main-content{{min-width:0;}}
.results-bar{{
  display:flex;
  align-items:center;
  justify-content:space-between;
  margin-bottom:24px;
  flex-wrap:wrap;
  gap:12px;
}}
.results-label{{font-size:13px;color:var(--text-light);}}
.results-label strong{{color:var(--forest);font-weight:700;}}
.sort-pills{{display:flex;gap:7px;flex-wrap:wrap;}}
.sort-pill{{
  background:#fff;
  border:1px solid var(--border);
  color:var(--text-mid);
  font-family:'Poppins',sans-serif;
  font-size:11px;
  font-weight:500;
  letter-spacing:0.3px;
  padding:6px 14px;
  border-radius:var(--r-pill);
  cursor:pointer;
  transition:all 0.18s;
}}
.sort-pill:hover{{border-color:var(--green-light);color:var(--green);}}
.sort-pill.active{{background:var(--green);border-color:var(--green);color:#fff;}}

/* ─── CARD GRID ─── */
.card-grid{{
  display:grid;
  grid-template-columns:repeat(auto-fill,minmax(260px,1fr));
  gap:22px;
  margin-bottom:48px;
}}

/* ─── DESTINATION CARD ─── */
.dest-card{{
  background:#fff;
  border-radius:var(--r-xl);
  border:1px solid var(--border);
  overflow:hidden;
  box-shadow:var(--shadow-sm);
  transition:transform 0.25s,box-shadow 0.25s;
  cursor:pointer;
}}
.dest-card:hover{{transform:translateY(-5px);box-shadow:var(--shadow);}}
.card-img-wrap{{position:relative;height:200px;overflow:hidden;background:linear-gradient(135deg,#d8f3dc,#b7e4c7);}}
.card-img{{
  width:100%;height:100%;
  object-fit:cover;
  display:block;
  transition:transform 0.5s ease;
}}
.dest-card:hover .card-img{{transform:scale(1.05);}}
.card-img-overlay{{
  position:absolute;inset:0;
  background:linear-gradient(180deg,transparent 50%,rgba(10,30,20,0.4) 100%);
  pointer-events:none;
}}
.card-img-fallback{{
  width:100%;height:100%;
  display:flex;align-items:center;justify-content:center;
  font-size:48px;
  background:linear-gradient(135deg,#d8f3dc,#b7e4c7);
}}
.card-badge{{
  position:absolute;top:12px;left:12px;
  background:rgba(255,255,255,0.95);
  color:var(--green);
  font-size:9px;font-weight:700;
  letter-spacing:1px;text-transform:uppercase;
  padding:4px 10px;
  border-radius:var(--r-pill);
  backdrop-filter:blur(8px);
}}
.card-fav{{
  position:absolute;top:12px;right:12px;
  width:30px;height:30px;
  border-radius:50%;
  background:rgba(255,255,255,0.92);
  display:flex;align-items:center;justify-content:center;
  font-size:13px;cursor:pointer;
  transition:transform 0.2s;
  border:none;
  backdrop-filter:blur(8px);
}}
.card-fav:hover{{transform:scale(1.15);}}
.card-fav.liked{{color:#e05c5c;}}
.card-loc-pill{{
  position:absolute;bottom:10px;left:12px;
  display:flex;align-items:center;gap:4px;
  background:rgba(0,0,0,0.35);
  backdrop-filter:blur(8px);
  color:#fff;font-size:10px;font-weight:500;
  padding:3px 8px;
  border-radius:var(--r-pill);
  border:1px solid rgba(255,255,255,0.15);
}}
.card-body{{padding:18px 18px 20px;}}
.card-top{{display:flex;align-items:flex-start;justify-content:space-between;gap:8px;margin-bottom:6px;}}
.card-name{{
  font-family:'Playfair Display',serif;
  font-size:16px;font-weight:500;
  color:var(--forest);line-height:1.25;
  flex:1;
}}
.card-rating{{
  display:flex;align-items:center;gap:4px;flex-shrink:0;
  background:var(--green-pale);
  border-radius:var(--r-pill);
  padding:4px 9px;
}}
.card-rating-star{{color:#C8A96E;font-size:11px;}}
.card-rating-val{{font-size:12px;font-weight:700;color:var(--forest);}}
.card-kecamatan{{
  font-size:12px;font-weight:300;
  color:var(--text-light);margin-bottom:12px;
}}
.card-footer{{
  display:flex;align-items:center;justify-content:space-between;
  padding-top:12px;
  border-top:1px solid var(--border);
}}
.card-reviews{{font-size:11px;color:var(--text-light);}}
.card-btn-row{{display:flex;gap:6px;}}
.card-btn{{
  display:inline-flex;align-items:center;gap:5px;
  background:var(--green-pale);
  color:var(--green);
  font-family:'Poppins',sans-serif;
  font-size:11px;font-weight:600;
  padding:7px 13px;
  border-radius:var(--r-pill);
  border:none;cursor:pointer;
  transition:all 0.2s;text-decoration:none;
}}
.card-btn:hover{{background:var(--green);color:#fff;}}
.card-btn.similar{{
  background:transparent;
  border:1px solid var(--border);
  color:var(--text-light);
}}
.card-btn.similar:hover{{border-color:var(--green-light);color:var(--green);background:var(--green-pale);}}

/* ─── EMPTY STATE ─── */
.empty-state{{
  grid-column:1/-1;
  text-align:center;
  padding:60px 20px;
  color:var(--text-light);
}}
.empty-state-icon{{font-size:48px;margin-bottom:16px;}}
.empty-state h3{{font-size:18px;color:var(--text-mid);margin-bottom:8px;}}
.empty-state p{{font-size:13px;}}

/* ══════════════════════════════════════
   FEATURED DETAIL PANEL
══════════════════════════════════════ */
.featured-panel{{
  background:#fff;
  border:1px solid var(--border);
  border-radius:var(--r-xl);
  padding:28px;
  box-shadow:var(--shadow-sm);
  margin-top:8px;
  display:none;
}}
.featured-panel.visible{{display:block;}}
.featured-panel-header{{
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:22px;
}}
.featured-panel-title{{
  font-family:'Playfair Display',serif;
  font-size:18px;font-weight:500;color:var(--forest);
}}
.featured-panel-close{{
  background:none;border:none;
  font-size:20px;color:var(--text-light);
  cursor:pointer;line-height:1;
  transition:color 0.2s;
}}
.featured-panel-close:hover{{color:var(--text-dark);}}
.featured-body{{
  display:grid;
  grid-template-columns:1fr 1.2fr;
  gap:24px;
}}
.featured-img-wrap{{
  border-radius:var(--r-lg);overflow:hidden;
  aspect-ratio:4/3;position:relative;
  background:linear-gradient(135deg,#d8f3dc,#b7e4c7);
}}
.featured-img{{width:100%;height:100%;object-fit:cover;display:block;}}
.featured-img-overlay{{
  position:absolute;inset:0;
  background:linear-gradient(180deg,transparent 55%,rgba(10,30,20,0.4) 100%);
}}
.featured-info{{display:flex;flex-direction:column;gap:12px;}}
.fi-name{{
  font-family:'Playfair Display',serif;
  font-size:22px;font-weight:500;color:var(--forest);line-height:1.2;
}}
.fi-location{{
  display:flex;align-items:center;gap:6px;
  font-size:12px;color:var(--text-light);
}}
.fi-meta{{display:flex;gap:12px;flex-wrap:wrap;}}
.fi-meta-item{{
  display:flex;flex-direction:column;gap:2px;
  background:var(--cream);
  padding:10px 14px;
  border-radius:var(--r-md);
  flex:1;min-width:70px;
}}
.fim-label{{font-size:9px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--text-light);}}
.fim-val{{font-size:15px;font-weight:700;color:var(--forest);}}
.fi-cta-row{{display:flex;gap:10px;margin-top:4px;}}
.fi-cta-primary{{
  flex:1;padding:12px;
  background:var(--green);color:#fff;
  border:none;border-radius:var(--r-md);
  font-family:'Poppins',sans-serif;
  font-size:13px;font-weight:600;
  cursor:pointer;transition:background 0.2s;
  text-decoration:none;text-align:center;
  display:inline-block;
}}
.fi-cta-primary:hover{{background:var(--forest);}}
.fi-cta-sec{{
  padding:12px 16px;
  background:transparent;
  color:var(--green);
  border:1.5px solid var(--green-light);
  border-radius:var(--r-md);
  font-family:'Poppins',sans-serif;
  font-size:13px;font-weight:600;
  cursor:pointer;transition:all 0.2s;
}}
.fi-cta-sec:hover{{background:var(--green-pale);}}
.fi-cta-sec.liked{{background:var(--green-pale);border-color:var(--green-mid);}}
.similar-rec-section{{margin-top:16px;}}
.similar-rec-title{{
  font-size:11px;font-weight:700;
  letter-spacing:1px;text-transform:uppercase;
  color:var(--text-light);margin-bottom:10px;
}}
.similar-chips{{display:flex;flex-wrap:wrap;gap:6px;}}
.similar-chip{{
  background:var(--sage-pale);
  border:1px solid var(--border);
  color:var(--text-mid);
  font-size:11px;font-weight:500;
  padding:5px 11px;border-radius:var(--r-pill);
  cursor:pointer;transition:all 0.18s;
  max-width:200px;overflow:hidden;
  white-space:nowrap;text-overflow:ellipsis;
}}
.similar-chip:hover{{background:var(--green-pale);border-color:var(--green-light);color:var(--green);}}

/* ══════════════════════════════════════
   FOOTER
══════════════════════════════════════ */
footer{{
  background:var(--forest);
  padding:52px 40px 28px;
  display:grid;
  grid-template-columns:1.5fr 1fr 1fr 1fr;
  gap:40px;
}}
.foot-brand .foot-logo{{
  font-size:18px;font-weight:800;
  letter-spacing:3px;text-transform:uppercase;
  color:#fff;margin-bottom:12px;
}}
.foot-tagline{{
  font-size:13px;font-weight:300;
  color:rgba(255,255,255,0.5);
  line-height:1.8;max-width:200px;
}}
.foot-col h4{{
  font-size:9px;font-weight:700;
  letter-spacing:1.5px;text-transform:uppercase;
  color:rgba(255,255,255,0.4);margin-bottom:14px;
}}
.foot-col a{{
  display:block;font-size:13px;font-weight:300;
  color:rgba(255,255,255,0.6);text-decoration:none;
  margin-bottom:8px;transition:color 0.18s;cursor:pointer;
}}
.foot-col a:hover{{color:#fff;}}
.foot-bottom{{
  background:var(--forest);
  border-top:1px solid rgba(255,255,255,0.07);
  padding:18px 40px;
  display:flex;align-items:center;justify-content:space-between;
}}
.foot-copy{{font-size:11px;color:rgba(255,255,255,0.3);}}
.foot-socials{{display:flex;gap:12px;}}
.social-pill{{
  width:30px;height:30px;border-radius:50%;
  border:1px solid rgba(255,255,255,0.12);
  display:flex;align-items:center;justify-content:center;
  color:rgba(255,255,255,0.5);font-size:12px;
  cursor:pointer;transition:all 0.2s;text-decoration:none;
}}
.social-pill:hover{{border-color:rgba(255,255,255,0.4);color:#fff;}}

/* ══════════════════════════════════════
   RESPONSIVE
══════════════════════════════════════ */
@media (max-width:1024px){{
  .explore-layout{{grid-template-columns:240px 1fr;padding:0 20px;}}
  footer{{grid-template-columns:1fr 1fr;}}
}}
@media (max-width:768px){{
  nav{{padding:14px 20px;}}
  .nav-links{{gap:20px;}}
  .hero-content{{padding:0 24px 110px;}}
  .hero-headline{{font-size:clamp(32px,8vw,52px);}}
  .hero-cta-inline{{padding:10px 18px;font-size:12px;}}
  .hero-stat{{padding:14px 16px;}}
  .hstat-num{{font-size:16px;}}
  #explore{{padding:48px 0 80px;}}
  .explore-layout{{
    grid-template-columns:1fr;
    padding:0 16px;
    gap:20px;
  }}
  .sidebar-card{{position:relative;top:0;}}
  .card-grid{{grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:16px;}}
  .featured-body{{grid-template-columns:1fr;}}
  footer{{grid-template-columns:1fr 1fr;padding:36px 20px 20px;gap:24px;}}
  .foot-bottom{{padding:14px 20px;flex-direction:column;gap:12px;}}
}}
@media (max-width:480px){{
  .nav-links li a{{font-size:12px;}}
  .nav-links{{gap:14px;}}
  .hero-content{{padding:0 18px 100px;}}
  .hero-headline{{font-size:clamp(28px,9vw,42px);}}
  .hero-stats-bar{{flex-wrap:wrap;}}
  .hero-stat{{min-width:50%;}}
  .card-grid{{grid-template-columns:1fr;}}
  footer{{grid-template-columns:1fr;}}
}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar{{width:5px;}}
::-webkit-scrollbar-track{{background:transparent;}}
::-webkit-scrollbar-thumb{{background:var(--green-light);border-radius:10px;}}

/* ── ANIMATION ── */
@keyframes fadeUp{{
  from{{opacity:0;transform:translateY(20px);}}
  to{{opacity:1;transform:translateY(0);}}
}}
.anim{{opacity:0;animation:fadeUp 0.6s ease forwards;}}
.anim-1{{animation-delay:0.1s;}}
.anim-2{{animation-delay:0.22s;}}
.anim-3{{animation-delay:0.36s;}}
.anim-4{{animation-delay:0.5s;}}
</style>
</head>
<body>

<!-- ══════════════════════════════════════
     NAVBAR
══════════════════════════════════════ -->
<nav id="navbar">
  <div class="nav-logo" onclick="scrollToSection('home')">WISTARA</div>
  <ul class="nav-links">
    <li><a onclick="scrollToSection('home')">Home</a></li>
    <li><a onclick="scrollToSection('explore')">Explore</a></li>
    <li><a onclick="scrollToSection('favorites')">Favorites</a></li>
  </ul>
</nav>

<!-- ══════════════════════════════════════
     HERO SECTION
══════════════════════════════════════ -->
<section id="home">
  <div class="hero-bg"></div>
  <div class="hero-overlay"></div>

  <div class="hero-content">
    <p class="hero-eyebrow anim anim-1">Explore Bali Like Never Before</p>
    <h1 class="hero-headline">
      <div class="hero-headline-row anim anim-2">
        DISCOVER
        <a class="hero-cta-inline" onclick="scrollToSection('explore')">
          Start Exploring
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>
        </a>
      </div>
      <div class="anim anim-3">BEAUTIFUL BALI</div>
    </h1>
  </div>

  <div class="hero-stats-bar anim anim-4">
    <div class="hero-stat">
      <span class="hstat-num">{total_destinations}+</span>
      <span class="hstat-label">Destinasi</span>
    </div>
    <div class="hero-stat">
      <span class="hstat-num">{avg_rating}★</span>
      <span class="hstat-label">Rata-rata Rating</span>
    </div>
    <div class="hero-stat">
      <span class="hstat-num">{total_kabupaten}</span>
      <span class="hstat-label">Kabupaten/Kota</span>
    </div>
  </div>
</section>

<!-- ══════════════════════════════════════
     EXPLORE SECTION
══════════════════════════════════════ -->
<section id="explore">
  <div class="explore-header">
    <div class="section-tag">
      <span class="section-tag-line"></span>
      Direkomendasikan untuk Kamu
      <span class="section-tag-line"></span>
    </div>
    <h2 class="section-h2">Temukan Destinasi <em>Terbaik</em> Bali</h2>
    <p class="section-sub">Didukung model machine learning Content-Based Filtering — rekomendasi berdasarkan kemiripan konten dan kategori wisata.</p>
  </div>

  <!-- Mode Banner (muncul kalau sedang lihat similar) -->
  <div class="mode-banner" id="mode-banner">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
    Menampilkan tempat <strong id="banner-name">serupa</strong> berdasarkan kemiripan konten (TF-IDF + Cosine Similarity)
    <button class="mode-banner-close" onclick="resetToExplore()">×</button>
  </div>

  <div class="explore-layout">

    <!-- SIDEBAR FILTER -->
    <aside class="sidebar">
      <div class="sidebar-card">
        <div class="sb-title">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>
          Filter Rekomendasi
        </div>

        <!-- Kategori -->
        <div class="filter-group">
          <span class="filter-label">Kategori Wisata</span>
          <div class="chip-wrap" id="cat-chips">
            <!-- Diisi oleh JS dari data Python -->
          </div>
        </div>

        <!-- Kabupaten -->
        <div class="filter-group">
          <span class="filter-label">Kabupaten / Kota</span>
          <select class="f-select" id="kab-select">
            <!-- Diisi oleh JS -->
          </select>
        </div>

        <!-- Rating -->
        <div class="filter-group">
          <span class="filter-label">Rating Minimum</span>
          <div class="rating-row">
            <div class="rating-display-val" id="rating-display">3.0</div>
            <div class="rating-stars" id="rating-stars">★★★☆☆</div>
          </div>
          <input type="range" min="1" max="5" step="0.5" value="3" id="rating-slider"/>
        </div>

        <button class="sb-apply" onclick="applyFilters()">
          🔍 Cari Rekomendasi
        </button>
        <button class="sb-reset" onclick="resetFilters()">Reset Filter</button>
      </div>
    </aside>

    <!-- MAIN CONTENT -->
    <div class="main-content">
      <div class="results-bar">
        <p class="results-label">
          Menampilkan <strong id="result-count">0</strong> destinasi
          <span id="filter-summary" style="color:var(--green-mid);font-size:12px;"></span>
        </p>
        <div class="sort-pills">
          <div class="sort-pill active" onclick="sortCards(this,'rating')">Rating Tertinggi</div>
          <div class="sort-pill" onclick="sortCards(this,'popular')">Terpopuler</div>
          <div class="sort-pill" onclick="sortCards(this,'az')">A → Z</div>
        </div>
      </div>

      <!-- Card Grid -->
      <div class="card-grid" id="card-grid">
        <!-- Diisi JS -->
      </div>

      <!-- Featured Detail Panel -->
      <div class="featured-panel" id="featured-panel">
        <div class="featured-panel-header">
          <div class="featured-panel-title">Detail Destinasi</div>
          <button class="featured-panel-close" onclick="closeFeatured()">×</button>
        </div>
        <div class="featured-body">
          <div class="featured-img-wrap">
            <img class="featured-img" id="fp-img" src="" alt=""/>
            <div class="featured-img-overlay"></div>
          </div>
          <div class="featured-info">
            <div>
              <div class="fi-name" id="fp-name">—</div>
              <div class="fi-location">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13S3 17 3 10a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
                <span id="fp-location">—</span>
              </div>
              <div class="fi-meta">
                <div class="fi-meta-item">
                  <span class="fim-label">Rating</span>
                  <span class="fim-val" id="fp-rating">—</span>
                </div>
                <div class="fi-meta-item">
                  <span class="fim-label">Ulasan</span>
                  <span class="fim-val" id="fp-reviews">—</span>
                </div>
                <div class="fi-meta-item">
                  <span class="fim-label">Kategori</span>
                  <span class="fim-val" id="fp-cat">—</span>
                </div>
                <div class="fi-meta-item">
                  <span class="fim-label">Kecamatan</span>
                  <span class="fim-val" id="fp-kec">—</span>
                </div>
              </div>
            </div>

            <!-- Rekomendasi Serupa (Content-Based) -->
            <div class="similar-rec-section">
              <div class="similar-rec-title"> Tempat Serupa </div>
              <div class="similar-chips" id="similar-chips">
                <!-- Diisi JS -->
              </div>
            </div>

            <div class="fi-cta-row">
              <a class="fi-cta-primary" id="fp-gmaps-btn" href="#" target="_blank" rel="noopener">
                📍 Buka Google Maps
              </a>
              <button class="fi-cta-sec" id="fp-fav-btn" onclick="toggleFavFeatured()">♡ Simpan</button>
            </div>
          </div>
        </div>
      </div>
    </div><!-- /main-content -->

  </div><!-- /explore-layout -->
</section>

<!-- Section Favorites (Anchor) -->
<div id="favorites" style="height:1px;"></div>

<!-- ══════════════════════════════════════
     FOOTER
══════════════════════════════════════ -->
<footer>
  <div class="foot-brand">
    <div class="foot-logo">WISTARA</div>
    <p class="foot-tagline">Panduan wisata Bali.</p>
  </div>
  <div class="foot-col">
    <h4>Kategori</h4>
    <a onclick="filterByKategori('Alam')">Alam</a>
    <a onclick="filterByKategori('Budaya')">Budaya</a>
    <a onclick="filterByKategori('Rekreasi')">Rekreasi</a>
    <a onclick="filterByKategori('Umum')">Umum</a>
  </div>
  <div class="foot-col">
    <h4>Kabupaten</h4>
    <a onclick="filterByKabupaten('Kabupaten Badung')">Badung</a>
    <a onclick="filterByKabupaten('Kabupaten Gianyar')">Gianyar</a>
    <a onclick="filterByKabupaten('Kabupaten Buleleng')">Buleleng</a>
    <a onclick="filterByKabupaten('Kota Denpasar')">Denpasar</a>
  </div>
  <div class="foot-col">
    <h4>Info</h4>
    <a>Tentang WISTARA</a>
    <a>Panduan</a>
    <a>Kontak</a>
  </div>
</footer>
<div class="foot-bottom">
  <span class="foot-copy">© 2025 WISTARA. Final Project — Sistem Rekomendasi Wisata Bali.</span>
  <div class="foot-socials">
    <a class="social-pill">in</a>
    <a class="social-pill">ig</a>
    <a class="social-pill">gh</a>
  </div>
</div>

<!-- ══════════════════════════════════════
     SCRIPT
══════════════════════════════════════ -->
<script>
// ════════════════════════════════════════
// DATA dari Python (diinjeksikan saat render)
// ════════════════════════════════════════

// Dataset hasil filter dari backend Python
const INITIAL_DATA = {cards_json};

// Pilihan kategori dan kabupaten dari df asli
const ALL_KATEGORI  = {kategori_json};
const ALL_KABUPATEN = {kabupaten_json};

// Mode awal: 'explore' atau 'similar'
const PAGE_MODE = '{page_mode}';
const SIMILAR_TO = `{selected_similar_to}`;

// ════════════════════════════════════════
// STATE
// ════════════════════════════════════════
let currentData  = [...INITIAL_DATA];
let sortedData   = [...INITIAL_DATA];
let likedSet     = new Set(); // simpan ID yang di-favoritkan
let activeCat    = '{selected_kategori}';
let activeKab    = '{selected_kabupaten}';
let activeRating = {selected_min_rating};
let featuredId   = null;

// ════════════════════════════════════════
// INISIALISASI
// ════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {{
  buildFilters();
  renderCards(currentData);
  setupNavbarScroll();
  setupRatingSlider();

  // Jika mode similar, tampilkan banner
  if (PAGE_MODE === 'similar' && SIMILAR_TO) {{
    const banner = document.getElementById('mode-banner');
    document.getElementById('banner-name').textContent = SIMILAR_TO;
    banner.classList.add('visible');
  }}
}});

// ════════════════════════════════════════
// NAVBAR
// ════════════════════════════════════════
function setupNavbarScroll() {{
  const nav = document.getElementById('navbar');
  window.addEventListener('scroll', () => {{
    nav.classList.toggle('scrolled', window.scrollY > 60);
  }}, {{passive: true}});
}}

// ── Smooth scroll ke section ──
// CATATAN: scrollTo adalah fungsi bawaan browser, kita pakai nama berbeda
function scrollToSection(id) {{
  const el = document.getElementById(id);
  if (el) el.scrollIntoView({{behavior: 'smooth'}});
}}

// ════════════════════════════════════════
// BUILD FILTER OPTIONS DARI DATA
// ════════════════════════════════════════
function buildFilters() {{
  // Chip kategori
  const chipWrap = document.getElementById('cat-chips');
  chipWrap.innerHTML = ALL_KATEGORI.map(kat => {{
    const isActive = kat === activeCat || (kat === 'Semua' && activeCat === 'Semua');
    return `<div class="chip${{isActive ? ' active' : ''}}" data-cat="${{kat}}">${{kat}}</div>`;
  }}).join('');

  // Event listener chip
  chipWrap.addEventListener('click', e => {{
    const chip = e.target.closest('.chip');
    if (!chip) return;
    chipWrap.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
    chip.classList.add('active');
    activeCat = chip.dataset.cat;
  }});

  // Dropdown kabupaten
  const kabSelect = document.getElementById('kab-select');
  kabSelect.innerHTML = ALL_KABUPATEN.map(kab =>
    `<option value="${{kab}}" ${{kab === activeKab ? 'selected' : ''}}>${{kab}}</option>`
  ).join('');
}}

// ════════════════════════════════════════
// RATING SLIDER
// ════════════════════════════════════════
function setupRatingSlider() {{
  const slider  = document.getElementById('rating-slider');
  const display = document.getElementById('rating-display');
  const stars   = document.getElementById('rating-stars');

  slider.value = activeRating;
  updateSliderUI(activeRating);

  slider.addEventListener('input', () => {{
    const v = parseFloat(slider.value);
    activeRating = v;
    updateSliderUI(v);
  }});
}}

function updateSliderUI(v) {{
  document.getElementById('rating-display').textContent = v.toFixed(1);
  const filled = Math.round(v);
  document.getElementById('rating-stars').textContent = '★'.repeat(filled) + '☆'.repeat(5 - filled);
  const pct = ((v - 1) / 4) * 100;
  document.getElementById('rating-slider').style.background =
    `linear-gradient(to right, var(--green-mid) 0%, var(--green-mid) ${{pct}}%, rgba(122,140,110,.18) ${{pct}}%)`;
}}

// ════════════════════════════════════════
// APPLY FILTER — Kirim ke Streamlit via URL
//
// CARA KERJA INTEGRASI ML:
// 1. User pilih filter di JS (frontend)
// 2. JS update URL dengan query params
// 3. Streamlit (Python) baca query params
// 4. Python jalankan fungsi rekomendasi dari model
// 5. Hasilnya diinjeksikan ke INITIAL_DATA
// 6. Halaman re-render dengan data baru
// ════════════════════════════════════════
function applyFilters() {{
  activeKab = document.getElementById('kab-select').value;

  const params = new URLSearchParams();

  if (activeCat && activeCat !== 'Semua') {{
    params.set('kategori', activeCat);
  }}

  if (activeKab && activeKab !== 'Semua') {{
    params.set('kabupaten', activeKab);
  }}

  if (activeRating > 1) {{
    params.set('min_rating', activeRating.toFixed(1));
  }}

  window.location.href =
    window.location.pathname +
    '?' +
    params.toString() +
    '#explore';
}}

function resetFilters() {{
  window.location.href =
    window.location.pathname + '#explore';
}}

function resetToExplore() {{
  window.location.href =
    window.location.pathname + '#explore';
}}

// Quick filter dari footer
function filterByKategori(kat) {{
  const params = new URLSearchParams();

  params.set('kategori', kat);

  window.location.href =
    window.location.pathname +
    '?' +
    params.toString() +
    '#explore';
}}
function filterByKabupaten(kab) {{
  const params = new URLSearchParams();

  params.set('kabupaten', kab);

  window.location.href =
    window.location.pathname +
    '?' +
    params.toString() +
    '#explore';
}}

// ════════════════════════════════════════
// RENDER CARDS
// ════════════════════════════════════════
function renderCards(data) {{
  const grid = document.getElementById('card-grid');
  document.getElementById('result-count').textContent = data.length;

  if (data.length === 0) {{
    grid.innerHTML = `
      <div class="empty-state">
        <div class="empty-state-icon">🌿</div>
        <h3>Tidak ada hasil ditemukan</h3>
        <p>Coba ubah filter kategori, kabupaten, atau rating minimum.</p>
      </div>`;
    return;
  }}

  // Update filter summary
  const parts = [];
  if (activeCat && activeCat !== 'Semua') parts.push(activeCat);
  if (activeKab && activeKab !== 'Semua') parts.push(activeKab);
  if (activeRating > 1) parts.push(`Rating ≥ ${{activeRating}}`);
  document.getElementById('filter-summary').textContent =
    parts.length > 0 ? '— ' + parts.join(' · ') : '';

  grid.innerHTML = data.map((d, idx) => {{
    const emoji = getCategoryEmoji(d.kategori);
    const reviewsStr = formatReviews(d.jumlah_rating);
    const imgHtml = d.img
      ? `<img class="card-img" src="${{d.img}}" alt="${{escHtml(d.name)}}" loading="lazy"
            onerror="this.style.display='none';this.parentElement.querySelector('.card-img-fallback').style.display='flex'"/>
         <div class="card-img-fallback" style="display:none">${{emoji}}</div>`
      : `<div class="card-img-fallback">${{emoji}}</div>`;

    return `
    <div class="dest-card" onclick="showFeatured('${{escAttr(d.id)}}')">
      <div class="card-img-wrap">
        ${{imgHtml}}
        <div class="card-img-overlay"></div>
        <div class="card-badge">${{d.kategori}}</div>
        <button class="card-fav ${{likedSet.has(d.id) ? 'liked' : ''}}"
          onclick="event.stopPropagation(); toggleLike('${{escAttr(d.id)}}', this)">
          ${{likedSet.has(d.id) ? '♥' : '♡'}}
        </button>
        <div class="card-loc-pill">
          <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13S3 17 3 10a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
          ${{d.kecamatan}}
        </div>
      </div>
      <div class="card-body">
        <div class="card-top">
          <div class="card-name">${{escHtml(d.name)}}</div>
          <div class="card-rating">
            <span class="card-rating-star">★</span>
            <span class="card-rating-val">${{d.rating}}</span>
          </div>
        </div>
        <div class="card-kecamatan">📍 ${{d.kecamatan}}, ${{d.kabupaten}}</div>
        <div class="card-footer">
          <span class="card-reviews">${{reviewsStr}} ulasan</span>
          <div class="card-btn-row">
            <button class="card-btn similar" onclick="event.stopPropagation(); findSimilar('${{escAttr(d.id)}}', '${{escAttr(d.name)}}')">
              Serupa
            </button>
            <button class="card-btn" onclick="event.stopPropagation(); showFeatured('${{escAttr(d.id)}}')">
              Detail →
            </button>
          </div>
        </div>
      </div>
    </div>`;
  }}).join('');
}}

// ════════════════════════════════════════
// FEATURED DETAIL PANEL
// ════════════════════════════════════════
function showFeatured(id) {{
  const d = currentData.find(x => x.id === id);
  if (!d) return;
  featuredId = id;

  document.getElementById('fp-name').textContent = d.name;
  document.getElementById('fp-location').textContent = `${{d.kecamatan}}, ${{d.kabupaten}}`;
  document.getElementById('fp-rating').textContent = d.rating + ' ★';
  document.getElementById('fp-reviews').textContent = formatReviews(d.jumlah_rating);
  document.getElementById('fp-cat').textContent = d.kategori;
  document.getElementById('fp-kec').textContent = d.kecamatan;

  const img = document.getElementById('fp-img');
  if (d.img) {{
    img.src = d.img;
    img.alt = d.name;
    img.style.display = 'block';
  }} else {{
    img.style.display = 'none';
  }}

  // Google Maps link
  const gmapsBtn = document.getElementById('fp-gmaps-btn');
  if (d.gmaps) {{
    gmapsBtn.href = d.gmaps;
    gmapsBtn.style.opacity = '1';
    gmapsBtn.style.pointerEvents = 'auto';
  }} else {{
    // Fallback ke koordinat
    gmapsBtn.href = `https://www.google.com/maps?q=${{d.lat}},${{d.lon}}`;
  }}

  // Favoritkan
  const favBtn = document.getElementById('fp-fav-btn');
  if (likedSet.has(id)) {{
    favBtn.textContent = '♥ Tersimpan';
    favBtn.classList.add('liked');
  }} else {{
    favBtn.textContent = '♡ Simpan';
    favBtn.classList.remove('liked');
  }}

  // Tampilkan similar berdasarkan kategori yang sama (dari INITIAL_DATA)
  const similars = currentData
    .filter(x => x.id !== id)
    .sort((a, b) => b.rating - a.rating)
    .slice(0, 5);

  const chipsEl = document.getElementById('similar-chips');
  if (similars.length > 0) {{
    chipsEl.innerHTML = similars.map(s =>
      `<div class="similar-chip" onclick="showFeatured('${{escAttr(s.id)}}')" title="${{escAttr(s.name)}}">
        ${{escHtml(s.name)}} (${{s.rating}}★)
      </div>`
    ).join('');
  }} else {{
    chipsEl.innerHTML = `<span style="font-size:12px;color:var(--text-light)">Tidak ada data serupa dalam kategori ini</span>`;
  }}

  const panel = document.getElementById('featured-panel');
  panel.classList.add('visible');
  panel.scrollIntoView({{behavior: 'smooth', block: 'nearest'}});
}}

function closeFeatured() {{
  document.getElementById('featured-panel').classList.remove('visible');
  featuredId = null;
}}

function toggleFavFeatured() {{
  if (!featuredId) return;
  const btn = document.getElementById('fp-fav-btn');
  if (likedSet.has(featuredId)) {{
    likedSet.delete(featuredId);
    btn.textContent = '♡ Simpan';
    btn.classList.remove('liked');
  }} else {{
    likedSet.add(featuredId);
    btn.textContent = '♥ Tersimpan';
    btn.classList.add('liked');
  }}
  // Update card fav button
  renderCards(sortedData);
  showFeatured(featuredId);
}}

// ════════════════════════════════════════
// FIND SIMILAR — Kirim ke Python via URL param
//
// CARA KERJA:
// 1. User klik "Serupa" di card
// 2. JS kirim nama tempat ke URL: ?similar_to=NamaTempat
// 3. Python baca param, jalankan get_similar_destinations()
// 4. Fungsi itu pakai similarity_df (cosine similarity matrix)
// 5. Hasilnya dikembalikan sebagai INITIAL_DATA baru
// ════════════════════════════════════════
function findSimilar(id, name) {{
  const params = new URLSearchParams();

  params.set('similar_to', name);

  window.location.href =
    window.location.pathname +
    '?' +
    params.toString() +
    '#explore';
}}

// ════════════════════════════════════════
// SORT
// ════════════════════════════════════════
function sortCards(el, mode) {{
  document.querySelectorAll('.sort-pill').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  let data = [...currentData];
  if (mode === 'rating')   data.sort((a, b) => b.rating - a.rating);
  if (mode === 'popular')  data.sort((a, b) => b.jumlah_rating - a.jumlah_rating);
  if (mode === 'az')       data.sort((a, b) => a.name.localeCompare(b.name));
  sortedData = data;
  renderCards(data);
}}

// ════════════════════════════════════════
// LIKE / FAVORIT
// ════════════════════════════════════════
function toggleLike(id, btn) {{
  if (likedSet.has(id)) {{
    likedSet.delete(id);
    btn.classList.remove('liked');
    btn.textContent = '♡';
  }} else {{
    likedSet.add(id);
    btn.classList.add('liked');
    btn.textContent = '♥';
  }}
}}

// ════════════════════════════════════════
// UTILS
// ════════════════════════════════════════
function getCategoryEmoji(kat) {{
  const map = {{
    'Alam': '🌿', 'Budaya': '🛕', 'Rekreasi': '🎡', 'Umum': '📍'
  }};
  return map[kat] || '🗺️';
}}

function formatReviews(n) {{
  if (!n) return '0';
  if (n >= 1000) return (n / 1000).toFixed(1) + 'rb';
  return n.toString();
}}

function escHtml(str) {{
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}}

function escAttr(str) {{
  return String(str).replace(/'/g, "\\'").replace(/"/g, '&quot;');
}}

// Init sort data
sortedData = [...currentData];
</script>

</body>
</html>"""

# ══════════════════════════════════════════════
# 6. RENDER KE STREAMLIT
#    height disesuaikan agar tidak terpotong.
#    scrolling=False karena kita pakai scroll internal HTML.
# ══════════════════════════════════════════════

# Estimasi tinggi halaman (hero + explore section + footer)
PAGE_HEIGHT = 6500

components.html(html_code, height=7000, scrolling=True)
