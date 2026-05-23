# SkinSPK — Sistem Pendukung Keputusan Pemilihan Ingredient Skincare

Sistem berbasis **Hybrid Decision Tree + Simple Additive Weighting (SAW)**
untuk rekomendasi ingredient skincare sesuai jenis kulit.

---

## Struktur Folder

```
skincare_app/
├── app.py                  ← Aplikasi Flask utama
├── setup_model.py          ← Script salin model dari Drive
├── requirements.txt        ← Daftar library Python
│
├── templates/
│   ├── base.html           ← Layout dasar (navbar, footer)
│   ├── index.html          ← Landing page
│   ├── kuesioner.html      ← Form 6 pertanyaan
│   ├── hasil.html          ← Halaman hasil rekomendasi
│   └── about.html          ← Penjelasan metode
│
├── model/                  ← Folder model (diisi lewat setup_model.py)
│   ├── decision_tree_best.pkl
│   ├── preprocessing_pipeline.pkl
│   ├── le_target.pkl
│   ├── saw_matrices.json
│   └── ...
│
└── data/
    └── dataset_skic.csv    ← Dataset produk skincare
```

---

## Langkah Setup

### 1. Pastikan notebook sudah dijalankan
Jalankan semua cell di `SPK_Skincare_NoSHAP_v2.ipynb` di Google Colab.
File model akan tersimpan di `/content/drive/MyDrive/SPK/model/`.

### 2. Download folder project ke laptop
Download folder `skincare_app/` ini ke laptop kamu.

### 3. Salin file model
Opsi A — Otomatis (jika di Colab):
```bash
python setup_model.py
```

Opsi B — Manual:
Salin file-file berikut dari Google Drive (`MyDrive/SPK/model/`)
ke folder `skincare_app/model/` di laptopmu:
- `decision_tree_best.pkl`
- `preprocessing_pipeline.pkl`
- `le_target.pkl`
- `saw_matrices.json`

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Jalankan website
```bash
python app.py
```

Buka browser: **http://localhost:5000**

---

## Alur Sistem

```
User isi kuesioner (6 pertanyaan)
         ↓
Preprocessing (OrdinalEncoder + RobustScaler)
         ↓
Decision Tree CART → Prediksi Jenis Kulit
         ↓
SAW (K1=60% Jenis Kulit + K2=40% Masalah Kulit)
         ↓
Ranking 8 Ingredient + Rekomendasi Produk
```

---

## Endpoint

| URL | Method | Keterangan |
|-----|--------|------------|
| `/` | GET | Landing page |
| `/kuesioner` | GET | Form kuesioner |
| `/proses` | POST | Proses input, jalankan DT + SAW |
| `/hasil` | GET | Tampilkan hasil rekomendasi |
| `/about` | GET | Penjelasan metode |

---

## Deployment ke Railway / Render (Opsional)

1. Buat repo GitHub dari folder ini
2. Tambahkan file `Procfile`:
   ```
   web: gunicorn app:app
   ```
3. Connect repo ke Railway.app atau Render.com
4. Deploy otomatis!
