"""
app.py — Aplikasi Web SPK Skincare
Sistem Pendukung Keputusan Pemilihan Ingredient Skincare
Metode: Hybrid Decision Tree + SAW
"""

from flask import Flask, render_template, request, jsonify, session
import joblib
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import os

app = Flask(__name__)
app.secret_key = 'spk_skincare_utY2024'

# ─── Path ────────────────────────────────────────────────────
BASE     = Path(__file__).parent
MODEL_DIR = BASE / 'model'
DATA_DIR  = BASE / 'data'

# ─── Load semua model saat startup ───────────────────────────
print("Memuat model...")

model        = joblib.load(MODEL_DIR / 'decision_tree_best.pkl')
preprocessor = joblib.load(MODEL_DIR / 'preprocessing_pipeline.pkl')
le_target    = joblib.load(MODEL_DIR / 'le_target.pkl')

with open(MODEL_DIR / 'saw_matrices.json', encoding='utf-8') as f:
    saw_cfg = json.load(f)

df_produk = pd.read_csv(DATA_DIR / 'dataset_skic.csv', sep=';')
df_produk = df_produk.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

print("Model berhasil dimuat!")

# ─── Konstanta SAW ───────────────────────────────────────────
K1          = saw_cfg['K1_matrix']
K2          = saw_cfg['K2_matrix']
INGREDIENTS = saw_cfg['ingredients']
W1, W2      = 0.60, 0.40

INFO_INGREDIENT = {
    'Niacinamide': {
        'manfaat'  : 'Mencerahkan kulit, mengontrol produksi minyak, memperkecil pori-pori',
        'cocok'    : 'Berminyak, Kombinasi, Normal',
        'icon'     : '💧',
        'color'    : '#4CAF50',
    },
    'Salicylic Acid': {
        'manfaat'  : 'Anti-jerawat, eksfoliasi pori, mencegah komedo',
        'cocok'    : 'Berminyak, Kombinasi',
        'icon'     : '🧪',
        'color'    : '#2196F3',
    },
    'Hyaluronic Acid': {
        'manfaat'  : 'Hidrasi intensif, efek plumping, menjaga kelembapan kulit',
        'cocok'    : 'Kering, Normal, Kombinasi',
        'icon'     : '💦',
        'color'    : '#00BCD4',
    },
    'Retinol': {
        'manfaat'  : 'Anti-aging, regenerasi sel, mengatasi bekas jerawat',
        'cocok'    : 'Normal, Berminyak',
        'icon'     : '⚡',
        'color'    : '#FF9800',
    },
    'Ceramide': {
        'manfaat'  : 'Memperkuat skin barrier, mencegah TEWL, melindungi kulit',
        'cocok'    : 'Kering, Sensitif, Kombinasi',
        'icon'     : '🛡️',
        'color'    : '#9C27B0',
    },
    'Vitamin C': {
        'manfaat'  : 'Antioksidan, mencerahkan flek hitam, merangsang produksi kolagen',
        'cocok'    : 'Normal, Kering, Kombinasi',
        'icon'     : '🍋',
        'color'    : '#FFC107',
    },
    'Tea Tree': {
        'manfaat'  : 'Antibakteri, anti-inflamasi, efektif melawan jerawat',
        'cocok'    : 'Berminyak, Kombinasi',
        'icon'     : '🌿',
        'color'    : '#8BC34A',
    },
    'Bakuchiol': {
        'manfaat'  : 'Alternatif retinol alami, anti-aging ringan, aman untuk kulit sensitif',
        'cocok'    : 'Normal, Sensitif, Kering',
        'icon'     : '🌸',
        'color'    : '#E91E63',
    },
}

INFO_JENIS_KULIT = {
    'Berminyak': {
        'deskripsi': 'Kulit kamu memproduksi sebum berlebih. Wajah cenderung terlihat mengkilap, terutama di area T-zone (dahi, hidung, dagu), pori-pori terlihat lebih besar.',
        'tips'     : 'Gunakan produk oil-free dan non-comedogenic. Cuci muka 2x sehari dengan cleanser yang lembut.',
        'icon'     : '✨',
        'color'    : '#FF6B35',
        'bg'       : '#FFF3EE',
    },
    'Kering': {
        'deskripsi': 'Kulit kamu kekurangan kelembapan alami. Terasa tertarik, kasar, dan bisa terlihat bersisik. Lebih rentan terhadap garis-garis halus.',
        'tips'     : 'Gunakan moisturizer yang kaya dan hindari produk berbahan alkohol tinggi. Minum air yang cukup.',
        'icon'     : '🌵',
        'color'    : '#8B6F47',
        'bg'       : '#FDF6EE',
    },
    'Normal': {
        'deskripsi': 'Kulit kamu berada dalam kondisi ideal — seimbang antara minyak dan kelembapan. Pori-pori tidak terlalu terlihat dan jarang bermasalah.',
        'tips'     : 'Pertahankan rutinitas skincare yang konsisten untuk menjaga kondisi kulit yang sudah baik ini.',
        'icon'     : '⭐',
        'color'    : '#4CAF50',
        'bg'       : '#F0FFF0',
    },
    'Kombinasi': {
        'deskripsi': 'Kulit kamu memiliki area berminyak di T-zone (dahi, hidung, dagu) namun normal atau kering di area pipi.',
        'tips'     : 'Gunakan produk yang berbeda untuk area berbeda, atau pilih produk yang diformulasikan untuk kulit kombinasi.',
        'icon'     : '⚖️',
        'color'    : '#2196F3',
        'bg'       : '#EEF5FF',
    },
}

# ─── Fungsi SAW ───────────────────────────────────────────────
def hitung_saw(jenis_kulit, masalah_kulit):
    rows = []
    for ing in INGREDIENTS:
        k1_val = K1[ing].get(jenis_kulit, 1)
        k2_val = K2[ing].get(masalah_kulit, 1)
        rows.append({'ingredient': ing, 'K1': k1_val, 'K2': k2_val})

    dm         = pd.DataFrame(rows)
    dm['r_K1'] = dm['K1'] / dm['K1'].max()
    dm['r_K2'] = dm['K2'] / dm['K2'].max()
    dm['Vi']   = W1 * dm['r_K1'] + W2 * dm['r_K2']
    dm         = dm.sort_values('Vi', ascending=False).reset_index(drop=True)
    dm['ranking'] = dm.index + 1

    # Tentukan status
    def get_status(row):
        if row['ranking'] <= 3:
            return 'recommended'
        elif row['Vi'] < 0.45:
            return 'avoid'
        return 'neutral'
    dm['status'] = dm.apply(get_status, axis=1)
    return dm

# ─── Fungsi cari produk ───────────────────────────────────────
def cari_produk(jenis_kulit, masalah_kulit, top_ingredients, max_produk=5):
    hasil = []
    for ing in top_ingredients:
        mask = (
            df_produk['Untuk Kulit'].str.contains(jenis_kulit, case=False, na=False) &
            (df_produk['Tipe Bahan Aktif'] == ing)
        )
        produk = df_produk[mask]

        # Fallback: cari berdasarkan bahan aktif saja
        if produk.empty:
            produk = df_produk[df_produk['Tipe Bahan Aktif'] == ing]

        for _, row in produk.head(2).iterrows():
            entry = row.to_dict()
            entry['ingredient_match'] = ing
            if entry not in hasil:
                hasil.append(entry)

    # Fallback: cari berdasarkan masalah kulit
    if not hasil:
        mask2 = df_produk['Masalah Kulit'].str.contains(
            masalah_kulit, case=False, na=False)
        for _, row in df_produk[mask2].head(max_produk).iterrows():
            hasil.append(row.to_dict())

    return hasil[:max_produk]

# ─── ROUTES ──────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/kuesioner')
def kuesioner():
    return render_template('kuesioner.html')

@app.route('/proses', methods=['POST'])
def proses():
    try:
        data = request.get_json()

        # Validasi input
        required = ['umur', 'jenis_kelamin', 'kelembapan', 'minyak',
                    'sensitivitas', 'masalah_kulit']
        for field in required:
            if not data.get(field):
                return jsonify({'error': f'Field {field} tidak boleh kosong'}), 400

        # ── Step 1: Preprocessing ──────────────────────────────
        df_input = pd.DataFrame([{
            'Umur'              : int(data['umur']),
            'Jenis Kelamin'     : data['jenis_kelamin'],
            'Tingkat Kelembapan': data['kelembapan'],
            'Tingkat Minyak'    : data['minyak'],
            'Sensitivitas'      : data['sensitivitas'],
        }])
        X_input = preprocessor.transform(df_input)

        # ── Step 2: Decision Tree → Prediksi Jenis Kulit ──────
        pred_idx    = model.predict(X_input)[0]
        pred_proba  = model.predict_proba(X_input)[0]
        jenis_kulit = le_target.inverse_transform([pred_idx])[0]

        confidence = {
            le_target.classes_[i]: round(float(p) * 100, 1)
            for i, p in enumerate(pred_proba)
        }

        # ── Step 3: SAW → Ranking Ingredient ──────────────────
        masalah_kulit = data['masalah_kulit']
        saw_result    = hitung_saw(jenis_kulit, masalah_kulit)

        # Ambil ingredient berdasarkan status
        recommended = saw_result[saw_result['status'] == 'recommended']['ingredient'].tolist()
        neutral     = saw_result[saw_result['status'] == 'neutral']['ingredient'].tolist()
        avoid       = saw_result[saw_result['status'] == 'avoid']['ingredient'].tolist()

        # Detail SAW dengan info ingredient
        saw_detail = []
        for _, row in saw_result.iterrows():
            ing  = row['ingredient']
            info = INFO_INGREDIENT.get(ing, {})
            saw_detail.append({
                'ranking'   : int(row['ranking']),
                'ingredient': ing,
                'K1'        : int(row['K1']),
                'K2'        : int(row['K2']),
                'r_K1'      : round(float(row['r_K1']), 3),
                'r_K2'      : round(float(row['r_K2']), 3),
                'Vi'        : round(float(row['Vi']), 3),
                'status'    : row['status'],
                'manfaat'   : info.get('manfaat', ''),
                'icon'      : info.get('icon', ''),
                'color'     : info.get('color', '#666'),
            })

        # ── Step 4: Cari Produk ────────────────────────────────
        produk_list  = cari_produk(jenis_kulit, masalah_kulit, recommended)
        produk_output = []
        for p in produk_list:
            produk_output.append({
                'brand'        : str(p.get('Brand', '')),
                'produk'       : str(p.get('Produk', '')),
                'jenis_produk' : str(p.get('Jenis Produk', '')),
                'untuk_kulit'  : str(p.get('Untuk Kulit', '')),
                'masalah'      : str(p.get('Masalah Kulit', '')),
                'ukuran'       : str(p.get('Ukuran', '')),
                'bahan_aktif'  : str(p.get('Tipe Bahan Aktif', '')),
                'tahun'        : str(p.get('Tahun Rilis', '')),
            })

        # ── Response ──────────────────────────────────────────
        result = {
            'success'       : True,
            'input'         : {
                'umur'         : data['umur'],
                'jenis_kelamin': data['jenis_kelamin'],
                'kelembapan'   : data['kelembapan'],
                'minyak'       : data['minyak'],
                'sensitivitas' : data['sensitivitas'],
                'masalah_kulit': masalah_kulit,
            },
            'jenis_kulit'   : jenis_kulit,
            'confidence'    : confidence,
            'info_kulit'    : INFO_JENIS_KULIT.get(jenis_kulit, {}),
            'saw_detail'    : saw_detail,
            'recommended'   : recommended,
            'neutral'       : neutral,
            'avoid'         : avoid,
            'produk'        : produk_output,
            'bobot'         : {'K1': W1, 'K2': W2},
        }

        # Simpan ke session untuk halaman hasil
        session['hasil'] = result
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/hasil')
def hasil():
    data = session.get('hasil')
    if not data:
        return render_template('index.html')
    return render_template('hasil.html', data=data)


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
