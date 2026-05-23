"""
setup_model.py
==============
Jalankan script ini SATU KALI untuk menyiapkan semua file model
dari folder Google Drive ke folder model/ di project website.

Cara pakai (dari folder skincare_app/):
    python setup_model.py

Atau jika path berbeda, ubah DRIVE_MODEL_PATH di bawah.
"""

import shutil
from pathlib import Path

# ─── SESUAIKAN PATH INI ──────────────────────────────────────
# Path folder model hasil dari notebook Jupyter di Google Drive
DRIVE_MODEL_PATH = Path("/content/drive/MyDrive/SPK/model")

# Path folder model di project website (tidak perlu diubah)
LOCAL_MODEL_PATH = Path(__file__).parent / "model"
LOCAL_DATA_PATH  = Path(__file__).parent / "data"

# File yang perlu disalin
MODEL_FILES = [
    "decision_tree_best.pkl",
    "preprocessing_pipeline.pkl",
    "le_target.pkl",
    "saw_matrices.json",
    "system_config.pkl",
    "feature_config.pkl",
    "hybrid_pipeline.pkl",
    "preprocessor.pkl",
]

# ─── JALANKAN ────────────────────────────────────────────────
def setup():
    LOCAL_MODEL_PATH.mkdir(exist_ok=True)
    LOCAL_DATA_PATH.mkdir(exist_ok=True)

    print("Menyalin file model...")
    missing = []
    for fname in MODEL_FILES:
        src = DRIVE_MODEL_PATH / fname
        dst = LOCAL_MODEL_PATH / fname
        if src.exists():
            shutil.copy2(src, dst)
            size = dst.stat().st_size / 1024
            print(f"  ✓ {fname:<45} ({size:.1f} KB)")
        else:
            missing.append(fname)
            print(f"  ✗ {fname:<45} (TIDAK DITEMUKAN)")

    if missing:
        print(f"\n⚠️  {len(missing)} file tidak ditemukan: {missing}")
        print("   Pastikan notebook sudah dijalankan dan model sudah tersimpan.")
    else:
        print(f"\n✅ Semua file model berhasil disalin ke {LOCAL_MODEL_PATH}")

    # Verifikasi file kritis
    critical = ["decision_tree_best.pkl", "preprocessing_pipeline.pkl",
                "le_target.pkl", "saw_matrices.json"]
    all_ok = all((LOCAL_MODEL_PATH / f).exists() for f in critical)
    if all_ok:
        print("✅ Verifikasi file kritis: PASSED — website siap dijalankan!")
    else:
        print("❌ Verifikasi GAGAL — file kritis tidak lengkap.")

if __name__ == "__main__":
    setup()
