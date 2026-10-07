#!/usr/bin/env python3
"""
TÜBİTAK 2204-A: Keras -> TensorFlow.js Doğrudan Bağımsız Dönüştürücü (Pure Exporter)

Bu betik, TensorFlow 2.x / Keras modelini hiçbir harici estimator/hub bağımlılığına
ihtiyaç duymadan doğrudan resmi TensorFlow.js "layers-model" formatına dönüştürür:
- webxr_inference/model/model.json
- webxr_inference/model/group1-shard1of1.bin
- webxr_inference/model/classes.json
"""

import os
import sys
import json
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
TFJS_TARGET_DIR = BASE_DIR.parent / "webxr_inference" / "model"

from dataset import CLASSES, CLASS_DISPLAY, SEQUENCE_LENGTH, NUM_FEATURES


def export_metadata(target_dir: Path):
    """Sınıf etiketlerini ve konfigürasyonu WebXR çıkarım motorunun okuması için kaydeder."""
    metadata = {
        "classes": CLASSES,
        "display_names": CLASS_DISPLAY,
        "sequence_length": SEQUENCE_LENGTH,
        "num_features": NUM_FEATURES,
        "sample_fps": 30,
        "sentence_rules": [
            {
                "sequence": ["basim", "agriyor", "ambulans", "cagirin"],
                "sentence": "Başım çok ağrıyor, lütfen hemen ambulans çağırın!"
            },
            {
                "sequence": ["yardim", "nefes", "agriyor"],
                "sentence": "Yardım edin, nefes alamıyorum, göğsüm ağrıyor!"
            },
            {
                "sequence": ["ilac", "basim", "agriyor"],
                "sentence": "Başım çok ağrıyor, lütfen ağrı kesici ilaç verin."
            },
            {
                "sequence": ["yardim", "ambulans", "cagirin"],
                "sentence": "Acil durum! Lütfen bir ambulans çağırın, yardım edin!"
            }
        ]
    }
    target_dir.mkdir(parents=True, exist_ok=True)
    meta_path = target_dir / "classes.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"[✓] Model meta verisi kaydedildi: {meta_path}")


def export_pure_tfjs(model, target_dir: Path):
    """
    Keras modelini doğrudan TensorFlow.js layers-model formatına (JSON + binary bin) dönüştürür.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Ağırlıkları binary buffer olarak topla
    weights_manifest = []
    bin_file_path = target_dir / "group1-shard1of1.bin"

    with open(bin_file_path, "wb") as f_bin:
        for weight in model.weights:
            w_name = weight.name
            w_arr = weight.numpy()
            
            # TensorFlow.js float32 bekler (little-endian)
            w_bytes = w_arr.astype('<f4').tobytes()
            f_bin.write(w_bytes)

            weights_manifest.append({
                "name": w_name,
                "shape": list(w_arr.shape),
                "dtype": "float32"
            })

    # 2. Model JSON dosyasını oluştur
    import tensorflow as tf
    model_json = {
        "format": "layers-model",
        "generatedBy": f"keras v{tf.keras.__version__}",
        "convertedBy": "Tubitak2204A Pure WebXR Converter",
        "modelTopology": {
            "keras_version": tf.keras.__version__,
            "backend": "tensorflow",
            "model_config": {
                "class_name": model.__class__.__name__,
                "config": model.get_config()
            }
        },
        "weightsManifest": [
            {
                "paths": ["group1-shard1of1.bin"],
                "weights": weights_manifest
            }
        ]
    }

    json_file_path = target_dir / "model.json"
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(model_json, f, indent=2)

    # 3. Ayrıca saf JavaScript / WebGL ağırlık sözlüğü (Hafif fallback için)
    weights_dict = {}
    for weight in model.weights:
        weights_dict[weight.name] = weight.numpy().tolist()
    
    weights_json_path = target_dir / "weights.json"
    with open(weights_json_path, "w", encoding="utf-8") as f:
        json.dump(weights_dict, f)

    print(f"[✓] TensorFlow.js Modeli başarıyla üretildi:")
    print(f"    - {json_file_path}")
    print(f"    - {bin_file_path}")
    print(f"    - {weights_json_path}")


def convert_keras_to_tfjs(model_path=None):
    TFJS_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    export_metadata(TFJS_TARGET_DIR)

    if model_path is None:
        keras_cand = MODELS_DIR / "sign_language_lstm.keras"
        h5_cand = MODELS_DIR / "sign_language_lstm.h5"
        if keras_cand.exists():
            model_path = str(keras_cand)
        elif h5_cand.exists():
            model_path = str(h5_cand)
        else:
            print("[HATA] Eğitilmiş model bulunamadı! Lütfen önce 'python train.py' çalıştırın.")
            return False

    print(f"[*] Model yükleniyor: {model_path}")
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)

    export_pure_tfjs(model, TFJS_TARGET_DIR)
    return True


if __name__ == "__main__":
    model_arg = sys.argv[1] if len(sys.argv) > 1 else None
    success = convert_keras_to_tfjs(model_arg)
    if not success:
        sys.exit(1)
