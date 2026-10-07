#!/usr/bin/env python3
"""
TÜBİTAK 2204-A: Veri Seti Yükleme, Ön İşleme, Veri Artırma ve Sentetik Veri Üretici

Bu modül:
1. WebXR üzerinden toplanan JSON el koordinatı verilerini yükler.
2. Sabit kare uzunluğuna (Zaman Penceresi T=45) enterpolasyon ile hizalar.
3. Veri artırma (Data Augmentation: gürültü ekleme, zaman esnetme, ölçekleme) uygular.
4. İlk test ve doğrulama için sentetik kinematik veri üreteci içerir.
"""

import json
import os
import glob
import numpy as np
from pathlib import Path
# Doğrusal enterpolasyon numpy ile gerçekleştirilmektedir

CLASSES = [
    "basim",      # 0: Başım
    "agriyor",    # 1: Ağrıyor
    "ambulans",   # 2: Ambulans
    "cagirin",    # 3: Çağırın
    "yardim",     # 4: Yardım
    "nefes",      # 5: Nefes
    "ilac",       # 6: İlaç
    "notr"        # 7: Nötr / Bekleme
]

CLASS_DISPLAY = {
    "basim": "Başım",
    "agriyor": "Ağrıyor",
    "ambulans": "Ambulans",
    "cagirin": "Çağırın",
    "yardim": "Yardım",
    "nefes": "Nefes",
    "ilac": "İlaç",
    "notr": "Nötr / Bekleme"
}

LABEL_TO_ID = {c: i for i, c in enumerate(CLASSES)}
ID_TO_LABEL = {i: c for i, c in enumerate(CLASSES)}

SEQUENCE_LENGTH = 45   # 30 FPS'de 1.5 saniyelik zaman penceresi
NUM_FEATURES = 126     # 2 El x 21 Eklem x 3 Koordinat (x, y, z)


def resample_sequence(sequence: np.ndarray, target_length: int = SEQUENCE_LENGTH) -> np.ndarray:
    """
    Farklı uzunluktaki zaman serisini doğrusal enterpolasyon (linear interpolation)
    ile tam olarak target_length uzunluğuna dönüştürür.
    Shape: (T, F) -> (target_length, F)
    """
    orig_len = len(sequence)
    if orig_len == target_length:
        return sequence.astype(np.float32)
    
    if orig_len < 2:
        # Tek kare veya boş ise tekrarla
        return np.repeat(sequence, target_length, axis=0)[:target_length].astype(np.float32)

    orig_indices = np.linspace(0, 1, orig_len)
    target_indices = np.linspace(0, 1, target_length)
    
    resampled = np.zeros((target_length, sequence.shape[1]), dtype=np.float32)
    for f in range(sequence.shape[1]):
        resampled[:, f] = np.interp(target_indices, orig_indices, sequence[:, f])
        
    return resampled


def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Kinematik zaman serisi verisini zenginleştirmek için veri artırma:
    1. Gauss Gürültüsü (Jittering): El titremesi simülasyonu.
    2. Ölçekleme (Scaling): El boyutunda ufak %5 varyasyon.
    3. Zaman Kaydırma / Eğme (Time warping): Hareketi biraz hızlandırma/yavaşlatma.
    """
    aug = sequence.copy()
    
    # 1. Hafif Gauss Gürültüsü
    noise = np.random.normal(0, 0.015, size=aug.shape)
    aug += noise
    
    # 2. Rastgele Ölçekleme Faktörü (0.95 - 1.05)
    scale_factor = np.random.uniform(0.95, 1.05)
    aug *= scale_factor
    
    # 3. Zaman Bozulması (Warping): Rastgele hızlanma/yavaşlama
    if np.random.rand() > 0.5:
        orig_indices = np.linspace(0, 1, len(aug))
        # Orta noktayı hafifçe kaydır
        mid_shift = np.random.uniform(0.4, 0.6)
        warped_indices = np.sort(np.concatenate(([0.0], np.random.beta(2, 2, size=len(aug)-2), [1.0])))
        for f in range(aug.shape[1]):
            aug[:, f] = np.interp(orig_indices, warped_indices, aug[:, f])

    return aug.astype(np.float32)


def load_dataset(data_dir: str, augment_factor: int = 2):
    """
    data_dir içindeki tüm JSON kayıtlarını okur, tensörlere dönüştürür.
    """
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))

    X_list = []
    y_list = []

    print(f"[*] {len(json_files)} adet veri dosyası taranıyor...")

    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            label_str = data.get("label", "").lower()
            if label_str not in LABEL_TO_ID:
                continue

            frames = data.get("frames", [])
            if not frames:
                continue

            # Karelerden öznitelik dizisi çıkar: Shape (T, 126)
            seq_features = []
            for frame in frames:
                feat = frame.get("features", [])
                if len(feat) == NUM_FEATURES:
                    seq_features.append(feat)
                else:
                    # Eksik veya fazla ise 126'ya tamamla
                    padded = (feat + [0.0] * NUM_FEATURES)[:NUM_FEATURES]
                    seq_features.append(padded)

            seq_arr = np.array(seq_features, dtype=np.float32)
            resampled = resample_sequence(seq_arr, SEQUENCE_LENGTH)

            # Orijinal örneği ekle
            X_list.append(resampled)
            y_list.append(LABEL_TO_ID[label_str])

            # Veri artırma (Data Augmentation)
            for _ in range(augment_factor):
                X_list.append(augment_sequence(resampled))
                y_list.append(LABEL_TO_ID[label_str])

        except Exception as e:
            print(f"[!] Hata ({file_path.name}): {e}")

    if not X_list:
        return np.empty((0, SEQUENCE_LENGTH, NUM_FEATURES)), np.empty((0,))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


def generate_synthetic_data(num_samples_per_class: int = 50, output_dir: str = None):
    """
    Gerçek VR donanımı henüz el altında değilken ya da modeli anında eğitip test edebilmek için
    sağlık temalı 8 hareket sınıfı için kinematik el trajektuvarları simüle eder.
    TÜBİTAK ön denemeleri ve pipeline testleri için kritik öneme sahiptir.
    """
    print(f"[*] Sentetik Kinematik Veri Üretiliyor: {len(CLASSES)} sınıf x {num_samples_per_class} örnek...")
    
    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    X_list = []
    y_list = []

    t = np.linspace(0, 1, SEQUENCE_LENGTH)

    for cls_idx, cls_name in enumerate(CLASSES):
        for s in range(num_samples_per_class):
            sequence = np.zeros((SEQUENCE_LENGTH, NUM_FEATURES), dtype=np.float32)
            
            # Sınıfa özel hareket karakteristikleri
            if cls_name == "basim":
                # Sağ el başa doğru yükselir (Y artar, X hafif daralır), şakakta titreşir
                hand_phase = np.sin(np.pi * t)
                for j in range(21):
                    # Sağ el (indeks 63 - 125)
                    sequence[:, 63 + j * 3 + 1] = hand_phase * 1.8 + np.random.normal(0, 0.05, SEQUENCE_LENGTH)
                    sequence[:, 63 + j * 3 + 0] = -0.3 * hand_phase + np.random.normal(0, 0.03, SEQUENCE_LENGTH)

            elif cls_name == "agriyor":
                # Sağ ve sol el parmakları hızla açılıp kapanır ve titrer (Yüksek frekanslı sinüs)
                shake = np.sin(12 * np.pi * t) * 0.4
                for j in range(21):
                    sequence[:, j * 3 + 1] += shake
                    sequence[:, 63 + j * 3 + 1] += shake

            elif cls_name == "ambulans":
                # Her iki el havada dairesel / siren şeklinde döner (Sinüs ve Kosinüs faz farkı)
                rot_x = np.cos(4 * np.pi * t) * 0.6
                rot_y = np.sin(4 * np.pi * t) * 0.6
                for j in range(21):
                    sequence[:, j * 3 + 0] = rot_x
                    sequence[:, j * 3 + 1] = 1.0 + rot_y
                    sequence[:, 63 + j * 3 + 0] = -rot_x
                    sequence[:, 63 + j * 3 + 1] = 1.0 + rot_y

            elif cls_name == "cagirin":
                # Sağ el dışarıdan içeriye (göğse doğru) çekilir (Z ekseninde belirgin hareket)
                pull = np.linspace(1.2, 0.2, SEQUENCE_LENGTH)
                for j in range(21):
                    sequence[:, 63 + j * 3 + 2] = pull + np.random.normal(0, 0.04, SEQUENCE_LENGTH)

            elif cls_name == "yardim":
                # Sol el yatay taban, sağ el yumruk şeklinde sol elin üzerine vurup kaldırılır
                impact = np.where(t < 0.5, 0.5 - t, (t - 0.5) * 1.5)
                for j in range(21):
                    # Sol el sabit taban
                    sequence[:, j * 3 + 1] = 0.2
                    # Sağ el yukarı aşağı hareket
                    sequence[:, 63 + j * 3 + 1] = impact

            elif cls_name == "nefes":
                # Eller göğüs üzerinde nefes alma ritminde yavaşça açılıp kapanır
                breath = np.sin(2 * np.pi * t) * 0.5
                for j in range(21):
                    sequence[:, j * 3 + 0] = -breath * 0.5
                    sequence[:, 63 + j * 3 + 0] = breath * 0.5

            elif cls_name == "ilac":
                # Sol el açık avuç, sağ el işaret/başparmak ilaç koyar gibi yaklaşır
                pinch_move = np.sin(np.pi * t) * 0.8
                for j in range(21):
                    sequence[:, 63 + j * 3 + 0] = -pinch_move * 0.3
                    sequence[:, 63 + j * 3 + 1] = pinch_move * 0.2

            elif cls_name == "notr":
                # Eller serbest aşağıda bekler, minimum hareket
                noise = np.random.normal(0, 0.02, size=(SEQUENCE_LENGTH, NUM_FEATURES))
                sequence = noise.astype(np.float32)

            # Ekstra veri artırma gürültüsü
            sequence = augment_sequence(sequence)

            X_list.append(sequence)
            y_list.append(cls_idx)

            if output_dir:
                sample_json = {
                    "label": cls_name,
                    "frameCount": SEQUENCE_LENGTH,
                    "featuresPerFrame": NUM_FEATURES,
                    "timestamp": 1700000000000 + s * 1000 + cls_idx,
                    "frames": [{"features": sequence[f].tolist()} for f in range(SEQUENCE_LENGTH)]
                }
                file_name = f"{cls_name}_synthetic_{s:03d}.json"
                with open(out_path / file_name, "w", encoding="utf-8") as f:
                    json.dump(sample_json, f)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    print(f"[✓] {len(X)} adet sentetik örnek hazırlandı (Shape: {X.shape})")
    return X, y

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data"
    X, y = load_dataset(data_dir)
    print(f"Yüklenen Veri: X={X.shape}, y={y.shape}")
    if len(X) == 0:
        print("[!] Gerçek veri bulunamadı, 400 adet sentetik örnek üretiliyor...")
        X_syn, y_syn = generate_synthetic_data(50, output_dir=data_dir)
        print(f"[✓] Sentetik veri üretildi ve '{data_dir}' dizinine kaydedildi.")
