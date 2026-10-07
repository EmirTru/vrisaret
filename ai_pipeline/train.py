#!/usr/bin/env python3
"""
TÜBİTAK 2204-A: Sürekli İşaret Dili Tanıma Modeli Eğitim Betiği (train.py)

Bu betik:
1. ai_pipeline/data klasöründeki gerçek VR verilerini yükler (yoksa sentetik üretir).
2. Verileri Train (%80) / Test (%20) olarak böler.
3. BiLSTM modelini eğitir.
4. Karmaşıklık Matrisi (Confusion Matrix) ve Eğitim Grafiğini kaydeder.
5. Modeli TensorFlow.js formatına dönüştürür.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

from dataset import load_dataset, generate_synthetic_data, CLASSES, CLASS_DISPLAY, SEQUENCE_LENGTH, NUM_FEATURES
from model import build_lstm_model

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    print("[HATA] TensorFlow bulunamadı! 'pip install tensorflow' çalıştırınız.")
    sys.exit(1)


def plot_and_save_metrics(history, cm, save_path):
    """Eğitim metriklerini ve Karmaşıklık Matrisini akademik rapor formatında çizer."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Doğruluk (Accuracy) Grafiği
    axes[0].plot(history.history['accuracy'], label='Eğitim Başarımı (Train)', color='#0284c7', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Doğrulama Başarımı (Val)', color='#10b981', linewidth=2)
    axes[0].set_title('Model Doğruluğu (Epoch vs Accuracy)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Doğruluk')
    axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[0].legend()

    # 2. Kayıp (Loss) Grafiği
    axes[1].plot(history.history['loss'], label='Eğitim Kaybı (Train)', color='#ef4444', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Doğrulama Kaybı (Val)', color='#f59e0b', linewidth=2)
    axes[1].set_title('Model Kaybı (Epoch vs Loss)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Kayıp (Cross-Entropy)')
    axes[1].grid(True, linestyle='--', alpha=0.6)
    axes[1].legend()

    # 3. Karmaşıklık Matrisi (Confusion Matrix)
    im = axes[2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[2].set_title('Karmaşıklık Matrisi (Confusion Matrix)', fontsize=12, fontweight='bold')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(CLASSES))
    axes[2].set_xticks(tick_marks)
    axes[2].set_yticks(tick_marks)
    axes[2].set_xticklabels([CLASS_DISPLAY[c] for c in CLASSES], rotation=45, ha="right", fontsize=9)
    axes[2].set_yticklabels([CLASS_DISPLAY[c] for c in CLASSES], fontsize=9)

    # Hücre değerlerini yaz
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[2].text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")

    axes[2].set_ylabel('Gerçek Sınıf')
    axes[2].set_xlabel('Tahmin Edilen Sınıf')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[✓] Eğitim ve Doğrulama Grafiği Kaydedildi: {save_path}")


def main():
    print("=" * 65)
    print(" TÜBİTAK 2204-A: BiLSTM İşaret Dili Modeli Eğitimi ")
    print("=" * 65)

    # 1. Veri Setini Yükle
    X, y = load_dataset(str(DATA_DIR), augment_factor=2)
    
    if len(X) == 0:
        print("[!] ai_pipeline/data içinde gerçek veri bulunamadı.")
        print("[*] Proje testi için 400 adet sentetik kinematik veri üretiliyor...")
        X, y = generate_synthetic_data(num_samples_per_class=50, output_dir=str(DATA_DIR))

    print(f"Toplam Örnek Sayısı: {len(X)} | Tensör Boyutu: {X.shape}")

    # 2. Eğitim / Test Ayrımı (%80 Eğitim, %20 Test)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"Eğitim Kümesi    : {X_train.shape[0]} örnek")
    print(f"Doğrulama Kümesi : {X_val.shape[0]} örnek")

    # 3. Model Oluşturma
    model = build_lstm_model(
        input_shape=(SEQUENCE_LENGTH, NUM_FEATURES),
        num_classes=len(CLASSES),
        model_type="bilstm"
    )
    model.summary()

    # 4. Eğitim Callback'leri
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1
        )
    ]

    # 5. Eğitimi Başlat
    epochs = 40
    batch_size = 16
    print(f"\n[*] Model eğitiliyor ({epochs} Epoch)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # 6. Değerlendirme & Metrikler
    y_pred_probs = model.predict(X_val)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n" + "=" * 65)
    print(" DOĞRULAMA KÜMESİ BAŞARIM RAPORU (TÜBİTAK 2204-A)")
    print("=" * 65)
    report = classification_report(
        y_val, y_pred,
        target_names=[CLASS_DISPLAY[c] for c in CLASSES],
        digits=4
    )
    print(report)

    # Raporu metin dosyasına kaydet
    with open(MODELS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    cm = confusion_matrix(y_val, y_pred)
    plot_and_save_metrics(history, cm, MODELS_DIR / "training_curves.png")

    # 7. Modeli Kaydet
    saved_keras_path = MODELS_DIR / "sign_language_lstm.keras"
    saved_h5_path = MODELS_DIR / "sign_language_lstm.h5"
    model.save(saved_keras_path)
    model.save(saved_h5_path)
    print(f"[✓] Keras Modeli Kaydedildi: {saved_keras_path}")

    # 8. TensorFlow.js Formatına Dönüştür
    print("\n[*] Model TensorFlow.js web formatına dönüştürülüyor...")
    try:
        from convert_to_tfjs import convert_keras_to_tfjs
        convert_keras_to_tfjs(str(saved_keras_path))
    except Exception as e:
        print(f"[!] Otomatik dönüştürme uyarısı: {e}")
        print("    'python convert_to_tfjs.py' komutunu manuel çalıştırabilirsiniz.")

    print("\n" + "=" * 65)
    print(" EĞİTİM VE MODEL DÖNÜŞTÜRME TAMAMLANDI! ")
    print("=" * 65)


if __name__ == "__main__":
    main()
