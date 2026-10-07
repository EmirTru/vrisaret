#!/usr/bin/env python3
"""
TÜBİTAK 2204-A: Sürekli İşaret Dili Tanıma için Çift Yönlü LSTM (BiLSTM) Mimarisi

Zaman Serisi El Kinematiği Girişi:
- Zaman Adımı (Sequence Length): 45 (1.5 saniye @ 30 FPS)
- Öznitelik Sayısı (Features): 126 (2 El x 21 Eklem x 3 Eksen)
- Çıkış: 8 Sağlık Temalı İşaret Dili Sınıfı
"""

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

def build_lstm_model(input_shape=(45, 126), num_classes=8, model_type="bilstm"):
    """
    Sürekli el takibi zaman serileri için optimize edilmiş LSTM/BiLSTM modeli.
    Meta Quest 2 ve mobil WebGL çıkarımı için hafif ve yüksek doğruluklu tasarlanmıştır.
    """
    if not HAS_TF:
        raise ImportError("TensorFlow kütüphanesi yüklü değil. 'pip install tensorflow' çalıştırın.")

    inputs = keras.Input(shape=input_shape, name="hand_kinematics_input")

    # Eksik el tespitinde sıfırları maskele
    x = layers.Masking(mask_value=0.0)(inputs)

    if model_type == "bilstm":
        # 1. Çift Yönlü LSTM Katmanı: Zamansal bağlamı hem ileri hem geri öğrenir
        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.0),
            name="bilstm_layer_1"
        )(x)
        # 2. Çift Yönlü LSTM Katmanı
        x = layers.Bidirectional(
            layers.LSTM(48, return_sequences=False, dropout=0.2, recurrent_dropout=0.0),
            name="bilstm_layer_2"
        )(x)
    elif model_type == "gru":
        # Daha hafif Snapdragon XR2 optimizasyonu için GRU seçeneği
        x = layers.Bidirectional(layers.GRU(48, return_sequences=True, dropout=0.2))(x)
        x = layers.Bidirectional(layers.GRU(32, return_sequences=False, dropout=0.2))(x)
    else:
        # Standart Tek Yönlü LSTM
        x = layers.LSTM(64, return_sequences=True, dropout=0.2)(x)
        x = layers.LSTM(32, return_sequences=False, dropout=0.2)(x)

    # Karar ve Sınıflandırma Katmanları
    x = layers.Dense(64, activation="relu", name="dense_features")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.3, name="dropout_classifier")(x)

    # Softmax Olasılık Dağılımı Çıkışı
    outputs = layers.Dense(num_classes, activation="softmax", name="action_probabilities")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="VR_SignLanguage_BiLSTM")

    # TÜBİTAK bilimsel değerlendirmesi için optimizer ve metrikler
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

if __name__ == "__main__":
    if HAS_TF:
        model = build_lstm_model()
        model.summary()
    else:
        print("[!] TensorFlow henüz import edilemedi.")
