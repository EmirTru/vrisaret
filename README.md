# TÜBİTAK 2204-A: WebXR ve VR El Takibi ile Sürekli İşaret Dili Tanıma

> **Proje Başlığı:** WebXR ve VR El Takibi ile Sürekli İşaret Dili Tanıma ve Gerçek Zamanlı Türkçe Tercüme Sistemi  
> **Donanım:** Meta Quest 2 (Dahili Kameralar, WebXR Hand Input API)  
> **Frontend:** Three.js, HTML5, WebGL, Web Audio API, Web Speech API (TTS)  
> **Backend / Yapay Zekâ:** Python, Çift Yönlü LSTM (BiLSTM), TensorFlow.js  
> **Hedef:** Sağlık temalı (örn. *"Başım ağrıyor, ambulans çağırın!"*) sürekli işaret dili hareketlerinin doğrudan VR gözlük içinde (%100 On-Device Edge AI) tanınması ve seslendirilmesi.

---

## 🌟 Projenin Öne Çıkan Özellikleri

1. **Hiçbir Sensörlü Eldiven veya Harici Kamera Gerektirmez:**
   - Meta Quest 2'nin dahili kameraları ve W3C **WebXR Hand Input API** standardı kullanılarak her iki eldeki 21'er eklem noktasının (toplam 42 eklem, 126 öznitelik) 3D $(X,Y,Z)$ kartezyen koordinatları 30 FPS hızında yakalanır.
2. **Matematiksel Koordinat Normalizasyonu:**
   - **Bilek Merkezli Öteleme:** $P' = P - P_{wrist}$ ile kullanıcının odadaki konumundan bağımsızlık (Translation Invariance).
   - **El Boyutu Ölçekleme:** $P'' = P' / ||P_{middle} - P_{wrist}||$ ile farklı el büyüklüklerine karşı dayanıklılık (Scale Invariance).
3. **Çift Yönlü LSTM (BiLSTM) Mimarisi:**
   - 45 karelik (1.5 saniye) kayan pencere (Sliding Window) üzerinde zamansal bağlamı hem ileri hem geri yönde analiz eder.
4. **Doğrudan VR Gözlükte Çalışan Edge AI (TensorFlow.js):**
   - Eğitilen model TensorFlow.js Layers formatına dönüştürülür. Meta Quest Browser içinde WebGL donanım hızlandırmasıyla 15-25 ms gecikmeyle çalışır.
5. **Sürekli İşaret Dili & Türkçe Cümle Birleştirme:**
   - Kelimeleri sırayla algılar (`basim` ➔ `agriyor` ➔ `ambulans` ➔ `cagirin`), durum makinesiyle filtreler ve akıcı Türkçe cümleye çevirerek sesli okur (Text-to-Speech).
6. **GitHub Pages ile Sıfır Kurulum:**
   - Statik WebXR yapısı sayesinde doğrudan GitHub Pages üzerinde barındırılabilir. Meta Quest 2'den ücretsiz SSL ile doğrudan erişilebilir.

---

## 📁 Proje Klasör Yapısı

```text
vrisaret/
├── index.html                   # Ana Karşılama Portalı (GitHub Pages giriş sayfası)
├── server.py                    # Yerel HTTPS geliştirme & veri kaydetme sunucusu
├── README.md                    # Proje genel dökümanı ve kurulum rehberi
├── TUBITAK_PROJE_RAPORU_REHBERI.md # 2204-A Akademik Rapor Şablonu ve Jüri Savunma Rehberi
├── .gitignore                   # Git için temiz commit kuralları
├── ssl/                         # Yerel HTTPS sertifikaları (cert.pem, key.pem)
│
├── webxr_collector/             # VERİ TOPLAMA MODÜLÜ
│   ├── index.html               # VR Veri Toplama Arayüzü & 3D el görselleştirme
│   └── collector.js             # 30 FPS örnekleme, normalizasyon, VR HUD, sesli geri sayım
│
├── ai_pipeline/                 # DERİN ÖĞRENME EĞİTİM HATTI
│   ├── dataset.py               # JSON veri yükleyici, veri artırma, sentetik veri üreteci
│   ├── model.py                 # Keras BiLSTM zaman serisi derin öğrenme mimarisi
│   ├── train.py                 # Eğitim scripti, metrikler, kayıp eğrileri, TF.js export
│   ├── convert_to_tfjs.py       # Keras modelini TensorFlow.js web modeline çevirme
│   ├── requirements.txt         # Python bağımlılıkları
│   ├── data/                    # Toplanan/üretilen el koordinatı JSON dosyaları
│   └── models/                  # Eğitilen .keras, .h5 modelleri ve başarım grafikleri
│
└── webxr_inference/             # VR CANLI ÇIKARIM VE TERCÜMAN
    ├── index.html               # VR Canlı Tercüman web arayüzü
    ├── app.js                   # TF.js motoru, 45-kare kayan pencere, cümle kuralları, TTS
    └── model/                   # TensorFlow.js model dosyaları
        ├── model.json           # Model mimarisi ve ağırlık tanımları
        ├── group1-shard1of1.bin # İkili ağırlık tensörleri
        └── classes.json         # Sınıf etiketleri ve Türkçe cümle kuralları
```

---

## 🚀 GitHub'a Yükleme ve GitHub Pages ile Yayınlama

Bu projeyi GitHub'a yükleyip **GitHub Pages** açtığınızda, Meta Quest 2 gözlüğünüzden kablosuz ve ek kurulumsuz olarak doğrudan çalıştırabilirsiniz!

### 1. GitHub'da Yeni Repo Oluşturun
1. [github.com](https://github.com) adresine gidin ve yeni bir public repository oluşturun (Örn: `vrisaret`).

### 2. Masaüstündeki Klasörü GitHub'a Gönderin
Terminali açın ve şu komutları girin:

```bash
cd /Users/emirtru/Desktop/vrisaret

# Git deposunu başlatın
git init

# Dosyaları ekleyin ve commit yapın
git add .
git commit -m "TÜBİTAK 2204-A WebXR VR Isaret Dili Projesi İlk Sürüm"

# GitHub reponuzu bağlayın (Kendi kullanıcı adınızı ve repo adınızı yazın)
git branch -M main
git remote add origin https://github.com/<KULLANICI_ADINIZ>/<REPO_ADINIZ>.git
git push -u origin main
```

### 3. GitHub Pages'i Aktifleştirin
1. GitHub reponuzda **Settings** (Ayarlar) sekmesine gidin.
2. Sol menüden **Pages** seçeneğine tıklayın.
3. **Build and deployment** altında:
   - **Source:** `Deploy from a branch`
   - **Branch:** `main` / `(root)` seçin ve **Save** butonuna tıklayın.
4. 1-2 dakika içinde GitHub size özel bir adres oluşturacaktır:
   👉 `https://<KULLANICI_ADINIZ>.github.io/<REPO_ADINIZ>/`

### 4. Meta Quest 2'de Çalıştırma
1. Meta Quest 2 gözlüğünüzü takın ve **Meta Quest Browser**'ı açın.
2. Adres çubuğuna GitHub Pages adresinizi yazın:
   - Ana Portal: `https://<KULLANICI_ADINIZ>.github.io/<REPO_ADINIZ>/`
   - Doğrudan VR Tercüman: `https://<KULLANICI_ADINIZ>.github.io/<REPO_ADINIZ>/`
3. Ekranda beliren **"👓 VR BAŞLAT (Meta Quest 2)"** butonuna tıklayın.
4. Gözlük kontrolcülerini bırakıp el takibine (Hand Tracking) geçin!

---

## 💻 Yerel Geliştirme (Local Development)

Yerel ağ üzerinden hızlı test ve yeni veri kaydetmek için:

```bash
cd /Users/emirtru/Desktop/vrisaret
python3 server.py
```

Sunucu açıldığında terminalde yerel IP adresinizi (örn. `https://192.168.1.35:8443`) gösterecektir.
Meta Quest 2 tarayıcısından bu adresi açarak veri toplayabilir ve modelleri anında test edebilirsiniz.

---

## 🧠 Model Eğitimi ve TensorFlow.js Dönüştürme

Eğer yeni işaret dili hareketleri kaydettiyseniz veya modeli yeniden eğitmek isterseniz:

```bash
cd /Users/emirtru/Desktop/vrisaret/ai_pipeline

# Bağımlılıkları kontrol edin
pip install -r requirements.txt

# Modeli eğitin (Otomatik olarak değerlendirir ve TensorFlow.js formatına çevirir)
python3 train.py
```

Eğitim tamamlandığında:
- `ai_pipeline/models/training_curves.png`: Eğitim/doğrulama başarım ve karmaşıklık matrisi grafiği.
- `ai_pipeline/models/classification_report.txt`: Sınıf bazlı hassasiyet (precision, recall, f1-score).
- `webxr_inference/model/`: Güncellenmiş `model.json` ve ağırlık dosyaları otomatik olarak buraya kopyalanır.

---

## 🏥 Sağlık Temalı İşaret Sözlüğü ve Cümleler

| Sınıf Etiketi | Anlamı | Tipik El Hareketi |
|---|---|---|
| `basim` | Başım | İşaret parmağı şakak/baş bölgesine dokunur |
| `agriyor` | Ağrıyor | Eller sıkılıp titretilerek acı ifadesi verilir |
| `ambulans` | Ambulans | Eller baş üstünde tepe lambası gibi döner |
| `cagirin` | Çağırın | El dışarıdan içeriye doğru çekilir |
| `yardim` | Yardım | Bir el diğer elin üzerine yumruk desteği yapar |
| `nefes` | Nefes | El göğüs kafesine yerleştirilip açılır |
| `ilac` | İlaç | Avuç içine hap koyma jesti yapılır |
| `notr` | Nötr / Bekleme | Eller serbest dinlenme pozisyonundadır |

**Birleştirilen Türkçe Cümle Örnekleri:**
- `basim` + `agriyor` + `ambulans` + `cagirin` ➔ *"Başım çok ağrıyor, lütfen hemen ambulans çağırın!"*
- `yardim` + `nefes` + `agriyor` ➔ *"Yardım edin, nefes alamıyorum, göğsüm ağrıyor!"*
- `ilac` + `basim` + `agriyor` ➔ *"Başım çok ağrıyor, lütfen ağrı kesici ilaç verin."*
