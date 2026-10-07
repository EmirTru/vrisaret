# TÜBİTAK 2204-A: WebXR ve VR El Takibi ile Sürekli İşaret Dili Tanıma

> **Proje Başlığı:** WebXR ve VR El Takibi ile Sürekli İşaret Dili Tanıma ve Gerçek Zamanlı Türkçe Tercüme Sistemi  
> **Donanım:** Meta Quest 2 (Dahili Kameralar, WebXR Hand Input API)  
> **Frontend:** Three.js, HTML5, WebGL, Web Audio API, Web Speech API (TTS)  
> **Backend / Yapay Zekâ:** Python, Çift Yönlü LSTM (BiLSTM), TensorFlow.js  
> **Çalışma Prensibi:** Sıfır Sunucu! %100 İstemci Taraflı (Edge AI). GitHub Pages üzerinden doğrudan Meta Quest Browser'da çalışır.

---

## 🌟 Projenin Amacı ve Çalışma Mantığı

Bu proje; işitme ve konuşma engelli bireylerin acil sağlık durumlarında (örn: *"Başım ağrıyor, ambulans çağırın!"*) herhangi bir özel veri eldivenine veya harici kameraya ihtiyaç duymadan, yalnızca bir **Meta Quest 2 VR başlığı** takarak iletişim kurabilmelerini sağlar.

1. **WebXR Hand Input API:** Meta Quest 2'nin dahili kameraları ile her iki eldeki 21'er eklem noktasının (toplam 42 eklem, 126 kinematik özellik) 3D uzaysal koordinatları 30 FPS hızında yakalanır.
2. **Koordinat Normalizasyonu:**
   - **Bilek Merkezli Öteleme:** $P' = P - P_{wrist}$ ile kullanıcının odadaki konumundan bağımsızlık (Translation Invariance).
   - **El Boyutu Ölçekleme:** $P'' = P' / ||P_{middle} - P_{wrist}||$ ile el boyutu farklılıklarından bağımsızlık (Scale Invariance).
3. **Çift Yönlü LSTM (BiLSTM):** 45 karelik (1.5 saniye) kayan pencere (Sliding Window) el hareketlerini analiz eder.
4. **Edge AI (On-Device TensorFlow.js):** Model doğrudan Meta Quest 2 tarayıcısında WebGL donanım hızlandırmasıyla çalışır (Gecikme süresi: 15-20 ms). Harici sunucu veya bilgisayar bağlantısı gerekmez.
5. **Cümle Birleştirme ve Seslendirme (TTS):** Algılanan işaretler (`basim` ➔ `agriyor` ➔ `ambulans` ➔ `cagirin`) akıcı Türkçe cümleye dönüştürülür ve gözlük hoparlöründen sesli olarak okunur.

---

## 📁 Proje Dosya Yapısı

```text
vrisaret/
├── index.html                   # Ana VR Uygulaması (Tercüman + Veri Toplayıcı)
├── app.js                       # WebXR el takibi, TensorFlow.js çıkarımı, TTS ve VR HUD
├── README.md                    # Bu doküman (GitHub kurulum & kullanım rehberi)
├── TUBITAK_PROJE_RAPORU_REHBERI.md # TÜBİTAK 2204-A Rapor Şablonu & Jüri Savunma Rehberi
├── .gitignore                   # Git commit kuralları
│
├── model/                       # EĞİTİLMİŞ TENSORFLOW.JS MODELİ (Tarayıcıda çalışır)
│   ├── model.json               # Model mimarisi
│   ├── group1-shard1of1.bin     # Model ağırlıkları (690 KB - çok hafif!)
│   ├── classes.json             # Sınıf etiketleri ve Türkçe cümle kuralları
│   └── weights.json             # Model ağırlık yedeği
│
└── ai_pipeline/                 # DERİN ÖĞRENME EĞİTİM HATTI (TÜBİTAK Jüri Dosyaları)
    ├── dataset.py               # Veri yükleyici, veri artırma & sentetik veri üreteci
    ├── model.py                 # Keras BiLSTM zaman serisi derin öğrenme mimarisi
    ├── train.py                 # Eğitim scripti, metrikler, kayıp eğrileri grafiği
    ├── convert_to_tfjs.py       # Modeli TensorFlow.js formatına ihraç eden script
    ├── requirements.txt         # Bağımlılıklar
    ├── data/                    # Başlangıç sentetik ve toplanan el verileri
    └── models/                  # Eğitilen .keras modeli ve training_curves.png grafiği
```

---

## 🚀 GitHub'a Yükleme ve Meta Quest 2'de Tek Tıkla Çalıştırma

Projeyi GitHub'a yükleyip **GitHub Pages** açtığınızda hiçbir sunucu çalıştırmanıza gerek kalmaz.

### Adım 1: GitHub'da Yeni Repo Oluşturun
1. [github.com](https://github.com) adresine girin ve `vrisaret` adında yeni bir public repository oluşturun.

### Adım 2: Masaüstündeki Klasörü GitHub'a Gönderin
Terminali açın ve şu komutları çalıştırın:

```bash
cd /Users/emirtru/Desktop/vrisaret

git add .
git commit -m "TÜBİTAK 2204-A WebXR VR İsaret Dili Projesi Tam Sürüm"
git branch -M main
git remote add origin https://github.com/<KULLANICI_ADINIZ>/vrisaret.git
git push -u origin main
```
*(<KULLANICI_ADINIZ> yerine kendi GitHub kullanıcı adınızı yazın)*

### Adım 3: GitHub Pages'i Aktifleştirin
1. GitHub reponuzda **Settings** (Ayarlar) sekmesine gidin.
2. Sol menüden **Pages** seçeneğine tıklayın.
3. **Build and deployment** altında:
   - **Source:** `Deploy from a branch`
   - **Branch:** `main` / `(root)` seçin ve **Save** (Kaydet) deyin.
4. 1 dakika içinde projenizin web adresi hazır olacaktır:
   👉 **`https://<KULLANICI_ADINIZ>.github.io/vrisaret/`**

### Adım 4: Meta Quest 2'den Açma (Kullanım)
1. Meta Quest 2 gözlüğünüzü takın ve **Meta Quest Browser**'ı açın.
2. Adres çubuğuna yukarıdaki linki yazın (`https://<KULLANICI_ADINIZ>.github.io/vrisaret/`).
3. Sayfadaki büyük mavi **"👓 VR BAŞLAT (Meta Quest 2)"** butonuna dokunun.
4. Kontrolcüleri yere bırakın. Ellerinizi havaya kaldırın.
5. İşaret dilini yaptıkça el iskeletinizi 3D olarak görecek, havada asılı duran panelde anlık çevrilen Türkçe cümleyi okuyacak ve gözlük hoparlöründen sesli olarak duyacaksınız!

---

## 🧪 Gözlüksüz Test (Jüri & Bilgisayar Simülasyonu)

Gözlüğünüz yanınızda değilken veya bilgisayar ekranında jüriye/öğretmeninize sunum yaparken:
- `index.html` dosyasını bilgisayarınızdaki tarayıcıda açın.
- Sağ paneldeki **"🧪 Gözlüksüz Test & Jüri Simülasyonu"** altındaki butonlara tıklayın:
  - *"Başım ağrıyor, ambulans çağırın!"* butonuna bastığınızda simüle edilmiş el hareketi modele girer, kelimeler sırayla tanınır ve Türkçe olarak seslendirilir.

---

## 🏥 Sağlık Temalı İşaret Sözlüğü ve Cümleler

| Sınıf Etiketi | Türkçe Anlamı | Tipik El Hareketi |
|---|---|---|
| `basim` | Başım | İşaret parmağı şakak/baş bölgesine dokunur |
| `agriyor` | Ağrıyor | Eller sıkılıp titretilerek acı ifadesi verilir |
| `ambulans` | Ambulans | Eller baş üstünde tepe lambası gibi döner |
| `cagirin` | Çağırın | El dışarıdan içeriye doğru çekilir |
| `yardim` | Yardım | Bir el diğer elin üzerine yumruk desteği yapar |
| `nefes` | Nefes | El göğüs kafesine yerleştirilip açılır |
| `ilac` | İlaç | Avuç içine hap koyma jesti yapılır |
| `notr` | Nötr / Bekleme | Eller serbest dinlenme pozisyonundadır |

**Birleştirilen Türkçe Cümleler:**
- `basim` + `agriyor` + `ambulans` + `cagirin` ➔ *"Başım çok ağrıyor, lütfen hemen ambulans çağırın!"*
- `yardim` + `nefes` + `agriyor` ➔ *"Yardım edin, nefes alamıyorum, göğsüm ağrıyor!"*
- `ilac` + `basim` + `agriyor` ➔ *"Başım çok ağrıyor, lütfen ağrı kesici ilaç verin."*
