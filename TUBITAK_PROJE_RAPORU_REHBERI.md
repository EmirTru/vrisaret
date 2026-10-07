# TÜBİTAK 2204-A Lise Araştırma Projeleri Yarışması
## Proje Raporu Şablonu ve Jüri Savunma Rehberi

---

### Proje Bilgileri
- **Proje Adı:** WebXR ve VR El Takibi ile Sürekli İşaret Dili Tanıma ve Gerçek Zamanlı Türkçe Tercüme Sistemi
- **Ana Alan:** Yazılım
- **Tematik Alan:** Yapay Zekâ, Giyilebilir Teknolojiler, Sağlık ve Erişilebilirlik
- **Donanım:** Meta Quest 2 (Dahili Kameralar ve WebXR Hand Input API)
- **Yazılım & AI Mimarisi:** Three.js, Python, Çift Yönlü LSTM (BiLSTM), TensorFlow.js, WebGL

---

## 1. Projenin Özeti (Abstract)
İşitme ve konuşma engelli bireylerin acil sağlık durumlarında (örneğin kaza, ani rahatsızlık, kalp krizi veya nefes darlığı) sağlık personeliyle hızlı ve doğru iletişim kuramaması hayati riskler doğurmaktadır. Geleneksel işaret dili tanıma sistemleri ya pahalı ve hareketi kısıtlayan sensörlü veri eldivenleri ya da sınırlı görüş açısına (FoV) sahip 2D RGB web kameraları kullanmaktadır. 2D kameralar derinlik (Z ekseni) kayıplarına, ışık değişimlerine ve arka plan parazitlerine karşı oldukça hassastır.

Bu projede, Meta Quest 2 sanal gerçeklik başlığının dahili kameraları ve **WebXR Hand Input API** kullanılarak kullanıcının her iki elindeki 21'er eklem noktasının (toplam 42 eklem, 126 kinematik özellik) 3 boyutlu $(X, Y, Z)$ uzaysal koordinatları 30 FPS hızında yakalanmıştır. Elde edilen veriler, kullanıcının başlık konumundan ve el anatomisinden bağımsız hale getirilmesi için **Bilek Merkezli Öteleme (Translation Invariance)** ve **Orta Parmak Referanslı Öklid Ölçekleme (Scale Invariance)** normalizasyonuna tabi tutulmuştur.

Zaman serisi verileri Python ortamında tasarlanan **Çift Yönlü Uzun-Kısa Süreli Bellek (Bidirectional LSTM)** derin öğrenme modeli ile eğitilmiş; sağlık temalı ardışık işaret dili hareketleri (Örn: *"Başım ağrıyor, ambulans çağırın"*) akıcı tam cümlelere dönüştürülmüştür. Eğitilen model **TensorFlow.js** formatına optimize edilerek doğrudan Meta Quest 2 tarayıcısında (%100 On-Device / Edge AI) sıfır sunucu gecikmesiyle çalıştırılmış, tanınan cümleler sanal gerçeklik içinde görselleştirilmiş ve Web Speech API ile seslendirilmiştir.

---

## 2. Giriş ve Literatür Taraması (Problem Tanımı & Özgün Değer)
### Problem Durumu:
- Türkiye'de yaklaşık 3 milyon işitme engelli birey bulunmaktadır.
- Acil servislerde, ambulanslarda ve afet anlarında işaret dili bilen tercüman bulmak çoğu zaman mümkün olmamaktadır.
- Mevcut mobil uygulamalar genellikle tek tek harf veya izole kelime tanımakta, sürekli (continuous) ve bağlamsal işaret dilini desteklememektedir.

### Projenin Özgünlüğü (İnovasyon Unsurları):
1. **Veri Eldivensiz ve Harici Kamerasız Mekansal Bilişim (Spatial Computing):** Kullanıcı ekstra hiçbir kablo, giyilebilir sensör veya harici kamera taşımak zorunda değildir. Quest 2'nin yerel kameraları kullanılır.
2. **3D Kinematik Derinlik:** 2D görüntülerde eller üst üste geldiğinde (occlusion) yaşanan takip kaybı, VR 3D eklem uzayında asgariye iner.
3. **Uçta Yapay Zekâ (On-Device Edge AI):** Veriler internete veya bulut sunucuya gönderilmez. Model doğrudan VR başlığının GPU'sunda (Snapdragon XR2) çalışır. Bu sayede **%100 veri gizliliği (KVKK/HIPAA uyumu)** ve internet kesintilerinde bile kesintisiz çalışma sağlanır.
4. **GitHub Pages ile Sıfır Kurulum Dağıtım:** Statik WebXR yapısı sayesinde proje GitHub Pages üzerinde barındırılabilir, kullanıcı Quest 2 tarayıcısından tek linkle anında sisteme erişebilir.

---

## 3. Yöntem ve Bilimsel Yaklaşım

### 3.1. Koordinat Yakalama ve Eklem Topolojisi
WebXR Hand Input API standardında her el için 21 temel eklem noktası seçilmiştir:
- Bilek ($P_0$)
- Başparmak ($P_1, P_2, P_3, P_4$)
- İşaret Parmağı ($P_5, P_6, P_7, P_8$)
- Orta Parmak ($P_9, P_{10}, P_{11}, P_{12}$)
- Yüzük Parmağı ($P_{13}, P_{14}, P_{15}, P_{16}$)
- Serçe Parmak ($P_{17}, P_{18}, P_{19}, P_{20}$)

Her bir karede iki el için $2 \times 21 \times 3 = 126$ öznitelikli tensör üretilir.

### 3.2. Matematiksel Koordinat Normalizasyonu
Kullanıcının odadaki konumu veya baş hareketleri ham koordinatları değiştireceğinden aşağıdaki iki aşamalı normalizasyon geliştirilmiştir:

1. **Bilek Referanslı Öteleme (Wrist-Centric Translation):**
   Her $i$ eklemi için bilek koordinatı ($P_{wrist}$) çıkarılır:
   $$\vec{P}'_i = \vec{P}_i - \vec{P}_{wrist}$$
   Böylece bilek $(0,0,0)$ orijinine taşınır.

2. **El Boyutu Ölçek Değişmezliği (Scale Invariance):**
   Kullanıcıların el boyut farkını yok etmek amacıyla, bilek ile orta parmak kök eklemi arasındaki Öklid mesafesi referans alınır:
   $$D_{ref} = \|\vec{P}_{middle\_mcp} - \vec{P}_{wrist}\|_2 = \sqrt{(x_{m} - x_{w})^2 + (y_{m} - y_{w})^2 + (z_{m} - z_{w})^2}$$
   Tüm bağıl eklem koordinatları bu değere bölünür:
   $$\vec{P}''_i = \frac{\vec{P}'_i}{D_{ref} + \epsilon}$$

### 3.3. Derin Öğrenme Mimarisi (BiLSTM)
- **Giriş Tensörü:** $(Batch, 45, 126)$ $\rightarrow$ 30 FPS'de 1.5 saniyelik kayan pencere.
- **Masking Katmanı:** Takip dışı kalan ellerin sıfır dolgularını (padding) ihmal eder.
- **BiLSTM Katmanı 1:** 64 Nöron, Dropout: 0.2 $\rightarrow$ İleri ve geri zamansal vektörleri birleştirir.
- **BiLSTM Katmanı 2:** 48 Nöron, Dropout: 0.2 $\rightarrow$ Üst düzey zamansal hareket şablonlarını soyutlar.
- **Dense & Batch Normalization:** 64 Nöron + ReLU aktivasyonu.
- **Softmax Çıkış Katmanı:** 8 Sınıflı olasılık dağılımı.

---

## 4. Sağlık Temalı Senaryolar ve Sözlük
1. `basim` (Başım - Şakak bölgesine dokunma)
2. `agriyor` (Ağrıyor - Acı/titreme hareketi)
3. `ambulans` (Ambulans - Siren/tepe lambası hareketi)
4. `cagirin` (Çağırın - Çağırma/gel jesti)
5. `yardim` (Yardım - El üstüne yumruk/destek hareketi)
6. `nefes` (Nefes - Göğüs kafesine dokunma)
7. `ilac` (İlaç - Avuç içine hap koyma hareketi)
8. `notr` (Nötr / Bekleme)

**Örnek Birleştirilen Cümleler:**
- `basim` + `agriyor` + `ambulans` + `cagirin` $\rightarrow$ *"Başım çok ağrıyor, lütfen hemen ambulans çağırın!"*
- `yardim` + `nefes` + `agriyor` $\rightarrow$ *"Yardım edin, nefes alamıyorum, göğsüm ağrıyor!"*
- `ilac` + `basim` + `agriyor` $\rightarrow$ *"Başım çok ağrıyor, lütfen ağrı kesici ilaç verin."*

---

## 5. TÜBİTAK Jürisi İçin Sıkça Sorulabilecek Sorular ve Yanıtlar

**Soru 1: Neden 2D görüntü işleme (OpenCV/YOLO) yerine WebXR Hand Input API kullandınız?**
> *Cevap:* 2D kameralarda Z ekseni (derinlik) bilgisi yoktur ve eller üst üste bindiğinde ya da kamera açısı değiştiğinde model başarımı ciddi oranda düşer. WebXR Hand Input API, Meta Quest 2'nin kızılötesi stereoskopik kameralarından doğrudan milimetrik 3D kartezyen koordinatlar sunar. Ayrıca arka plan ışıklandırmasından ve karmaşık ortamlardan etkilenmez.

**Soru 2: Neden standart LSTM yerine Çift Yönlü LSTM (BiLSTM) tercih ettiniz?**
> *Cevap:* İşaret dilindeki hareketler izole bir andan ibaret değildir; bir hareketin anlamı hem kendisinden önceki hazırlık evresinden (anticipation) hem de kendisinden sonraki toparlanma evresinden (recovery) etkilenir. BiLSTM, zaman serisini hem baştan sona hem sondan başa tarayarak çift yönlü bağlamı yakalar.

**Soru 3: Gözlük üzerinde yapay zekâ çalıştırmak pili ve performansı nasıl etkiliyor?**
> *Cevap:* Çıkarım işlemi her karede değil, kayan pencere mantığıyla 4 karede bir (yaklaşık 120 ms aralıkla) çalıştırılmaktadır. TensorFlow.js WebGL hızlandırması sayesinde çıkarım gecikmesi 15-25 ms civarındadır. Bu optimizasyon gözlüğün aşırı ısınmasını engeller ve akıcı 72/90 FPS VR deneyimini korur.

**Soru 4: Projenin toplumsal ve ekonomik faydası nedir?**
> *Cevap:* Pahalı yabancı veri eldivenlerine (10.000$+ maliyet) olan bağımlılığı ortadan kaldırmaktadır. Tüketici elektroniği sınıfındaki standart bir VR başlığıyla acil durumlarda işitme engelliler için hayat kurtarıcı bir iletişim köprüsü kurmaktadır.
