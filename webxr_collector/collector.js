/**
 * TÜBİTAK 2204-A: WebXR El Takibi Veri Toplayıcı (collector.js)
 * Meta Quest 2 WebXR Hand Input API ile 30 FPS el kinematik verisi toplama,
 * bilek normalizasyonu ve otomatik kaydetme motoru.
 */

// 21 Standart Eklem İsimleri (WebXR XRHand -> 21 Eklem Eşleşmesi)
const SELECTED_JOINTS = [
  "wrist",
  // Başparmak (4)
  "thumb-metacarpal", "thumb-phalanx-proximal", "thumb-phalanx-distal", "thumb-tip",
  // İşaret Parmağı (4)
  "index-finger-phalanx-proximal", "index-finger-phalanx-intermediate", "index-finger-phalanx-distal", "index-finger-tip",
  // Orta Parmak (4)
  "middle-finger-phalanx-proximal", "middle-finger-phalanx-intermediate", "middle-finger-phalanx-distal", "middle-finger-tip",
  // Yüzük Parmağı (4)
  "ring-finger-phalanx-proximal", "ring-finger-phalanx-intermediate", "ring-finger-phalanx-distal", "ring-finger-tip",
  // Serçe Parmak (4)
  "pinky-finger-phalanx-proximal", "pinky-finger-phalanx-intermediate", "pinky-finger-phalanx-distal", "pinky-finger-tip"
];

// Uygulama Durumu (State)
const state = {
  selectedLabel: "basim",
  targetFrames: 45, // 1.5 saniye @ 30 FPS
  isRecording: false,
  isCountingDown: false,
  recordedSamples: [],
  currentSampleFrames: [],
  lastFrameTime: 0,
  frameInterval: 1000 / 30, // 33.33ms (30 FPS hedefi)
  stats: {},
  vrHudCanvas: null,
  vrHudTexture: null,
  lastPinchTime: 0
};

// Web Audio API ile VR içi ses efektleri
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function playBeep(freq = 440, duration = 0.1, type = 'sine') {
  try {
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {
    console.warn("Ses çalınamadı:", e);
  }
}

// Three.js Değişkenleri
let scene, camera, renderer, hudMesh;
let leftJointSpheres = [], rightJointSpheres = [];

function initThree() {
  const container = document.getElementById('canvas-container');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f172a);

  camera = new THREE.PerspectiveCamera(70, container.clientWidth / container.clientHeight, 0.05, 50);
  camera.position.set(0, 1.4, 1.2);

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambientLight);
  const dirLight = new THREE.DirectionalLight(0x38bdf8, 0.8);
  dirLight.position.set(2, 4, 2);
  scene.add(dirLight);

  // Zemin Izgarası
  const grid = new THREE.GridHelper(10, 20, 0x0284c7, 0x334155);
  grid.position.y = 0;
  scene.add(grid);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.xr.enabled = true;
  container.appendChild(renderer.domElement);

  // VR Butonu Oluşturma
  createVRButton();

  // El Görselleştirme Küreleri (21 adet sol, 21 adet sağ)
  const leftMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.3 });
  const rightMat = new THREE.MeshStandardMaterial({ color: 0x34d399, roughness: 0.3 });
  const sphereGeo = new THREE.SphereGeometry(0.008, 8, 8);

  for (let i = 0; i < 21; i++) {
    const lSphere = new THREE.Mesh(sphereGeo, leftMat);
    lSphere.visible = false;
    scene.add(lSphere);
    leftJointSpheres.push(lSphere);

    const rSphere = new THREE.Mesh(sphereGeo, rightMat);
    rSphere.visible = false;
    scene.add(rSphere);
    rightJointSpheres.push(rSphere);
  }

  // VR Havada Asılı HUD Paneli
  initVRHUD();

  window.addEventListener('resize', onWindowResize);
  renderer.setAnimationLoop(onXRFrame);
}

function createVRButton() {
  const btnContainer = document.getElementById('vr-btn-container');
  if ('xr' in navigator) {
    navigator.xr.isSessionSupported('immersive-vr').then((supported) => {
      if (supported) {
        const btn = document.createElement('button');
        btn.textContent = '👓 VR BAŞLAT (Meta Quest 2)';
        btn.className = 'btn-primary';
        btn.style.padding = '12px 24px';
        btn.style.fontSize = '1.1rem';
        btn.style.boxShadow = '0 0 15px rgba(2, 132, 199, 0.6)';
        btn.onclick = () => {
          navigator.xr.requestSession('immersive-vr', {
            optionalFeatures: ['local-floor', 'bounded-floor', 'hand-tracking']
          }).then(onSessionStarted);
        };
        btnContainer.appendChild(btn);
        document.getElementById('connection-status').innerHTML = '🔌 WebXR: <strong style="color:#34d399">VR Destekleniyor</strong>';
      } else {
        document.getElementById('connection-status').innerHTML = '🔌 WebXR: <span style="color:#f87171">VR Desteklenmiyor (Tarayıcı Simülasyonu)</span>';
      }
    });
  }
}

function onSessionStarted(session) {
  renderer.xr.setSession(session);
  document.getElementById('connection-status').innerHTML = '🔌 WebXR: <strong style="color:#38bdf8">VR Oturumu Aktif</strong>';
  session.addEventListener('end', () => {
    document.getElementById('connection-status').innerHTML = '🔌 WebXR: <span>VR Oturumu Kapandı</span>';
  });
}

function initVRHUD() {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 512;
  state.vrHudCanvas = canvas;
  state.vrHudTexture = new THREE.CanvasTexture(canvas);

  const hudGeo = new THREE.PlaneGeometry(1.0, 0.5);
  const hudMat = new THREE.MeshBasicMaterial({
    map: state.vrHudTexture,
    transparent: true,
    opacity: 0.92,
    side: THREE.DoubleSide
  });

  hudMesh = new THREE.Mesh(hudGeo, hudMat);
  hudMesh.position.set(0, 1.35, -0.9);
  scene.add(hudMesh);

  updateVRHUD("HAZIR", "Kayıt için Pinch yapın veya butona tıklayın");
}

function updateVRHUD(mainText, subText, extra = "") {
  const ctx = state.vrHudCanvas.getContext('2d');
  ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
  ctx.fillRect(0, 0, 1024, 512);

  // Dış Çerçeve
  ctx.strokeStyle = state.isRecording ? '#ef4444' : '#38bdf8';
  ctx.lineWidth = 10;
  ctx.strokeRect(10, 10, 1004, 492);

  // Başlık
  ctx.fillStyle = '#94a3b8';
  ctx.font = 'bold 32px sans-serif';
  ctx.fillText("TÜBİTAK 2204-A: VR İŞARET DİLİ VERİ TOPLAYICI", 50, 70);

  // Mevcut Etiket
  ctx.fillStyle = '#f8fafc';
  ctx.font = 'bold 44px sans-serif';
  ctx.fillText(`Seçili İşaret: [ ${state.selectedLabel.toUpperCase()} ]`, 50, 140);

  // Ana Durum Mesajı
  ctx.fillStyle = state.isRecording ? '#f87171' : (state.isCountingDown ? '#facc15' : '#38bdf8');
  ctx.font = 'bold 64px sans-serif';
  ctx.fillText(mainText, 50, 240);

  // Alt Açıklama
  ctx.fillStyle = '#cbd5e1';
  ctx.font = '32px sans-serif';
  ctx.fillText(subText, 50, 310);

  // Sayaç ve Detaylar
  ctx.fillStyle = '#34d399';
  ctx.font = '28px sans-serif';
  const count = state.stats[state.selectedLabel] || 0;
  ctx.fillText(`Bu İşaret İçin Kayıt: ${count} adet | Toplam: ${state.recordedSamples.length} | ${extra}`, 50, 420);

  state.vrHudTexture.needsUpdate = true;
}

// Normalizasyon Fonksiyonu: Bilek Merkezli & Ölçek Değişmezliği
function normalizeHandCoordinates(rawJoints) {
  // rawJoints: [{ x, y, z }, ... 21 adet]
  if (!rawJoints || rawJoints.length !== 21) {
    return new Array(21 * 3).fill(0);
  }

  const wrist = rawJoints[0];
  // Orta parmak proximal eklemi (indeks 9: "middle-finger-phalanx-proximal")
  const middleProximal = rawJoints[9];
  
  // El boyutu referans mesafesi (Scale Invariance)
  const scale = Math.sqrt(
    Math.pow(middleProximal.x - wrist.x, 2) +
    Math.pow(middleProximal.y - wrist.y, 2) +
    Math.pow(middleProximal.z - wrist.z, 2)
  ) || 0.1; // sıfıra bölünmeyi engellemek için epsilon 0.1

  const normalized = [];
  for (let i = 0; i < 21; i++) {
    const joint = rawJoints[i];
    // Bileğe göre öteleme ve el boyutuna bölme
    normalized.push((joint.x - wrist.x) / scale);
    normalized.push((joint.y - wrist.y) / scale);
    normalized.push((joint.z - wrist.z) / scale);
  }
  return normalized;
}

// WebXR Döngüsü (Her Render Karesi)
function onXRFrame(time, frame) {
  if (frame) {
    const referenceSpace = renderer.xr.getReferenceSpace();
    const session = frame.session;

    let leftHandJoints = null;
    let rightHandJoints = null;

    // VR Gözlük Takip Edilen Giriş Kaynakları
    for (const inputSource of session.inputSources) {
      if (inputSource.hand) {
        const handedness = inputSource.handedness; // 'left' veya 'right'
        const jointsData = [];

        SELECTED_JOINTS.forEach((jointName, index) => {
          const jointSpace = inputSource.hand.get(jointName);
          if (jointSpace) {
            const jointPose = frame.getJointPose(jointSpace, referenceSpace);
            if (jointPose) {
              const pos = jointPose.transform.position;
              jointsData.push({ x: pos.x, y: pos.y, z: pos.z });

              // 3D Küreleri Güncelle
              const spheres = handedness === 'left' ? leftJointSpheres : rightJointSpheres;
              if (spheres[index]) {
                spheres[index].position.set(pos.x, pos.y, pos.z);
                spheres[index].visible = true;
              }
            }
          }
        });

        if (jointsData.length === 21) {
          if (handedness === 'left') leftHandJoints = jointsData;
          if (handedness === 'right') rightHandJoints = jointsData;
        }
      }
    }

    // UI Güncelleme (El Görünürlüğü)
    document.getElementById('left-hand-state').textContent = leftHandJoints ? 'Takipte' : 'Yok';
    document.getElementById('right-hand-state').textContent = rightHandJoints ? 'Takipte' : 'Yok';
    document.getElementById('left-hand-state').style.color = leftHandJoints ? '#34d399' : '#f87171';
    document.getElementById('right-hand-state').style.color = rightHandJoints ? '#34d399' : '#f87171';

    // VR İçi Pinch Tetikleyici Kontrolü (Sağ el baş ve işaret parmağı ucu)
    if (rightHandJoints && !state.isRecording && !state.isCountingDown) {
      const thumbTip = rightHandJoints[4];
      const indexTip = rightHandJoints[8];
      const dist = Math.sqrt(
        Math.pow(thumbTip.x - indexTip.x, 2) +
        Math.pow(thumbTip.y - indexTip.y, 2) +
        Math.pow(thumbTip.z - indexTip.z, 2)
      );
      if (dist < 0.025 && (Date.now() - state.lastPinchTime > 2500)) {
        state.lastPinchTime = Date.now();
        playBeep(880, 0.15, 'triangle');
        startCountdown();
      }
    }

    // Kayıt Aktif İse Örnekleme (30 FPS Sınırı)
    if (state.isRecording && (time - state.lastFrameTime >= state.frameInterval)) {
      state.lastFrameTime = time;

      // Sol ve Sağ el koordinatlarını normalize et
      const normLeft = normalizeHandCoordinates(leftHandJoints);
      const normRight = normalizeHandCoordinates(rightHandJoints);

      // İki el bilek bağıl mesafesi
      let interWristVec = [0, 0, 0];
      if (leftHandJoints && rightHandJoints) {
        interWristVec = [
          rightHandJoints[0].x - leftHandJoints[0].x,
          rightHandJoints[0].y - leftHandJoints[0].y,
          rightHandJoints[0].z - leftHandJoints[0].z
        ];
      }

      // Kare Tensör Vektörü: 63 (Sol) + 63 (Sağ) = 126 Öznitelik
      const frameFeatures = [...normLeft, ...normRight];

      state.currentSampleFrames.push({
        t: state.currentSampleFrames.length,
        features: frameFeatures,
        interWrist: interWristVec,
        hasLeft: !!leftHandJoints,
        hasRight: !!rightHandJoints
      });

      const progress = Math.round((state.currentSampleFrames.length / state.targetFrames) * 100);
      updateVRHUD("🔴 KAYDEDİLİYOR...", `İlerleme: %${progress} (${state.currentSampleFrames.length}/${state.targetFrames} Kare)`);

      if (state.currentSampleFrames.length >= state.targetFrames) {
        finishRecording();
      }
    }
  }

  renderer.render(scene, camera);
}

function startCountdown() {
  if (state.isRecording || state.isCountingDown) return;
  state.isCountingDown = true;
  let counter = 3;

  const overlay = document.getElementById('countdown-overlay');
  overlay.style.display = 'block';

  function tick() {
    if (counter > 0) {
      overlay.textContent = counter;
      playBeep(440 + (3 - counter) * 110, 0.12);
      updateVRHUD(`⏳ GERİ SAYIM: ${counter}`, "Ellerinizi başlangıç pozisyonuna getirin!");
      counter--;
      setTimeout(tick, 1000);
    } else {
      overlay.style.display = 'none';
      state.isCountingDown = false;
      startRecording();
    }
  }
  tick();
}

function startRecording() {
  state.isRecording = true;
  state.currentSampleFrames = [];
  playBeep(880, 0.25, 'sine');
  
  const btn = document.getElementById('btn-trigger-record');
  btn.classList.add('recording');
  btn.textContent = '🔴 KAYIT YAPILIYOR...';
  updateVRHUD("🔴 KAYIT BAŞLADI!", "İşareti akıcı bir şekilde tamamlayın...");
}

async function finishRecording() {
  state.isRecording = false;
  playBeep(587.33, 0.15);
  setTimeout(() => playBeep(880, 0.25), 160);

  const btn = document.getElementById('btn-trigger-record');
  btn.classList.remove('recording');
  btn.textContent = '🔴 Kaydı Başlat (3 sn Geri Sayım)';

  const sampleData = {
    label: state.selectedLabel,
    frameCount: state.currentSampleFrames.length,
    featuresPerFrame: 126,
    timestamp: Date.now(),
    frames: state.currentSampleFrames
  };

  state.recordedSamples.push(sampleData);
  state.stats[state.selectedLabel] = (state.stats[state.selectedLabel] || 0) + 1;
  saveToLocalStorage();
  updateStatsUI();

  updateVRHUD("✅ KAYDEDİLDİ!", `İşaret: ${state.selectedLabel} | Sunucuya Gönderiliyor...`);

  // Yerel Sunucuya (server.py) Otomatik Kaydet
  try {
    const res = await fetch('/api/save_sample', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(sampleData)
    });
    if (res.ok) {
      updateVRHUD("✅ BAŞARIYLA KAYDEDİLDİ", "Veri diske yazıldı. Sıradaki kayıt için hazır.");
    } else {
      updateVRHUD("⚠️ YERELDE SAKLANDI", "Sunucuya ulaşılamadı, tarayıcıda tutuluyor.");
    }
  } catch (err) {
    console.warn("Sunucu kaydetme hatası (yerel hafızada tutuldu):", err);
    updateVRHUD("⚠️ YERELDE SAKLANDI", "Veri tarayıcı önbelleğine kaydedildi.");
  }
}

function saveToLocalStorage() {
  try {
    localStorage.setItem('vrisaret_stats', JSON.stringify(state.stats));
  } catch (e) {}
}

function loadFromLocalStorage() {
  try {
    const s = localStorage.getItem('vrisaret_stats');
    if (s) state.stats = JSON.parse(s);
  } catch (e) {}
}

function updateStatsUI() {
  const total = Object.values(state.stats).reduce((a, b) => a + b, 0);
  document.getElementById('stat-total').textContent = total;
  document.getElementById('stat-current').textContent = state.stats[state.selectedLabel] || 0;
}

function onWindowResize() {
  const container = document.getElementById('canvas-container');
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}

// UI Olay Dinleyicileri
document.getElementById('sign-select').addEventListener('change', (e) => {
  state.selectedLabel = e.target.value;
  updateStatsUI();
  updateVRHUD("HAZIR", `Seçili İşaret: ${state.selectedLabel}`);
});

document.getElementById('sample-duration').addEventListener('change', (e) => {
  state.targetFrames = parseInt(e.target.value);
});

document.getElementById('btn-trigger-record').addEventListener('click', () => {
  startCountdown();
});

document.getElementById('btn-download-all').addEventListener('click', () => {
  if (state.recordedSamples.length === 0) {
    alert("Henüz kaydedilmiş örnek bulunmamaktadır!");
    return;
  }
  const blob = new Blob([JSON.stringify(state.recordedSamples, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `vrisaret_dataset_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

document.getElementById('btn-sync-server').addEventListener('click', async () => {
  try {
    const res = await fetch('/api/stats');
    if (res.ok) {
      const data = await res.json();
      state.stats = data.classes || {};
      updateStatsUI();
      alert(`Sunucu ile senkronize edildi! Sunucuda toplam ${data.total_samples} örnek var.`);
    }
  } catch (e) {
    alert("Sunucuya bağlanılamadı. server.py çalışıyor mu?");
  }
});

document.getElementById('btn-clear-local').addEventListener('click', () => {
  if (confirm("Yerel kayıt sayacını sıfırlamak istediğinize emin misiniz? (Diskteki dosyalar silinmez)")) {
    state.recordedSamples = [];
    state.stats = {};
    localStorage.removeItem('vrisaret_stats');
    updateStatsUI();
  }
});

// Başlatma
loadFromLocalStorage();
updateStatsUI();
initThree();
