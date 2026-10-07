/**
 * TÜBİTAK 2204-A: VR Gerçek Zamanlı Sürekli İşaret Dili Tanıma Motoru (app.js)
 * Meta Quest 2 üzerinde WebGL/TensorFlow.js ile Edge AI çıkarımı,
 * Kayan Pencere (Sliding Window), Cümle Birleştirme ve Türkçe Seslendirme (TTS).
 */

const SELECTED_JOINTS = [
  "wrist",
  "thumb-metacarpal", "thumb-phalanx-proximal", "thumb-phalanx-distal", "thumb-tip",
  "index-finger-phalanx-proximal", "index-finger-phalanx-intermediate", "index-finger-phalanx-distal", "index-finger-tip",
  "middle-finger-phalanx-proximal", "middle-finger-phalanx-intermediate", "middle-finger-phalanx-distal", "middle-finger-tip",
  "ring-finger-phalanx-proximal", "ring-finger-phalanx-intermediate", "ring-finger-phalanx-distal", "ring-finger-tip",
  "pinky-finger-phalanx-proximal", "pinky-finger-phalanx-intermediate", "pinky-finger-phalanx-distal", "pinky-finger-tip"
];

// Varsayılan Sınıflar (model/classes.json yüklenemezse yedek)
let CLASSES = ["basim", "agriyor", "ambulans", "cagirin", "yardim", "nefes", "ilac", "notr"];
let CLASS_DISPLAY = {
  "basim": "Başım",
  "agriyor": "Ağrıyor",
  "ambulans": "Ambulans",
  "cagirin": "Çağırın",
  "yardim": "Yardım",
  "nefes": "Nefes",
  "ilac": "İlaç",
  "notr": "Nötr / Bekleme"
};

let SENTENCE_RULES = [
  {
    sequence: ["basim", "agriyor", "ambulans", "cagirin"],
    sentence: "Başım çok ağrıyor, lütfen hemen ambulans çağırın!"
  },
  {
    sequence: ["yardim", "nefes", "agriyor"],
    sentence: "Yardım edin, nefes alamıyorum, göğsüm ağrıyor!"
  },
  {
    sequence: ["ilac", "basim", "agriyor"],
    sentence: "Başım çok ağrıyor, lütfen ağrı kesici ilaç verin."
  },
  {
    sequence: ["yardim", "ambulans", "cagirin"],
    sentence: "Acil durum! Lütfen bir ambulans çağırın, yardım edin!"
  }
];

const CONFIG = {
  sequenceLength: 45,
  numFeatures: 126,
  inferenceIntervalFrames: 4, // Her 4 karede bir çıkarım (Quest 2 GPU optimizasyonu)
  confidenceThreshold: 0.80,
  debounceCount: 3
};

// Uygulama Durumu
const appState = {
  model: null,
  isModelLoaded: false,
  slidingBuffer: [], // FIFO boyutu: 45
  frameCount: 0,
  recognizedWords: [],
  currentSentence: "İşaret dili bekleniyor...",
  lastDetectedWord: null,
  consecutiveDetections: 0,
  lastInferenceMs: 0,
  vrHudCanvas: null,
  vrHudTexture: null,
  topPredictions: []
};

// Web Audio API
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function playChime() {
  try {
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.setValueAtTime(523.25, audioCtx.currentTime); // C5
    osc.frequency.exponentialRampToValueAtTime(659.25, audioCtx.currentTime + 0.1); // E5
    gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.25);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.25);
  } catch (e) {}
}

// Türkçe Metin Seslendirme (Text-to-Speech)
function speakTurkish(text) {
  if (!('speechSynthesis' in window)) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'tr-TR';
  utterance.rate = 1.0;
  window.speechSynthesis.speak(utterance);
}

// Three.js Değişkenleri
let scene, camera, renderer, hudMesh;
let leftJointSpheres = [], rightJointSpheres = [];

async function init() {
  initThree();
  await loadModelAndMetadata();
  setupUIEventListeners();
}

async function loadModelAndMetadata() {
  const modelStatusEl = document.getElementById('model-status');
  
  // 1. Meta Veriyi Yükle
  try {
    const metaRes = await fetch('./model/classes.json');
    if (metaRes.ok) {
      const meta = await metaRes.json();
      CLASSES = meta.classes || CLASSES;
      CLASS_DISPLAY = meta.display_names || CLASS_DISPLAY;
      if (meta.sentence_rules) SENTENCE_RULES = meta.sentence_rules;
      if (meta.sequence_length) CONFIG.sequenceLength = meta.sequence_length;
      console.log("[✓] Model meta verisi yüklendi:", CLASSES);
    }
  } catch (e) {
    console.warn("classes.json bulunamadı, varsayılan sınıflar kullanılıyor.");
  }

  // 2. TensorFlow.js Modelini Yükle
  try {
    modelStatusEl.innerHTML = '🧠 AI Modeli: <span style="color:#facc15">Yükleniyor (TF.js)...</span>';
    appState.model = await tf.loadLayersModel('./model/model.json');
    appState.isModelLoaded = true;
    modelStatusEl.innerHTML = '🧠 AI Modeli: <strong style="color:#34d399">Aktif (WebGL)</strong>';
    console.log("[✓] TensorFlow.js Modeli başarıyla yüklendi!");

    // Isınma (Warmup) çıkarımı
    const dummy = tf.zeros([1, CONFIG.sequenceLength, CONFIG.numFeatures]);
    appState.model.predict(dummy).dispose();
    dummy.dispose();
  } catch (err) {
    console.warn("[!] Model dosyası bulunamadı. train.py çalıştırıldıktan sonra model aktifleşecektir:", err);
    modelStatusEl.innerHTML = '🧠 AI Modeli: <span style="color:#f87171">Model Bekleniyor (train.py)</span>';
  }
}

function initThree() {
  const container = document.getElementById('canvas-container');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x090d16);

  camera = new THREE.PerspectiveCamera(70, container.clientWidth / container.clientHeight, 0.05, 50);
  camera.position.set(0, 1.4, 1.2);

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
  scene.add(ambientLight);
  const dirLight = new THREE.DirectionalLight(0x38bdf8, 0.8);
  dirLight.position.set(2, 4, 2);
  scene.add(dirLight);

  const grid = new THREE.GridHelper(10, 20, 0x0284c7, 0x1f2937);
  grid.position.y = 0;
  scene.add(grid);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.xr.enabled = true;
  container.appendChild(renderer.domElement);

  createVRButton();

  // El İskeleti Küreleri
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

  initVRHUD();

  window.addEventListener('resize', () => {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  });

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
        btn.style.padding = '12px 28px';
        btn.style.fontSize = '1.1rem';
        btn.style.boxShadow = '0 0 20px rgba(2, 132, 199, 0.6)';
        btn.onclick = () => {
          navigator.xr.requestSession('immersive-vr', {
            optionalFeatures: ['local-floor', 'bounded-floor', 'hand-tracking']
          }).then((session) => renderer.xr.setSession(session));
        };
        btnContainer.appendChild(btn);
      }
    });
  }
}

function initVRHUD() {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 512;
  appState.vrHudCanvas = canvas;
  appState.vrHudTexture = new THREE.CanvasTexture(canvas);

  const hudGeo = new THREE.PlaneGeometry(1.1, 0.55);
  const hudMat = new THREE.MeshBasicMaterial({
    map: appState.vrHudTexture,
    transparent: true,
    opacity: 0.94,
    side: THREE.DoubleSide
  });

  hudMesh = new THREE.Mesh(hudGeo, hudMat);
  hudMesh.position.set(0, 1.35, -0.95);
  scene.add(hudMesh);

  renderVRHUD();
}

function renderVRHUD() {
  const ctx = appState.vrHudCanvas.getContext('2d');
  ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
  ctx.fillRect(0, 0, 1024, 512);

  // Dış Çerçeve
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 8;
  ctx.strokeRect(8, 8, 1008, 496);

  // Başlık
  ctx.fillStyle = '#94a3b8';
  ctx.font = 'bold 28px sans-serif';
  ctx.fillText("TÜBİTAK 2204-A | GERÇEK ZAMANLI İŞARET DİLİ TERCÜMANI", 40, 55);

  // Anlık Cümle
  ctx.fillStyle = '#38bdf8';
  ctx.font = 'bold 46px sans-serif';
  const displaySentence = appState.currentSentence.length > 36 
    ? appState.currentSentence.substring(0, 36) + "..." 
    : appState.currentSentence;
  ctx.fillText(displaySentence, 40, 130);

  // Algılanan Kelimeler
  ctx.fillStyle = '#cbd5e1';
  ctx.font = '26px sans-serif';
  const wordsStr = appState.recognizedWords.length > 0 
    ? appState.recognizedWords.map(w => CLASS_DISPLAY[w] || w).join("  ➔  ")
    : "(Hareket bekleniyor...)";
  ctx.fillText(`Kelimeler: ${wordsStr}`, 40, 190);

  // Top Olasılık Barları (VR İçi Canlı Görselleştirme)
  ctx.fillStyle = '#64748b';
  ctx.font = 'bold 22px sans-serif';
  ctx.fillText("CANLI TAHMİN OLASILIKLARI:", 40, 250);

  const top = appState.topPredictions.slice(0, 3);
  top.forEach((pred, idx) => {
    const yPos = 290 + idx * 45;
    const name = CLASS_DISPLAY[pred.label] || pred.label;
    const pct = Math.round(pred.prob * 100);

    ctx.fillStyle = '#f8fafc';
    ctx.font = '24px sans-serif';
    ctx.fillText(`${name}:`, 40, yPos);

    // Bar Arka Plan
    ctx.fillStyle = '#334155';
    ctx.fillRect(200, yPos - 20, 450, 24);

    // Bar Dolgu
    ctx.fillStyle = pred.prob >= CONFIG.confidenceThreshold ? '#34d399' : '#38bdf8';
    ctx.fillRect(200, yPos - 20, 450 * pred.prob, 24);

    // Yüzde Yazısı
    ctx.fillStyle = '#f8fafc';
    ctx.fillText(`%${pct}`, 670, yPos);
  });

  // Alt Bilgi
  ctx.fillStyle = '#64748b';
  ctx.font = '20px sans-serif';
  ctx.fillText(`Gecikme: ${appState.lastInferenceMs}ms | Sol el pinch: Temizle`, 40, 470);

  appState.vrHudTexture.needsUpdate = true;
}

// Normalizasyon (Bilek Merkezli + El Boyutu)
function normalizeHandCoordinates(rawJoints) {
  if (!rawJoints || rawJoints.length !== 21) {
    return new Array(21 * 3).fill(0);
  }
  const wrist = rawJoints[0];
  const middleProximal = rawJoints[9];
  const scale = Math.sqrt(
    Math.pow(middleProximal.x - wrist.x, 2) +
    Math.pow(middleProximal.y - wrist.y, 2) +
    Math.pow(middleProximal.z - wrist.z, 2)
  ) || 0.1;

  const normalized = [];
  for (let i = 0; i < 21; i++) {
    const j = rawJoints[i];
    normalized.push((j.x - wrist.x) / scale);
    normalized.push((j.y - wrist.y) / scale);
    normalized.push((j.z - wrist.z) / scale);
  }
  return normalized;
}

// Ana Render ve WebXR Çıkarım Döngüsü
async function onXRFrame(time, frame) {
  if (frame) {
    const referenceSpace = renderer.xr.getReferenceSpace();
    const session = frame.session;

    let leftHandJoints = null;
    let rightHandJoints = null;

    for (const inputSource of session.inputSources) {
      if (inputSource.hand) {
        const handedness = inputSource.handedness;
        const jointsData = [];

        SELECTED_JOINTS.forEach((jointName, index) => {
          const jointSpace = inputSource.hand.get(jointName);
          if (jointSpace) {
            const jointPose = frame.getJointPose(jointSpace, referenceSpace);
            if (jointPose) {
              const pos = jointPose.transform.position;
              jointsData.push({ x: pos.x, y: pos.y, z: pos.z });

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

    // VR İçi Sol El Pinch = Cümleyi Temizle
    if (leftHandJoints) {
      const thumbTip = leftHandJoints[4];
      const indexTip = leftHandJoints[8];
      const dist = Math.sqrt(
        Math.pow(thumbTip.x - indexTip.x, 2) +
        Math.pow(thumbTip.y - indexTip.y, 2) +
        Math.pow(thumbTip.z - indexTip.z, 2)
      );
      if (dist < 0.025) {
        clearSentence();
      }
    }

    // Kare Normalizasyonu ve Kayan Pencereye (Buffer) Ekleme
    const normLeft = normalizeHandCoordinates(leftHandJoints);
    const normRight = normalizeHandCoordinates(rightHandJoints);
    const frameFeatures = [...normLeft, ...normRight];

    appState.slidingBuffer.push(frameFeatures);
    if (appState.slidingBuffer.length > CONFIG.sequenceLength) {
      appState.slidingBuffer.shift();
    }

    // Çıkarım Zamanlaması
    appState.frameCount++;
    if (
      appState.isModelLoaded &&
      appState.slidingBuffer.length === CONFIG.sequenceLength &&
      appState.frameCount % CONFIG.inferenceIntervalFrames === 0
    ) {
      await runInference();
    }
  }

  renderer.render(scene, camera);
}

// TensorFlow.js Çıkarımı ve Sürekli İşaret Tanıma Durum Makinesi
async function runInference() {
  const t0 = performance.now();

  try {
    const inputTensor = tf.tensor3d([appState.slidingBuffer], [1, CONFIG.sequenceLength, CONFIG.numFeatures]);
    const prediction = appState.model.predict(inputTensor);
    const probabilities = await prediction.data();

    inputTensor.dispose();
    prediction.dispose();

    appState.lastInferenceMs = Math.round(performance.now() - t0);
    document.getElementById('fps-status').textContent = `⚡ Çıkarım Gecikmesi: ${appState.lastInferenceMs} ms`;

    // Olasılıkları Sırala
    const predList = [];
    for (let i = 0; i < CLASSES.length; i++) {
      predList.push({ label: CLASSES[i], prob: probabilities[i] });
    }
    predList.sort((a, b) => b.prob - a.prob);
    appState.topPredictions = predList;

    updateProbabilityUI(predList);

    // Durum Makinesi (State Machine & Debouncing)
    const top = predList[0];
    if (top.label !== 'notr' && top.prob >= CONFIG.confidenceThreshold) {
      if (top.label === appState.lastDetectedWord) {
        appState.consecutiveDetections++;
      } else {
        appState.lastDetectedWord = top.label;
        appState.consecutiveDetections = 1;
      }

      // Eşik ve Ardışık Doğrulama Sağlandıysa Kelimeyi Onayla
      if (appState.consecutiveDetections === CONFIG.debounceCount) {
        onWordRecognized(top.label);
      }
    } else {
      // Nötr durumunda sayacı sıfırla
      if (top.label === 'notr') {
        appState.lastDetectedWord = null;
        appState.consecutiveDetections = 0;
      }
    }

    renderVRHUD();
  } catch (err) {
    console.error("Çıkarım hatası:", err);
  }
}

// Yeni Bir Kelime Algılandığında Cümle Kuralları & TTS
function onWordRecognized(wordLabel) {
  playChime();

  // Son algılanan kelime ile aynı değilse ekle
  const lastWord = appState.recognizedWords[appState.recognizedWords.length - 1];
  if (lastWord !== wordLabel) {
    appState.recognizedWords.push(wordLabel);
  }

  // Cümle Kuralı Kontrolü
  let matchedSentence = null;
  const currentWordsStr = appState.recognizedWords.join(",");

  for (const rule of SENTENCE_RULES) {
    const ruleStr = rule.sequence.join(",");
    if (currentWordsStr.includes(ruleStr)) {
      matchedSentence = rule.sentence;
      break;
    }
  }

  if (matchedSentence) {
    appState.currentSentence = matchedSentence;
  } else {
    // Kural eşleşmezse ardışık kelimelerden Türkçe cümle oluştur
    const turkishWords = appState.recognizedWords.map(w => CLASS_DISPLAY[w] || w);
    appState.currentSentence = turkishWords.join(" ") + "...";
  }

  updateSentenceUI();
  renderVRHUD();

  // Otomatik Sesli Okuma
  const autoTTS = document.getElementById('chk-auto-tts').checked;
  if (autoTTS) {
    speakTurkish(matchedSentence || CLASS_DISPLAY[wordLabel] || wordLabel);
  }
}

function updateSentenceUI() {
  document.getElementById('translated-sentence').textContent = appState.currentSentence;
  
  const wordsContainer = document.getElementById('words-tag-container');
  wordsContainer.innerHTML = '';
  appState.recognizedWords.forEach(w => {
    const span = document.createElement('span');
    span.className = 'word-tag';
    span.textContent = CLASS_DISPLAY[w] || w;
    wordsContainer.appendChild(span);
  });
}

function updateProbabilityUI(preds) {
  const container = document.getElementById('prob-bars-container');
  container.innerHTML = '';

  preds.slice(0, 4).forEach(p => {
    const row = document.createElement('div');
    row.className = 'prob-bar-container';

    const pct = Math.round(p.prob * 100);
    const isHigh = p.prob >= CONFIG.confidenceThreshold;

    row.innerHTML = `
      <div class="prob-label-row">
        <span>${CLASS_DISPLAY[p.label] || p.label}</span>
        <span style="font-weight:700; color:${isHigh ? '#34d399' : '#94a3b8'}">%${pct}</span>
      </div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill ${isHigh ? 'high' : ''}" style="width: ${pct}%"></div>
      </div>
    `;
    container.appendChild(row);
  });
}

function clearSentence() {
  appState.recognizedWords = [];
  appState.currentSentence = "İşaret dili bekleniyor...";
  appState.lastDetectedWord = null;
  appState.consecutiveDetections = 0;
  updateSentenceUI();
  renderVRHUD();
}

function setupUIEventListeners() {
  document.getElementById('btn-speak').addEventListener('click', () => {
    speakTurkish(appState.currentSentence);
  });

  document.getElementById('btn-clear').addEventListener('click', clearSentence);

  const threshSlider = document.getElementById('thresh-slider');
  threshSlider.addEventListener('input', (e) => {
    CONFIG.confidenceThreshold = parseFloat(e.target.value);
    document.getElementById('thresh-val').textContent = CONFIG.confidenceThreshold.toFixed(2);
  });

  // Jüri & Gözlüksüz Demo Simülasyonları
  const simAmbulanceBtn = document.getElementById('btn-sim-ambulance');
  if (simAmbulanceBtn) {
    simAmbulanceBtn.addEventListener('click', () => {
      clearSentence();
      const sequence = ["basim", "agriyor", "ambulans", "cagirin"];
      sequence.forEach((word, idx) => {
        setTimeout(() => {
          onWordRecognized(word);
        }, (idx + 1) * 800);
      });
    });
  }

  const simHelpBtn = document.getElementById('btn-sim-help');
  if (simHelpBtn) {
    simHelpBtn.addEventListener('click', () => {
      clearSentence();
      const sequence = ["yardim", "nefes", "agriyor"];
      sequence.forEach((word, idx) => {
        setTimeout(() => {
          onWordRecognized(word);
        }, (idx + 1) * 800);
      });
    });
  }
}

window.addEventListener('DOMContentLoaded', init);
