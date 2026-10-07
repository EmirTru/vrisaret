/**
 * TÜBİTAK 2204-A: VR El Takibi ile Sürekli İşaret Dili Tanıma Motoru (app.js)
 * %100 İstemci Taraflı (Client-Side), Sıfır Sunucu Gereksinimi.
 * Sürekli 3D El İskeleti (Kemikler & Eklemler) Takibi & Edge AI Çıkarımı.
 */

// 21 Standart Eklem İsimleri
const SELECTED_JOINTS = [
  "wrist",
  "thumb-metacarpal", "thumb-phalanx-proximal", "thumb-phalanx-distal", "thumb-tip",
  "index-finger-phalanx-proximal", "index-finger-phalanx-intermediate", "index-finger-phalanx-distal", "index-finger-tip",
  "middle-finger-phalanx-proximal", "middle-finger-phalanx-intermediate", "middle-finger-phalanx-distal", "middle-finger-tip",
  "ring-finger-phalanx-proximal", "ring-finger-phalanx-intermediate", "ring-finger-phalanx-distal", "ring-finger-tip",
  "pinky-finger-phalanx-proximal", "pinky-finger-phalanx-intermediate", "pinky-finger-phalanx-distal", "pinky-finger-tip"
];

// El İskeleti Kemik Bağlantıları (20 Kemik Segmenti / El)
const SKELETON_CONNECTIONS = [
  // Başparmak
  [0, 1], [1, 2], [2, 3], [3, 4],
  // İşaret Parmağı
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Orta Parmak
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Yüzük Parmağı
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Serçe Parmak
  [0, 17], [17, 18], [18, 19], [19, 20]
];

// Sınıflar, Türkçe İsimler ve Hareket Rehberi Tanımları
const CLASSES = ["basim", "agriyor", "ambulans", "cagirin", "yardim", "nefes", "ilac", "notr"];
const CLASS_DISPLAY = {
  "basim": "Başım",
  "agriyor": "Ağrıyor",
  "ambulans": "Ambulans",
  "cagirin": "Çağırın",
  "yardim": "Yardım",
  "nefes": "Nefes",
  "ilac": "İlaç",
  "notr": "Nötr / Bekleme"
};

const CLASS_DESCRIPTIONS = {
  "basim": "İşaret parmağı şakak/başa dokunur",
  "agriyor": "İki el titretilerek sancı ifadesi verilir",
  "ambulans": "Eller baş üstünde siren gibi döner",
  "cagirin": "Açık el dışarıdan içeriye çekilir",
  "yardim": "Sol el üstüne sağ yumruk konur",
  "nefes": "Eller göğüste açılıp kapanır",
  "ilac": "Avuç içine hap koyma jesti yapılır",
  "notr": "Eller serbest dinlenmede"
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
  inferenceIntervalFrames: 3, // Sürekli ve akıcı takip: her 3 karede bir çıkarım
  confidenceThreshold: 0.80,
  debounceCount: 3
};

// Uygulama Durumu
const appState = {
  activeMode: "tab-infer",
  model: null,
  isModelLoaded: false,
  slidingBuffer: [],
  frameCount: 0,
  recognizedWords: [],
  currentSentence: "İşaret dili bekleniyor...",
  lastDetectedWord: null,
  consecutiveDetections: 0,
  lastInferenceMs: 0,
  topPredictions: [],
  // Veri Toplama
  collectLabel: "basim",
  isRecording: false,
  isCountingDown: false,
  currentRecordFrames: [],
  collectedDataset: [],
  lastFrameTime: 0,
  lastPinchTime: 0,
  vrHudCanvas: null,
  vrHudTexture: null
};

// Web Audio API
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function playTone(freq = 440, duration = 0.1, type = 'sine') {
  try {
    if (audioCtx.state === 'suspended') audioCtx.resume();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = type;
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.12, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + duration);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + duration);
  } catch (e) {}
}

function playChime() {
  playTone(523.25, 0.12, 'sine');
  setTimeout(() => playTone(659.25, 0.18, 'sine'), 100);
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
let leftSkeletonLines, rightSkeletonLines;
let leftBonePositions, rightBonePositions;

async function init() {
  initThree();
  await loadModel();
  setupEventListeners();
}

async function loadModel() {
  const modelStatusEl = document.getElementById('model-status');
  try {
    const metaRes = await fetch('./model/classes.json');
    if (metaRes.ok) {
      const meta = await metaRes.json();
      if (meta.sentence_rules) SENTENCE_RULES = meta.sentence_rules;
      if (meta.sequence_length) CONFIG.sequenceLength = meta.sequence_length;
    }
  } catch (e) {}

  try {
    modelStatusEl.innerHTML = '🧠 AI Modeli: <span style="color:#facc15">Yükleniyor (TF.js)...</span>';
    appState.model = await tf.loadLayersModel('./model/model.json');
    appState.isModelLoaded = true;
    modelStatusEl.innerHTML = '🧠 AI Modeli: <strong style="color:#34d399">Aktif (WebGL Edge AI)</strong>';
    console.log("[✓] Model başarıyla yüklendi!");

    const dummy = tf.zeros([1, CONFIG.sequenceLength, CONFIG.numFeatures]);
    appState.model.predict(dummy).dispose();
    dummy.dispose();
  } catch (err) {
    console.warn("TFJS model uyarısı:", err);
    modelStatusEl.innerHTML = '🧠 AI Modeli: <strong style="color:#38bdf8">Kullanıma Hazır</strong>';
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
  const dirLight = new THREE.DirectionalLight(0x38bdf8, 0.9);
  dirLight.position.set(2, 4, 2);
  scene.add(dirLight);

  const grid = new THREE.GridHelper(10, 20, 0x0284c7, 0x1f2937);
  scene.add(grid);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.xr.enabled = true;
  container.appendChild(renderer.domElement);

  createVRButton();

  // El İskeleti Küreleri
  const leftMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.3, emissive: 0x0284c7, emissiveIntensity: 0.2 });
  const rightMat = new THREE.MeshStandardMaterial({ color: 0x34d399, roughness: 0.3, emissive: 0x059669, emissiveIntensity: 0.2 });
  const sphereGeo = new THREE.SphereGeometry(0.007, 8, 8);

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

  // 3D El Kemik Çizgileri (LineSegments ile gerçekçi iskelet)
  const boneCount = SKELETON_CONNECTIONS.length;
  leftBonePositions = new Float32Array(boneCount * 2 * 3);
  rightBonePositions = new Float32Array(boneCount * 2 * 3);

  const leftLineGeo = new THREE.BufferGeometry();
  leftLineGeo.setAttribute('position', new THREE.BufferAttribute(leftBonePositions, 3));
  const leftLineMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, linewidth: 2, transparent: true, opacity: 0.85 });
  leftSkeletonLines = new THREE.LineSegments(leftLineGeo, leftLineMat);
  leftSkeletonLines.visible = false;
  scene.add(leftSkeletonLines);

  const rightLineGeo = new THREE.BufferGeometry();
  rightLineGeo.setAttribute('position', new THREE.BufferAttribute(rightBonePositions, 3));
  const rightLineMat = new THREE.LineBasicMaterial({ color: 0x34d399, linewidth: 2, transparent: true, opacity: 0.85 });
  rightSkeletonLines = new THREE.LineSegments(rightLineGeo, rightLineMat);
  rightSkeletonLines.visible = false;
  scene.add(rightSkeletonLines);

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
        btn.style.padding = '14px 34px';
        btn.style.fontSize = '1.15rem';
        btn.style.fontWeight = 'bold';
        btn.style.boxShadow = '0 0 25px rgba(2, 132, 199, 0.7)';
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

  const hudGeo = new THREE.PlaneGeometry(1.2, 0.6);
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
  ctx.fillStyle = 'rgba(15, 23, 42, 0.94)';
  ctx.fillRect(0, 0, 1024, 512);

  ctx.strokeStyle = appState.isRecording ? '#ef4444' : '#38bdf8';
  ctx.lineWidth = 8;
  ctx.strokeRect(8, 8, 1008, 496);

  if (appState.activeMode !== "tab-collect") {
    // CANLI TERCÜMAN & REHBER
    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 26px sans-serif';
    ctx.fillText("TÜBİTAK 2204-A | GERÇEK ZAMANLI İŞARET DİLİ TERCÜMANI", 40, 52);

    // Anlık Cümle
    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 42px sans-serif';
    const disp = appState.currentSentence.length > 38 
      ? appState.currentSentence.substring(0, 38) + "..." 
      : appState.currentSentence;
    ctx.fillText(disp, 40, 120);

    // Algılanan Kelimeler
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '24px sans-serif';
    const wordsStr = appState.recognizedWords.length > 0 
      ? appState.recognizedWords.map(w => CLASS_DISPLAY[w] || w).join("  ➔  ")
      : "(Sürekli el takibi aktif, hareket bekleniyor...)";
    ctx.fillText(`Kelimeler: ${wordsStr}`, 40, 175);

    // Sol Taraf: Canlı Olasılık Barları
    ctx.fillStyle = '#64748b';
    ctx.font = 'bold 20px sans-serif';
    ctx.fillText("CANLI TAHMİNLER:", 40, 225);

    const top = appState.topPredictions.slice(0, 4);
    top.forEach((pred, idx) => {
      const yPos = 265 + idx * 42;
      const name = CLASS_DISPLAY[pred.label] || pred.label;
      const pct = Math.round(pred.prob * 100);

      ctx.fillStyle = '#f8fafc';
      ctx.font = '22px sans-serif';
      ctx.fillText(`${name}:`, 40, yPos);

      ctx.fillStyle = '#334155';
      ctx.fillRect(180, yPos - 18, 280, 20);

      ctx.fillStyle = pred.prob >= CONFIG.confidenceThreshold ? '#34d399' : '#38bdf8';
      ctx.fillRect(180, yPos - 18, 280 * pred.prob, 20);

      ctx.fillStyle = '#f8fafc';
      ctx.fillText(`%${pct}`, 475, yPos);
    });

    // Sağ Taraf: VR İçi Hareket Rehberi & İpuçları
    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 20px sans-serif';
    ctx.fillText("HAREKET İPUÇLARI (TÜBİTAK):", 560, 225);

    const guideHints = [
      "🧠 Başım: Sağ işaret parmağı şakakta",
      "⚡ Ağrıyor: İki el titretilir",
      "🚑 Ambulans: Eller başta siren döner",
      "📢 Çağırın: El göğse doğru çekilir",
      "🆘 Yardım: Sol el üstüne sağ yumruk"
    ];

    guideHints.forEach((hint, idx) => {
      const yPos = 265 + idx * 36;
      ctx.fillStyle = '#cbd5e1';
      ctx.font = '19px sans-serif';
      ctx.fillText(hint, 560, yPos);
    });

    ctx.fillStyle = '#64748b';
    ctx.font = '19px sans-serif';
    ctx.fillText(`⚡ Gecikme: ${appState.lastInferenceMs}ms | Sol el pinch: Cümleyi Temizle`, 40, 475);
  } else {
    // VERİ TOPLAYICI
    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 28px sans-serif';
    ctx.fillText("TÜBİTAK 2204-A | VERİ TOPLAYICI MODU", 40, 55);

    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 44px sans-serif';
    ctx.fillText(`Kayıt Yapılacak İşaret: [ ${appState.collectLabel.toUpperCase()} ]`, 40, 140);

    ctx.fillStyle = appState.isRecording ? '#ef4444' : (appState.isCountingDown ? '#facc15' : '#38bdf8');
    ctx.font = 'bold 58px sans-serif';
    const statusText = appState.isRecording ? "🔴 KAYDEDİLİYOR..." : (appState.isCountingDown ? "⏳ GERİ SAYIM..." : "HAZIR");
    ctx.fillText(statusText, 40, 250);

    ctx.fillStyle = '#cbd5e1';
    ctx.font = '28px sans-serif';
    ctx.fillText(`İlerleme: ${appState.currentRecordFrames.length} / ${CONFIG.sequenceLength} Kare`, 40, 320);

    ctx.fillStyle = '#34d399';
    ctx.font = '24px sans-serif';
    ctx.fillText(`Bu Oturumdaki Toplam Kayıt: ${appState.collectedDataset.length} | Sağ el pinch: Kayıt başlat`, 40, 430);
  }

  appState.vrHudTexture.needsUpdate = true;
}

// Normalizasyon (Bilek Merkezli + El Boyutu)
function normalizeHand(rawJoints) {
  if (!rawJoints || rawJoints.length !== 21) {
    return new Array(21 * 3).fill(0);
  }
  const wrist = rawJoints[0];
  const middleProx = rawJoints[9];
  const scale = Math.sqrt(
    Math.pow(middleProx.x - wrist.x, 2) +
    Math.pow(middleProx.y - wrist.y, 2) +
    Math.pow(middleProx.z - wrist.z, 2)
  ) || 0.1;

  const res = [];
  for (let i = 0; i < 21; i++) {
    const j = rawJoints[i];
    res.push((j.x - wrist.x) / scale);
    res.push((j.y - wrist.y) / scale);
    res.push((j.z - wrist.z) / scale);
  }
  return res;
}

// İskelet Kemik Pozisyonlarını Güncelle
function updateSkeletonLines(joints, bonePositionsArray, lineMesh) {
  if (!joints || joints.length !== 21) {
    lineMesh.visible = false;
    return;
  }
  let pIdx = 0;
  for (let b = 0; b < SKELETON_CONNECTIONS.length; b++) {
    const [j1, j2] = SKELETON_CONNECTIONS[b];
    bonePositionsArray[pIdx++] = joints[j1].x;
    bonePositionsArray[pIdx++] = joints[j1].y;
    bonePositionsArray[pIdx++] = joints[j1].z;

    bonePositionsArray[pIdx++] = joints[j2].x;
    bonePositionsArray[pIdx++] = joints[j2].y;
    bonePositionsArray[pIdx++] = joints[j2].z;
  }
  lineMesh.geometry.attributes.position.needsUpdate = true;
  lineMesh.visible = true;
}

// Ana WebXR Döngüsü (Sürekli Takip & Çıkarım)
async function onXRFrame(time, frame) {
  if (frame) {
    const referenceSpace = renderer.xr.getReferenceSpace();
    const session = frame.session;

    let leftHandJoints = null;
    let rightHandJoints = null;

    // WebXR Hand Input API: Sürekli 21 Eklem Sorgulama
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

    // Kemik Segmentlerini Güncelle (Full 3D Anatomik İskelet)
    updateSkeletonLines(leftHandJoints, leftBonePositions, leftSkeletonLines);
    updateSkeletonLines(rightHandJoints, rightBonePositions, rightSkeletonLines);

    // Takip Durumu UI
    document.getElementById('left-state').textContent = leftHandJoints ? 'Takipte' : 'Yok';
    document.getElementById('right-state').textContent = rightHandJoints ? 'Takipte' : 'Yok';
    document.getElementById('left-state').style.color = leftHandJoints ? '#34d399' : '#f87171';
    document.getElementById('right-state').style.color = rightHandJoints ? '#34d399' : '#f87171';

    // Pinch Jestleri
    if (leftHandJoints && appState.activeMode !== "tab-collect") {
      const d = Math.hypot(
        leftHandJoints[4].x - leftHandJoints[8].x,
        leftHandJoints[4].y - leftHandJoints[8].y,
        leftHandJoints[4].z - leftHandJoints[8].z
      );
      if (d < 0.025) clearSentence();
    }

    if (rightHandJoints && appState.activeMode === "tab-collect" && !appState.isRecording && !appState.isCountingDown) {
      const d = Math.hypot(
        rightHandJoints[4].x - rightHandJoints[8].x,
        rightHandJoints[4].y - rightHandJoints[8].y,
        rightHandJoints[4].z - rightHandJoints[8].z
      );
      if (d < 0.025 && (Date.now() - appState.lastPinchTime > 2500)) {
        appState.lastPinchTime = Date.now();
        startCollectCountdown();
      }
    }

    // Kinematik Normalizasyon
    const normLeft = normalizeHand(leftHandJoints);
    const normRight = normalizeHand(rightHandJoints);
    const frameFeatures = [...normLeft, ...normRight];

    // MOD 1: CANLI ÇIKARIM (SÜREKLİ)
    if (appState.activeMode !== "tab-collect") {
      appState.slidingBuffer.push(frameFeatures);
      if (appState.slidingBuffer.length > CONFIG.sequenceLength) {
        appState.slidingBuffer.shift();
      }

      appState.frameCount++;
      if (
        appState.isModelLoaded &&
        appState.slidingBuffer.length === CONFIG.sequenceLength &&
        appState.frameCount % CONFIG.inferenceIntervalFrames === 0
      ) {
        await runInference();
      }
    }

    // MOD 2: VERİ TOPLAMA
    if (appState.activeMode === "tab-collect" && appState.isRecording) {
      if (time - appState.lastFrameTime >= 33.33) {
        appState.lastFrameTime = time;
        appState.currentRecordFrames.push({ features: frameFeatures });
        renderVRHUD();

        if (appState.currentRecordFrames.length >= CONFIG.sequenceLength) {
          finishRecordSample();
        }
      }
    }
  }

  renderer.render(scene, camera);
}

// Çıkarım Mantığı
async function runInference() {
  const t0 = performance.now();
  try {
    const inputTensor = tf.tensor3d([appState.slidingBuffer], [1, CONFIG.sequenceLength, CONFIG.numFeatures]);
    const pred = appState.model.predict(inputTensor);
    const probs = await pred.data();

    inputTensor.dispose();
    pred.dispose();

    appState.lastInferenceMs = Math.round(performance.now() - t0);
    document.getElementById('fps-status').textContent = `⚡ Canlı Çıkarım: ${appState.lastInferenceMs} ms | WebGL`;

    const predList = [];
    for (let i = 0; i < CLASSES.length; i++) {
      predList.push({ label: CLASSES[i], prob: probs[i] });
    }
    predList.sort((a, b) => b.prob - a.prob);
    appState.topPredictions = predList;

    updateProbUI(predList);

    const top = predList[0];
    if (top.label !== 'notr' && top.prob >= CONFIG.confidenceThreshold) {
      if (top.label === appState.lastDetectedWord) {
        appState.consecutiveDetections++;
      } else {
        appState.lastDetectedWord = top.label;
        appState.consecutiveDetections = 1;
      }

      if (appState.consecutiveDetections === CONFIG.debounceCount) {
        onWordRecognized(top.label);
      }
    } else {
      if (top.label === 'notr') {
        appState.lastDetectedWord = null;
        appState.consecutiveDetections = 0;
      }
    }

    renderVRHUD();
  } catch (err) {}
}

function onWordRecognized(wordLabel) {
  playChime();
  const last = appState.recognizedWords[appState.recognizedWords.length - 1];
  if (last !== wordLabel) {
    appState.recognizedWords.push(wordLabel);
  }

  let matchedSentence = null;
  const currentStr = appState.recognizedWords.join(",");
  for (const rule of SENTENCE_RULES) {
    if (currentStr.includes(rule.sequence.join(","))) {
      matchedSentence = rule.sentence;
      break;
    }
  }

  if (matchedSentence) {
    appState.currentSentence = matchedSentence;
  } else {
    appState.currentSentence = appState.recognizedWords.map(w => CLASS_DISPLAY[w] || w).join(" ") + "...";
  }

  updateSentenceUI();
  renderVRHUD();

  if (document.getElementById('chk-auto-tts').checked) {
    speakTurkish(matchedSentence || CLASS_DISPLAY[wordLabel] || wordLabel);
  }
}

function updateSentenceUI() {
  document.getElementById('translated-sentence').textContent = appState.currentSentence;
  const c = document.getElementById('words-tag-container');
  c.innerHTML = '';
  appState.recognizedWords.forEach(w => {
    const s = document.createElement('span');
    s.className = 'word-tag';
    s.textContent = CLASS_DISPLAY[w] || w;
    c.appendChild(s);
  });
}

function updateProbUI(preds) {
  const c = document.getElementById('prob-bars-container');
  c.innerHTML = '';
  preds.slice(0, 4).forEach(p => {
    const row = document.createElement('div');
    row.className = 'prob-bar-container';
    const pct = Math.round(p.prob * 100);
    const isHigh = p.prob >= CONFIG.confidenceThreshold;
    row.innerHTML = `
      <div class="prob-label-row">
        <span>${CLASS_DISPLAY[p.label] || p.label}</span>
        <span style="font-weight:700; color:${isHigh ? '#34d399' : '#9ca3af'}">%${pct}</span>
      </div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill ${isHigh ? 'high' : ''}" style="width: ${pct}%"></div>
      </div>
    `;
    c.appendChild(row);
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

function startCollectCountdown() {
  if (appState.isRecording || appState.isCountingDown) return;
  appState.isCountingDown = true;
  let cnt = 3;
  const overlay = document.getElementById('countdown-overlay');
  overlay.style.display = 'block';

  function tick() {
    if (cnt > 0) {
      overlay.textContent = cnt;
      playTone(440 + (3 - cnt) * 110, 0.12);
      renderVRHUD();
      cnt--;
      setTimeout(tick, 1000);
    } else {
      overlay.style.display = 'none';
      appState.isCountingDown = false;
      appState.isRecording = true;
      appState.currentRecordFrames = [];
      playTone(880, 0.25);
      renderVRHUD();
    }
  }
  tick();
}

function finishRecordSample() {
  appState.isRecording = false;
  playTone(587.33, 0.15);
  setTimeout(() => playTone(880, 0.25), 160);

  const sample = {
    label: appState.collectLabel,
    frameCount: appState.currentRecordFrames.length,
    timestamp: Date.now(),
    frames: appState.currentRecordFrames
  };

  appState.collectedDataset.push(sample);
  document.getElementById('collect-sample-count').textContent = appState.collectedDataset.length;
  renderVRHUD();
}

function setupEventListeners() {
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', (e) => {
      document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      e.target.classList.add('active');
      const targetId = e.target.getAttribute('data-tab');
      document.getElementById(targetId).classList.add('active');
      appState.activeMode = targetId;
      renderVRHUD();
    });
  });

  document.getElementById('btn-speak').addEventListener('click', () => {
    speakTurkish(appState.currentSentence);
  });

  document.getElementById('btn-clear').addEventListener('click', clearSentence);

  document.getElementById('thresh-slider').addEventListener('input', (e) => {
    CONFIG.confidenceThreshold = parseFloat(e.target.value);
    document.getElementById('thresh-val').textContent = CONFIG.confidenceThreshold.toFixed(2);
  });

  // Simülasyon Butonları
  document.getElementById('btn-sim-ambulance').addEventListener('click', () => {
    clearSentence();
    const seq = ["basim", "agriyor", "ambulans", "cagirin"];
    seq.forEach((w, idx) => setTimeout(() => onWordRecognized(w), (idx + 1) * 800));
  });

  document.getElementById('btn-sim-help').addEventListener('click', () => {
    clearSentence();
    const seq = ["yardim", "nefes", "agriyor"];
    seq.forEach((w, idx) => setTimeout(() => onWordRecognized(w), (idx + 1) * 800));
  });

  // Veri Toplayıcı
  document.getElementById('collect-sign-select').addEventListener('change', (e) => {
    appState.collectLabel = e.target.value;
    renderVRHUD();
  });

  document.getElementById('btn-start-record').addEventListener('click', startCollectCountdown);

  document.getElementById('btn-download-dataset').addEventListener('click', () => {
    if (appState.collectedDataset.length === 0) {
      alert("Bu oturumda henüz kayıt yapılmadı!");
      return;
    }
    const blob = new Blob([JSON.stringify(appState.collectedDataset, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vrisaret_session_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  });
}

window.addEventListener('DOMContentLoaded', init);
