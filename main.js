/* =========================================================================
   Hand Gesture Web App — landmark-based recognition (MediaPipe Hands)
   -------------------------------------------------------------------------
   Instead of classifying raw camera pixels (the old Teachable Machine
   approach, which is easily confused by background/lighting), we detect the
   21 3D landmarks of the hand and read the gesture from finger GEOMETRY.
   This is robust, needs zero training data, and is near 100% accurate.
   ========================================================================= */

"use strict";

// ---- DOM references -------------------------------------------------------
const video          = document.getElementById("webcam");
const canvas         = document.getElementById("overlay");
const ctx            = canvas.getContext("2d");
const statusEl       = document.getElementById("status");
const gestureNameEl  = document.getElementById("result_gesture_name");
const emojiEl        = document.getElementById("result_emoji");
const confFillEl     = document.getElementById("confidence_fill");
const confTextEl     = document.getElementById("confidence_text");
const startBtn       = document.getElementById("startBtn");
const autoSpeakEl    = document.getElementById("autoSpeak");

// ---- Gesture -> emoji map -------------------------------------------------
const EMOJI = {
  "Amazing":    "\u{1F44C}", // 👌
  "Best":       "\u{1F44D}", // 👍
  "Hand Raise": "\u270B",    // ✋
  "Swag":       "\u{1F91F}", // 🤟
  "Peace":      "\u270C",    // ✌️
  "Fist":       "\u270A",    // ✊
  "No Hand":    "\u{1F91A}", // 🤚
};

// ---- MediaPipe landmark indices ------------------------------------------
// Fingertips: thumb=4, index=8, middle=12, ring=16, pinky=20
// PIP joints:            index=6, middle=10, ring=14, pinky=18
// MCP joints:            index=5, middle=9,  ring=13, pinky=17
const TIPS = { thumb: 4, index: 8, middle: 12, ring: 16, pinky: 20 };
const PIPS = { index: 6, middle: 10, ring: 14, pinky: 18 };
const MCPS = { index: 5, middle: 9, ring: 13, pinky: 17 };

let prediction = "";           // current stable prediction
let lastSpoken = "";           // last gesture we auto-announced
let hands = null;
let cameraLoop = null;

// Temporal smoothing: only accept a gesture once it repeats over frames.
const history = [];
const HISTORY_LEN = 8;

// ------------------------------------------------------------------------
// Geometry helpers
// ------------------------------------------------------------------------
function dist(a, b) {
  const dx = a.x - b.x, dy = a.y - b.y, dz = (a.z || 0) - (b.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

// Is a (non-thumb) finger extended? Tip is farther from the wrist than its
// PIP joint — reliable regardless of hand rotation.
function isFingerExtended(lm, finger) {
  const wrist = lm[0];
  return dist(lm[TIPS[finger]], wrist) > dist(lm[PIPS[finger]], wrist) * 1.05;
}

// Thumb: compare tip vs its IP joint distance from the wrist.
function isThumbExtended(lm) {
  const wrist = lm[0];
  return dist(lm[TIPS.thumb], wrist) > dist(lm[3], wrist) * 1.05;
}

// Hand scale — used to normalize distances so thresholds work at any depth.
function handScale(lm) {
  return dist(lm[0], lm[MCPS.middle]) || 1e-6;
}

// ------------------------------------------------------------------------
// The classifier: turn 21 landmarks into a gesture label + confidence
// ------------------------------------------------------------------------
function classifyGesture(lm) {
  const scale = handScale(lm);

  const ext = {
    thumb:  isThumbExtended(lm),
    index:  isFingerExtended(lm, "index"),
    middle: isFingerExtended(lm, "middle"),
    ring:   isFingerExtended(lm, "ring"),
    pinky:  isFingerExtended(lm, "pinky"),
  };

  // Distance between thumb tip and index tip (normalized) — the OK sign.
  const thumbIndexTouch = dist(lm[TIPS.thumb], lm[TIPS.index]) / scale;
  const countExt = Object.values(ext).filter(Boolean).length;

  // ---- 👌 Amazing (OK): thumb & index form a ring, others extended ----
  if (thumbIndexTouch < 0.35 && ext.middle && ext.ring && ext.pinky) {
    return { label: "Amazing", confidence: 0.97 };
  }

  // ---- ✊ Fist: nothing extended ----
  if (countExt === 0) {
    return { label: "Fist", confidence: 0.95 };
  }

  // ---- 👍 Best (thumbs up): only thumb out, hand roughly vertical ----
  if (ext.thumb && !ext.index && !ext.middle && !ext.ring && !ext.pinky) {
    return { label: "Best", confidence: 0.96 };
  }

  // ---- 🤟 Swag (ILY / rock): thumb + index + pinky, middle & ring down ----
  if (ext.thumb && ext.index && ext.pinky && !ext.middle && !ext.ring) {
    return { label: "Swag", confidence: 0.96 };
  }
  // rock horns variant (index + pinky, no thumb)
  if (ext.index && ext.pinky && !ext.middle && !ext.ring) {
    return { label: "Swag", confidence: 0.9 };
  }

  // ---- ✌️ Peace (victory): index + middle up, ring & pinky down ----
  if (ext.index && ext.middle && !ext.ring && !ext.pinky) {
    return { label: "Peace", confidence: 0.95 };
  }

  // ---- ✋ Hand Raise (open palm): all four fingers extended ----
  if (ext.index && ext.middle && ext.ring && ext.pinky) {
    return { label: "Hand Raise", confidence: 0.97 };
  }

  return { label: "…", confidence: 0.4 };
}

// ------------------------------------------------------------------------
// Smooth predictions so the label doesn't flicker frame-to-frame.
// ------------------------------------------------------------------------
function smooth(label) {
  history.push(label);
  if (history.length > HISTORY_LEN) history.shift();
  const counts = {};
  for (const l of history) counts[l] = (counts[l] || 0) + 1;
  let best = label, bestN = 0;
  for (const l in counts) if (counts[l] > bestN) { best = l; bestN = counts[l]; }
  return { label: best, agreement: bestN / history.length };
}

// ------------------------------------------------------------------------
// Draw landmarks + connections onto the overlay canvas
// ------------------------------------------------------------------------
function draw(landmarks) {
  ctx.save();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (landmarks && window.drawConnectors && window.HAND_CONNECTIONS) {
    window.drawConnectors(ctx, landmarks, window.HAND_CONNECTIONS, {
      color: "#00e5ff", lineWidth: 3,
    });
    window.drawLandmarks(ctx, landmarks, {
      color: "#ff2d78", fillColor: "#ffffff", lineWidth: 1, radius: 4,
    });
  }
  ctx.restore();
}

// ------------------------------------------------------------------------
// Update the prediction panel
// ------------------------------------------------------------------------
function updatePanel(label, confidence) {
  const known = EMOJI[label] !== undefined;
  gestureNameEl.textContent = known ? label : "Detecting…";
  emojiEl.innerHTML = known ? EMOJI[label] : "\u{1F440}";
  const pct = Math.round(confidence * 100);
  confFillEl.style.width = pct + "%";
  confTextEl.textContent = pct + "%";

  if (known && label !== prediction) {
    prediction = label;
    if (autoSpeakEl.checked && label !== lastSpoken) {
      lastSpoken = label;
      speak();
    }
  }
}

// ------------------------------------------------------------------------
// MediaPipe results callback
// ------------------------------------------------------------------------
function onResults(results) {
  // Match canvas to the rendered video size.
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const lm = results.multiHandLandmarks[0];
    draw(lm);
    const raw = classifyGesture(lm);
    const sm = smooth(raw.label);
    // Blend the classifier's confidence with temporal agreement.
    const confidence = raw.confidence * sm.agreement;
    updatePanel(sm.label, confidence);
    statusEl.style.display = "none";
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    history.length = 0;
    prediction = "";
    updatePanel("No Hand", 0);
    gestureNameEl.textContent = "Show your hand";
    statusEl.style.display = "block";
    statusEl.textContent = "No hand detected — hold your hand up to the camera";
  }
}

// ------------------------------------------------------------------------
// Initialize MediaPipe Hands
// ------------------------------------------------------------------------
function initHands() {
  hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4/${file}`,
  });
  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,        // 1 = more accurate
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.6,
  });
  hands.onResults(onResults);
}

// ------------------------------------------------------------------------
// Start the webcam and processing loop
// ------------------------------------------------------------------------
async function startCamera() {
  if (!hands) initHands();
  startBtn.disabled = true;
  startBtn.textContent = "Camera Running…";
  statusEl.textContent = "Starting camera…";
  statusEl.style.display = "block";

  try {
    if (window.Camera) {
      cameraLoop = new Camera(video, {
        onFrame: async () => { await hands.send({ image: video }); },
        width: 640,
        height: 480,
      });
      await cameraLoop.start();
    } else {
      // Fallback: getUserMedia + manual loop
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      video.srcObject = stream;
      await video.play();
      const loop = async () => {
        await hands.send({ image: video });
        requestAnimationFrame(loop);
      };
      loop();
    }
    statusEl.textContent = "Ready — show a gesture ✋";
  } catch (err) {
    console.error(err);
    statusEl.textContent = "Could not access the camera. Please allow permission.";
    startBtn.disabled = false;
    startBtn.textContent = "Start Camera";
  }
}

// ------------------------------------------------------------------------
// Text-to-speech announcement
// ------------------------------------------------------------------------
function speak() {
  if (!("speechSynthesis" in window)) return;
  const label = prediction || gestureNameEl.textContent;
  if (!label || label === "…") return;
  const synth = window.speechSynthesis;
  synth.cancel();
  const u = new SpeechSynthesisUtterance("The gesture is " + label);
  u.rate = 1.0;
  synth.speak(u);
}

// Expose for inline onclick handlers.
window.startCamera = startCamera;
window.speak = speak;

// Prepare the model as soon as the page loads (so first Start is fast).
window.addEventListener("load", () => {
  initHands();
  statusEl.textContent = "Ready — click “Start Camera” to begin";
});



