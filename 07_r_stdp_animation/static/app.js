/**
 * R-STDP Animation Frontend
 *
 * Visualizes a reward-modulated STDP network with:
 * - 1 Pre-synaptic neuron
 * - 2 Post-synaptic neurons (A and B)
 * - Eligibility traces
 * - Reward signals
 * - Two synaptic weights
 *
 * All neurons and synapses are drawn on a single canvas.
 */

// ═══════════════════════════════════════════════════════════════════════════════
// CANVAS ELEMENTS
// ═══════════════════════════════════════════════════════════════════════════════

const networkCanvas = document.getElementById("network-canvas");
const networkCtx = networkCanvas.getContext("2d");

// Membrane potential canvases
const membraneCanvases = {
  pre: document.getElementById("membrane-pre"),
  postA: document.getElementById("membrane-post-a"),
  postB: document.getElementById("membrane-post-b"),
};

const membraneCtx = {
  pre: membraneCanvases.pre?.getContext("2d"),
  postA: membraneCanvases.postA?.getContext("2d"),
  postB: membraneCanvases.postB?.getContext("2d"),
};

// R-STDP chart canvases
const rstdpCanvases = {
  pre: document.getElementById("pre-trace"),
  post: document.getElementById("post-trace"),
  eligibility: document.getElementById("eligibility-trace"),
  reward: document.getElementById("reward-trace"),
  weight: document.getElementById("weight-trace"),
};

const rstdpCtx = {
  pre: rstdpCanvases.pre?.getContext("2d"),
  post: rstdpCanvases.post?.getContext("2d"),
  eligibility: rstdpCanvases.eligibility?.getContext("2d"),
  reward: rstdpCanvases.reward?.getContext("2d"),
  weight: rstdpCanvases.weight?.getContext("2d"),
};

// ═══════════════════════════════════════════════════════════════════════════════
// COLORS
// ═══════════════════════════════════════════════════════════════════════════════

const COLORS = {
  pre: "#ff5c5c",
  postA: "#4db3ff",
  postB: "#ff9912",
  reward: "#3ecf3e",
  synapse: "#888",
  bg: "#161821",
  text: "#e8edf3",
  muted: "#8c95a6",
};

// ═══════════════════════════════════════════════════════════════════════════════
// NEURON POSITIONS (on 1200x340 canvas)
// ═══════════════════════════════════════════════════════════════════════════════

const NEURON_RADIUS = 70;
const NEURONS = {
  pre:   { x: 200, y: 170, r: NEURON_RADIUS, color: COLORS.pre, label: "Pre" },
  postA: { x: 900, y: 90,  r: NEURON_RADIUS, color: COLORS.postA, label: "Post A" },
  postB: { x: 900, y: 250, r: NEURON_RADIUS, color: COLORS.postB, label: "Post B" },
};

// ═══════════════════════════════════════════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════════════════════════════════════════

const state = {
  stepMs: 50,
  dx: 3.0,
  viewSamples: 320,
  maxBuffer: 2048,
  reconnectMs: 1200,
  lastFrameTs: 0,
  phase: 0,
  ws: null,
  reconnectTimer: null,

  // Current frame data
  frame: {
    v: [0, 0, 0],
    spike: [0, 0, 0],
    reward: [0, 0],
    weight: [0.5, 0.5],
  },

  // Pulse animations (list of active pulses traveling along synapses)
  pulses: [],

  // Spike animations (expanding rings when neurons fire)
  spikeAnims: {
    pre: null,
    postA: null,
    postB: null,
  },

  // Membrane potential history buffers
  membrane: {
    pre: [],
    postA: [],
    postB: [],
    threshold: 1.0,
  },

  // R-STDP data buffers
  rstdp: {
    step: -1,
    preTrace: [],
    postTraceA: [],
    postTraceB: [],
    eligibilityA: [],
    eligibilityB: [],
    rewardA: [],
    rewardB: [],
    weightA: [],
    weightB: [],
  },
};

// ═══════════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

function updateStatus(text) {
  const statusEl = document.getElementById("status-main");
  if (statusEl) statusEl.textContent = text;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

// ═══════════════════════════════════════════════════════════════════════════════
// WEBSOCKET CONNECTION
// ═══════════════════════════════════════════════════════════════════════════════

function connectLive() {
  const host = location.hostname || "localhost";
  const defaultUrl = `ws://${host}:8765`;
  const wsUrl = window.LIF_WS || defaultUrl;

  clearTimeout(state.reconnectTimer);
  updateStatus(`Connecting to ${wsUrl}...`);

  state.ws = new WebSocket(wsUrl);

  state.ws.onopen = () => updateStatus(`Live: ${wsUrl}`);
  state.ws.onerror = () => updateStatus("WebSocket error, reconnecting...");
  state.ws.onclose = () => {
    updateStatus("Connection closed, reconnecting...");
    state.reconnectTimer = setTimeout(connectLive, state.reconnectMs);
  };

  state.ws.onmessage = (ev) => {
    const frame = JSON.parse(ev.data);
    if (frame.delay_ms) state.stepMs = frame.delay_ms;
    state.lastFrameTs = performance.now();
    state.phase = 0;

    // Update current frame data
    state.frame.v = frame.v || [0, 0, 0];
    state.frame.spike = frame.spike || [0, 0, 0];
    state.frame.reward = frame.reward || [0, 0];
    state.frame.weight = frame.weight || [0.5, 0.5];

    // Update threshold
    if (frame.threshold) {
      state.membrane.threshold = frame.threshold;
    }

    // Store membrane potential history
    state.membrane.pre.push(state.frame.v[0]);
    state.membrane.postA.push(state.frame.v[1]);
    state.membrane.postB.push(state.frame.v[2]);

    // Trim membrane buffers
    ["pre", "postA", "postB"].forEach((k) => {
      if (state.membrane[k].length > state.maxBuffer) {
        state.membrane[k].splice(0, state.membrane[k].length - state.maxBuffer);
      }
    });

    // Trigger pulse animations on spikes
    if (frame.spike[0]) {
      // Pre spike -> pulses to both post neurons
      addPulse("pre", "postA");
      addPulse("pre", "postB");
      addSpikeAnim("pre");
    }
    if (frame.spike[1]) {
      addSpikeAnim("postA");
    }
    if (frame.spike[2]) {
      addSpikeAnim("postB");
    }

    // Update R-STDP buffers
    const rstdp = state.rstdp;
    rstdp.preTrace.push(frame.pre_trace || 0);
    rstdp.postTraceA.push(frame.post_trace ? frame.post_trace[0] : 0);
    rstdp.postTraceB.push(frame.post_trace ? frame.post_trace[1] : 0);
    rstdp.eligibilityA.push(frame.eligibility ? frame.eligibility[0] : 0);
    rstdp.eligibilityB.push(frame.eligibility ? frame.eligibility[1] : 0);
    rstdp.rewardA.push(frame.reward ? frame.reward[0] : 0);
    rstdp.rewardB.push(frame.reward ? frame.reward[1] : 0);
    rstdp.weightA.push(frame.weight ? frame.weight[0] : 0.5);
    rstdp.weightB.push(frame.weight ? frame.weight[1] : 0.5);

    // Trim R-STDP buffers
    const keys = ["preTrace", "postTraceA", "postTraceB", "eligibilityA", "eligibilityB", "rewardA", "rewardB", "weightA", "weightB"];
    keys.forEach((k) => {
      if (rstdp[k].length > state.maxBuffer) {
        rstdp[k].splice(0, rstdp[k].length - state.maxBuffer);
      }
    });

    rstdp.step = rstdp.preTrace.length - 1;
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// PULSE ANIMATIONS
// ═══════════════════════════════════════════════════════════════════════════════

function addPulse(from, to) {
  state.pulses.push({
    from,
    to,
    progress: 0,
    speed: 0.03, // progress per frame
  });
}

function updatePulses() {
  state.pulses = state.pulses.filter((p) => {
    p.progress += p.speed;
    return p.progress < 1;
  });
}

function addSpikeAnim(neuronKey) {
  state.spikeAnims[neuronKey] = {
    progress: 0,
    speed: 0.04, // how fast the ring expands
  };
}

function updateSpikeAnims() {
  for (const key of Object.keys(state.spikeAnims)) {
    const anim = state.spikeAnims[key];
    if (anim) {
      anim.progress += anim.speed;
      if (anim.progress >= 1) {
        state.spikeAnims[key] = null;
      }
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// NETWORK CANVAS DRAWING
// ═══════════════════════════════════════════════════════════════════════════════

function drawNetwork() {
  const ctx = networkCtx;
  const { width: w, height: h } = networkCanvas;

  ctx.clearRect(0, 0, w, h);

  // Draw synapses (lines from Pre to Post A and Post B)
  drawSynapse(ctx, NEURONS.pre, NEURONS.postA, state.frame.weight[0], COLORS.postA);
  drawSynapse(ctx, NEURONS.pre, NEURONS.postB, state.frame.weight[1], COLORS.postB);

  // Draw pulses
  state.pulses.forEach((p) => {
    const fromN = NEURONS[p.from];
    const toN = NEURONS[p.to];
    const x = lerp(fromN.x, toN.x, p.progress);
    const y = lerp(fromN.y, toN.y, p.progress);

    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fillStyle = COLORS.pre;
    ctx.shadowColor = COLORS.pre;
    ctx.shadowBlur = 15;
    ctx.fill();
    ctx.shadowBlur = 0;
  });

  // Draw neurons
  drawNeuron(ctx, NEURONS.pre, state.frame.v[0], state.frame.spike[0]);
  drawNeuron(ctx, NEURONS.postA, state.frame.v[1], state.frame.spike[1], state.frame.reward[0]);
  drawNeuron(ctx, NEURONS.postB, state.frame.v[2], state.frame.spike[2], state.frame.reward[1]);

  // Draw spike animations (expanding rings)
  drawSpikeAnim(ctx, NEURONS.pre, state.spikeAnims.pre);
  drawSpikeAnim(ctx, NEURONS.postA, state.spikeAnims.postA);
  drawSpikeAnim(ctx, NEURONS.postB, state.spikeAnims.postB);

  // Draw reward indicators
  drawRewardIndicator(ctx, 820, 30, "Reward A", state.frame.reward[0]);
  drawRewardIndicator(ctx, 820, 290, "Reward B", state.frame.reward[1]);

  // Draw weight labels
  drawWeightLabel(ctx, NEURONS.pre, NEURONS.postA, state.frame.weight[0], "w₁");
  drawWeightLabel(ctx, NEURONS.pre, NEURONS.postB, state.frame.weight[1], "w₂");
}

function drawSynapse(ctx, from, to, weight, color) {
  // Line from edge of 'from' neuron to edge of 'to' neuron
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const nx = dx / dist;
  const ny = dy / dist;

  const x1 = from.x + nx * from.r;
  const y1 = from.y + ny * from.r;
  const x2 = to.x - nx * to.r;
  const y2 = to.y - ny * to.r;

  // Line thickness based on weight
  const lineWidth = 2 + weight * 6;

  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.globalAlpha = 0.6 + weight * 0.4;
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawNeuron(ctx, neuron, voltage, spiking, reward = 0) {
  const { x, y, r, color, label } = neuron;

  // Outer glow when spiking
  if (spiking) {
    ctx.beginPath();
    ctx.arc(x, y, r + 15, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 30;
    ctx.globalAlpha = 0.5;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.shadowBlur = 0;
  }

  // Reward glow (green)
  if (reward > 0) {
    ctx.beginPath();
    ctx.arc(x, y, r + 10, 0, Math.PI * 2);
    ctx.strokeStyle = COLORS.reward;
    ctx.lineWidth = 4;
    ctx.shadowColor = COLORS.reward;
    ctx.shadowBlur = 20;
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // Neuron body (filled based on voltage)
  const grad = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, 0, x, y, r);
  grad.addColorStop(0, "#3a3f4d");
  grad.addColorStop(1, "#1a1d26");

  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = grad;
  ctx.fill();

  // Border
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.globalAlpha = 0.5 + Math.min(voltage, 1) * 0.5;
  ctx.stroke();
  ctx.globalAlpha = 1;

  // Voltage fill (arc from bottom)
  const fillAngle = Math.PI * 2 * Math.min(Math.max(voltage, 0), 1);
  if (voltage > 0) {
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.arc(x, y, r - 5, Math.PI / 2, Math.PI / 2 + fillAngle);
    ctx.closePath();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.3;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  // Label
  ctx.fillStyle = COLORS.text;
  ctx.font = "bold 16px 'Segoe UI', sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x, y);
}

function drawRewardIndicator(ctx, x, y, label, active) {
  // Label
  ctx.fillStyle = active > 0 ? COLORS.reward : COLORS.muted;
  ctx.font = "12px 'Segoe UI', sans-serif";
  ctx.textAlign = "left";
  ctx.fillText(label, x, y);

  // Light
  const lightX = x + 70;
  ctx.beginPath();
  ctx.arc(lightX, y - 2, 7, 0, Math.PI * 2);

  if (active > 0) {
    ctx.fillStyle = COLORS.reward;
    ctx.shadowColor = COLORS.reward;
    ctx.shadowBlur = 15;
  } else {
    ctx.fillStyle = "#333";
    ctx.shadowBlur = 0;
  }
  ctx.fill();
  ctx.shadowBlur = 0;
}

function drawWeightLabel(ctx, from, to, weight, label) {
  const mx = (from.x + to.x) / 2;
  const my = (from.y + to.y) / 2;

  ctx.fillStyle = COLORS.text;
  ctx.font = "12px 'Segoe UI', sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(`${label}=${weight.toFixed(2)}`, mx - 30, my);
}

function drawSpikeAnim(ctx, neuron, anim) {
  if (!anim) return;

  const { x, y, r, color } = neuron;
  const progress = anim.progress;

  // Expanding ring
  const ringRadius = r + 10 + progress * 60;
  const alpha = 1 - progress;

  ctx.beginPath();
  ctx.arc(x, y, ringRadius, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 4 * (1 - progress * 0.7);
  ctx.globalAlpha = alpha;
  ctx.shadowColor = color;
  ctx.shadowBlur = 20 * alpha;
  ctx.stroke();
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;

  // Inner flash
  if (progress < 0.3) {
    const flashAlpha = (0.3 - progress) / 0.3;
    ctx.beginPath();
    ctx.arc(x, y, r + 5, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = flashAlpha * 0.4;
    ctx.fill();
    ctx.globalAlpha = 1;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MEMBRANE POTENTIAL DRAWING
// ═══════════════════════════════════════════════════════════════════════════════

function drawMembrane(ctx, canvas, data, color, threshold) {
  if (!ctx || !canvas || !data.length) return;

  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);

  const marginY = 12;
  const marginRight = 10;
  const baseY = h - marginY;
  const maxV = Math.max(threshold * 1.3, ...data);
  const scale = (h - marginY * 2) / Math.max(maxV, 1e-3);
  const dx = state.dx;
  const step = data.length - 1;
  const view = Math.min(state.viewSamples, data.length);

  // Grid
  ctx.strokeStyle = "#1f2430";
  ctx.lineWidth = 1;
  for (let j = 0; j <= 4; j++) {
    const y = marginY + (j / 4) * (h - marginY * 2);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Threshold line
  const thY = baseY - threshold * scale;
  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = "#e27272";
  ctx.beginPath();
  ctx.moveTo(0, thY);
  ctx.lineTo(w, thY);
  ctx.stroke();
  ctx.setLineDash([]);

  // Membrane potential line
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let j = 0; j < view; j++) {
    const idx = (step - j + data.length) % data.length;
    const x = w - marginRight - j * dx - state.phase * dx;
    const y = baseY - data[idx] * scale;
    if (x < 0) continue;
    if (j === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

// ═══════════════════════════════════════════════════════════════════════════════
// R-STDP TRACE DRAWING
// ═══════════════════════════════════════════════════════════════════════════════

function drawSingleTrace(ctx, canvas, data, color) {
  if (!ctx || !canvas || !data.length) return;

  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);

  const marginY = 10;
  const marginRight = 10;
  const baseY = h - marginY;
  const maxV = Math.max(...data, 0.1);
  const scale = (h - marginY * 2) / Math.max(maxV, 1e-6);
  const dx = state.dx;
  const step = state.rstdp.step;
  const view = Math.min(state.viewSamples, data.length);

  // Grid
  ctx.strokeStyle = "#1f2430";
  ctx.lineWidth = 1;
  for (let j = 0; j <= 3; j++) {
    const y = marginY + (j / 3) * (h - marginY * 2);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Trace
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let j = 0; j < view; j++) {
    const idx = (step - j + data.length) % data.length;
    const x = w - marginRight - j * dx - state.phase * dx;
    const y = baseY - data[idx] * scale;
    if (x < 0) continue;
    if (j === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function drawDualTrace(ctx, canvas, dataA, dataB, colorA, colorB, allowNegative = false) {
  if (!ctx || !canvas) return;

  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);

  const marginY = 10;
  const marginRight = 10;
  const dx = state.dx;
  const step = state.rstdp.step;
  const viewA = Math.min(state.viewSamples, dataA.length);
  const viewB = Math.min(state.viewSamples, dataB.length);

  let minV = 0;
  let maxV = 0.1;
  if (dataA.length) {
    maxV = Math.max(maxV, ...dataA);
    if (allowNegative) minV = Math.min(minV, ...dataA);
  }
  if (dataB.length) {
    maxV = Math.max(maxV, ...dataB);
    if (allowNegative) minV = Math.min(minV, ...dataB);
  }
  const span = maxV - minV || 1e-3;
  const scale = (h - marginY * 2) / span;
  const baseY = h - marginY;

  // Grid
  ctx.strokeStyle = "#1f2430";
  ctx.lineWidth = 1;
  for (let j = 0; j <= 3; j++) {
    const y = marginY + (j / 3) * (h - marginY * 2);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Zero line
  if (allowNegative && minV < 0) {
    const zeroY = baseY - (0 - minV) * scale;
    ctx.setLineDash([6, 6]);
    ctx.strokeStyle = "#555";
    ctx.beginPath();
    ctx.moveTo(0, zeroY);
    ctx.lineTo(w, zeroY);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Trace A
  if (dataA.length) {
    ctx.strokeStyle = colorA;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let j = 0; j < viewA; j++) {
      const idx = (step - j + dataA.length) % dataA.length;
      const x = w - marginRight - j * dx - state.phase * dx;
      const y = baseY - (dataA[idx] - minV) * scale;
      if (x < 0) continue;
      if (j === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Trace B
  if (dataB.length) {
    ctx.strokeStyle = colorB;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let j = 0; j < viewB; j++) {
      const idx = (step - j + dataB.length) % dataB.length;
      const x = w - marginRight - j * dx - state.phase * dx;
      const y = baseY - (dataB[idx] - minV) * scale;
      if (x < 0) continue;
      if (j === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
}

function drawWeights() {
  const ctx = rstdpCtx.weight;
  const canvas = rstdpCanvases.weight;
  if (!ctx || !canvas) return;

  const dataA = state.rstdp.weightA;
  const dataB = state.rstdp.weightB;
  if (!dataA.length && !dataB.length) return;

  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);

  const marginY = 14;
  const marginRight = 14;
  const baseY = h - marginY;

  // Fixed scale [0, 1]
  const minV = 0.0;
  const maxV = 1.0;
  const span = maxV - minV;
  const scale = (h - marginY * 2) / span;

  const dx = state.dx;
  const step = state.rstdp.step;
  const viewA = Math.min(state.viewSamples, dataA.length);
  const viewB = Math.min(state.viewSamples, dataB.length);

  // Grid
  ctx.strokeStyle = "#1f2430";
  ctx.lineWidth = 1;
  for (let j = 0; j <= 4; j++) {
    const y = marginY + (j / 4) * (h - marginY * 2);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Weight A (blue)
  if (dataA.length) {
    ctx.strokeStyle = COLORS.postA;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let j = 0; j < viewA; j++) {
      const idx = (step - j + dataA.length) % dataA.length;
      const x = w - marginRight - j * dx - state.phase * dx;
      const y = baseY - (dataA[idx] - minV) * scale;
      if (x < 0) continue;
      if (j === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Weight B (orange)
  if (dataB.length) {
    ctx.strokeStyle = COLORS.postB;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let j = 0; j < viewB; j++) {
      const idx = (step - j + dataB.length) % dataB.length;
      const x = w - marginRight - j * dx - state.phase * dx;
      const y = baseY - (dataB[idx] - minV) * scale;
      if (x < 0) continue;
      if (j === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Legend
  ctx.font = "12px sans-serif";
  ctx.fillStyle = COLORS.postA;
  ctx.fillText("w₁", 10, 20);
  ctx.fillStyle = COLORS.postB;
  ctx.fillText("w₂", 40, 20);
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN ANIMATION LOOP
// ═══════════════════════════════════════════════════════════════════════════════

function tick(ts) {
  if (!state.lastFrameTs) state.lastFrameTs = ts;
  const elapsed = ts - state.lastFrameTs;
  state.phase = Math.min(elapsed / state.stepMs, 1);

  // Update animations
  updatePulses();
  updateSpikeAnims();

  // Draw network (neurons + synapses)
  drawNetwork();

  // Draw membrane potential charts
  drawMembrane(membraneCtx.pre, membraneCanvases.pre, state.membrane.pre, COLORS.pre, state.membrane.threshold);
  drawMembrane(membraneCtx.postA, membraneCanvases.postA, state.membrane.postA, COLORS.postA, state.membrane.threshold);
  drawMembrane(membraneCtx.postB, membraneCanvases.postB, state.membrane.postB, COLORS.postB, state.membrane.threshold);

  // Draw R-STDP charts
  drawSingleTrace(rstdpCtx.pre, rstdpCanvases.pre, state.rstdp.preTrace, COLORS.pre);
  drawDualTrace(rstdpCtx.post, rstdpCanvases.post, state.rstdp.postTraceA, state.rstdp.postTraceB, COLORS.postA, COLORS.postB);
  drawDualTrace(rstdpCtx.eligibility, rstdpCanvases.eligibility, state.rstdp.eligibilityA, state.rstdp.eligibilityB, COLORS.postA, COLORS.postB, true);
  drawDualTrace(
    rstdpCtx.reward,
    rstdpCanvases.reward,
    state.rstdp.rewardA,
    state.rstdp.rewardB,
    COLORS.postA,
    COLORS.postB
  );  
  drawWeights();

  requestAnimationFrame(tick);
}

// ═══════════════════════════════════════════════════════════════════════════════
// STARTUP
// ═══════════════════════════════════════════════════════════════════════════════

connectLive();
requestAnimationFrame(tick);
