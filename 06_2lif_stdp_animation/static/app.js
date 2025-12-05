const neuronCount = 2;
const dendriteAngles = [-32, 0, 28];
const dendriteLengths = [200, 180, 190];
const dendriteOffsets = [-42, -4, 36];

const cards = Array.from({ length: neuronCount }, (_, i) => {
  const card = document.querySelectorAll(".neuron-card")[i];
  return {
    dendritesRoot: card.querySelector(`#dendrites-${i}`),
    axonRoot: card.querySelector(`#axon-${i}`),
    statusEl: card.querySelector(`#status-${i}`),
    memCanvas: card.querySelector(`#membrane-${i}`),
    spikeCanvas: card.querySelector(`#spiketrain-${i}`),
  };
}).map((c) => ({
  ...c,
  memCtx: c.memCanvas.getContext("2d"),
  spikeCtx: c.spikeCanvas.getContext("2d"),
}));

const stdpCanvases = {
  pre: document.getElementById("pre-trace"),
  post: document.getElementById("post-trace"),
  weight: document.getElementById("weight-trace"),
};
const stdpCtx = {
  pre: stdpCanvases.pre.getContext("2d"),
  post: stdpCanvases.post.getContext("2d"),
  weight: stdpCanvases.weight.getContext("2d"),
};

const state = {
  stepMs: 80,
  dx: 3.0,
  viewSamples: 320,  // Increased to fill full width of STDP charts (960px / 3.0 ≈ 320)
  maxBuffer: 2048,
  reconnectMs: 1200,
  lastFrameTs: 0,
  phase: 0,
  ws: null,
  reconnectTimer: null,
  neurons: Array.from({ length: neuronCount }, () => ({
    step: -1,
    buffer: { v: [], spikes: [], inputs: [], inputsDetail: [], threshold: 1 },
  })),
  stdp: {
    step: -1,
    pre: [],
    post: [],
    weight: [],
  },
};

function createLines() {
  cards.forEach((card) => {
    dendriteAngles.forEach((angle, i) => {
      const line = document.createElement("div");
      line.className = "dendrite-line";
      line.style.setProperty("--len", `${dendriteLengths[i]}px`);
      line.style.transform = `rotate(${angle}deg)`;
      line.style.top = `calc(50% + ${dendriteOffsets[i]}px)`;
      card.dendritesRoot.appendChild(line);
    });

    const axon = document.createElement("div");
    axon.className = "axon-line";
    axon.style.setProperty("--len", "240px");
    card.axonRoot.appendChild(axon);
  });
}

function updateStatus(text) {
  cards.forEach((c) => (c.statusEl.textContent = text));
}

function connectLive() {
  const host = location.hostname || "localhost";
  const defaultUrl = `ws://${host}:8765/stream`;
  const wsUrl = window.LIF_WS || defaultUrl;

  clearTimeout(state.reconnectTimer);
  updateStatus(`Подключение к ${wsUrl}...`);

  state.ws = new WebSocket(wsUrl);

  state.ws.onopen = () => updateStatus(`Live: ${wsUrl}`);
  state.ws.onerror = () => updateStatus("WebSocket ошибка, пробуем переподключиться...");
  state.ws.onclose = () => {
    updateStatus("Соединение закрыто, переподключение...");
    state.reconnectTimer = setTimeout(connectLive, state.reconnectMs);
  };

  state.ws.onmessage = (ev) => {
    const frame = JSON.parse(ev.data);
    if (frame.delay_ms) state.stepMs = frame.delay_ms;
    state.lastFrameTs = performance.now();
    state.phase = 0;

    cards.forEach((_, i) => {
      const buf = state.neurons[i].buffer;
      buf.v.push(frame.v[i]);
      buf.spikes.push(frame.spike[i]);
      buf.inputs.push(frame.input[i]);
      const detail = frame.input_detail ? frame.input_detail[i] : null;
      buf.inputsDetail.push(detail);
      buf.threshold = frame.threshold;

      ["v", "spikes", "inputs"].forEach((k) => {
        if (buf[k].length > state.maxBuffer) buf[k].splice(0, buf[k].length - state.maxBuffer);
      });
      if (buf.inputsDetail.length > state.maxBuffer) buf.inputsDetail.splice(0, buf.inputsDetail.length - state.maxBuffer);

      state.neurons[i].step = buf.v.length - 1;

      if (detail && Array.isArray(detail)) {
        detail.forEach((val, idx) => {
          if (val) fireDendrite(i, idx);
        });
      } else if (frame.input[i]) {
        fireDendrite(i);
      }
      if (frame.spike[i]) {
        fireAxon(i);
        if (i === 0) fireSynapse();
      }
    });

    state.stdp.pre.push(frame.pre_trace || 0);
    state.stdp.post.push(frame.post_trace || 0);
    state.stdp.weight.push(frame.weight || state.stdp.weight[state.stdp.weight.length - 1] || 0);
    ["pre", "post", "weight"].forEach((k) => {
      const arr = state.stdp[k];
      if (arr.length > state.maxBuffer) arr.splice(0, arr.length - state.maxBuffer);
    });
    state.stdp.step = state.stdp.weight.length - 1;
  };
}

function fireDendrite(i, forcedIdx = null) {
  const lines = cards[i].dendritesRoot.querySelectorAll(".dendrite-line");
  if (!lines.length) return;
  const idx = forcedIdx !== null ? forcedIdx % lines.length : Math.floor(Math.random() * lines.length);
  const line = lines[idx];
  const pulse = document.createElement("div");
  pulse.className = "pulse";
  pulse.style.setProperty("--len", line.style.getPropertyValue("--len"));
  pulse.style.setProperty("--dur", "0.9s");
  pulse.addEventListener("animationend", () => pulse.remove());
  line.appendChild(pulse);
}

function fireAxon(i) {
  const line = cards[i].axonRoot.querySelector(".axon-line");
  if (!line) return;
  const pulse = document.createElement("div");
  pulse.className = "pulse";
  pulse.style.setProperty("--len", line.style.getPropertyValue("--len"));
  pulse.style.setProperty("--dur", "0.7s");
  pulse.addEventListener("animationend", () => pulse.remove());
  line.appendChild(pulse);
}

function fireSynapse() {
  const line = document.getElementById("synapse-line");
  if (!line) return;
  const pulse = document.createElement("div");
  pulse.className = "pulse";
  pulse.style.setProperty("--len", "100%");
  pulse.style.setProperty("--dur", "0.7s");
  pulse.addEventListener("animationend", () => pulse.remove());
  line.appendChild(pulse);
}

function drawMembrane(i) {
  const { memCtx, memCanvas } = cards[i];
  const buf = state.neurons[i].buffer;
  const v = buf.v;
  if (!v.length) return;
  const threshold = buf.threshold;
  const { width: w, height: h } = memCanvas;
  memCtx.clearRect(0, 0, w, h);
  const margin = 18;
  const baseY = h - margin;
  const maxV = Math.max(threshold * 1.2, ...v);
  const scale = (h - margin * 2) / Math.max(maxV, 1e-3);

  memCtx.strokeStyle = "#1f2430";
  memCtx.lineWidth = 1;
  for (let j = 0; j <= 4; j++) {
    const y = margin + (j / 4) * (h - margin * 2);
    memCtx.beginPath();
    memCtx.moveTo(0, y);
    memCtx.lineTo(w, y);
    memCtx.stroke();
  }

  memCtx.setLineDash([6, 6]);
  memCtx.strokeStyle = "#e27272";
  const thY = baseY - threshold * scale;
  memCtx.beginPath();
  memCtx.moveTo(0, thY);
  memCtx.lineTo(w, thY);
  memCtx.stroke();
  memCtx.setLineDash([]);

  const dx = state.dx;
  const view = Math.min(state.viewSamples, v.length);
  memCtx.strokeStyle = "#4db3ff";
  memCtx.lineWidth = 2;
  memCtx.beginPath();
  for (let j = 0; j < view; j++) {
    const idx = (state.neurons[i].step - j + v.length) % v.length;
    const x = w - margin - j * dx - state.phase * dx;
    const y = baseY - v[idx] * scale;
    if (j === 0) memCtx.moveTo(x, y);
    else memCtx.lineTo(x, y);
  }
  memCtx.stroke();
}

function drawSpikes(i) {
  const { spikeCtx, spikeCanvas } = cards[i];
  const spikes = state.neurons[i].buffer.spikes;
  if (!spikes.length) return;
  const { width: w, height: h } = spikeCanvas;
  const margin = 18;
  const baseY = h - margin;
  const dx = state.dx;
  const view = Math.min(state.viewSamples, spikes.length);

  spikeCtx.clearRect(0, 0, w, h);
  spikeCtx.strokeStyle = "#1f2430";
  spikeCtx.lineWidth = 1;
  for (let j = 0; j <= 3; j++) {
    const y = margin + (j / 3) * (h - margin * 2);
    spikeCtx.beginPath();
    spikeCtx.moveTo(0, y);
    spikeCtx.lineTo(w, y);
    spikeCtx.stroke();
  }

  spikeCtx.strokeStyle = "#a0a7b5";
  spikeCtx.lineWidth = 1.5;
  spikeCtx.beginPath();
  spikeCtx.moveTo(margin, baseY);
  spikeCtx.lineTo(w - margin, baseY);
  spikeCtx.stroke();

  spikeCtx.strokeStyle = "#ff5c5c";
  spikeCtx.lineWidth = 3;
  for (let j = 0; j < view; j++) {
    const idx = (state.neurons[i].step - j + spikes.length) % spikes.length;
    if (spikes[idx]) {
      const x = w - margin - j * dx - state.phase * dx;
      spikeCtx.beginPath();
      spikeCtx.moveTo(x, baseY);
      spikeCtx.lineTo(x, margin + 10);
      spikeCtx.stroke();
    }
  }
}

function drawTrace(kind, color, dashed = false) {
  const ctx = stdpCtx[kind];
  const canvas = stdpCanvases[kind];
  const data = state.stdp[kind];
  if (!data.length) return;
  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);
  const marginY = 14;
  const marginRight = 14;
  const baseY = h - marginY;
  const maxV = Math.max(...data, 1);
  const scale = (h - marginY * 2) / Math.max(maxV, 1e-6);
  const dx = state.dx;
  const view = Math.min(state.viewSamples, data.length);

  ctx.strokeStyle = "#1f2430";
  ctx.lineWidth = 1;
  for (let j = 0; j <= 3; j++) {
    const y = marginY + (j / 3) * (h - marginY * 2);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  if (dashed) ctx.setLineDash([6, 6]);
  ctx.beginPath();
  for (let j = 0; j < view; j++) {
    const idx = (state.stdp.step - j + data.length) % data.length;
    const x = w - marginRight - j * dx - state.phase * dx;
    const y = baseY - data[idx] * scale;
    if (x < 0) continue;  // Skip points outside canvas
    if (j === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawWeight() {
  const ctx = stdpCtx.weight;
  const canvas = stdpCanvases.weight;
  const data = state.stdp.weight;
  if (!data.length) return;
  const { width: w, height: h } = canvas;
  ctx.clearRect(0, 0, w, h);
  const marginY = 16;
  const marginRight = 16;
  const baseY = h - marginY;
  const minV = Math.min(...data);
  const maxV = Math.max(...data);
  const span = Math.max(maxV - minV, 1e-3);
  const scale = (h - marginY * 2) / span;
  const dx = state.dx;
  const view = Math.min(state.viewSamples, data.length);
  const zeroY = baseY - (-minV) * scale;

  ctx.strokeStyle = "#1f2430";
  ctx.lineWidth = 1;
  for (let j = 0; j <= 3; j++) {
    const y = marginY + (j / 3) * (h - marginY * 2);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = "#888";
  ctx.beginPath();
  ctx.moveTo(0, zeroY);
  ctx.lineTo(w, zeroY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "#ffa24d";
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  for (let j = 0; j < view; j++) {
    const idx = (state.stdp.step - j + data.length) % data.length;
    const x = w - marginRight - j * dx - state.phase * dx;
    const y = baseY - (data[idx] - minV) * scale;
    if (x < 0) continue;  // Skip points outside canvas
    if (j === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

function tick(ts) {
  if (!state.lastFrameTs) state.lastFrameTs = ts;
  const elapsed = ts - state.lastFrameTs;
  state.phase = Math.min(elapsed / state.stepMs, 1);

  for (let i = 0; i < neuronCount; i++) {
    drawMembrane(i);
    drawSpikes(i);
  }
  drawTrace("pre", "#ff5c5c");
  drawTrace("post", "#4db3ff");
  drawWeight();

  requestAnimationFrame(tick);
}

createLines();
connectLive();
requestAnimationFrame(tick);
