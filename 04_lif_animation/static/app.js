const dendriteAngles = [-32, 0, 28];
const dendriteLengths = [200, 180, 190];
const dendriteOffsets = [-42, -4, 36];
const dendritesRoot = document.getElementById("dendrites");
const axonRoot = document.getElementById("axon");
const statusEl = document.getElementById("status");
const memCanvas = document.getElementById("membrane");
const spikeCanvas = document.getElementById("spiketrain");
const memCtx = memCanvas.getContext("2d");
const spikeCtx = spikeCanvas.getContext("2d");

const state = {
  step: -1,
  phase: 0,
  lastTs: 0,
  lastFrameTs: 0,
  stepMs: 80,
  dx: 3.2,
  viewSamples: 130,
  maxBuffer: 600,
  reconnectMs: 1200,
  buffer: { v: [], spikes: [], inputs: [], threshold: 1 },
  ws: null,
  reconnectTimer: null,
};

function createLines() {
  dendriteAngles.forEach((angle, i) => {
    const line = document.createElement("div");
    line.className = "dendrite-line";
    line.style.setProperty("--len", `${dendriteLengths[i]}px`);
    line.style.transform = `rotate(${angle}deg)`;
    line.style.top = `calc(50% + ${dendriteOffsets[i]}px)`;
    dendritesRoot.appendChild(line);
  });

  const axon = document.createElement("div");
  axon.className = "axon-line";
  axon.style.setProperty("--len", "240px");
  axonRoot.appendChild(axon);
}

function connectLive() {
  const defaultUrl = `ws://${location.hostname}:8765/stream`;
  const wsUrl = window.LIF_WS || defaultUrl;

  clearTimeout(state.reconnectTimer);
  statusEl.textContent = `Подключаемся к ${wsUrl}...`;

  state.ws = new WebSocket(wsUrl);

  state.ws.onopen = () => {
    statusEl.textContent = `Live: ${wsUrl}`;
  };

  state.ws.onerror = () => {
    statusEl.textContent = "WebSocket недоступен, пробуем переподключиться...";
  };

  state.ws.onclose = () => {
    statusEl.textContent = "Соединение закрыто, переподключение...";
    state.reconnectTimer = setTimeout(connectLive, state.reconnectMs);
  };

  state.ws.onmessage = (ev) => {
    const frame = JSON.parse(ev.data);
    const buf = state.buffer;
    buf.v.push(frame.v);
    buf.spikes.push(frame.spike);
    buf.inputs.push(frame.input);
    buf.threshold = frame.threshold;

    ["v", "spikes", "inputs"].forEach((k) => {
      if (buf[k].length > state.maxBuffer) buf[k].splice(0, buf[k].length - state.maxBuffer);
    });

    state.step = buf.v.length - 1;
    state.phase = 0;
    state.lastFrameTs = performance.now();

    if (frame.input) fireDendrite();
    if (frame.spike) fireAxon();
  };
}

function fireDendrite() {
  const lines = dendritesRoot.querySelectorAll('.dendrite-line');
  if (!lines.length) return;
  const idx = Math.floor(Math.random() * lines.length);
  const line = lines[idx];
  const pulse = document.createElement('div');
  pulse.className = 'pulse';
  pulse.style.setProperty('--len', line.style.getPropertyValue('--len'));
   pulse.style.setProperty('--dur', '0.9s');
  pulse.addEventListener('animationend', () => pulse.remove());
  line.appendChild(pulse);
}

function fireAxon() {
  const line = axonRoot.querySelector('.axon-line');
  if (!line) return;
  const pulse = document.createElement('div');
  pulse.className = 'pulse';
  pulse.style.setProperty('--len', line.style.getPropertyValue('--len'));
   pulse.style.setProperty('--dur', '0.7s');
  pulse.addEventListener('animationend', () => pulse.remove());
  line.appendChild(pulse);
}

function drawMembrane() {
  const v = state.buffer.v;
  const threshold = state.buffer.threshold;
  const { width: w, height: h } = memCanvas;
  memCtx.clearRect(0, 0, w, h);
  const margin = 18;
  const baseY = h - margin;
  const maxV = Math.max(threshold * 1.2, ...v);
  const scale = (h - margin * 2) / Math.max(maxV, 1e-3);

  memCtx.strokeStyle = "#1f2430";
  memCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = margin + (i / 4) * (h - margin * 2);
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
  for (let i = 0; i < view; i++) {
    const idx = (state.step - i + v.length) % v.length;
    const x = w - margin - i * dx - state.phase * dx;
    const y = baseY - v[idx] * scale;
    if (i === 0) memCtx.moveTo(x, y);
    else memCtx.lineTo(x, y);
  }
  memCtx.stroke();
}

function drawSpikes() {
  const spikes = state.buffer.spikes;
  const { width: w, height: h } = spikeCanvas;
  const margin = 18;
  const baseY = h - margin;
  const dx = state.dx;
  const view = Math.min(state.viewSamples, spikes.length);

  spikeCtx.clearRect(0, 0, w, h);
  spikeCtx.strokeStyle = "#1f2430";
  spikeCtx.lineWidth = 1;
  for (let i = 0; i <= 3; i++) {
    const y = margin + (i / 3) * (h - margin * 2);
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
  for (let i = 0; i < view; i++) {
    const idx = (state.step - i + spikes.length) % spikes.length;
    if (spikes[idx]) {
      const x = w - margin - i * dx - state.phase * dx;
      spikeCtx.beginPath();
      spikeCtx.moveTo(x, baseY);
      spikeCtx.lineTo(x, margin + 10);
      spikeCtx.stroke();
    }
  }
}

function tick(ts) {
  if (!state.buffer.v.length) {
    state.lastTs = ts;
    requestAnimationFrame(tick);
    return;
  }

  if (!state.lastFrameTs) state.lastFrameTs = ts;
  const elapsed = ts - state.lastFrameTs;
  state.phase = Math.min(elapsed / state.stepMs, 1);

  drawMembrane();
  drawSpikes();

  requestAnimationFrame(tick);
}

createLines();
connectLive();
requestAnimationFrame(tick);
