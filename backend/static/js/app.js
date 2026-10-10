/**
 * OceanEmbed Frontend Core Application Engine.
 * SIH Problem Statement 26066.
 */

// Application Global State
const appState = {
  currentTab: 'dashboard',
  overview: null,
  metrics: null,
  currentSlice: null,
  currentProfile: null,
  currentTransect: null,
  embeddings: null,
  argoFloats: [],
  selectedArgoFloat: null,
  pipelineSpec: null,
  
  // Controls
  dayIdx: 0,
  variable: 'temperature',
  depthM: 0.0,
  mode: 'pred', // 'pred', 'target', 'error'
  colormap: 'thermal', // 'thermal', 'turbo', 'coolwarm', 'viridis', 'jet'
  
  // Coordinate Selection for vertical profile
  selectedLat: 15.0,
  selectedLon: 70.0,
  
  // Transect Controls
  transectAxis: 'lat',
  transectCoord: 15.0,
  
  // Simulator State
  simParams: {
    base_day_idx: 0,
    sst_delta_degC: 1.5,
    sss_delta_psu: 0.0,
    ssh_delta_m: 0.05,
    wind_scale_factor: 1.2,
    current_scale_factor: 1.0,
  },
  simResult: null,
  simLoading: false,
};

// Available standard depth levels
const STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000];

// Colormap Interpolators
function getColormapColor(val, minVal, maxVal, cmap = 'thermal') {
  if (val === null || val === undefined || isNaN(val)) {
    return [18, 28, 48, 255]; // Dark slate land mask
  }
  
  let t = (val - minVal) / (maxVal - minVal + 1e-6);
  t = Math.max(0, Math.min(1, t));

  if (cmap === 'thermal') {
    // Deep blue -> cyan -> yellow -> orange -> dark red
    if (t < 0.25) {
      const u = t / 0.25;
      return [Math.round(10 * (1-u) + 0 * u), Math.round(20 * (1-u) + 180 * u), Math.round(120 * (1-u) + 240 * u), 255];
    } else if (t < 0.5) {
      const u = (t - 0.25) / 0.25;
      return [Math.round(0 * (1-u) + 0 * u), Math.round(180 * (1-u) + 245 * u), Math.round(240 * (1-u) + 160 * u), 255];
    } else if (t < 0.75) {
      const u = (t - 0.5) / 0.25;
      return [Math.round(0 * (1-u) + 255 * u), Math.round(245 * (1-u) + 200 * u), Math.round(160 * (1-u) + 20 * u), 255];
    } else {
      const u = (t - 0.75) / 0.25;
      return [Math.round(255 * (1-u) + 220 * u), Math.round(200 * (1-u) + 20 * u), Math.round(20 * (1-u) + 20 * u), 255];
    }
  } else if (cmap === 'turbo') {
    // Turbo palette
    const r = Math.sin(t * Math.PI - Math.PI/2) * 127 + 128;
    const g = Math.sin(t * Math.PI) * 200 + 30;
    const b = Math.cos(t * Math.PI - Math.PI/2) * 127 + 128;
    return [Math.round(r), Math.round(g), Math.round(b), 255];
  } else if (cmap === 'coolwarm') {
    // Coolwarm (Blue -> White -> Red)
    if (t < 0.5) {
      const u = t / 0.5;
      return [Math.round(59 + 196 * u), Math.round(76 + 179 * u), Math.round(192 + 63 * u), 255];
    } else {
      const u = (t - 0.5) / 0.5;
      return [Math.round(255 - 75 * u), Math.round(255 - 217 * u), Math.round(255 - 217 * u), 255];
    }
  } else {
    // Viridis
    return [
      Math.round(68 + 180 * t),
      Math.round(1 + 220 * t),
      Math.round(84 + 100 * (1-t)),
      255
    ];
  }
}

// Initialize Application
document.addEventListener('DOMContentLoaded', async () => {
  setupNavigation();
  setupEventListeners();
  await loadInitialData();
});

// Setup Navigation Tabs
function setupNavigation() {
  const tabs = document.querySelectorAll('.nav-tab');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const tabId = tab.dataset.tab;
      switchTab(tabId);
    });
  });
}

function switchTab(tabId) {
  appState.currentTab = tabId;
  document.querySelectorAll('.nav-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === tabId);
  });
  document.querySelectorAll('.tab-view').forEach(view => {
    view.style.display = (view.id === `view-${tabId}`) ? 'block' : 'none';
  });

  // Trigger tab-specific refresh
  if (tabId === 'explorer') {
    renderSlice();
    renderProfile();
  } else if (tabId === 'transect') {
    loadAndRenderTransect();
  } else if (tabId === 'embeddings') {
    loadAndRenderEmbeddings();
  } else if (tabId === 'argo') {
    loadAndRenderArgo();
  } else if (tabId === 'simulator') {
    if (!appState.simResult) runSimulator();
  } else if (tabId === 'pipeline') {
    loadAndRenderPipeline();
  }
}

// Initial Data Loader
async function loadInitialData() {
  try {
    const [overview, metrics, argo] = await Promise.all([
      fetch('/api/overview').then(r => r.json()),
      fetch('/api/metrics').then(r => r.json()),
      fetch('/api/argo?limit=360').then(r => r.json()),
    ]);

    appState.overview = overview;
    appState.metrics = metrics;
    appState.argoFloats = argo;

    renderDashboard();
    await loadSlice();
    await loadProfile();
  } catch (err) {
    console.error('Failed to load initial data:', err);
  }
}

// Setup Interactive UI Event Listeners
function setupEventListeners() {
  // Day slider
  const daySlider = document.getElementById('day-slider');
  const dayVal = document.getElementById('day-slider-val');
  if (daySlider) {
    daySlider.addEventListener('input', (e) => {
      appState.dayIdx = parseInt(e.target.value);
      if (dayVal) dayVal.textContent = `Day +${appState.dayIdx + 1} Snapshot`;
      loadSlice();
      loadProfile();
    });
  }

  // Variable selector
  const varSelect = document.getElementById('var-select');
  if (varSelect) {
    varSelect.addEventListener('change', (e) => {
      appState.variable = e.target.value;
      const depthContainer = document.getElementById('depth-select-container');
      if (depthContainer) {
        depthContainer.style.display = (appState.variable === 'temperature') ? 'block' : 'none';
      }
      loadSlice();
    });
  }

  // Depth selector
  const depthSelect = document.getElementById('depth-select');
  if (depthSelect) {
    depthSelect.addEventListener('change', (e) => {
      appState.depthM = parseFloat(e.target.value);
      loadSlice();
    });
  }

  // Mode buttons (pred, target, error)
  document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('btn-primary'));
      document.querySelectorAll('.mode-btn').forEach(b => b.classList.add('btn-outline'));
      btn.classList.remove('btn-outline');
      btn.classList.add('btn-primary');
      appState.mode = btn.dataset.mode;
      loadSlice();
    });
  });

  // Colormap selector
  const cmapSelect = document.getElementById('cmap-select');
  if (cmapSelect) {
    cmapSelect.addEventListener('change', (e) => {
      appState.colormap = e.target.value;
      renderSlice();
    });
  }

  // Canvas interaction
  const canvas = document.getElementById('ocean-canvas');
  if (canvas) {
    canvas.addEventListener('mousemove', handleCanvasHover);
    canvas.addEventListener('click', handleCanvasClick);
  }

  // Transect controls
  const transectAxis = document.getElementById('transect-axis');
  const transectCoord = document.getElementById('transect-coord');
  if (transectAxis && transectCoord) {
    transectAxis.addEventListener('change', (e) => {
      appState.transectAxis = e.target.value;
      transectCoord.min = (appState.transectAxis === 'lat') ? 5 : 45;
      transectCoord.max = (appState.transectAxis === 'lat') ? 30 : 105;
      transectCoord.value = (appState.transectAxis === 'lat') ? 15 : 75;
      appState.transectCoord = parseFloat(transectCoord.value);
      loadAndRenderTransect();
    });
    transectCoord.addEventListener('input', (e) => {
      appState.transectCoord = parseFloat(e.target.value);
      document.getElementById('transect-coord-val').textContent = `${appState.transectCoord}°${appState.transectAxis === 'lat' ? 'N' : 'E'}`;
      loadAndRenderTransect();
    });
  }

  // Simulator controls
  const simSst = document.getElementById('sim-sst');
  if (simSst) {
    simSst.addEventListener('input', (e) => {
      appState.simParams.sst_delta_degC = parseFloat(e.target.value);
      document.getElementById('sim-sst-val').textContent = `${e.target.value > 0 ? '+' : ''}${e.target.value}°C`;
    });
  }
  const simSsh = document.getElementById('sim-ssh');
  if (simSsh) {
    simSsh.addEventListener('input', (e) => {
      appState.simParams.ssh_delta_m = parseFloat(e.target.value);
      document.getElementById('sim-ssh-val').textContent = `${e.target.value > 0 ? '+' : ''}${e.target.value}m`;
    });
  }
  const simWind = document.getElementById('sim-wind');
  if (simWind) {
    simWind.addEventListener('input', (e) => {
      appState.simParams.wind_scale_factor = parseFloat(e.target.value);
      document.getElementById('sim-wind-val').textContent = `${Math.round(e.target.value * 100)}%`;
    });
  }
  const simBtn = document.getElementById('run-sim-btn');
  if (simBtn) {
    simBtn.addEventListener('click', runSimulator);
  }
}

// Render Executive Dashboard
function renderDashboard() {
  if (!appState.metrics || !appState.overview) return;

  const s = appState.metrics.summary;
  document.getElementById('kpi-rmse').textContent = `${s.mean_rmse.toFixed(3)} °C`;
  document.getElementById('kpi-corr').textContent = `${s.mean_corr.toFixed(3)}`;
  document.getElementById('kpi-d20').textContent = `${s.thermocline_d20_rmse_m.toFixed(2)} m`;
  document.getElementById('kpi-argo').textContent = `${s.argo_float_count} Floats`;
  document.getElementById('kpi-argo-rmse').textContent = `${s.argo_mean_rmse.toFixed(3)} °C`;

  // Render Depth-wise Benchmark Table
  const tableBody = document.getElementById('metrics-table-body');
  if (tableBody && appState.metrics.depth_metrics) {
    tableBody.innerHTML = appState.metrics.depth_metrics.map(d => `
      <tr class="border-b border-ocean-800/40 hover:bg-white/5 transition-colors">
        <td class="py-2.5 px-4 font-mono font-semibold text-cyan-400">${d.depth_m} m</td>
        <td class="py-2.5 px-4 font-mono">${d.rmse.toFixed(4)} °C</td>
        <td class="py-2.5 px-4 font-mono text-emerald-400 font-semibold">${d.correlation.toFixed(4)}</td>
        <td class="py-2.5 px-4 font-mono">${d.bias > 0 ? '+' : ''}${d.bias.toFixed(4)} °C</td>
        <td class="py-2.5 px-4 font-mono">${d.mae.toFixed(4)} °C</td>
      </tr>
    `).join('');
  }

  // Render Benchmark Models Comparison Table
  const benchBody = document.getElementById('benchmarks-table-body');
  if (benchBody && appState.metrics.benchmarks) {
    benchBody.innerHTML = appState.metrics.benchmarks.map(b => `
      <tr class="border-b border-ocean-800/40 ${b.is_current ? 'bg-cyan-500/10 border-cyan-500/30' : 'hover:bg-white/5'} transition-colors">
        <td class="py-3 px-4 font-semibold ${b.is_current ? 'text-cyan-300' : 'text-slate-200'}">
          ${b.model_name}
          ${b.is_current ? '<span class="ml-2 badge badge-cyan">Our Proposed Net</span>' : ''}
        </td>
        <td class="py-3 px-4 text-xs text-slate-400">${b.type}</td>
        <td class="py-3 px-4 font-mono font-bold ${b.is_current ? 'text-emerald-400' : 'text-slate-300'}">${b.mean_rmse_degC.toFixed(4)} °C</td>
        <td class="py-3 px-4 font-mono font-bold ${b.is_current ? 'text-cyan-400' : 'text-slate-300'}">${b.upper_200m_corr.toFixed(4)}</td>
        <td class="py-3 px-4 font-mono">${b.thermocline_d20_rmse_m.toFixed(2)} m</td>
        <td class="py-3 px-4 font-mono text-xs text-slate-400">${b.inference_time_ms} ms</td>
      </tr>
    `).join('');
  }
}

// Load and Render 2D Ocean Grid Slice
async function loadSlice() {
  try {
    const url = `/api/slice?day_idx=${appState.dayIdx}&variable=${appState.variable}&depth_m=${appState.depthM}&mode=${appState.mode}`;
    const res = await fetch(url);
    appState.currentSlice = await res.json();
    renderSlice();
  } catch (err) {
    console.error('Error fetching slice:', err);
  }
}

function renderSlice() {
  const slice = appState.currentSlice;
  if (!slice || !slice.grid) return;

  const canvas = document.getElementById('ocean-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const rows = slice.grid.length; // 101
  const cols = slice.grid[0].length; // 241

  canvas.width = cols;
  canvas.height = rows;

  const imgData = ctx.createImageData(cols, rows);
  const data = imgData.data;

  const minV = slice.min_val;
  const maxV = slice.max_val;

  // Render pixels (flipping Y so North is up)
  for (let r = 0; r < rows; r++) {
    const gridY = rows - 1 - r; // Invert latitude row
    for (let c = 0; c < cols; c++) {
      const val = slice.grid[gridY][c];
      const [red, green, blue, alpha] = getColormapColor(val, minV, maxV, appState.colormap);
      const idx = (r * cols + c) * 4;
      data[idx] = red;
      data[idx + 1] = green;
      data[idx + 2] = blue;
      data[idx + 3] = alpha;
    }
  }

  ctx.putImageData(imgData, 0, 0);

  // Overlay selected coordinate crosshair
  if (appState.selectedLat && appState.selectedLon) {
    const latFrac = (appState.selectedLat - 5.0) / (30.0 - 5.0);
    const lonFrac = (appState.selectedLon - 45.0) / (105.0 - 45.0);
    const cx = Math.round(lonFrac * cols);
    const cy = Math.round((1 - latFrac) * rows);

    ctx.strokeStyle = '#00f2fe';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
    ctx.stroke();
  }

  // Update Slice Metadata Labels
  document.getElementById('slice-min-val').textContent = `${minV.toFixed(2)} ${slice.variable === 'temperature' ? '°C' : ''}`;
  document.getElementById('slice-max-val').textContent = `${maxV.toFixed(2)} ${slice.variable === 'temperature' ? '°C' : ''}`;
  document.getElementById('slice-mean-val').textContent = `${slice.mean_val.toFixed(2)} ${slice.variable === 'temperature' ? '°C' : ''}`;
}

// Canvas Hover and Click Handlers
function handleCanvasHover(e) {
  const canvas = e.target;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;

  const lonFrac = x / rect.width;
  const latFrac = 1 - (y / rect.height);

  const lon = 45.0 + lonFrac * (105.0 - 45.0);
  const lat = 5.0 + latFrac * (30.0 - 5.0);

  const tooltip = document.getElementById('canvas-tooltip');
  if (tooltip) {
    tooltip.style.display = 'block';
    tooltip.style.left = `${x}px`;
    tooltip.style.top = `${y}px`;
    tooltip.innerHTML = `<strong>${lat.toFixed(2)}°N, ${lon.toFixed(2)}°E</strong><br><span class="text-xs text-cyan-300">Click to inspect 3D profile</span>`;
  }
}

function handleCanvasClick(e) {
  const canvas = e.target;
  const rect = canvas.getBoundingClientRect();
  const lonFrac = (e.clientX - rect.left) / rect.width;
  const latFrac = 1 - ((e.clientY - rect.top) / rect.height);

  appState.selectedLon = Math.round((45.0 + lonFrac * 60.0) * 4) / 4;
  appState.selectedLat = Math.round((5.0 + latFrac * 25.0) * 4) / 4;

  renderSlice();
  loadProfile();
}

// Load and Render 1D Vertical Temperature Profile
async function loadProfile() {
  try {
    const url = `/api/profile?day_idx=${appState.dayIdx}&lat=${appState.selectedLat}&lon=${appState.selectedLon}`;
    const res = await fetch(url);
    appState.currentProfile = await res.json();
    renderProfile();
  } catch (err) {
    console.error('Error fetching profile:', err);
  }
}

function renderProfile() {
  const prof = appState.currentProfile;
  if (!prof) return;

  document.getElementById('profile-coord-label').textContent = `${prof.grid_lat.toFixed(2)}°N, ${prof.grid_lon.toFixed(2)}°E`;
  document.getElementById('profile-ocean-badge').innerHTML = prof.is_ocean 
    ? '<span class="badge badge-emerald">Ocean Cell</span>' 
    : '<span class="badge badge-amber">Land Mask</span>';

  document.getElementById('prof-d20-pred').textContent = prof.pred_d20_m ? `${prof.pred_d20_m} m` : 'N/A';
  document.getElementById('prof-d20-target').textContent = prof.target_d20_m ? `${prof.target_d20_m} m` : 'N/A';
  document.getElementById('prof-sst-val').textContent = `${prof.surface_inputs.sst}°C`;
  document.getElementById('prof-sss-val').textContent = `${prof.surface_inputs.sss} PSU`;
  document.getElementById('prof-ssh-val').textContent = `${prof.surface_inputs.ssh} m`;
  document.getElementById('prof-wind-val').textContent = `${prof.surface_inputs.u_wind}, ${prof.surface_inputs.v_wind} m/s`;

  // Draw Vertical Profile Chart on Canvas
  const canvas = document.getElementById('profile-chart-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = 440;
  const h = canvas.height = 360;

  ctx.clearRect(0, 0, w, h);

  const padLeft = 45;
  const padRight = 20;
  const padTop = 25;
  const padBottom = 35;

  const chartW = w - padLeft - padRight;
  const chartH = h - padTop - padBottom;

  const minT = 0.0;
  const maxT = 32.0;
  const maxZ = 1000.0;

  // Draw Grid Lines
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.lineWidth = 1;
  ctx.font = '10px JetBrains Mono';
  ctx.fillStyle = '#64748b';

  // Temperature X Grid
  for (let t = 0; t <= 30; t += 5) {
    const x = padLeft + (t / maxT) * chartW;
    ctx.beginPath();
    ctx.moveTo(x, padTop);
    ctx.lineTo(x, padTop + chartH);
    ctx.stroke();
    ctx.fillText(`${t}°`, x - 6, padTop + chartH + 15);
  }

  // Depth Y Grid (Non-linear squish for upper ocean emphasis)
  const mapZ = (z) => padTop + Math.pow(z / maxZ, 0.6) * chartH;

  const gridDepths = [0, 50, 100, 200, 500, 1000];
  gridDepths.forEach(z => {
    const y = mapZ(z);
    ctx.beginPath();
    ctx.moveTo(padLeft, y);
    ctx.lineTo(padLeft + chartW, y);
    ctx.stroke();
    ctx.fillText(`${z}m`, 5, y + 3);
  });

  const mapX = (t) => padLeft + Math.max(0, Math.min(1, t / maxT)) * chartW;

  // Draw GLORYS Target Profile (Blue line)
  ctx.strokeStyle = '#4facfe';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  prof.depths.forEach((z, i) => {
    const px = mapX(prof.target_temperature[i]);
    const py = mapZ(z);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();

  // Draw OceanEmbed Predicted Profile (Red/Orange dashed line)
  ctx.strokeStyle = '#ff5858';
  ctx.lineWidth = 2.0;
  ctx.setLineDash([5, 3]);
  ctx.beginPath();
  prof.depths.forEach((z, i) => {
    const px = mapX(prof.pred_temperature[i]);
    const py = mapZ(z);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.stroke();
  ctx.setLineDash([]); // Reset line dash

  // Draw Nearest ARGO Float (Green circles)
  if (prof.nearest_argo) {
    ctx.fillStyle = '#00f5a0';
    prof.depths.forEach((z, i) => {
      const px = mapX(prof.nearest_argo.temperature[i]);
      const py = mapZ(z);
      ctx.beginPath();
      ctx.arc(px, py, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    });
    document.getElementById('argo-match-info').textContent = `ARGO float within ${prof.nearest_argo.distance_deg}° match`;
  } else {
    document.getElementById('argo-match-info').textContent = 'No ARGO float within 3.0°';
  }

  // Draw D20 Thermocline Depth line if exists
  if (prof.pred_d20_m) {
    const d20Y = mapZ(prof.pred_d20_m);
    ctx.strokeStyle = 'rgba(246, 211, 101, 0.7)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(padLeft, d20Y);
    ctx.lineTo(padLeft + chartW, d20Y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#f6d365';
    ctx.fillText(`D20: ${prof.pred_d20_m}m`, padLeft + chartW - 75, d20Y - 4);
  }
}

// Load and Render Transect
async function loadAndRenderTransect() {
  try {
    const url = `/api/transect?day_idx=${appState.dayIdx}&axis=${appState.transectAxis}&coord_val=${appState.transectCoord}`;
    const res = await fetch(url);
    appState.currentTransect = await res.json();
    renderTransect();
  } catch (err) {
    console.error('Error fetching transect:', err);
  }
}

function renderTransect() {
  const tr = appState.currentTransect;
  if (!tr || !tr.pred_transect) return;

  const canvas = document.getElementById('transect-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  const nDepths = tr.depths.length; // 15
  const nCoords = tr.coords.length;

  const w = canvas.width = 750;
  const h = canvas.height = 280;

  ctx.clearRect(0, 0, w, h);

  const colW = w / nCoords;
  const rowH = h / nDepths;

  // Render depth vs coordinate heatmap
  for (let d = 0; d < nDepths; d++) {
    for (let c = 0; c < nCoords; c++) {
      const val = tr.pred_transect[d][c];
      const [red, green, blue, alpha] = getColormapColor(val, 4.0, 31.0, 'thermal');
      ctx.fillStyle = `rgba(${red}, ${green}, ${blue}, ${alpha / 255})`;
      ctx.fillRect(c * colW, d * rowH, colW + 0.5, rowH + 0.5);
    }
  }

  // Draw Isotherm contour lines (28C, 20C, 15C)
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
  ctx.lineWidth = 1.2;
}

// Load and Render Latent Satellite Embeddings
async function loadAndRenderEmbeddings() {
  try {
    const res = await fetch(`/api/embeddings?day_idx=${appState.dayIdx}`);
    appState.embeddings = await res.json();
    renderEmbeddings();
  } catch (err) {
    console.error('Error fetching embeddings:', err);
  }
}

function renderEmbeddings() {
  const emb = appState.embeddings;
  if (!emb) return;

  // Render Top 6 Latent Embedding Channels
  const gridContainer = document.getElementById('embedding-channels-grid');
  if (gridContainer && emb.top_channels) {
    gridContainer.innerHTML = emb.top_channels.map(ch => `
      <div class="glass-panel p-3">
        <div class="flex justify-between items-center mb-2">
          <span class="text-xs font-mono font-semibold text-cyan-300">Channel #${ch.channel_id}</span>
          <span class="text-xs text-slate-400">Var: ${ch.variance}</span>
        </div>
        <div class="h-24 bg-black/50 rounded flex items-center justify-center border border-white/5 overflow-hidden">
          <canvas id="emb-canvas-${ch.channel_id}" class="w-full h-full"></canvas>
        </div>
      </div>
    `).join('');

    // Draw each channel thumbnail
    emb.top_channels.forEach(ch => {
      const cv = document.getElementById(`emb-canvas-${ch.channel_id}`);
      if (cv && ch.grid) {
        const ctx = cv.getContext('2d');
        const rows = ch.grid.length;
        const cols = ch.grid[0].length;
        cv.width = cols;
        cv.height = rows;
        const imgData = ctx.createImageData(cols, rows);
        const d = imgData.data;
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const val = ch.grid[r][c];
            const [red, green, blue] = getColormapColor(val, -2.5, 2.5, 'turbo');
            const idx = (r * cols + c) * 4;
            d[idx] = red; d[idx+1] = green; d[idx+2] = blue; d[idx+3] = 255;
          }
        }
        ctx.putImageData(imgData, 0, 0);
      }
    });
  }

  // Render Depth Attention Weights
  const attnContainer = document.getElementById('attention-weights-list');
  if (attnContainer && emb.depth_attention_weights) {
    attnContainer.innerHTML = emb.depth_attention_weights.map(a => `
      <div class="mb-3">
        <div class="flex justify-between text-xs mb-1 font-mono">
          <span class="text-cyan-400 font-bold">${a.depth}m Depth</span>
          <span class="text-slate-400">SSH: ${Math.round(a.ssh_weight*100)}% | SST: ${Math.round(a.sst_weight*100)}% | SSS: ${Math.round(a.sss_weight*100)}%</span>
        </div>
        <div class="h-3.5 bg-ocean-900 rounded-full flex overflow-hidden border border-white/10">
          <div class="bg-rose-500 h-full" style="width: ${a.sst_weight*100}%" title="SST: ${Math.round(a.sst_weight*100)}%"></div>
          <div class="bg-amber-400 h-full" style="width: ${a.sss_weight*100}%" title="SSS: ${Math.round(a.sss_weight*100)}%"></div>
          <div class="bg-cyan-400 h-full" style="width: ${a.ssh_weight*100}%" title="SSH: ${Math.round(a.ssh_weight*100)}%"></div>
          <div class="bg-emerald-400 h-full" style="width: ${a.curr_weight*100}%" title="Currents: ${Math.round(a.curr_weight*100)}%"></div>
          <div class="bg-indigo-400 h-full" style="width: ${a.wind_weight*100}%" title="Winds: ${Math.round(a.wind_weight*100)}%"></div>
        </div>
      </div>
    `).join('');
  }
}

// Load and Render ARGO Validation Suite
async function loadAndRenderArgo() {
  const container = document.getElementById('argo-floats-table');
  if (!container || !appState.argoFloats) return;

  container.innerHTML = appState.argoFloats.slice(0, 50).map(f => `
    <tr class="border-b border-ocean-800/40 hover:bg-cyan-500/10 cursor-pointer transition-colors" onclick="selectArgoFloat('${f.id}')">
      <td class="py-2 px-3 font-mono text-xs text-cyan-300">${f.id}</td>
      <td class="py-2 px-3 font-mono text-xs">${f.lat.toFixed(2)}°N, ${f.lon.toFixed(2)}°E</td>
      <td class="py-2 px-3 font-mono text-xs">${f.surface_temp.toFixed(2)}°C</td>
      <td class="py-2 px-3 font-mono text-xs">${f.thermocline_temp_100m.toFixed(2)}°C</td>
      <td class="py-2 px-3 font-mono text-xs">${f.deep_temp_1000m.toFixed(2)}°C</td>
      <td class="py-2 px-3 text-xs text-right">
        <button class="btn btn-outline py-1 px-2 text-xs">Inspect CTD</button>
      </td>
    </tr>
  `).join('');
}

window.selectArgoFloat = function(floatId) {
  const f = appState.argoFloats.find(item => item.id === floatId);
  if (!f) return;
  appState.selectedLat = f.lat;
  appState.selectedLon = f.lon;
  appState.dayIdx = f.day_idx;
  switchTab('explorer');
  loadSlice();
  loadProfile();
};

// Run Climate What-If Simulator
async function runSimulator() {
  const btn = document.getElementById('run-sim-btn');
  if (btn) btn.innerHTML = '<span class="pulse-dot mr-2"></span> Reconstructing 3D Field...';

  try {
    const res = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(appState.simParams),
    });
    appState.simResult = await res.json();
    renderSimulatorResults();
  } catch (err) {
    console.error('Simulation error:', err);
  } finally {
    if (btn) btn.innerHTML = '⚡ Run Deep Learning Model Forward Pass';
  }
}

function renderSimulatorResults() {
  const res = appState.simResult;
  if (!res) return;

  const imp = res.subsurface_impact;
  document.getElementById('sim-delta-surface').textContent = `${imp.mean_surface_delta_degC > 0 ? '+' : ''}${imp.mean_surface_delta_degC}°C`;
  document.getElementById('sim-delta-thermo').textContent = `${imp.mean_thermocline_delta_degC > 0 ? '+' : ''}${imp.mean_thermocline_delta_degC}°C`;
  document.getElementById('sim-delta-deep').textContent = `${imp.mean_deep_delta_degC > 0 ? '+' : ''}${imp.mean_deep_delta_degC}°C`;
  document.getElementById('sim-penetration').textContent = `${imp.thermal_penetration_depth_m} meters`;

  // Draw Simulation Response Profiles
  const container = document.getElementById('sim-stations-container');
  if (container && res.sample_profiles_at_key_stations) {
    container.innerHTML = res.sample_profiles_at_key_stations.map((st, idx) => `
      <div class="glass-panel p-4">
        <h4 class="font-bold text-sm text-cyan-300 mb-2">${st.station}</h4>
        <div class="text-xs space-y-1 font-mono">
          <div class="flex justify-between"><span>Surface:</span><span class="text-rose-400 font-bold">${st.baseline_temp[0]}°C → ${st.simulated_temp[0]}°C</span></div>
          <div class="flex justify-between"><span>100m Core:</span><span class="text-amber-400 font-bold">${st.baseline_temp[7]}°C → ${st.simulated_temp[7]}°C</span></div>
          <div class="flex justify-between"><span>500m Layer:</span><span class="text-cyan-400">${st.baseline_temp[12]}°C → ${st.simulated_temp[12]}°C</span></div>
        </div>
      </div>
    `).join('');
  }
}

// Load and Render Data Harmonization Pipeline
async function loadAndRenderPipeline() {
  try {
    const res = await fetch('/api/pipeline/spec');
    appState.pipelineSpec = await res.json();
    const p = appState.pipelineSpec;

    const srcContainer = document.getElementById('pipeline-sources-container');
    if (srcContainer && p.sources) {
      srcContainer.innerHTML = p.sources.map(s => `
        <div class="glass-panel p-4 flex flex-col justify-between">
          <div>
            <div class="flex justify-between items-start mb-2">
              <span class="badge badge-cyan">${s.variable}</span>
              <span class="badge badge-emerald">${s.status}</span>
            </div>
            <h5 class="font-bold text-sm text-slate-100">${s.product}</h5>
            <p class="text-xs text-slate-400 mt-1">Native: ${s.native_res}</p>
          </div>
          <div class="mt-3 pt-2 border-t border-white/5 text-xs text-cyan-300/80">
            Harmonization: ${s.interpolation}
          </div>
        </div>
      `).join('');
    }
  } catch (err) {
    console.error('Error loading pipeline spec:', err);
  }
}
