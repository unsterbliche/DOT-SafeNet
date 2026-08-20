const colors = ["#2563eb", "#059669", "#dc2626", "#d97706", "#7c3aed", "#475569"];
const MAX_MOLECULES = 10;

let currentResult = null;
let currentJobId = null;
let currentDisplayIndex = 0;
let sampleCatalog = [];
let pollTimer = null;

const $ = (selector) => document.querySelector(selector);

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    const message = Array.isArray(detail.detail)
      ? detail.detail.map(item => `${item.loc?.join(".") || "request"}: ${item.msg}`).join("; ")
      : detail.detail || response.statusText;
    throw new Error(message);
  }
  return response.json();
}

async function svgApi(path, body) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail || response.statusText);
  }
  return response.text();
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(value)) return "NA";
  return Number(value).toFixed(digits);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function init() {
  await loadSamples();
  await loadJobs();
  bindNavigation();
  $("#manualSubmit").addEventListener("click", submitManual);
  $("#previewMolecule").addEventListener("click", previewMolecule);
  $("#addMolecule").addEventListener("click", addMoleculeToBatch);
  $("#structureFile").addEventListener("change", handleFileUpload);
  $("#refreshJobs").addEventListener("click", loadJobs);
  $("#jobSelect").addEventListener("change", event => {
    if (event.target.value) openJob(event.target.value);
  });
  window.addEventListener("hashchange", () => showPage(pageFromHash()));
  showPage(pageFromHash());
  previewMolecule().catch(() => {});
}

function bindNavigation() {
  document.querySelectorAll("[data-page-link]").forEach(link => {
    link.addEventListener("click", event => {
      const page = link.dataset.pageLink;
      if (!page) return;
      event.preventDefault();
      history.pushState(null, "", `#${page}`);
      showPage(page);
    });
  });
}

function pageFromHash() {
  const page = window.location.hash.replace("#", "") || "home";
  return ["home", "predict", "jobs", "help"].includes(page) ? page : "home";
}

function showPage(page) {
  document.querySelectorAll(".app-page").forEach(section => {
    section.classList.toggle("active", section.dataset.page === page);
  });
  document.querySelectorAll("[data-page-link]").forEach(link => {
    link.classList.toggle("active", link.dataset.pageLink === page);
  });
  if (page === "jobs") loadJobs().catch(() => {});
  window.scrollTo({ top: 0, behavior: "instant" });
}

async function loadSlides() {
  const slides = await api("/api/assets/slides").catch(() => []);
  const first = slides.find(slide => slide.name === "slide_1" && slide.url);
  if (first) $("#figure1").src = first.url;

  const selected = slides.filter(slide => ["slide_1", "slide_2", "slide_3", "slide_4", "slide_5"].includes(slide.name) && slide.url);
  $("#slideGallery").innerHTML = selected.map(slide => `
    <figure>
      <img src="${escapeHtml(slide.url)}" alt="${escapeHtml(slide.name)}">
      <figcaption>${escapeHtml(slide.name.replace("_", " "))}</figcaption>
    </figure>
  `).join("");
}

async function loadSamples() {
  sampleCatalog = await api("/api/samples");
  $("#samples").innerHTML = sampleCatalog.map(sample => `
    <button class="sample-card" type="button" data-sample="${escapeHtml(sample.key)}">
      <h4>${escapeHtml(sample.name)}</h4>
      <div class="dose-tags">${sample.dose_panel_mg.map(d => `<span>${d}mg</span>`).join("")}</div>
    </button>
  `).join("");
  document.querySelectorAll(".sample-card[data-sample]").forEach(card => {
    card.addEventListener("click", () => loadSampleIntoForm(card.dataset.sample));
  });
  loadDefaultSamplesIntoForm();
}

async function loadJobs() {
  const jobs = await api("/api/jobs").catch(() => []);
  if (!jobs.length) {
    $("#jobSelect").innerHTML = `<option value="">No saved tasks yet</option>`;
    return;
  }
  const seenNames = new Map();
  $("#jobSelect").innerHTML = `<option value="">Select a saved prediction</option>` + jobs.map((job, index) => {
    const date = job.finished_at || job.created_at;
    const when = date ? new Date(date * 1000).toLocaleString() : "pending";
    const doses = (job.dose_panel_mg || []).join(", ");
    const name = job.compound_name || "Prediction";
    const duplicateIndex = (seenNames.get(name) || 0) + 1;
    seenNames.set(name, duplicateIndex);
    const count = job.batch_size > 1 ? `${job.batch_size} compounds` : "1 compound";
    const warningText = (job.warnings || []).length ? ` | ${job.warnings[0]}` : "";
    return `<option value="${escapeHtml(job.job_id)}">${escapeHtml(name)} (${duplicateIndex}) | ${escapeHtml(count)} | ${escapeHtml(job.status || "unknown")} | ${escapeHtml(doses)} mg | ${escapeHtml(when)}${escapeHtml(warningText)}</option>`;
  }).join("");
  if (currentJobId) $("#jobSelect").value = currentJobId;
}

async function runSample(key) {
  setResultMessage(`Running ${key} sample...`);
  const job = await api("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sample_key: key }),
  });
  await openJob(job.job_id);
  await loadJobs();
  history.pushState(null, "", "#jobs");
  showPage("jobs");
}

function loadSampleIntoForm(key) {
  const sample = sampleCatalog.find(item => item.key === key);
  if (!sample) return;
  $("#drawName").value = sample.name;
  $("#drawSmiles").value = sample.smiles;
  $("#manualDoses").value = sample.dose_panel_mg.join(",");
  $("#manualSmiles").value = sampleInputLine(sample);
  setInputWarning(`${sample.name} loaded. Click Submit prediction to run.`);
  previewMolecule().catch(() => {});
  document.querySelectorAll(".sample-card[data-sample]").forEach(card => {
    card.classList.toggle("active", card.dataset.sample === key);
  });
}

function loadDefaultSamplesIntoForm() {
  if (!sampleCatalog.length) return;
  const example = sampleCatalog.find(item => item.key === "citalopram") || sampleCatalog[0];
  $("#drawName").value = example.name;
  $("#drawSmiles").value = example.smiles;
  $("#manualDoses").value = example.dose_panel_mg.join(",");
  $("#manualSmiles").value = sampleCatalog.map(sampleInputLine).join("\n");
}

function sampleInputLine(sample) {
  return `${sample.name},${sample.smiles},${sample.dose_panel_mg.join("|")}`;
}

async function submitManual() {
  try {
    const parsed = collectInputItems();
    if (!parsed.items.length) throw new Error("Enter at least one molecule, for example: erdafitinib,COc1cc(OC)cc(N(CCNC(C)C)c2ccc3ncc(-c4cnn(C)c4)nc3c2)c1");
    const dosePanel = parseDoses($("#manualDoses").value);
    if (!dosePanel.length) throw new Error("Enter at least one dose in mg, for example: 5,10,25");
    if (parsed.warning) setInputWarning(parsed.warning);

    setResultMessage("Running prediction. Live inference can take a while for new molecules.");
    const job = await api("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ items: parsed.items, dose_panel_mg: dosePanel }),
    });
    currentJobId = job.job_id;
    history.pushState(null, "", "#jobs");
    showPage("jobs");
    renderWaitingJob({
      status: "queued",
      stage: "prediction submitted",
      progress: 0,
      job_id: job.job_id,
    });
    await loadJobs();
    await openJob(job.job_id);
  } catch (error) {
    $("#jobStatus").textContent = "Prediction input needs attention";
    $("#resultPanel").className = "result-panel empty";
    $("#resultPanel").textContent = error.message;
  }
}

async function openJob(jobId) {
  currentJobId = jobId;
  const status = await api(`/api/jobs/${jobId}`);
  if (status.status !== "succeeded") {
    $("#jobStatus").textContent = `Task ${status.status}`;
    renderWaitingJob(status);
    markActiveJob(jobId);
    if (status.status === "queued" || status.status === "running") scheduleJobPoll(jobId);
    return;
  }
  stopJobPoll();
  currentResult = await api(`/api/jobs/${jobId}/results`);
  currentDisplayIndex = 0;
  renderResult(currentResult);
  markActiveJob(jobId);
  $("#resultPanel").scrollIntoView({ behavior: "smooth", block: "start" });
}

function scheduleJobPoll(jobId) {
  stopJobPoll();
  pollTimer = window.setTimeout(async () => {
    try {
      await loadJobs();
      await openJob(jobId);
    } catch (error) {
      $("#jobStatus").textContent = "Task status refresh failed";
      $("#resultPanel").className = "result-panel empty";
      $("#resultPanel").textContent = error.message;
    }
  }, 3000);
}

function stopJobPoll() {
  if (pollTimer) {
    window.clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function renderWaitingJob(status) {
  const progress = Number(status.progress || 0);
  const stage = status.error || status.stage || "Waiting for worker";
  $("#resultPanel").className = `result-panel empty ${status.status === "failed" ? "failed" : "waiting"}`;
  $("#resultPanel").innerHTML = `
    <div class="waiting-card">
      <h3>${escapeHtml(status.status === "failed" ? "Prediction failed" : "Prediction is running")}</h3>
      <p>${escapeHtml(stage)}</p>
      <div class="progress-track"><div style="width: ${Math.max(5, Math.min(100, progress))}%"></div></div>
      <p class="muted">Job ID: ${escapeHtml(status.job_id || currentJobId || "")}</p>
    </div>
  `;
}

function markActiveJob(jobId) {
  if ($("#jobSelect")) $("#jobSelect").value = jobId;
}

function setResultMessage(message) {
  $("#jobStatus").textContent = message;
  $("#resultPanel").className = "result-panel empty";
  $("#resultPanel").textContent = message;
}

function parseDoses(text) {
  return text.replace(/\uFF0C/g, ",").split(/[,\|]+/)
    .map(value => Number.parseFloat(value.trim()))
    .filter(value => Number.isFinite(value) && value > 0);
}

function collectInputItems() {
  const items = parseManualItems($("#manualSmiles").value);
  const unique = [];
  const seen = new Set();
  items.forEach((item, index) => {
    const smiles = (item.smiles || "").trim();
    if (!smiles) return;
    const key = smiles.toLowerCase();
    if (seen.has(key)) return;
    seen.add(key);
    const payload = { name: (item.name || `compound_${index + 1}`).trim(), smiles };
    if (item.dose_panel_mg?.length) payload.dose_panel_mg = item.dose_panel_mg;
    unique.push(payload);
  });
  if (unique.length > MAX_MOLECULES) {
    return {
      items: unique.slice(0, MAX_MOLECULES),
      warning: `Received ${unique.length} molecules; only the first ${MAX_MOLECULES} will be predicted.`,
    };
  }
  return { items: unique, warning: "" };
}

function parseManualItems(text) {
  return text
    .replace(/\uFF1B/g, ";")
    .split(/\r?\n|;/)
    .map(line => line.trim())
    .filter(Boolean)
    .map(parseManualLine)
    .filter(Boolean);
}

function parseManualLine(line) {
  const normalized = line.replace(/\uFF0C/g, ",").trim();
  if (/^name\s*,\s*smiles$/i.test(normalized) || /^smiles$/i.test(normalized)) return null;
  if (normalized.includes(",")) {
    const [name, smiles, ...rest] = normalized.split(",");
    const dosePanel = rest.length ? parseDoses(rest.join(",")) : [];
    return smiles
      ? { name: name.trim() || undefined, smiles: smiles.trim(), dose_panel_mg: dosePanel }
      : { smiles: name.trim(), dose_panel_mg: dosePanel };
  }
  const parts = normalized.split(/\s+/);
  if (parts.length >= 2) return { name: parts[0], smiles: parts.slice(1).join("") };
  return { smiles: normalized };
}

function setInputWarning(message) {
  $("#inputWarning").textContent = message || "";
}

async function previewMolecule() {
  const smiles = $("#drawSmiles").value.trim();
  if (!smiles) {
    $("#moleculePreview").textContent = "Enter a SMILES to preview.";
    return;
  }
  $("#moleculePreview").textContent = "Rendering molecule...";
  try {
    const svg = await svgApi("/api/depict", { smiles });
    $("#moleculePreview").innerHTML = svg;
  } catch (error) {
    $("#moleculePreview").textContent = error.message;
  }
}

function addMoleculeToBatch() {
  const name = $("#drawName").value.trim() || "drawn_molecule";
  const smiles = $("#drawSmiles").value.trim();
  if (!smiles) {
    setInputWarning("Enter a SMILES before adding it to the batch.");
    return;
  }
  appendManualItems([{ name, smiles }]);
  setInputWarning(`${name} added to the batch list.`);
}

function appendManualItems(items) {
  const current = $("#manualSmiles").value.trim();
  const addition = items.map(item => `${item.name || "compound"},${item.smiles}`).join("\n");
  $("#manualSmiles").value = [current, addition].filter(Boolean).join("\n");
}

async function handleFileUpload(event) {
  const file = event.target.files?.[0];
  if (!file) return;
  const text = await file.text();
  const suffix = file.name.split(".").pop().toLowerCase();
  try {
    let items = [];
    if (suffix === "sdf" || suffix === "mol2") {
      const payload = await api("/api/structures/parse", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: file.name, content: text }),
      });
      items = payload.items || [];
    } else {
      items = parseManualItems(text);
    }
    if (!items.length) throw new Error("No molecules were found in the uploaded file.");
    const warning = items.length > MAX_MOLECULES
      ? `Uploaded ${items.length} molecules; only the first ${MAX_MOLECULES} will be predicted.`
      : `Loaded ${items.length} molecule${items.length === 1 ? "" : "s"} from ${file.name}.`;
    appendManualItems(items.slice(0, MAX_MOLECULES));
    setInputWarning(warning);
  } catch (error) {
    setInputWarning(error.message);
  } finally {
    event.target.value = "";
  }
}

function renderResult(result) {
  const displayResult = getDisplayResult(result);
  const batchText = result.results
    ? `Completed ${result.results.length} compounds; viewing ${displayResult.compound.name}`
    : `Completed ${displayResult.compound.name}`;
  const warnings = result.warnings || [];
  $("#jobStatus").textContent = `${batchText} (${displayResult.dose_panel_mg.join(", ")} mg)`;

  const batchPicker = result.results ? `
    <label class="batch-picker">
      Compound
      <select id="resultPicker">
        ${result.results.map((item, index) => `<option value="${index}" ${index === currentDisplayIndex ? "selected" : ""}>${escapeHtml(item.compound.name)}</option>`).join("")}
      </select>
    </label>
  ` : "";

  $("#resultPanel").className = "result-panel";
  $("#resultPanel").innerHTML = `
    <div class="result-stack">
      <div class="result-header">
        <div class="result-actions result-actions-top">
          ${batchPicker}
          <a class="download-link" href="/api/jobs/${currentJobId}/results.csv">Download CSV</a>
        </div>
        <div class="result-identity">
          <h3>${escapeHtml(displayResult.compound.name)}</h3>
          <p class="muted smiles-line">${escapeHtml(displayResult.compound.canonical_smiles)}</p>
          <p class="muted">${escapeHtml(displayResult.inference_engine || result.inference_engine || "DOT-SafeNet v1.0.0")}</p>
        </div>
      </div>
      ${warnings.map(w => `<div class="notice">${escapeHtml(w)}</div>`).join("")}
      <div class="result-layout">
        <section class="data-block chart-block radar-block">
          <h3>ADR Radar</h3>
          <canvas id="radarCanvas" width="720" height="720"></canvas>
        </section>
        <section class="data-block chart-block">
          <h3>Free Cmax Curve</h3>
          <svg id="cmaxSvg" class="chart square-chart" viewBox="0 0 720 720"></svg>
        </section>
        <section class="data-block network-block">
          <h3>Target Attribution</h3>
          <svg id="networkSvg" class="chart" viewBox="0 0 760 520"></svg>
        </section>
        <section class="data-block table-block">
          <h3>Top Risks <span class="info-tip" tabindex="0">?</span></h3>
          <div class="tooltip-text">SOC risks are ranked by the five-fold mean score at the highest submitted dose. Percentiles are calculated against the development background for the same SOC.</div>
          <div id="riskTable"></div>
        </section>
        <section class="data-block table-block">
          <h3>Top Attributed Targets <span class="info-tip" tabindex="0">?</span></h3>
          <div class="tooltip-text">Targets are ranked by the largest positive change in an ADR score after target replacement. Direct and secondary evidence annotate the target-SOC relation.</div>
          <div id="offtargetTable"></div>
        </section>
      </div>
    </div>
  `;

  if (result.results) {
    $("#resultPicker").addEventListener("change", event => {
      currentDisplayIndex = Number(event.target.value);
      renderResult(currentResult);
    });
  }
  drawRadar(displayResult);
  drawCmaxCurve(displayResult);
  drawNetwork(displayResult);
  renderTables(displayResult);
}

function getDisplayResult(result) {
  if (!result.results) return result;
  return result.results[currentDisplayIndex] || result.results[0];
}

function drawRadar(result) {
  const canvas = $("#radarCanvas");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const cssSize = Math.max(520, Math.min(760, rect.width || 620));
  canvas.width = Math.round(cssSize * dpr);
  canvas.height = Math.round(cssSize * dpr);
  canvas.style.height = `${cssSize}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const size = cssSize;
  const cx = size / 2;
  const cy = size / 2 + 12;
  const radius = size * 0.34;
  const axes = result.adr_axes.map(axis => axis.abbr);
  const n = axes.length;

  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, size, size);

  [0.2, 0.4, 0.6, 0.8, 1.0].forEach(grid => {
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const angle = -Math.PI / 2 + (Math.PI * 2 * i / n);
      const x = cx + Math.cos(angle) * radius * grid;
      const y = cy + Math.sin(angle) * radius * grid;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.strokeStyle = grid === 1 ? "#bcc8d4" : "#e3e8ef";
    ctx.lineWidth = grid === 1 ? 1.2 : 1;
    ctx.stroke();
    ctx.fillStyle = "#8a98a8";
    ctx.font = "12px Segoe UI, Arial";
    ctx.fillText(grid.toFixed(1), cx + radius * grid + 18, cy - 4);
  });

  axes.forEach((axis, i) => {
    const angle = -Math.PI / 2 + (Math.PI * 2 * i / n);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(angle) * radius, cy + Math.sin(angle) * radius);
    ctx.strokeStyle = "#edf1f5";
    ctx.stroke();
    ctx.fillStyle = "#334155";
    ctx.font = "700 13px Segoe UI, Arial";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(axis, cx + Math.cos(angle) * (radius + 34), cy + Math.sin(angle) * (radius + 34));
  });

  result.dose_results.forEach((dose, doseIndex) => {
    const color = colors[doseIndex % colors.length];
    const values = dose.adr.map(item => Math.max(0, Math.min(1, item.mean_probability || 0)));
    ctx.beginPath();
    values.forEach((value, i) => {
      const angle = -Math.PI / 2 + (Math.PI * 2 * i / n);
      const x = cx + Math.cos(angle) * radius * value;
      const y = cy + Math.sin(angle) * radius * value;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.strokeStyle = color;
    ctx.lineWidth = doseIndex === result.dose_results.length - 1 ? 2.8 : 1.8;
    ctx.globalAlpha = doseIndex === result.dose_results.length - 1 ? 0.95 : 0.62;
    if (doseIndex === result.dose_results.length - 1) {
      ctx.fillStyle = `${color}22`;
      ctx.fill();
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.fillRect(24, 24 + doseIndex * 22, 11, 11);
    ctx.fillStyle = "#475569";
    ctx.font = "600 12px Segoe UI, Arial";
    ctx.textAlign = "left";
    ctx.fillText(`${dose.dose_mg}mg`, 42, 30 + doseIndex * 22);
  });
}

function drawCmaxCurve(result) {
  const svg = $("#cmaxSvg");
  const points = result.dose_results.map(d => ({ x: d.dose_mg, y: Number(d.pk.cmax_free_uM || 0) }));
  const doses = points.map(p => p.x);
  const minDose = Math.min(...doses, 0);
  const maxDose = Math.max(...doses, 1);
  const maxY = Math.max(...points.map(p => p.y), 0.001);
  const left = 86;
  const right = 660;
  const top = 72;
  const bottom = 622;
  const sx = value => {
    if (maxDose === minDose) return (left + right) / 2;
    return left + (right - left) * ((value - minDose) / (maxDose - minDose));
  };
  const sy = value => bottom - (bottom - top) * (value / maxY);
  const path = points.map((p, i) => `${i ? "L" : "M"} ${sx(p.x).toFixed(1)} ${sy(p.y).toFixed(1)}`).join(" ");
  const circles = points.map(p => `
    <circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="6" fill="#126c78"/>
    <text x="${sx(p.x)}" y="${bottom + 34}" text-anchor="middle" font-size="16" font-weight="700" fill="#475569">${p.x}mg</text>
  `).join("");
  svg.innerHTML = `
    <rect x="0" y="0" width="720" height="720" fill="#fff"/>
    <line x1="${left}" y1="${bottom}" x2="${right}" y2="${bottom}" stroke="#cbd5e1"/>
    <line x1="${left}" y1="${top}" x2="${left}" y2="${bottom}" stroke="#cbd5e1"/>
    ${[0, 0.25, 0.5, 0.75, 1].map(t => {
      const y = bottom - (bottom - top) * t;
      const label = (maxY * t).toFixed(maxY < 0.1 ? 3 : 2);
      return `<line x1="${left}" y1="${y}" x2="${right}" y2="${y}" stroke="#eef2f7"/><text x="24" y="${y + 5}" font-size="15" font-weight="650" fill="#64748b">${label}</text>`;
    }).join("")}
    <path d="${path}" fill="none" stroke="#126c78" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>
    ${circles}
    <text x="360" y="692" text-anchor="middle" font-size="16" font-weight="700" fill="#64748b">Dose</text>
    <text x="24" y="44" font-size="16" font-weight="700" fill="#64748b">Cmax_free (uM)</text>
  `;
}

function drawNetwork(result) {
  const svg = $("#networkSvg");
  const network = result.target_adr_network || { nodes: [], edges: [] };
  const targetNodes = network.nodes.filter(node => node.type === "target").slice(0, 10);
  const adrNodes = network.nodes.filter(node => node.type === "adr");
  const nodeById = new Map();
  const targetStep = targetNodes.length > 1 ? 360 / (targetNodes.length - 1) : 0;
  const adrStep = adrNodes.length > 1 ? 360 / (adrNodes.length - 1) : 0;

  targetNodes.forEach((node, index) => nodeById.set(node.id, { ...node, x: 150, y: 84 + index * targetStep }));
  adrNodes.forEach((node, index) => nodeById.set(node.id, { ...node, x: 600, y: 84 + index * adrStep }));

  let html = `<rect x="0" y="0" width="760" height="520" fill="#fff"/>`;
  html += `<text x="42" y="36" font-size="14" font-weight="700" fill="#1f2937">Largest positive target-replacement contributions</text>`;
  const maxEdge = Math.max(...network.edges.map(edge => edge.score), 0.001);
  network.edges.forEach(edge => {
    const source = nodeById.get(edge.source);
    const target = nodeById.get(edge.target);
    if (!source || !target) return;
    const width = 0.8 + 5.2 * (edge.score / maxEdge);
    const opacity = 0.16 + 0.52 * (edge.score / maxEdge);
    html += `<path d="M ${source.x + 46} ${source.y} C ${source.x + 150} ${source.y}, ${target.x - 150} ${target.y}, ${target.x - 54} ${target.y}" fill="none" stroke="#64748b" stroke-width="${width.toFixed(2)}" opacity="${opacity.toFixed(2)}" stroke-linecap="round"/>`;
  });

  targetNodes.forEach(node => {
    const pos = nodeById.get(node.id);
    const label = node.target_name || node.label || node.id;
    html += `<circle cx="${pos.x}" cy="${pos.y}" r="12" fill="#126c78" opacity="0.92"/>`;
    html += `<text x="${pos.x - 18}" y="${pos.y - 2}" text-anchor="end" font-size="12" font-weight="700" fill="#334155">${truncate(label, 16)}</text>`;
    html += `<text x="${pos.x - 18}" y="${pos.y + 13}" text-anchor="end" font-size="10" fill="#64748b">${node.accession || node.id}</text>`;
    html += `<text x="${pos.x + 22}" y="${pos.y + 4}" font-size="11" fill="#64748b">${fmt(node.score, 2)}</text>`;
  });

  adrNodes.forEach(node => {
    const pos = nodeById.get(node.id);
    html += `<circle cx="${pos.x}" cy="${pos.y}" r="13" fill="#c65d3b" opacity="0.9"/>`;
    html += `<text x="${pos.x + 24}" y="${pos.y - 3}" font-size="13" font-weight="800" fill="#334155">${node.label}</text>`;
    html += `<text x="${pos.x + 24}" y="${pos.y + 14}" font-size="11" fill="#64748b">${truncate(node.full_label, 34)} | ${fmt(node.score, 2)}</text>`;
  });

  if (!network.edges.length) {
    html += `<text x="380" y="260" text-anchor="middle" font-size="14" fill="#64748b">No target-ADR associations available for this result.</text>`;
  }
  svg.innerHTML = html;
}

function truncate(text, maxLength) {
  if (!text || text.length <= maxLength) return text || "";
  return `${text.slice(0, maxLength - 1)}...`;
}

function renderTables(result) {
  const latest = result.dose_results[result.dose_results.length - 1];
  const topOfftargets = (latest.top_offtargets || []).slice(0, 10);
  const riskCount = Math.max(topOfftargets.length, 1);
  const topRisks = [...(latest.adr || [])]
    .filter(row => row.mean_probability !== null && row.mean_probability !== undefined)
    .sort((a, b) => b.mean_probability - a.mean_probability)
    .slice(0, riskCount);
  $("#riskTable").innerHTML = table(
    ["ADR", "Score", "Percentile", "SD"],
    topRisks.map(r => [
      `${r.abbr} ${r.task_name}`,
      fmt(r.mean_probability),
      r.background_percentile === null || r.background_percentile === undefined ? "NA" : `${fmt(r.background_percentile, 1)}%`,
      fmt(r.std_probability),
    ])
  );
  $("#offtargetTable").innerHTML = table(
    ["Target", "pAC50", "log10 margin", "Max ΔADR", "Evidence"],
    topOfftargets.map(t => [
      `${t.target_name || t.gene_name || t.target} (${t.target})`,
      fmt(t.value),
      fmt(t.margin_log10),
      fmt(t.max_delta_probability),
      t.evidence_level || "none",
    ])
  );
}

function table(headers, rows) {
  return `
    <table>
      <thead><tr>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}</tbody>
    </table>
  `;
}

init().catch(error => {
  $("#jobStatus").textContent = "Initialization failed";
  $("#resultPanel").textContent = error.message;
});
