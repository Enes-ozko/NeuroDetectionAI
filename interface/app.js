const API = "http://localhost:8000";

Chart.defaults.color = "#9090a8";
Chart.defaults.font.family = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";
Chart.defaults.font.size = 12;

const CLASS_COLORS = {
  "Gliome":     "#e74c3c",
  "Meningiome": "#2ecc71",
  "Pituitaire": "#4f8ef7",
};

let files = [];
let session = [];

async function loadSession() {
  const r = await fetch(`${API}/session`);
  session = await r.json();
}

async function saveSession(newResults) {
  const light = newResults.map(({ src, annotated_src, ...rest }) => rest);
  await fetch(`${API}/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(light),
  });
}

async function deleteFromSession(filename) {
  await fetch(`${API}/session/${encodeURIComponent(filename)}`, { method: "DELETE" });
  session = session.filter(r => r.filename !== filename);
}

async function checkModels() {
  const statusEl = document.getElementById("modelStatus");
  if (!statusEl) return;
  const r = await fetch(`${API}/status`);
  const d = await r.json();
  document.getElementById("navDevice").textContent = d.device;
  statusEl.innerHTML = `
    <div class="model-pill ${d.e2 ? "ok" : "missing"}"><span class="dot"></span>Etape 2 — ${d.e2 ? "pret" : "manquant"}</div>
    <div class="model-pill ${d.e3 ? "ok" : "missing"}"><span class="dot"></span>Etape 3 — ${d.e3 ? "pret" : "manquant"}</div>
  `;
}

function fmtSize(b) {
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}

function readAsDataURL(file) {
  return new Promise(res => {
    const r = new FileReader();
    r.onload = e => res(e.target.result);
    r.readAsDataURL(file);
  });
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

function animateProgress(el, target, ms) {
  return new Promise(res => {
    const start = parseFloat(el.style.width || "0");
    const t0 = performance.now();
    function step(now) {
      const p = Math.min((now - t0) / ms, 1);
      el.style.width = (start + (target - start) * p) + "%";
      if (p < 1) requestAnimationFrame(step); else res();
    }
    requestAnimationFrame(step);
  });
}

function trashIcon() {
  return `<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/>
  </svg>`;
}

function addFiles(newFiles) {
  const imgs = [...newFiles].filter(f => f.type.startsWith("image/"));
  files = [...files, ...imgs];
  renderQueue();
  const toolbar = document.getElementById("toolbar");
  if (toolbar) toolbar.style.display = files.length ? "flex" : "none";
  const info = document.getElementById("toolbarInfo");
  if (info) info.textContent = `${files.length} image${files.length > 1 ? "s" : ""} en attente`;
}

function renderQueue() {
  const q = document.getElementById("queue");
  if (!q) return;
  if (!files.length) { q.innerHTML = ""; return; }
  q.innerHTML = files.map((f, i) => `
    <div class="queue-item" id="qi-${i}">
      <div class="qi-thumb" id="qi-thumb-${i}">&#8212;</div>
      <div class="qi-info">
        <div class="qi-name">${f.name}</div>
        <div class="qi-size">${fmtSize(f.size)}</div>
        <div class="qi-progress" id="qi-pb-${i}">
          <div class="qi-progress-fill" id="qi-pf-${i}" style="width:0%"></div>
        </div>
      </div>
      <span class="badge badge-wait" id="qi-badge-${i}">En attente</span>
    </div>
  `).join("");
  files.forEach(async (f, i) => {
    const src = await readAsDataURL(f);
    const el = document.getElementById(`qi-thumb-${i}`);
    if (el) el.innerHTML = `<img src="${src}">`;
  });
}

async function analyzeFile(file, idx) {
  const pb    = document.getElementById(`qi-pb-${idx}`);
  const pf    = document.getElementById(`qi-pf-${idx}`);
  const badge = document.getElementById(`qi-badge-${idx}`);
  pb.style.display = "block";
  badge.className = "badge badge-run";
  badge.textContent = "Analyse...";
  await animateProgress(pf, 40, 300);

  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/analyze`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`Erreur API : ${r.status}`);
  const result = await r.json();
  await animateProgress(pf, 100, 200);

  result.filename = file.name;

  badge.textContent = result.scenario === "sain" ? "Sain"
    : result.scenario === "ambig" ? "Ambigu"
    : result.ood ? "OOD" : result.type || "Tumeur";
  badge.className = "badge badge-" + (result.scenario === "sain" ? "sain"
    : result.scenario === "ambig" ? "ambig"
    : result.ood ? "ood" : "tumeur");

  return result;
}

function imgTag(url, alt) {
  if (!url) return `<div class="no-img">Image non disponible</div>`;
  const fullUrl = url.startsWith("http") ? url : `${API}${url}`;
  return `<img src="${fullUrl}" alt="${alt}">`;
}

function buildResultBlock(result) {
  const container = document.createElement("div");
  container.className = "result-block";
  container.dataset.filename = result.filename;

  const scenario   = result.scenario;
  const badgeClass = scenario === "sain" ? "badge-sain"
    : scenario === "ambig" ? "badge-ambig"
    : result.ood ? "badge-ood" : "badge-tumeur";
  const badgeText  = scenario === "sain" ? "Sain"
    : scenario === "ambig" ? "Ambigu"
    : result.ood ? "Type inconnu" : result.type || "Tumeur";

  const header = document.createElement("div");
  header.className = "result-block-header";
  header.innerHTML = `
    <div>
      <span class="result-block-name">${result.filename}</span>
      <span class="result-block-score">— score E2 : ${(result.proba * 100).toFixed(1)}%</span>
    </div>
    <div style="display:flex;align-items:center;gap:10px;margin-left:auto">
      <span class="badge ${badgeClass}">${badgeText}</span>
      <button class="btn-trash" title="Supprimer">${trashIcon()}</button>
    </div>
  `;
  header.querySelector(".btn-trash").addEventListener("click", async () => {
    await deleteFromSession(result.filename);
    container.remove();
  });
  container.appendChild(header);

  if (scenario === "sain" || scenario === "ambig") {
    const body = document.createElement("div");
    body.className = "result-sain";
    body.innerHTML = `
      ${imgTag(result.src_url, "IRM")}
      <div>
        <div class="graph-title" style="margin-bottom:12px">Score de detection</div>
        <div class="detection-gauge" style="margin-bottom:24px">
          <div class="gauge-track">
            <div class="gauge-fill" style="width:${result.proba * 100}%;background:${scenario === "ambig" ? "var(--amber)" : "var(--green)"}"></div>
          </div>
          <div class="gauge-markers">
            <span class="gauge-marker" style="left:30%">30%</span>
            <span class="gauge-marker" style="left:70%">70%</span>
          </div>
          <div class="gauge-labels"><span>Sain</span><span>Zone ambigue</span><span>Tumeur</span></div>
        </div>
        <div class="verdict-box ${scenario === "sain" ? "verdict-sain" : "verdict-ambig"}">
          ${scenario === "sain"
            ? `Aucune anomalie detectee. Score : ${(result.proba * 100).toFixed(1)}%, sous le seuil (30%).`
            : `Signal non concluant. Score : ${(result.proba * 100).toFixed(1)}%, zone d'ambiguite (30-70%). Relecture recommandee.`}
        </div>
      </div>
    `;
    container.appendChild(body);
    return container;
  }

  const imagesSection = document.createElement("div");
  imagesSection.className = "result-images";
  imagesSection.innerHTML = `
    <div class="result-image-panel">
      <div class="result-image-label">Image originale</div>
      ${imgTag(result.src_url, "IRM originale")}
    </div>
    <div class="result-image-panel">
      <div class="result-image-label">Activation Grad-CAM++</div>
      ${imgTag(result.annotated_url, "Grad-CAM")}
    </div>
  `;
  container.appendChild(imagesSection);

  const classes     = ["Gliome", "Meningiome", "Pituitaire"];
  const probs       = result.probs || [0.33, 0.33, 0.34];
  const predIdx     = result.predIdx ?? probs.indexOf(Math.max(...probs));
  const entropy     = result.entropy ?? 0;
  const entropyNorm = result.entropy_norm ?? 0;
  const entropyColor = entropyNorm > 0.75 ? "var(--red)" : entropyNorm > 0.5 ? "var(--amber)" : "var(--green)";
  const entropyVerdict = entropyNorm > 0.75
    ? "Distribution quasi-uniforme — cas OOD probable"
    : entropyNorm > 0.5 ? "Incertitude moderee — reviser manuellement"
    : "Confiance elevee — prediction stable";
  const entropyVerdictClass = entropyNorm > 0.75 ? "verdict-ood" : entropyNorm > 0.5 ? "verdict-ambig" : "verdict-sain";

  const barsHTML = classes.map((c, i) => `
    <div class="bar-row">
      <span class="bar-label" style="${i === predIdx ? `color:${CLASS_COLORS[c]}` : ""}">${c}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width:${(probs[i] * 100).toFixed(1)}%;background:${i === predIdx ? CLASS_COLORS[c] : "var(--bg-3)"}"></div>
      </div>
      <span class="bar-pct" style="${i === predIdx ? `color:${CLASS_COLORS[c]}` : ""}">${(probs[i] * 100).toFixed(1)}%</span>
    </div>
  `).join("");

  const bbox = result.bbox;
  const bboxHTML = bbox
    ? `<div class="bbox-info">
        <div class="bbox-val"><div class="bbox-key">X</div><div class="bbox-num">${bbox.x}</div></div>
        <div class="bbox-val"><div class="bbox-key">Y</div><div class="bbox-num">${bbox.y}</div></div>
        <div class="bbox-val"><div class="bbox-key">Largeur</div><div class="bbox-num">${bbox.w} px</div></div>
        <div class="bbox-val"><div class="bbox-key">Hauteur</div><div class="bbox-num">${bbox.h} px</div></div>
      </div>`
    : `<div style="color:var(--text-3);font-size:13px">Aucune region isolee (entropie trop elevee)</div>`;

  const graphSection = document.createElement("div");
  graphSection.className = "result-graphs";
  graphSection.innerHTML = `
    <div class="result-graph-panel">
      <div class="graph-title">Probabilites par type</div>
      <div class="bar-chart">${barsHTML}</div>
    </div>
    <div class="result-graph-panel">
      <div class="graph-title">Entropie de Shannon</div>
      <div class="entropy-display">
        <div>
          <div class="entropy-value" style="color:${entropyColor}">${entropy.toFixed(3)}</div>
          <div class="entropy-norm">normalise : ${(entropyNorm * 100).toFixed(1)}% du max (ln 3 = 1.099)</div>
        </div>
        <div class="entropy-bar-track">
          <div class="entropy-bar-fill" style="width:${(entropyNorm * 100).toFixed(1)}%;background:${entropyColor}"></div>
        </div>
        <div class="verdict-box ${entropyVerdictClass}" style="font-size:12px">${entropyVerdict}</div>
      </div>
    </div>
    <div class="result-graph-panel">
      <div class="graph-title">Region tumorale</div>
      ${bboxHTML}
    </div>
  `;
  container.appendChild(graphSection);

  const infoRow = document.createElement("div");
  infoRow.className = "result-info-row";
  infoRow.innerHTML = `
    <div class="result-info-cell">Score detection <strong>${(result.proba * 100).toFixed(1)}%</strong></div>
    <div class="result-info-cell">Type <strong>${result.ood ? "Inconnu (OOD)" : result.type || "—"}</strong></div>
    <div class="result-info-cell">Confiance <strong>${result.confidence ? (result.confidence * 100).toFixed(1) + "%" : "—"}</strong></div>
    <div class="result-info-cell">Scenario E3 <strong>${result.ood ? "B — OOD" : result.confidence > 0.55 ? "A — confirme" : "C — incertain"}</strong></div>
  `;
  container.appendChild(infoRow);

  return container;
}

function renderAllResults() {
  const resultsEl = document.getElementById("results");
  if (!resultsEl) return;
  resultsEl.innerHTML = "";
  session.forEach(result => resultsEl.appendChild(buildResultBlock(result)));
}

async function runAnalysis() {
  const runBtn    = document.getElementById("runBtn");
  runBtn.disabled = true;

  const allResults = [];
  for (let i = 0; i < files.length; i++) {
    try {
      const result = await analyzeFile(files[i], i);
      allResults.push(result);
    } catch (err) {
      const badge = document.getElementById(`qi-badge-${i}`);
      if (badge) { badge.className = "badge badge-missing"; badge.textContent = "Erreur"; }
      console.error(err);
    }
  }

  if (allResults.length) {
    session = [...session, ...allResults];
    await saveSession(allResults);
    renderAllResults();
  }

  files = [];
  renderQueue();
  const toolbar = document.getElementById("toolbar");
  if (toolbar) toolbar.style.display = "none";

  runBtn.disabled = false;
}

function clearAll() {
  files = [];
  renderQueue();
  const toolbar = document.getElementById("toolbar");
  if (toolbar) toolbar.style.display = "none";
}

async function initIndex() {
  await loadSession();
  checkModels();
  renderAllResults();

  const dz = document.getElementById("dropZone");
  const fi = document.getElementById("fileInput");
  if (!dz || !fi) return;

  dz.addEventListener("click", () => fi.click());
  fi.addEventListener("change", e => addFiles(e.target.files));
  dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("over"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("over"));
  dz.addEventListener("drop", e => {
    e.preventDefault();
    dz.classList.remove("over");
    addFiles(e.dataTransfer.files);
  });

  document.getElementById("runBtn").addEventListener("click", runAnalysis);
  document.getElementById("clearBtn").addEventListener("click", clearAll);
}

if (document.getElementById("dropZone")) initIndex();