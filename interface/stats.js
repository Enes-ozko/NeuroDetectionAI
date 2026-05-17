const API = "http://localhost:8000";

const CHART_GRID = "#2a2a38";
const CHART_TEXT = "#9090a8";

let session = [];
let charts  = [];

function pct(n, total) {
  if (!total) return "—";
  return (n / total * 100).toFixed(1) + "%";
}

function trashIcon() {
  return `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/>
  </svg>`;
}

async function deleteEntry(filename) {
  try {
    await fetch(`${API}/session/${encodeURIComponent(filename)}`, { method: "DELETE" });
    session = session.filter(r => r.filename !== filename);
    rebuildAll();
  } catch { console.error("Suppression échouée"); }
}

function destroyCharts() {
  charts.forEach(c => c.destroy());
  charts = [];
}

function rebuildAll() {
  destroyCharts();
  buildStats();
}

function buildStats() {
  const total   = session.length;
  const tumeurs = session.filter(r => r.scenario === "tumeur");
  const ambigs  = session.filter(r => r.scenario === "ambig");
  const sains   = session.filter(r => r.scenario === "sain");

  document.getElementById("statTotal").textContent    = total;
  document.getElementById("statTumeur").textContent   = tumeurs.length;
  document.getElementById("statAmbig").textContent    = ambigs.length;
  document.getElementById("statSain").textContent     = sains.length;
  document.getElementById("statTumeurPct").textContent = pct(tumeurs.length, total);
  document.getElementById("statAmbigPct").textContent  = pct(ambigs.length, total);
  document.getElementById("statSainPct").textContent   = pct(sains.length, total);

  charts.push(buildDonut(tumeurs.length, ambigs.length, sains.length));
  charts.push(buildTypes(tumeurs));
  charts.push(buildScoreDist());
  charts.push(buildEntropy(tumeurs));
  charts.push(buildConfClass(tumeurs));
  buildTable();
}

function baseOptions(extra = {}) {
  return {
    responsive: true,
    animation: { duration: 600 },
    plugins: {
      legend: { labels: { color: CHART_TEXT, boxWidth: 12, padding: 12 } },
      tooltip: { backgroundColor: "#1a1a24", titleColor: "#e8e8f0", bodyColor: "#9090a8" },
      ...extra.plugins,
    },
    ...extra,
  };
}

function buildDonut(t, a, s) {
  return new Chart(document.getElementById("chartDonut"), {
    type: "doughnut",
    data: {
      labels: ["Tumeur", "Ambigu", "Sain"],
      datasets: [{ data: [t, a, s], backgroundColor: ["#e74c3c", "#f0a500", "#2ecc71"], borderColor: "#0a0a0f", borderWidth: 3 }],
    },
    options: baseOptions({
      cutout: "65%",
      plugins: {
        legend: { position: "bottom", labels: { color: CHART_TEXT, boxWidth: 12, padding: 16 } },
        tooltip: { backgroundColor: "#1a1a24", titleColor: "#e8e8f0", bodyColor: "#9090a8" },
      },
    }),
  });
}

function buildTypes(tumeurs) {
  const counts = { Gliome: 0, Meningiome: 0, Pituitaire: 0, Inconnu: 0 };
  tumeurs.forEach(r => { if (r.ood) counts["Inconnu"]++; else if (r.type) counts[r.type]++; });
  return new Chart(document.getElementById("chartTypes"), {
    type: "bar",
    data: {
      labels: Object.keys(counts),
      datasets: [{ data: Object.values(counts), backgroundColor: ["#e74c3c", "#2ecc71", "#4f8ef7", "#f0a500"], borderRadius: 4, borderSkipped: false }],
    },
    options: baseOptions({
      scales: {
        x: { grid: { color: CHART_GRID }, ticks: { color: CHART_TEXT } },
        y: { grid: { color: CHART_GRID }, ticks: { color: CHART_TEXT, stepSize: 1 }, beginAtZero: true },
      },
      plugins: { legend: { display: false }, tooltip: { backgroundColor: "#1a1a24", titleColor: "#e8e8f0", bodyColor: "#9090a8" } },
    }),
  });
}

function buildScoreDist() {
  const bins = Array(10).fill(0);
  session.forEach(r => { const idx = Math.min(Math.floor(r.proba * 10), 9); bins[idx]++; });
  return new Chart(document.getElementById("chartScoreDist"), {
    type: "bar",
    data: {
      labels: bins.map((_, i) => `${i * 10}-${i * 10 + 10}%`),
      datasets: [{
        label: "Nb images",
        data: bins,
        backgroundColor: bins.map((_, i) => i < 3 ? "#2ecc71" : i < 7 ? "#f0a500" : "#e74c3c"),
        borderRadius: 3, borderSkipped: false,
      }],
    },
    options: baseOptions({
      scales: {
        x: { grid: { color: CHART_GRID }, ticks: { color: CHART_TEXT, maxRotation: 45 } },
        y: { grid: { color: CHART_GRID }, ticks: { color: CHART_TEXT, stepSize: 1 }, beginAtZero: true },
      },
      plugins: { legend: { display: false }, tooltip: { backgroundColor: "#1a1a24", titleColor: "#e8e8f0", bodyColor: "#9090a8" } },
    }),
  });
}

function buildEntropy(tumeurs) {
  return new Chart(document.getElementById("chartEntropy"), {
    type: "line",
    data: {
      labels: tumeurs.map(r => r.filename?.substring(0, 14) || "—"),
      datasets: [{
        label: "Entropie",
        data: tumeurs.map(r => r.entropy ? parseFloat(r.entropy.toFixed(3)) : 0),
        borderColor: "#4f8ef7", backgroundColor: "rgba(79,142,247,.1)",
        fill: true, tension: 0.3, pointRadius: 4, pointBackgroundColor: "#4f8ef7",
      }],
    },
    options: baseOptions({
      scales: {
        x: { display: false },
        y: { grid: { color: CHART_GRID }, ticks: { color: CHART_TEXT }, beginAtZero: true, max: 1.1 },
      },
      plugins: { legend: { display: false }, tooltip: { backgroundColor: "#1a1a24", titleColor: "#e8e8f0", bodyColor: "#9090a8" } },
    }),
  });
}

function buildConfClass(tumeurs) {
  const sums = { Gliome: 0, Meningiome: 0, Pituitaire: 0 };
  const cnts = { Gliome: 0, Meningiome: 0, Pituitaire: 0 };
  tumeurs.forEach(r => {
    if (!r.ood && r.type && sums[r.type] !== undefined) { sums[r.type] += r.confidence || 0; cnts[r.type]++; }
  });
  return new Chart(document.getElementById("chartConfClass"), {
    type: "bar",
    data: {
      labels: ["Gliome", "Meningiome", "Pituitaire"],
      datasets: [{
        label: "Confiance moy. (%)",
        data: Object.keys(sums).map(k => cnts[k] ? parseFloat((sums[k] / cnts[k] * 100).toFixed(1)) : 0),
        backgroundColor: ["#e74c3c", "#2ecc71", "#4f8ef7"], borderRadius: 4, borderSkipped: false,
      }],
    },
    options: baseOptions({
      scales: {
        x: { grid: { color: CHART_GRID }, ticks: { color: CHART_TEXT } },
        y: { grid: { color: CHART_GRID }, ticks: { color: CHART_TEXT }, beginAtZero: true, max: 100 },
      },
      plugins: { legend: { display: false }, tooltip: { backgroundColor: "#1a1a24", titleColor: "#e8e8f0", bodyColor: "#9090a8" } },
    }),
  });
}

function buildTable() {
  const body  = document.getElementById("histBody");
  const empty = document.getElementById("histEmpty");
  if (!session.length) {
    body.innerHTML = "";
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";
  body.innerHTML = session.map(r => {
    const bbox = r.bbox ? `${r.bbox.x},${r.bbox.y} — ${r.bbox.w}×${r.bbox.h}` : "—";
    const badgeClass = r.scenario === "sain" ? "badge-sain" : r.scenario === "ambig" ? "badge-ambig" : r.ood ? "badge-ood" : "badge-tumeur";
    const verdict    = r.scenario === "sain" ? "Sain" : r.scenario === "ambig" ? "Ambigu" : "Tumeur";
    return `<tr data-filename="${r.filename}">
      <td>${r.filename || "—"}</td>
      <td>${(r.proba * 100).toFixed(1)}%</td>
      <td><span class="badge ${badgeClass}" style="font-size:11px">${verdict}</span></td>
      <td>${r.ood ? "Inconnu" : r.type || "—"}</td>
      <td>${r.confidence ? (r.confidence * 100).toFixed(1) + "%" : "—"}</td>
      <td>${r.entropy ? parseFloat(r.entropy).toFixed(3) : "—"}</td>
      <td>${r.ood ? "Oui" : "Non"}</td>
      <td style="font-family:monospace;font-size:12px">${bbox}</td>
      <td>
        <button class="btn-trash" title="Supprimer" onclick="deleteEntry('${r.filename}')">${trashIcon()}</button>
      </td>
    </tr>`;
  }).join("");
}

async function init() {
  try {
    const r = await fetch(`${API}/session`);
    if (r.ok) session = await r.json();
  } catch { session = []; }
  buildStats();
}

init();