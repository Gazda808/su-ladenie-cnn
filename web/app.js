(function () {
  "use strict";

  const DATA = (window.RESULTS && window.RESULTS.experiments) ? window.RESULTS : {
    generated_at: "—",
    categories: {},
    experiments: [],
  };

  const EXPS = DATA.experiments.filter(e => !e.error);
  const CATEGORIES_LABELS = DATA.categories || {};

  const DATASET_LABELS = {
    mnist: "MNIST",
    fashion_mnist: "Fashion-MNIST",
    cifar10: "CIFAR-10",
  };

  const CAT_COLORS = {
    topology: "#a8c4e8",
    optimizer: "#f0c9a8",
    lr: "#c4b5f7",
    activation: "#a8d8c8",
    regularization: "#f0b8c4",
    kernel: "#e8e0a8",
    batch_size: "#b8e0a8",
  };

  function catColor(cat) {
    return CAT_COLORS[cat] || "#9aa3b2";
  }

  function fmtAcc(v)   { return (v == null) ? "–" : (v * 100).toFixed(2) + " %"; }
  function fmtNum(v, d=4) { return (v == null) ? "–" : Number(v).toFixed(d); }
  function fmtInt(v)   { return (v == null) ? "–" : Number(v).toLocaleString("sk-SK"); }
  function fmtTime(s)  {
    if (s == null) return "–";
    if (s < 60) return s.toFixed(1) + " s";
    const m = Math.floor(s / 60); const r = s - 60 * m;
    return `${m} min ${r.toFixed(0)} s`;
  }

  function shortHParams(e) {
    const m = e.model_config || {};
    const t = e.training_config || {};
    const parts = [];
    parts.push(`filters=[${(m.filters || []).join(",")}]`);
    parts.push(`act=${m.activation}`);
    if (m.use_batch_norm) parts.push("BN");
    if (m.dropout)        parts.push(`drop=${m.dropout}`);
    parts.push(`opt=${t.optimizer}`);
    parts.push(`lr=${t.lr}`);
    parts.push(`bs=${t.batch_size}`);
    if (t.weight_decay)   parts.push(`wd=${t.weight_decay}`);
    return parts.join(" · ");
  }

  const STATE = {
    view: "overview",
    filter: { dataset: "all", category: "all", sort: "accuracy_desc", search: "" },
    compareIds: [],
    detailId: EXPS.length ? EXPS[0].id : null,
  };

  document.addEventListener("DOMContentLoaded", () => {
    const brandEl = document.getElementById("brand-home");
    if (brandEl) {
      brandEl.addEventListener("click", (e) => {
        e.preventDefault();
        switchView("overview");
      });
    }

    initTabs();
    initOverview();
    initCompare();
    initDetail();

    renderOverview();
  });

  function initTabs() {
    document.querySelectorAll(".tab").forEach(btn => {
      btn.addEventListener("click", () => {
        const view = btn.dataset.view;
        switchView(view);
      });
    });
  }

  function switchView(view) {
    STATE.view = view;
    document.querySelectorAll(".tab").forEach(b => {
      b.classList.toggle("active", b.dataset.view === view);
    });
    document.querySelectorAll(".view").forEach(v => {
      v.classList.toggle("active", v.id === "view-" + view);
    });
    if (view === "overview")     renderOverview();
    if (view === "compare")      renderCompare();
    if (view === "detail")       renderDetail();
    if (view === "konfigurator") renderKonfigurator();
  }


  let chartOverview = null;

  function initOverview() {
    const dsSet = new Set(EXPS.map(e => e.dataset));
    const dsSel = document.getElementById("filter-dataset");
    [...dsSet].forEach(d => {
      const o = document.createElement("option");
      o.value = d; o.textContent = DATASET_LABELS[d] || d;
      dsSel.appendChild(o);
    });

    const catSet = new Set(EXPS.map(e => e.category));
    const catSel = document.getElementById("filter-category");
    [...catSet].forEach(c => {
      const o = document.createElement("option");
      o.value = c; o.textContent = CATEGORIES_LABELS[c] || c;
      catSel.appendChild(o);
    });

    document.getElementById("filter-dataset").addEventListener("change", e => {
      STATE.filter.dataset = e.target.value; renderOverview();
    });
    document.getElementById("filter-category").addEventListener("change", e => {
      STATE.filter.category = e.target.value; renderOverview();
    });
    document.getElementById("filter-sort").addEventListener("change", e => {
      STATE.filter.sort = e.target.value; renderOverview();
    });
    document.getElementById("filter-search").addEventListener("input", e => {
      STATE.filter.search = e.target.value.toLowerCase(); renderOverview();
    });
  }

  function getFilteredExps() {
    const f = STATE.filter;
    let list = EXPS.slice();
    if (f.dataset !== "all")  list = list.filter(e => e.dataset === f.dataset);
    if (f.category !== "all") list = list.filter(e => e.category === f.category);
    if (f.search) {
      const q = f.search;
      list = list.filter(e =>
        (e.id + " " + e.name + " " + (e.description || "") + " " + shortHParams(e))
          .toLowerCase().includes(q));
    }
    list.sort((a, b) => {
      const aa = a.final?.test_accuracy ?? 0;
      const bb = b.final?.test_accuracy ?? 0;
      switch (f.sort) {
        case "accuracy_asc":  return aa - bb;
        case "accuracy_desc": return bb - aa;
        case "params_asc":    return (a.num_parameters || 0) - (b.num_parameters || 0);
        case "time_asc":      return (a.training_time_sec || 0) - (b.training_time_sec || 0);
        case "id":            return a.id.localeCompare(b.id);
        default:              return bb - aa;
      }
    });
    return list;
  }

  function renderOverview() {
    const list = getFilteredExps();

    document.getElementById("kpi-count").textContent = list.length;
    if (list.length) {
      const accs = list.map(e => e.final?.test_accuracy ?? 0);
      const best = list.reduce((a, b) =>
        (b.final?.test_accuracy ?? 0) > (a.final?.test_accuracy ?? 0) ? b : a, list[0]);
      const avg = accs.reduce((a, b) => a + b, 0) / accs.length;
      const total = list.reduce((s, e) => s + (e.training_time_sec || 0), 0);
      document.getElementById("kpi-best-acc").textContent = fmtAcc(best.final?.test_accuracy);
      document.getElementById("kpi-best-name").textContent = best.name + " (" + best.id + ")";
      document.getElementById("kpi-avg-acc").textContent  = fmtAcc(avg);
      document.getElementById("kpi-total-time").textContent = fmtTime(total);
    } else {
      document.getElementById("kpi-best-acc").textContent = "–";
      document.getElementById("kpi-best-name").textContent = "–";
      document.getElementById("kpi-avg-acc").textContent  = "–";
      document.getElementById("kpi-total-time").textContent = "–";
    }

    renderOverviewChart(list);
    renderOverviewTable(list);
  }

  function renderOverviewChart(list) {
    const canvas = document.getElementById("chart-overview");
    if (!canvas) return;
    if (chartOverview) chartOverview.destroy();

    const n = list.length;
    const wrap = canvas.parentElement;
    const h = n <= 4 ? 260 : n <= 10 ? 320 : n <= 20 ? 380 : 440;
    wrap.style.height = h + "px";
    canvas.style.height = h + "px";

    const labels = list.map(e => e.id);
    const data = list.map(e => (e.final?.test_accuracy ?? 0) * 100);
    const colors = list.map(e => catColor(e.category));
    const bestIdx = data.indexOf(Math.max(...data));

    chartOverview = new Chart(canvas, {
      type: "bar",
      data: {
        labels,
        datasets: [{
          label: "Test presnosť (%)",
          data,
          backgroundColor: colors.map((c, i) => i === bestIdx ? c : c + "bb"),
          borderColor: colors.map((c, i) => i === bestIdx ? c : c + "66"),
          borderWidth: 1,
          maxBarThickness: 72,
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              afterLabel: (ctx) => {
                const e = list[ctx.dataIndex];
                return [
                  CATEGORIES_LABELS[e.category] || e.category,
                  DATASET_LABELS[e.dataset] || e.dataset,
                  shortHParams(e),
                ];
              }
            }
          }
        },
        scales: {
          x: {
            ticks: {
              color: "#888",
              maxRotation: n > 12 ? 75 : 0,
              minRotation: n > 12 ? 45 : 0,
              autoSkip: false,
            },
            grid: { color: "#232323" }
          },
          y: {
            ticks: { color: "#888", callback: v => v + " %" },
            grid: { color: "#232323" },
            beginAtZero: true, max: 100
          }
        },
        onClick: (_evt, elements) => {
          if (!elements.length) return;
          const idx = elements[0].index;
          STATE.detailId = list[idx].id;
          switchView("detail");
        }
      }
    });
  }

  function renderOverviewTable(list) {
    const tbody = document.getElementById("overview-tbody");
    tbody.innerHTML = "";
    const bestAcc = list.reduce((m, e) => Math.max(m, e.final?.test_accuracy ?? 0), 0);

    list.forEach(e => {
      const tr = document.createElement("tr");
      if ((e.final?.test_accuracy ?? 0) === bestAcc && bestAcc > 0) tr.classList.add("best");
      tr.style.cursor = "pointer";
      tr.addEventListener("click", () => {
        STATE.detailId = e.id;
        switchView("detail");
      });

      const catLabel = CATEGORIES_LABELS[e.category] || e.category;
      tr.innerHTML = `
        <td><code>${e.id}</code></td>
        <td>${e.name}</td>
        <td><span class="badge cat-${e.category}">${catLabel}</span></td>
        <td>${DATASET_LABELS[e.dataset] || e.dataset}</td>
        <td class="num"><strong>${fmtAcc(e.final?.test_accuracy)}</strong></td>
        <td class="num">${fmtNum(e.final?.f1_macro, 3)}</td>
        <td class="num">${fmtNum(e.final?.test_loss, 3)}</td>
        <td class="num">${fmtInt(e.num_parameters)}</td>
        <td class="num">${e.epochs_run}</td>
        <td class="num">${(e.training_time_sec || 0).toFixed(1)}</td>
      `;
      tbody.appendChild(tr);
    });

    if (!list.length) {
      tbody.innerHTML = `<tr><td colspan="10" class="muted" style="text-align:center;padding:24px">Žiadne experimenty nezodpovedajú filtru.</td></tr>`;
    }
  }

  let chartCmpAcc = null, chartCmpLoss = null;

  function initCompare() {
    const picker = document.getElementById("compare-picker");
    if (!picker) return;
    const sorted = EXPS.slice().sort((a, b) =>
      a.category.localeCompare(b.category) || a.id.localeCompare(b.id));

    sorted.forEach(e => {
      const item = document.createElement("label");
      item.className = "compare-item";
      item.style.borderLeftWidth = "4px";
      item.style.borderLeftColor = catColor(e.category);
      item.innerHTML = `
        <input type="checkbox" value="${e.id}">
        <div>
          <div class="ci-title">${e.name}</div>
          <div class="ci-meta">
            <code>${e.id}</code>
            · ${DATASET_LABELS[e.dataset] || e.dataset}
            · <strong>${fmtAcc(e.final?.test_accuracy)}</strong>
          </div>
        </div>
      `;
      picker.appendChild(item);
    });

    function syncCheckedClass() {
      picker.querySelectorAll(".compare-item").forEach(it => {
        const cb = it.querySelector("input[type=checkbox]");
        it.classList.toggle("checked", !!(cb && cb.checked));
      });
    }

    picker.addEventListener("change", () => {
      const checked = [...picker.querySelectorAll("input:checked")].map(i => i.value);
      if (checked.length > 6) {
        const lastBox = picker.querySelector(`input[value="${checked[checked.length - 1]}"]`);
        if (lastBox) lastBox.checked = false;
        alert("Max 6 experimentov naraz.");
        syncCheckedClass();
        return;
      }
      STATE.compareIds = checked;
      syncCheckedClass();
      renderCompare();
    });

    if (EXPS.length >= 3) {
      const seed = sorted.slice(0, 3).map(e => e.id);
      STATE.compareIds = seed;
      seed.forEach(id => {
        const cb = picker.querySelector(`input[value="${id}"]`);
        if (cb) cb.checked = true;
      });
      syncCheckedClass();
    }
  }

  function renderCompare() {
    const ids = STATE.compareIds;
    const selected = ids.map(id => EXPS.find(e => e.id === id)).filter(Boolean);

    drawCompareLineChart("chart-compare-acc", selected, "accuracy", "Presnosť");
    drawCompareLineChart("chart-compare-loss", selected, "loss", "Loss");

    const tbody = document.querySelector("#compare-table tbody");
    tbody.innerHTML = "";
    selected.forEach(e => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>
          <div><strong>${e.name}</strong></div>
          <div class="muted"><code>${e.id}</code> · ${DATASET_LABELS[e.dataset] || e.dataset}</div>
        </td>
        <td class="num"><strong>${fmtAcc(e.final?.test_accuracy)}</strong></td>
        <td class="num">${fmtNum(e.final?.f1_macro, 3)}</td>
        <td class="num">${fmtNum(e.final?.test_loss, 3)}</td>
        <td class="num">${fmtInt(e.num_parameters)}</td>
        <td class="num">${(e.training_time_sec || 0).toFixed(1)} s</td>
        <td class="muted">${shortHParams(e)}</td>
      `;
      tbody.appendChild(tr);
    });
    if (!selected.length) {
      tbody.innerHTML = `<tr><td colspan="7" class="muted" style="text-align:center;padding:24px">Vyberte experimenty zo zoznamu vyššie.</td></tr>`;
    }
  }

  function drawCompareLineChart(canvasId, selected, kind, label) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    const palette = ["#a8c4e8", "#34d399", "#f0c9a8", "#c4b5f7", "#fbbf24", "#f472b6"];

    const DASHES  = [[], [6,3], [2,4], [8,3,2,3], [4,2,4,2], [12,3]];

    const SHAPES  = ["circle","rect","triangle","rectRot","star","cross"];

    const datasets = [];
    selected.forEach((e, i) => {
      const color  = palette[i % palette.length];
      const shape  = SHAPES[i % SHAPES.length];
      const epochs = (e.history || []).map(h => h.epoch);
      let trainKey, valKey;
      if (kind === "accuracy") { trainKey = "train_accuracy"; valKey = "val_accuracy"; }
      else                     { trainKey = "train_loss";     valKey = "val_loss"; }
      const tr = (e.history || []).map(h => kind === "accuracy" ? h[trainKey] * 100 : h[trainKey]);
      const va = (e.history || []).map(h => kind === "accuracy" ? h[valKey]  * 100 : h[valKey]);

      datasets.push({
        label: e.id + " – val",
        data: va.map((y, j) => ({ x: epochs[j], y })),
        borderColor: color,
        backgroundColor: color + "18",
        borderWidth: 2.5,
        borderDash: [],
        pointStyle: shape,
        pointRadius: 5,
        pointHoverRadius: 7,
        tension: 0.3,
        fill: false,
      });

      datasets.push({
        label: e.id + " – train",
        data: tr.map((y, j) => ({ x: epochs[j], y })),
        borderColor: color,
        backgroundColor: "transparent",
        borderWidth: 1.5,
        borderDash: DASHES[i % DASHES.length].length ? DASHES[i % DASHES.length] : [5,4],
        pointStyle: shape,
        pointRadius: 3,
        pointHoverRadius: 5,
        tension: 0.3,
        fill: false,
      });
    });

    if (canvasId === "chart-compare-acc"  && chartCmpAcc)  chartCmpAcc.destroy();
    if (canvasId === "chart-compare-loss" && chartCmpLoss) chartCmpLoss.destroy();
    const chart = new Chart(canvas, {
      type: "line",
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: "index", intersect: false },
        plugins: {
          legend: {
            labels: {
              color: "#f5f5f5",
              boxWidth: 22,
              usePointStyle: true,
              font: { size: 12 },
            }
          },
          tooltip: {
            mode: "index",
            intersect: false,
            callbacks: {
              title: items => "Epocha " + (items[0] && items[0].parsed ? items[0].parsed.x : ""),
              label: item => {
                const val = kind === "accuracy"
                  ? item.parsed.y.toFixed(2) + " %"
                  : item.parsed.y.toFixed(4);
                return "  " + item.dataset.label + ": " + val;
              }
            }
          }
        },
        scales: {
          x: {
            type: "linear",
            title: { display: true, text: "Epocha", color: "#888" },
            ticks: { color: "#888", stepSize: 1 },
            grid: { color: "#1e1e1e" }
          },
          y: {
            title: { display: true, text: kind === "accuracy" ? "Presnosť (%)" : "Loss", color: "#888" },
            ticks: { color: "#888", callback: v => kind === "accuracy" ? v.toFixed(1) + " %" : v },
            grid: { color: "#1e1e1e" }
          }
        }
      }
    });
    if (canvasId === "chart-compare-acc")  chartCmpAcc  = chart;
    if (canvasId === "chart-compare-loss") chartCmpLoss = chart;
  }

  let chartDAcc = null, chartDLoss = null, chartDPerClass = null;

  function initDetail() {
    const sel = document.getElementById("detail-select");
    if (!sel) return;

    const byCat = {};
    EXPS.forEach(e => {
      (byCat[e.category] = byCat[e.category] || []).push(e);
    });

    Object.keys(byCat).sort().forEach(cat => {
      const og = document.createElement("optgroup");
      og.label = CATEGORIES_LABELS[cat] || cat;
      byCat[cat].forEach(e => {
        const o = document.createElement("option");
        o.value = e.id;
        o.textContent = `${e.id} – ${e.name}`;
        og.appendChild(o);
      });
      sel.appendChild(og);
    });

    sel.addEventListener("change", () => {
      STATE.detailId = sel.value;
      renderDetail();
    });
  }

  function renderDetail() {
    if (!STATE.detailId && EXPS.length) STATE.detailId = EXPS[0].id;
    const e = EXPS.find(x => x.id === STATE.detailId);
    if (!e) return;

    document.getElementById("detail-select").value = e.id;
    const catEl = document.getElementById("detail-category");
    catEl.textContent = CATEGORIES_LABELS[e.category] || e.category;
    catEl.style.background = catColor(e.category) + "33";
    catEl.style.borderColor = catColor(e.category);
    document.getElementById("detail-dataset").textContent = DATASET_LABELS[e.dataset] || e.dataset;
    document.getElementById("detail-description").textContent = e.description || "";

    document.getElementById("d-acc").textContent    = fmtAcc(e.final?.test_accuracy);
    document.getElementById("d-f1").textContent     = fmtNum(e.final?.f1_macro, 3);
    document.getElementById("d-loss").textContent   = fmtNum(e.final?.test_loss, 3);
    document.getElementById("d-params").textContent = fmtInt(e.num_parameters);
    document.getElementById("d-epochs").textContent = e.epochs_run;
    document.getElementById("d-time").textContent   = fmtTime(e.training_time_sec);

    drawDetailCurves(e);
    drawConfusion(e);
    drawPerClassChart(e);
    renderHParams(e);
    renderArchitecture(e);
    renderSamples(e);
  }

  function drawDetailCurves(e) {
    const epochs = (e.history || []).map(h => h.epoch);
    const tAcc = (e.history || []).map(h => h.train_accuracy * 100);
    const vAcc = (e.history || []).map(h => h.val_accuracy * 100);
    const tLoss = (e.history || []).map(h => h.train_loss);
    const vLoss = (e.history || []).map(h => h.val_loss);

    if (chartDAcc) chartDAcc.destroy();
    chartDAcc = new Chart(document.getElementById("d-chart-acc"), {
      type: "line",
      data: {
        labels: epochs,
        datasets: [
          { label: "Train", data: tAcc, borderColor: "#a8d8c8", backgroundColor: "#a8d8c833", tension: 0.25 },
          { label: "Validation", data: vAcc, borderColor: "#a8c4e8", backgroundColor: "#a8c4e833", tension: 0.25 },
        ]
      },
      options: detailLineOptions("Presnosť (%)", true)
    });

    if (chartDLoss) chartDLoss.destroy();
    chartDLoss = new Chart(document.getElementById("d-chart-loss"), {
      type: "line",
      data: {
        labels: epochs,
        datasets: [
          { label: "Train", data: tLoss, borderColor: "#f0c9a8", backgroundColor: "#f0c9a833", tension: 0.25 },
          { label: "Validation", data: vLoss, borderColor: "#f0b8c4", backgroundColor: "#f0b8c433", tension: 0.25 },
        ]
      },
      options: detailLineOptions("Loss", false)
    });
  }

  function detailLineOptions(yLabel, isPercent) {
    return {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color: "#f5f5f5" } } },
      scales: {
        x: { ticks: { color: "#888" }, grid: { color: "#232323" }, title: { display: true, text: "Epocha", color: "#888" } },
        y: { ticks: { color: "#888", callback: v => isPercent ? v + " %" : v }, grid: { color: "#232323" }, title: { display: true, text: yLabel, color: "#888" } }
      }
    };
  }

  function drawConfusion(e) {
    const div = document.getElementById("d-confusion");
    div.innerHTML = "";
    const cm = e.final?.confusion_matrix;
    const classes = e.dataset_info?.classes || [];
    if (!cm || !cm.length) {
      div.innerHTML = `<p class="muted">Matica zámen nie je k dispozícii.</p>`;
      return;
    }
    const n = cm.length;
    const max = Math.max(...cm.flat(), 1);

    div.style.gridTemplateColumns = "1fr";

    const header = document.createElement("div");
    header.className = "cm-row";
    header.style.gridTemplateColumns = `28px repeat(${n}, 1fr)`;
    header.appendChild(document.createElement("div"));
    for (let j = 0; j < n; j++) {
      const lbl = document.createElement("div");
      lbl.className = "cm-label";
      lbl.textContent = classes[j] ?? j;
      header.appendChild(lbl);
    }
    div.appendChild(header);

    for (let i = 0; i < n; i++) {
      const row = document.createElement("div");
      row.className = "cm-row";
      row.style.gridTemplateColumns = `28px repeat(${n}, 1fr)`;
      const lbl = document.createElement("div");
      lbl.className = "cm-label";
      lbl.textContent = classes[i] ?? i;
      row.appendChild(lbl);
      for (let j = 0; j < n; j++) {
        const v = cm[i][j];
        const intensity = v / max;
        const cell = document.createElement("div");
        cell.className = "cm-cell";
        const isDiag = i === j;
        const r = isDiag ? Math.round( 80 + 175 * intensity)        : Math.round(255 * intensity);
        const g = isDiag ? Math.round(225 * intensity + 30)         : Math.round( 80 * (1 - intensity));
        const b = isDiag ? Math.round(196 * intensity + 30)         : Math.round(126 * (1 - intensity));
        cell.style.background = `rgba(${r},${g},${b},${0.18 + 0.72 * intensity})`;
        cell.style.color = intensity > 0.45 ? "#111111" : "#f5f5f5";
        cell.textContent = v;
        cell.title = `${classes[i] ?? i} → ${classes[j] ?? j}: ${v}`;
        row.appendChild(cell);
      }
      div.appendChild(row);
    }
  }

  function drawPerClassChart(e) {
    const canvas = document.getElementById("d-chart-perclass");
    if (!canvas) return;
    if (chartDPerClass) chartDPerClass.destroy();
    const classes = e.dataset_info?.classes || [];
    const acc = e.final?.per_class_accuracy || [];
    const f1  = e.final?.per_class_f1 || [];
    chartDPerClass = new Chart(canvas, {
      type: "bar",
      data: {
        labels: classes,
        datasets: [
          { label: "Presnosť (%)", data: acc.map(v => v * 100), backgroundColor: "#a8c4e8cc", borderColor: "#a8c4e8", borderWidth: 1 },
          { label: "F1 (%)",        data: f1.map(v => v * 100), backgroundColor: "#a8d8c8cc", borderColor: "#a8d8c8", borderWidth: 1 },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend: { labels: { color: "#f5f5f5" } } },
        scales: {
          x: { ticks: { color: "#888" }, grid: { color: "#232323" } },
          y: { ticks: { color: "#888", callback: v => v + " %" }, grid: { color: "#232323" }, beginAtZero: true, max: 100 }
        }
      }
    });
  }

  function renderHParams(e) {
    const div = document.getElementById("d-hparams");
    div.innerHTML = "";
    const m = e.model_config || {};
    const t = e.training_config || {};
    const rows = [
      ["Filtre", `[${(m.filters || []).join(", ")}]`],
      ["Veľkosť jadra", m.kernel_size],
      ["Aktivácia", m.activation],
      ["Batch normalization", m.use_batch_norm ? "áno" : "nie"],
      ["Pooling", m.pooling],
      ["Dropout", m.dropout],
      ["Dense (FC) jednotky", m.fc_units],
      ["Optimalizátor", t.optimizer],
      ["Rýchlosť učenia", t.lr],
      ["Veľkosť dávky", t.batch_size],
      ["Weight decay", t.weight_decay ?? 0],
      ["Plánovaný počet epoch", t.epochs],
      ["Skutočne odbehnutých", e.epochs_run],
      ["Počet parametrov", fmtInt(e.num_parameters)],
      ["Čas trénovania", fmtTime(e.training_time_sec)],
    ];
    rows.forEach(([k, v]) => {
      const ck = document.createElement("div");
      ck.className = "k";
      ck.textContent = k;
      const cv = document.createElement("div");
      cv.className = "v";
      cv.textContent = v;
      div.appendChild(ck);
      div.appendChild(cv);
    });
  }

  function renderArchitecture(e) {
    const div = document.getElementById("d-architecture");
    div.innerHTML = "";
    (e.architecture || []).forEach(layer => {
      const row = document.createElement("div");
      row.className = "layer";
      row.innerHTML = `
        <span class="lt">[${layer.index}] ${layer.type}</span>
        <span class="ld">${escapeHTML(layer.details)}</span>
      `;
      div.appendChild(row);
    });
  }

  function renderSamples(e) {
    const div = document.getElementById("d-samples");
    div.innerHTML = "";
    const classes = e.dataset_info?.classes || [];
    const samples = e.samples || [];
    samples.forEach(s => {
      const cell = document.createElement("div");
      const correct = s.true_label === s.pred_label;
      cell.className = "sample" + (correct ? " correct" : " wrong");
      const cw = 84;
      const canvas = document.createElement("canvas");
      canvas.width = cw; canvas.height = cw;
      drawImageOnCanvas(canvas, s.image);
      cell.appendChild(canvas);

      const meta = document.createElement("div");
      meta.className = "meta";
      const tName = classes[s.true_label] ?? s.true_label;
      const pName = classes[s.pred_label] ?? s.pred_label;
      meta.innerHTML = `
        <div class="pred">pred.: ${pName}</div>
        <div class="true">skut.: ${tName}</div>
        <div class="true">istota: ${(s.confidence * 100).toFixed(1)} %</div>
      `;
      cell.appendChild(meta);
      div.appendChild(cell);
    });
    if (!samples.length) {
      div.innerHTML = `<p class="muted">Pre tento experiment nie sú uložené ukážkové predikcie.</p>`;
    }
  }

  function drawImageOnCanvas(canvas, image) {
    const ctx = canvas.getContext("2d");
    if (!image || !image.length) return;
    const C = image.length;
    const H = image[0].length;
    const W = image[0][0].length;
    const off = document.createElement("canvas");
    off.width = W; off.height = H;
    const offCtx = off.getContext("2d");
    const imgData = offCtx.createImageData(W, H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = (y * W + x) * 4;
        let r, g, b;
        if (C === 1) {
          const v = clamp01(image[0][y][x]) * 255;
          r = g = b = v;
        } else {
          r = clamp01(image[0][y][x]) * 255;
          g = clamp01(image[1][y][x]) * 255;
          b = clamp01(image[2][y][x]) * 255;
        }
        imgData.data[i]   = r;
        imgData.data[i+1] = g;
        imgData.data[i+2] = b;
        imgData.data[i+3] = 255;
      }
    }
    offCtx.putImageData(imgData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  }

  function clamp01(v) { return v < 0 ? 0 : (v > 1 ? 1 : v); }

  function escapeHTML(s) {
    return String(s).replace(/[&<>"']/g, c =>
      ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[c]));
  }


  const KFG = {
    FILTERS: [
      { id: "f0", label: "1 blok  [16]",            filters: [16] },
      { id: "f1", label: "2 bloky  [16, 32]",        filters: [16, 32] },
      { id: "f2", label: "2 bloky  [32, 64]",        filters: [32, 64] },
      { id: "f3", label: "2 bloky  [64, 128]",       filters: [64, 128] },
      { id: "f4", label: "3 bloky  [32, 64, 128]",   filters: [32, 64, 128] },
    ],
    ACTS: [
      { id: "relu",       label: "ReLU" },
      { id: "leaky_relu", label: "Leaky ReLU" },
      { id: "elu",        label: "ELU" },
      { id: "gelu",       label: "GELU" },
    ],
    OPTS: [
      { id: "sgd",     label: "SGD" },
      { id: "adam",    label: "Adam" },
      { id: "adamw",   label: "AdamW" },
      { id: "rmsprop", label: "RMSprop" },
    ],
    LRS: [
      { id: "0.0001", label: "0.0001", val: 0.0001 },
      { id: "0.001",  label: "0.001",  val: 0.001 },
      { id: "0.01",   label: "0.01",   val: 0.01 },
      { id: "0.1",    label: "0.1",    val: 0.1 },
    ],
    DROPOUTS: [
      { id: "0",    label: "0  (vypnutý)", val: 0.0 },
      { id: "0.25", label: "0.25",         val: 0.25 },
      { id: "0.5",  label: "0.5",          val: 0.5 },
    ],
    KERNELS: [
      { id: "3", label: "3 × 3", val: 3 },
      { id: "5", label: "5 × 5", val: 5 },
      { id: "7", label: "7 × 7", val: 7 },
    ],
    BSS: [
      { id: "32",  label: "32",  val: 32 },
      { id: "64",  label: "64",  val: 64 },
      { id: "128", label: "128", val: 128 },
      { id: "256", label: "256", val: 256 },
    ],
    state: {
      filterIdx: 2, activation: "relu", optimizer: "adam",
      lr: 0.001, dropout: 0.0, use_batch_norm: false,
      kernel_size: 3, batch_size: 64,
    },
    chart: null,
    animFrame: null,
    initialized: false,
  };


  function kfgParamGroup(title, name, options, defaultId) {
    const btns = options.map(o => `
      <label class="kfg-radio ${o.id === defaultId ? "checked" : ""}">
        <input type="radio" name="kfg-${name}" value="${o.id}" ${o.id === defaultId ? "checked" : ""}>
        <span class="kfg-radio-dot"></span>
        <span class="kfg-radio-label">${o.label}</span>
      </label>`).join("");
    return `
      <div class="kfg-group">
        <div class="kfg-group-title">${title}</div>
        <div class="kfg-options">${btns}</div>
      </div>`;
  }

  function kfgBuildHTML() {
    const mnistCount = EXPS.filter(e => e.dataset === "mnist" && e.final?.test_accuracy).length;
    return `
    <div class="kfg-intro card">
      <div class="card-header">
        <h2>Konfigurátor CNN</h2>
        <p class="muted">Nastavte hyperparametre a sledujte predikovaný výsledok v reálnom čase. Systém porovná vaše nastavenia s <strong>${mnistCount} MNIST experimentmi</strong> (z celkových 34 — zvyšok sú Fashion-MNIST a CIFAR-10) a odhadne výkon siete.</p>
      </div>
    </div>
    <div class="kfg-layout">

      <div class="kfg-left card">
        <div class="kfg-panel-title">Nastavenia siete</div>
        ${kfgParamGroup("Topológia", "filters", KFG.FILTERS, "f2")}
        ${kfgParamGroup("Aktivačná funkcia", "activation", KFG.ACTS, "relu")}
        ${kfgParamGroup("Optimalizátor", "optimizer", KFG.OPTS, "adam")}
        ${kfgParamGroup("Learning rate", "lr", KFG.LRS, "0.001")}
        ${kfgParamGroup("Dropout", "dropout", KFG.DROPOUTS, "0")}
        <div class="kfg-group">
          <div class="kfg-group-title">Batch Normalization</div>
          <label class="kfg-toggle">
            <input type="checkbox" id="kfg-bn">
            <span class="kfg-track"><span class="kfg-thumb"></span></span>
            <span id="kfg-bn-lbl" class="kfg-toggle-txt">Vypnutý</span>
          </label>
        </div>
        ${kfgParamGroup("Veľkosť jadra", "kernel", KFG.KERNELS, "3")}
        ${kfgParamGroup("Batch size", "bs", KFG.BSS, "64")}
      </div>

      <div class="kfg-right">

        <div class="kfg-acc-card card">
          <div class="kfg-acc-label">Predikovaná presnosť</div>
          <div class="kfg-acc-num" id="kfg-acc-num">–</div>
          <div class="kfg-acc-range" id="kfg-acc-range"></div>
          <div class="kfg-acc-track"><div class="kfg-acc-fill" id="kfg-acc-fill"></div></div>
        </div>

        <div class="card">
          <div class="card-header">
            <h2>Krivky učenia — predikcia</h2>
            <p class="muted">Animovaná simulácia na základe najbližšieho experimentu.</p>
          </div>
          <div class="chart-wrap" style="height:210px">
            <canvas id="kfg-canvas"></canvas>
          </div>
        </div>

        <div class="card" id="kfg-match-card">
          <div class="card-header"><h2>Najbližší experiment</h2></div>
          <div id="kfg-match-body"></div>
        </div>

        <div class="card">
          <div class="card-header"><h2>Top 3 podobné experimenty</h2></div>
          <div id="kfg-top3"></div>
        </div>

      </div>
    </div>`;
  }


  function kfgAttach() {
    document.querySelectorAll("#view-konfigurator input[type=radio]").forEach(inp => {
      inp.addEventListener("change", () => {
        document.querySelectorAll(`input[name="${inp.name}"]`).forEach(s => {
          s.closest(".kfg-radio").classList.toggle("checked", s === inp);
        });
        kfgReadState();
        kfgUpdate();
      });
    });
    const bn = document.getElementById("kfg-bn");
    if (bn) bn.addEventListener("change", () => {
      const lbl = document.getElementById("kfg-bn-lbl");
      if (lbl) lbl.textContent = bn.checked ? "Zapnutý" : "Vypnutý";
      KFG.state.use_batch_norm = bn.checked;
      kfgUpdate();
    });

    const section = document.getElementById("view-konfigurator");
    if (section) {
      section.addEventListener("click", (ev) => {
        const target = ev.target.closest("[data-goto]");
        if (!target) return;
        const expId = target.dataset.goto;
        switchView("detail");
        setTimeout(() => {
          const sel = document.getElementById("detail-select");
          if (sel) {
            sel.value = expId;
            sel.dispatchEvent(new Event("change"));
          }
        }, 80);
      });
    }
  }

  function kfgReadState() {
    const r = n => { const el = document.querySelector(`input[name="kfg-${n}"]:checked`); return el ? el.value : null; };
    const fids = KFG.FILTERS.map(f => f.id);
    KFG.state.filterIdx   = Math.max(0, fids.indexOf(r("filters") || "f2"));
    KFG.state.activation  = r("activation") || "relu";
    KFG.state.optimizer   = r("optimizer")  || "adam";
    KFG.state.lr          = parseFloat(r("lr") || "0.001");
    KFG.state.dropout     = parseFloat(r("dropout") || "0");
    KFG.state.kernel_size = parseInt(r("kernel") || "3");
    KFG.state.batch_size  = parseInt(r("bs") || "64");
  }


  const ACT_ORD = ["relu","leaky_relu","elu","gelu"];
  const OPT_ORD = ["sgd","rmsprop","adam","adamw"];

  function kfgFiltersComplexity(f) {
    return Math.min(f.reduce((a,b)=>a+b,0) / 224, 1);
  }
  function kfgExpVec(e) {
    const m = e.model_config, t = e.training_config;
    return [
      kfgFiltersComplexity(m.filters || [32,64]),
      ACT_ORD.indexOf(m.activation || "relu") / 3,
      OPT_ORD.indexOf(t.optimizer  || "adam") / 3,
      (Math.log10(t.lr || 0.001) + 4) / 3,
      (m.dropout || 0) / 0.5,
      m.use_batch_norm ? 1 : 0,
      ((m.kernel_size || 3) - 3) / 4,
      Math.log2((t.batch_size || 64) / 32) / 3,
    ];
  }
  function kfgUserVec() {
    const s = KFG.state;
    return [
      kfgFiltersComplexity(KFG.FILTERS[s.filterIdx].filters),
      ACT_ORD.indexOf(s.activation) / 3,
      OPT_ORD.indexOf(s.optimizer)  / 3,
      (Math.log10(s.lr) + 4) / 3,
      s.dropout / 0.5,
      s.use_batch_norm ? 1 : 0,
      (s.kernel_size - 3) / 4,
      Math.log2(s.batch_size / 32) / 3,
    ];
  }
  function kfgDist(a, b) {
    return Math.sqrt(a.reduce((s,x,i) => s + (x - b[i])**2, 0));
  }
  function kfgFindNearest(n = 3) {
    const pool = EXPS.filter(e => e.dataset === "mnist" && e.final?.test_accuracy);
    const uv   = kfgUserVec();
    return pool
      .map(e => ({ e, d: kfgDist(uv, kfgExpVec(e)) }))
      .sort((a, b) => a.d - b.d)
      .slice(0, n);
  }


  function kfgUpdate() {
    const top = kfgFindNearest(3);
    if (!top.length) return;

    const invD  = top.map(m => 1 / (m.d + 0.0001));
    const sumW  = invD.reduce((a,b)=>a+b,0);
    const pred  = top.reduce((s,m,i) => s + m.e.final.test_accuracy * invD[i], 0) / sumW;
    const minA  = Math.min(...top.map(m => m.e.final.test_accuracy));
    const maxA  = Math.max(...top.map(m => m.e.final.test_accuracy));

    kfgUpdateAcc(pred, minA, maxA);
    kfgAnimCurve(top[0].e);
    kfgUpdateMatch(top[0]);
    kfgUpdateTop3(top);
  }

  function kfgUpdateAcc(acc, min, max) {
    const numEl  = document.getElementById("kfg-acc-num");
    const rngEl  = document.getElementById("kfg-acc-range");
    const fillEl = document.getElementById("kfg-acc-fill");
    if (!numEl) return;
    const from = parseFloat(numEl.dataset.cur || "0");
    const to   = acc * 100;
    kfgCountUp(numEl, from, to, 450);
    numEl.dataset.cur = String(to);
    const col = acc > 0.98 ? "#34d399" : acc > 0.90 ? "#fbbf24" : "#f472b6";
    numEl.style.color = col;
    if (rngEl) rngEl.textContent = `Rozsah: ${(min*100).toFixed(2)} % – ${(max*100).toFixed(2)} %`;
    if (fillEl) {
      fillEl.style.width = (acc * 100) + "%";
      fillEl.style.background = col;
    }
  }

  function kfgCountUp(el, from, to, ms) {
    const t0 = performance.now();
    function frame(now) {
      const p = Math.min((now - t0) / ms, 1);
      const e = 1 - (1 - p) ** 3;
      el.textContent = (from + (to - from) * e).toFixed(2) + " %";
      if (p < 1) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  }

  function kfgAnimCurve(exp) {
    if (!exp.history?.length) return;
    const canvas = document.getElementById("kfg-canvas");
    if (!canvas) return;

    if (KFG.animFrame) cancelAnimationFrame(KFG.animFrame);
    if (KFG.chart)   { KFG.chart.destroy(); KFG.chart = null; }

    const hist    = exp.history;
    const epochs  = hist.map(h => "Epocha " + h.epoch);
    const trAcc   = hist.map(h => +(h.train_accuracy * 100).toFixed(2));
    const vaAcc   = hist.map(h => +(h.val_accuracy   * 100).toFixed(2));
    const allAcc  = [...trAcc, ...vaAcc];
    const yMin    = Math.max(0, Math.min(...allAcc) - 3);

    let step = 0;
    const STEPS = 48;

    function tick() {
      step++;
      const t  = Math.min(step / STEPS, 1);
      const n  = Math.max(1, Math.ceil(t * epochs.length));
      const ls = epochs.slice(0, n);
      const td = trAcc.slice(0, n);
      const vd = vaAcc.slice(0, n);

      if (!KFG.chart) {
        KFG.chart = new Chart(canvas, {
          type: "line",
          data: {
            labels: ls,
            datasets: [
              { label: "Tréning",   data: td, borderColor: "#34d399", backgroundColor: "rgba(52,211,153,0.07)", tension: 0.35, pointRadius: 4, borderWidth: 2 },
              { label: "Validácia", data: vd, borderColor: "#818cf8", backgroundColor: "rgba(129,140,248,0.07)", tension: 0.35, pointRadius: 4, borderWidth: 2 },
            ]
          },
          options: {
            responsive: true, maintainAspectRatio: false, animation: false,
            plugins: { legend: { labels: { color: "#888", boxWidth: 12, font: { size: 12 } } } },
            scales: {
              x: { ticks: { color: "#888", font: { size: 11 } }, grid: { color: "#1e1e1e" } },
              y: { min: yMin, max: 100, ticks: { color: "#888", callback: v => v + " %" }, grid: { color: "#1e1e1e" } },
            }
          }
        });
      } else {
        KFG.chart.data.labels = ls;
        KFG.chart.data.datasets[0].data = td;
        KFG.chart.data.datasets[1].data = vd;
        KFG.chart.update("none");
      }
      if (step < STEPS) KFG.animFrame = requestAnimationFrame(tick);
    }
    KFG.animFrame = requestAnimationFrame(tick);
  }

  function kfgUpdateMatch(match) {
    const el = document.getElementById("kfg-match-body");
    if (!el) return;
    const e  = match.e;
    const sim = Math.max(0, Math.min(100, (1 - match.d / 2.5) * 100)).toFixed(0);
    const catColor = CAT_COLORS[e.category] || "#818cf8";
    el.innerHTML = `
      <div class="kfg-match-row">
        <div class="kfg-match-info">
          <div class="kfg-match-id" style="color:${catColor}">${e.id}</div>
          <div class="kfg-match-name">${e.name}</div>
          <span class="badge cat-${e.category}">${CATEGORIES_LABELS[e.category] || e.category}</span>
        </div>
        <div class="kfg-match-stats">
          <div class="kfg-stat">
            <div class="kfg-stat-val" style="color:#34d399">${(e.final.test_accuracy*100).toFixed(2)} %</div>
            <div class="kfg-stat-lbl">Presnosť</div>
          </div>
          <div class="kfg-stat">
            <div class="kfg-stat-val">${sim} %</div>
            <div class="kfg-stat-lbl">Zhoda</div>
          </div>
          <div class="kfg-stat">
            <div class="kfg-stat-val">${e.num_parameters ? (e.num_parameters/1000).toFixed(0)+"k" : "–"}</div>
            <div class="kfg-stat-lbl">Parametre</div>
          </div>
        </div>
      </div>
      <button class="kfg-goto-btn" data-goto="${e.id}">
        Zobraziť detail experimentu →
      </button>`;
  }

  function kfgUpdateTop3(matches) {
    const el = document.getElementById("kfg-top3");
    if (!el) return;
    el.innerHTML = matches.map((m, i) => {
      const e   = m.e;
      const acc = (e.final.test_accuracy * 100).toFixed(2);
      const col = CAT_COLORS[e.category] || "#818cf8";
      return `
        <div class="kfg-t3-row" data-goto="${e.id}">
          <div class="kfg-t3-rank" style="color:${col}">#${i+1}</div>
          <div class="kfg-t3-mid">
            <div class="kfg-t3-id">${e.id}</div>
            <div class="kfg-t3-bar-bg">
              <div class="kfg-t3-bar" style="width:${e.final.test_accuracy*100}%;background:${col}"></div>
            </div>
          </div>
          <div class="kfg-t3-acc">${acc} %</div>
        </div>`;
    }).join("");
  }

  function renderKonfigurator() {
    const section = document.getElementById("view-konfigurator");
    if (!section) return;
    if (KFG.initialized) return;
    KFG.initialized = true;
    section.innerHTML = kfgBuildHTML();
    kfgAttach();
    kfgUpdate();
  }


})();
