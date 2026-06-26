/* Shared visualizations for the champion/challenger system.
 *   AltCredit.loadShapeFunctions()                  -> {feature: curve}
 *   AltCredit.renderModelPanel(el, panel)           -> agreement card
 *   AltCredit.renderShapeViewer(el, shapeMap, trace, drivers)
 *
 * The shape viewer renders the EBM champion's own decision curves (the model
 * itself, not a SHAP approximation) and marks where this borrower sits.
 */
(function () {
  const SVGNS = "http://www.w3.org/2000/svg";
  const bandColor = (b) => (b === "APPROVE" ? "#16a34a" : b === "REVIEW" ? "#d97706" : "#dc2626");
  const modelLabel = (n) =>
    ({ ebm: "EBM", catboost: "CatBoost", logistic: "Logistic" }[n] || n);

  function fmtValue(v, fmt) {
    if (v === null || v === undefined) return "—";
    switch (fmt) {
      case "rupee": return "₹" + Math.round(v).toLocaleString("en-IN");
      case "percent": return (v * 100).toFixed(0) + "%";
      case "rating": return v.toFixed(2);
      case "count": return Math.round(v).toLocaleString("en-IN");
      case "days": return v.toFixed(1) + " days";
      case "score01": return v.toFixed(2);
      case "bool": return v >= 0.5 ? "Yes" : "No";
      default: return (+v).toFixed(2);
    }
  }

  function injectStyle() {
    if (document.getElementById("acv-style")) return;
    const s = document.createElement("style");
    s.id = "acv-style";
    s.textContent = `
      .acv-panel{font-family:inherit}
      .acv-models{display:flex;flex-direction:column;gap:8px;margin:10px 0}
      .acv-model{display:flex;align-items:center;gap:12px;padding:10px 14px;border:1px solid #e2e8f0;border-radius:10px;background:#f8fafc}
      .acv-model.champ{border-color:#2563eb;background:#eff6ff}
      .acv-mrole{font-size:11px;font-weight:700;letter-spacing:.04em;color:#64748b;width:90px;text-transform:uppercase}
      .acv-mname{font-weight:700;color:#0f172a;flex:1}
      .acv-mpd{font-variant-numeric:tabular-nums;color:#475569;font-size:13px;width:96px}
      .acv-chip{padding:4px 12px;border-radius:20px;color:#fff;font-size:12px;font-weight:700}
      .acv-verdict{display:flex;align-items:center;gap:10px;padding:12px 14px;border-radius:10px;font-weight:700;margin-top:6px}
      .acv-verdict small{font-weight:500;opacity:.85}
      .acv-tabs{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 12px}
      .acv-tab{padding:6px 12px;border:1px solid #cbd5e1;border-radius:20px;background:#fff;cursor:pointer;font-size:12px;font-weight:600;color:#334155}
      .acv-tab.on{background:#2563eb;border-color:#2563eb;color:#fff}
      .acv-cap{font-size:13px;color:#475569;margin-top:8px;line-height:1.5}
      .acv-cap b{color:#0f172a}
    `;
    document.head.appendChild(s);
  }

  let _shapePromise = null;
  function loadShapeFunctions() {
    if (!_shapePromise) {
      _shapePromise = fetch("/score/model/explanations")
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          const map = {};
          (d && d.features ? d.features : []).forEach((f) => (map[f.feature] = f));
          return map;
        })
        .catch(() => ({}));
    }
    return _shapePromise;
  }

  function renderModelPanel(el, panel) {
    if (!el || !panel || !panel.champion) { if (el) el.style.display = "none"; return; }
    injectStyle();
    const ch = panel.champion;
    const rows = [
      `<div class="acv-model champ">
        <span class="acv-mrole">Champion</span>
        <span class="acv-mname">${modelLabel(ch.name)} <span style="font-weight:500;color:#2563eb">· glass-box</span></span>
        <span class="acv-mpd">PD ${(ch.pd * 100).toFixed(1)}%</span>
        <span class="acv-chip" style="background:${bandColor(ch.band)}">${ch.band}</span>
      </div>`,
    ];
    (panel.challengers || []).forEach((c) => {
      rows.push(`<div class="acv-model">
        <span class="acv-mrole">Challenger</span>
        <span class="acv-mname">${modelLabel(c.name)}</span>
        <span class="acv-mpd">PD ${(c.pd * 100).toFixed(1)}%</span>
        <span class="acv-chip" style="background:${bandColor(c.band)}">${c.band}</span>
      </div>`);
    });

    let vb, vc, vt;
    if (panel.unanimous) {
      vb = "#dcfce7"; vc = "#166534";
      vt = `✓ Unanimous panel — auto-decision permitted <small>all ${panel.n_models} models agree</small>`;
    } else if (panel.hard_conflict) {
      vb = "#fee2e2"; vc = "#991b1b";
      vt = `⚠ Hard conflict — one model would approve while another would reject <small>routed to human review</small>`;
    } else {
      vb = "#fef3c7"; vc = "#854d0e";
      vt = `~ Partial agreement <small>${panel.agreement_pct}% with champion · adjacent bands, not a conflict</small>`;
    }

    el.style.display = "block";
    el.innerHTML = `<div class="acv-panel">
      <div class="acv-models">${rows.join("")}</div>
      <div class="acv-verdict" style="background:${vb};color:${vc}">${vt}</div>
      <div class="acv-cap">The <b>glass-box champion</b> decides and explains; the challengers audit it.
        Agreement is measured on the action (approve / review / reject), not raw PD.
        PD spread across the panel: <b>${(panel.pd_min * 100).toFixed(1)}%–${(panel.pd_max * 100).toFixed(1)}%</b>.</div>
    </div>`;
  }

  // ---- shape-function viewer ----------------------------------------------
  function traceLookup(trace) {
    const map = {};
    const add = (s) => { if (s && s.feature) map[s.feature] = s; };
    (trace.top_drivers || []).forEach(add);
    (trace.by_source || []).forEach((g) => (g.signals || []).forEach(add));
    return map;
  }

  function el(tag, attrs) {
    const n = document.createElementNS(SVGNS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }

  function drawCurve(svg, curve, borrower) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const W = 640, H = 280, m = { l: 56, r: 18, t: 18, b: 48 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const x = curve.x, pts = curve.points;
    const xMin = x[0], xMax = x[x.length - 1] || x[0] + 1;
    let yMin = Math.min(0, ...pts), yMax = Math.max(0, ...pts);
    if (borrower) { yMin = Math.min(yMin, borrower.points); yMax = Math.max(yMax, borrower.points); }
    const pad = (yMax - yMin) * 0.12 || 1; yMin -= pad; yMax += pad;
    const sx = (v) => m.l + ((v - xMin) / (xMax - xMin || 1)) * (W - m.l - m.r);
    const sy = (v) => m.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (H - m.t - m.b);

    // zero baseline
    svg.appendChild(el("line", { x1: m.l, y1: sy(0), x2: W - m.r, y2: sy(0), stroke: "#cbd5e1", "stroke-dasharray": "4 4" }));
    svg.appendChild(el("text", { x: m.l - 8, y: sy(0) + 4, "text-anchor": "end", "font-size": 11, fill: "#94a3b8" }, ));
    // axis labels (y)
    [yMax, yMin].forEach((yv) => {
      const t = el("text", { x: m.l - 8, y: sy(yv) + 4, "text-anchor": "end", "font-size": 11, fill: "#94a3b8" });
      t.textContent = (yv >= 0 ? "+" : "") + yv.toFixed(0);
      svg.appendChild(t);
    });

    // step segments, colored by sign (green = raises score, red = lowers)
    for (let i = 0; i < pts.length; i++) {
      const c = pts[i] >= 0 ? "#16a34a" : "#dc2626";
      svg.appendChild(el("line", { x1: sx(x[i]), y1: sy(pts[i]), x2: sx(x[i + 1]), y2: sy(pts[i]), stroke: c, "stroke-width": 3, "stroke-linecap": "round" }));
      if (i > 0) svg.appendChild(el("line", { x1: sx(x[i]), y1: sy(pts[i - 1]), x2: sx(x[i]), y2: sy(pts[i]), stroke: "#e2e8f0", "stroke-width": 1.5 }));
    }

    // borrower marker
    if (borrower && isFinite(borrower.value)) {
      const bx = sx(Math.max(xMin, Math.min(xMax, borrower.value)));
      svg.appendChild(el("line", { x1: bx, y1: m.t, x2: bx, y2: H - m.b, stroke: "#2563eb", "stroke-dasharray": "5 4", "stroke-width": 1.5 }));
      const dot = el("circle", { cx: bx, cy: sy(borrower.points), r: 6, fill: "#2563eb", stroke: "#fff", "stroke-width": 2 });
      svg.appendChild(dot);
      const lab = el("text", { x: Math.min(bx + 8, W - m.r - 70), y: sy(borrower.points) - 12, "font-size": 12, "font-weight": 700, fill: "#1d4ed8" });
      lab.textContent = "you";
      svg.appendChild(lab);
    }
    // x label
    const xl = el("text", { x: (m.l + W - m.r) / 2, y: H - 12, "text-anchor": "middle", "font-size": 12, fill: "#475569" });
    xl.textContent = curve.label;
    svg.appendChild(xl);
  }

  function renderShapeViewer(host, shapeMap, trace, drivers) {
    if (!host) return;
    injectStyle();
    const look = traceLookup(trace);
    const feats = (drivers || [])
      .map((d) => d.feature)
      .filter((f) => shapeMap[f])
      .slice(0, 6);
    if (!feats.length) { host.style.display = "none"; return; }
    host.style.display = "block";
    host.innerHTML = `<div class="acv-tabs" id="acv-tabs"></div>
      <svg id="acv-svg" width="100%" style="max-width:640px;display:block;margin:0 auto"></svg>
      <div class="acv-cap" id="acv-cap"></div>`;
    const tabs = host.querySelector("#acv-tabs");
    const svg = host.querySelector("#acv-svg");
    const cap = host.querySelector("#acv-cap");

    function show(feat) {
      const curve = shapeMap[feat];
      const sig = look[feat] || {};
      const borrower = sig.value !== undefined ? { value: +sig.value, points: +(sig.points || 0) } : null;
      drawCurve(svg, curve, borrower);
      tabs.querySelectorAll(".acv-tab").forEach((t) => t.classList.toggle("on", t.dataset.f === feat));
      const vtxt = borrower ? fmtValue(borrower.value, curve.fmt) : "—";
      const ptxt = borrower ? (borrower.points >= 0 ? "+" : "") + borrower.points.toFixed(1) : "—";
      cap.innerHTML = `This curve <b>is the model</b> — the EBM's own term for <b>${curve.label}</b>, not a SHAP estimate.
        At this borrower's value <b>${vtxt}</b>, the model contributes <b>${ptxt} points</b> to the score.
        Green raises the score, red lowers it.`;
    }

    feats.forEach((f) => {
      const b = document.createElement("button");
      b.className = "acv-tab";
      b.dataset.f = f;
      b.textContent = shapeMap[f].label;
      b.onclick = () => show(f);
      tabs.appendChild(b);
    });
    show(feats[0]);
  }

  window.AltCredit = { loadShapeFunctions, renderModelPanel, renderShapeViewer };
})();
