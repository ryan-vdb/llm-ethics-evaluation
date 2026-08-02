const MODEL_LABELS = {
  claude_opus: "Claude Opus",
  claude_sonnet: "Claude Sonnet",
  deepseek: "DeepSeek",
  gemini_flash: "Gemini Flash",
  gpt_55: "GPT · gpt_55",
  grok: "Grok",
};

export function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

export function modelLabel(value) {
  return MODEL_LABELS[value] || String(value).replaceAll("_", " ");
}

export function formatNumber(value, digits = 3) {
  if (!Number.isFinite(value)) return "—";
  return Number(value).toFixed(digits).replace("-0.", "−0.");
}

export function formatSigned(value, digits = 3) {
  if (!Number.isFinite(value)) return "—";
  const formatted = Math.abs(value).toFixed(digits);
  if (value > 0) return `+${formatted}`;
  if (value < 0) return `−${formatted}`;
  return Number(0).toFixed(digits);
}

export function formatPercent(value, digits = 1) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : "—";
}

function scale(value, domainMin, domainMax, rangeMin, rangeMax) {
  return rangeMin + ((value - domainMin) / (domainMax - domainMin)) * (rangeMax - rangeMin);
}

function ticks(min, max, count = 5) {
  return Array.from({ length: count }, (_, index) => min + ((max - min) * index) / (count - 1));
}

function accessibleTable(label, headers, rows) {
  if (!headers?.length || !rows?.length) return "";
  return `
    <details class="chart-data">
      <summary>View chart data</summary>
      <div class="table-scroll">
        <table>
          <caption>${escapeHtml(label)}</caption>
          <thead><tr>${headers.map((header) => `<th scope="col">${escapeHtml(header)}</th>`).join("")}</tr></thead>
          <tbody>${rows.map((row) => `<tr>${row.map((cell, index) => index === 0 ? `<th scope="row">${escapeHtml(cell)}</th>` : `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`).join("")}</tbody>
        </table>
      </div>
    </details>`;
}

function chartFrame({ body, width, height, label, className = "", headers = [], rows = [] }) {
  return `
    <div class="chart-scroll" tabindex="0" aria-label="Scrollable chart: ${escapeHtml(label)}">
      <svg class="chart ${className}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(label)}">
        <title>${escapeHtml(label)}</title>
        <desc>Exact values are available in the chart data disclosure immediately below.</desc>
        ${body}
      </svg>
    </div>
    ${accessibleTable(label, headers, rows)}`;
}

function axisMarkup({ min, max, left, right, top, bottom, width, height, digits = 1, zero = false }) {
  const tickValues = ticks(min, max, 5);
  let output = "";
  for (const tick of tickValues) {
    const x = scale(tick, min, max, left, width - right);
    output += `<line class="chart-grid" x1="${x}" x2="${x}" y1="${top}" y2="${height - bottom}" />`;
    output += `<text class="chart-tick" x="${x}" y="${height - 8}" text-anchor="middle">${formatNumber(tick, digits)}</text>`;
  }
  if (zero && min < 0 && max > 0) {
    const x = scale(0, min, max, left, width - right);
    output += `<line class="chart-zero" x1="${x}" x2="${x}" y1="${top - 4}" y2="${height - bottom + 2}" />`;
  }
  return output;
}

export function forestPlot(rows, options = {}) {
  const width = 760;
  const rowHeight = 54;
  const top = 28;
  const bottom = 38;
  const left = 146;
  const right = 50;
  const height = top + bottom + rows.length * rowHeight;
  const min = options.min ?? 0.65;
  const max = options.max ?? 0.9;
  let body = axisMarkup({ min, max, left, right, top, bottom, width, height, digits: 2 });
  if (Number.isFinite(options.reference)) {
    const referenceX = scale(options.reference, min, max, left, width - right);
    body += `<line class="chart-reference" x1="${referenceX}" x2="${referenceX}" y1="${top - 8}" y2="${height - bottom + 2}" />`;
    body += `<text class="chart-note" x="${referenceX + 6}" y="15">mean ${formatNumber(options.reference, 3)}</text>`;
  }
  rows.forEach((row, index) => {
    const y = top + rowHeight * index + rowHeight / 2;
    const low = scale(row.low, min, max, left, width - right);
    const high = scale(row.high, min, max, left, width - right);
    const point = scale(row.rho, min, max, left, width - right);
    body += `<text class="chart-label" x="${left - 14}" y="${y + 5}" text-anchor="end">${escapeHtml(modelLabel(row.model))}</text>`;
    body += `<line class="interval-line" x1="${low}" x2="${high}" y1="${y}" y2="${y}" />`;
    body += `<line class="interval-cap" x1="${low}" x2="${low}" y1="${y - 6}" y2="${y + 6}" />`;
    body += `<line class="interval-cap" x1="${high}" x2="${high}" y1="${y - 6}" y2="${y + 6}" />`;
    body += `<circle class="chart-point" cx="${point}" cy="${y}" r="7"><title>${escapeHtml(modelLabel(row.model))}: ρ ${formatNumber(row.rho)}; node-bootstrap CI [${formatNumber(row.low)}, ${formatNumber(row.high)}]; Holm p ${formatNumber(row.holm, 4)}</title></circle>`;
    body += `<text class="chart-value" x="${width - right + 10}" y="${y + 5}">${formatNumber(row.rho)}</text>`;
  });
  const label = options.label || "Held-out model correlations with node-bootstrap intervals";
  return chartFrame({
    body, width, height, label, className: "forest-chart",
    headers: ["Model", "Partial Spearman rho", "95% CI low", "95% CI high", "Holm p"],
    rows: rows.map((row) => [modelLabel(row.model), formatNumber(row.rho), formatNumber(row.low), formatNumber(row.high), formatNumber(row.holm, 4)]),
  });
}

export function pairedDotPlot(rows, options = {}) {
  const width = 760;
  const rowHeight = 52;
  const top = 28;
  const bottom = 42;
  const left = 146;
  const right = 44;
  const height = top + bottom + rows.length * rowHeight;
  const min = options.min ?? 0;
  const max = options.max ?? 0.5;
  let body = axisMarkup({ min, max, left, right, top, bottom, width, height, digits: 1 });
  rows.forEach((row, index) => {
    const y = top + rowHeight * index + rowHeight / 2;
    const before = scale(row.before, min, max, left, width - right);
    const after = scale(row.after, min, max, left, width - right);
    body += `<text class="chart-label" x="${left - 14}" y="${y + 5}" text-anchor="end">${escapeHtml(modelLabel(row.label))}</text>`;
    body += `<line class="dumbbell-line" x1="${before}" x2="${after}" y1="${y}" y2="${y}" />`;
    body += `<circle class="dumbbell-before" cx="${before}" cy="${y}" r="6"><title>Raw answer geometry association: ${formatNumber(row.before)}</title></circle>`;
    body += `<circle class="dumbbell-after" cx="${after}" cy="${y}" r="7"><title>Question-projected residual geometry association: ${formatNumber(row.after)}</title></circle>`;
  });
  body += `<g class="chart-legend"><circle class="dumbbell-before" cx="${left}" cy="14" r="5"/><text x="${left + 10}" y="18">raw</text><circle class="dumbbell-after" cx="${left + 62}" cy="14" r="5"/><text x="${left + 72}" y="18">residual</text></g>`;
  const label = options.label || "Raw versus question-projected answer geometry association by model";
  return chartFrame({
    body, width, height, label,
    headers: ["Model", "Raw answer geometry", "Question-projected residual geometry"],
    rows: rows.map((row) => [modelLabel(row.label), formatNumber(row.before), formatNumber(row.after)]),
  });
}

export function bulletPlot({ observed, nullMean, null99, min = 0, max = 0.9, label }) {
  const width = 760;
  const height = 150;
  const left = 52;
  const right = 52;
  const axisY = 86;
  const nullStart = scale(min, min, max, left, width - right);
  const nullEnd = scale(null99, min, max, left, width - right);
  const meanX = scale(nullMean, min, max, left, width - right);
  const observedX = scale(observed, min, max, left, width - right);
  let body = axisMarkup({ min, max, left, right, top: 48, bottom: 38, width, height, digits: 1 });
  body += `<rect class="null-band" x="${nullStart}" y="${axisY - 14}" width="${nullEnd - nullStart}" height="28" rx="14"><title>Null range through the 99th percentile: ${formatNumber(null99)}</title></rect>`;
  body += `<line class="null-mean" x1="${meanX}" x2="${meanX}" y1="${axisY - 22}" y2="${axisY + 22}" />`;
  body += `<circle class="observed-marker" cx="${observedX}" cy="${axisY}" r="10"><title>Observed: ${formatNumber(observed)}</title></circle>`;
  body += `<text class="chart-note" x="${meanX}" y="50" text-anchor="middle">null mean ${formatNumber(nullMean)}</text>`;
  body += `<text class="chart-value strong" x="${observedX}" y="50" text-anchor="middle">observed ${formatNumber(observed)}</text>`;
  const chartLabel = label || "Observed result compared with projection-artifact null";
  return chartFrame({
    body, width, height, label: chartLabel,
    headers: ["Statistic", "Value"],
    rows: [["Observed", formatNumber(observed)], ["Null mean", formatNumber(nullMean)], ["Null 99th percentile", formatNumber(null99)]],
  });
}

export function divergingBarPlot(rows, options = {}) {
  const width = 760;
  const rowHeight = options.rowHeight || 50;
  const top = 26;
  const bottom = 42;
  const left = options.left || 165;
  const right = 58;
  const height = top + bottom + rows.length * rowHeight;
  const min = options.min ?? Math.min(-0.02, ...rows.map((row) => row.value));
  const max = options.max ?? Math.max(0.07, ...rows.map((row) => row.value));
  const zeroX = scale(0, min, max, left, width - right);
  let body = axisMarkup({ min, max, left, right, top, bottom, width, height, digits: options.digits ?? 2, zero: true });
  rows.forEach((row, index) => {
    const y = top + rowHeight * index + rowHeight / 2;
    const valueX = scale(row.value, min, max, left, width - right);
    const x = Math.min(zeroX, valueX);
    const barWidth = Math.max(2, Math.abs(valueX - zeroX));
    const className = row.highlight ? "diverging-bar highlight" : "diverging-bar";
    body += `<text class="chart-label" x="${left - 14}" y="${y + 5}" text-anchor="end">${escapeHtml(row.label)}</text>`;
    body += `<rect class="${className}" x="${x}" y="${y - 9}" width="${barWidth}" height="18" rx="4"><title>${escapeHtml(row.tooltip || `${row.label}: ${formatSigned(row.value)}`)}</title></rect>`;
    body += `<circle class="diverging-point${row.highlight ? " highlight" : ""}" cx="${valueX}" cy="${y}" r="5" />`;
    body += `<text class="chart-value" x="${row.value >= 0 ? valueX + 9 : valueX - 9}" y="${y + 5}" text-anchor="${row.value >= 0 ? "start" : "end"}">${formatSigned(row.value, options.digits ?? 3)}</text>`;
  });
  const label = options.label || "Diverging effect chart";
  return chartFrame({
    body, width, height, label,
    headers: ["Group", "Effect"],
    rows: rows.map((row) => [row.label, formatSigned(row.value, options.digits ?? 3)]),
  });
}

export function horizontalBarPlot(rows, options = {}) {
  const width = 760;
  const rowHeight = options.rowHeight || 50;
  const top = 24;
  const bottom = 42;
  const left = options.left || 210;
  const right = 58;
  const height = top + bottom + rows.length * rowHeight;
  const min = options.min ?? 0;
  const max = options.max ?? Math.max(...rows.map((row) => row.value)) * 1.1;
  let body = axisMarkup({ min, max, left, right, top, bottom, width, height, digits: options.digits ?? 1 });
  if (Number.isFinite(options.reference)) {
    const x = scale(options.reference, min, max, left, width - right);
    body += `<line class="chart-reference" x1="${x}" x2="${x}" y1="${top}" y2="${height - bottom}" />`;
  }
  rows.forEach((row, index) => {
    const y = top + rowHeight * index + rowHeight / 2;
    const valueX = scale(row.value, min, max, left, width - right);
    body += `<text class="chart-label" x="${left - 14}" y="${y + 5}" text-anchor="end">${escapeHtml(row.label)}</text>`;
    body += `<rect class="horizontal-bar${row.highlight ? " highlight" : ""}" x="${left}" y="${y - 9}" width="${Math.max(2, valueX - left)}" height="18" rx="4"><title>${escapeHtml(row.tooltip || `${row.label}: ${formatNumber(row.value)}`)}</title></rect>`;
    body += `<text class="chart-value" x="${valueX + 9}" y="${y + 5}">${options.percent ? formatPercent(row.value) : formatNumber(row.value, options.digits ?? 3)}</text>`;
  });
  const label = options.label || "Horizontal bar chart";
  return chartFrame({
    body, width, height, label,
    headers: ["Check", "Value"],
    rows: rows.map((row) => [row.label, options.percent ? formatPercent(row.value) : formatNumber(row.value, options.digits ?? 3)]),
  });
}

export function estimatePlot({ estimate, low, high, min = -0.04, max = 0.08, label }) {
  const width = 760;
  const height = 150;
  const left = 62;
  const right = 62;
  const y = 75;
  const estimateX = scale(estimate, min, max, left, width - right);
  const lowX = scale(low, min, max, left, width - right);
  const highX = scale(high, min, max, left, width - right);
  let body = axisMarkup({ min, max, left, right, top: 38, bottom: 38, width, height, digits: 2, zero: true });
  body += `<line class="estimate-interval" x1="${lowX}" x2="${highX}" y1="${y}" y2="${y}" />`;
  body += `<line class="interval-cap" x1="${lowX}" x2="${lowX}" y1="${y - 10}" y2="${y + 10}" />`;
  body += `<line class="interval-cap" x1="${highX}" x2="${highX}" y1="${y - 10}" y2="${y + 10}" />`;
  body += `<circle class="estimate-point" cx="${estimateX}" cy="${y}" r="10"><title>Estimate ${formatSigned(estimate)}; crossed 95% interval [${formatSigned(low)}, ${formatSigned(high)}]</title></circle>`;
  body += `<text class="chart-value strong" x="${estimateX}" y="45" text-anchor="middle">${formatSigned(estimate)}</text>`;
  const chartLabel = label || "Estimate with crossed model and scenario bootstrap interval";
  return chartFrame({
    body, width, height, label: chartLabel,
    headers: ["Statistic", "Value"],
    rows: [["Estimate", formatSigned(estimate)], ["95% interval low", formatSigned(low)], ["95% interval high", formatSigned(high)]],
  });
}
