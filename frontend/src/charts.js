const MODEL_LABELS = {
  claude_opus: "Claude Opus",
  claude_sonnet: "Claude Sonnet",
  deepseek: "DeepSeek",
  gemini_flash: "Gemini Flash",
  gpt_55: "GPT-5.5",
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

function wrapLabel(value, maximumCharacters) {
  const label = String(value);
  if (!Number.isFinite(maximumCharacters) || label.length <= maximumCharacters) return [label];
  const words = label.split(/\s+/);
  const lines = [];
  let line = "";
  for (const word of words) {
    const candidate = line ? `${line} ${word}` : word;
    if (line && candidate.length > maximumCharacters) {
      lines.push(line);
      line = word;
    } else {
      line = candidate;
    }
  }
  if (line) lines.push(line);
  return lines;
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

export function embeddingSpaceFigure({
  dimensions,
  rawQuestionCosine,
  maximumResidualQuestionCosine,
}) {
  const dimensionLabel = Number(dimensions).toLocaleString();
  const residualLabel = Number.isFinite(maximumResidualQuestionCosine)
    ? maximumResidualQuestionCosine.toExponential(1).replace("e-", "e−")
    : "—";
  return `
    <figure class="embedding-primer-figure" aria-labelledby="embedding-primer-title" aria-describedby="embedding-primer-caption">
      <header class="embedding-primer-heading">
        <span>60-second primer · schematic</span>
        <h2 id="embedding-primer-title">A response becomes a point in a very large space.</h2>
        <p>The embedder translates meaning into coordinates. Directional similarity between those coordinates lets us compare responses mathematically.</p>
      </header>
      <div class="embedding-flow">
        <section class="embedding-step embedding-text-step">
          <span>01 · Response text</span>
          <div class="response-snippet"><b>A</b><q>Respect informed choice and consent.</q></div>
          <div class="response-snippet"><b>B</b><q>Autonomy should guide the decision.</q></div>
          <div class="response-snippet distant"><b>C</b><q>Prevent irreversible harm to future generations.</q></div>
        </section>
        <section class="embedding-step embedding-vector-step">
          <span>02 · Encode meaning</span>
          <div class="vector-row"><b>A</b><code>[a₁, a₂, …, a<sub>${dimensionLabel}</sub>]</code></div>
          <div class="vector-row"><b>B</b><code>[b₁, b₂, …, b<sub>${dimensionLabel}</sub>]</code></div>
          <div class="vector-row distant"><b>C</b><code>[c₁, c₂, …, c<sub>${dimensionLabel}</sub>]</code></div>
          <p>${dimensionLabel} learned coordinates per response—not human-assigned categories.</p>
        </section>
        <section class="embedding-step embedding-projection-step">
          <span>03 · Remove the question direction</span>
          <svg viewBox="0 0 230 128" aria-hidden="true">
            <defs><marker id="primer-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z"/></marker></defs>
            <line class="primer-axis" x1="22" y1="106" x2="214" y2="106"/>
            <line class="primer-question-vector" x1="28" y1="106" x2="187" y2="48" marker-end="url(#primer-arrow)"/>
            <line class="primer-response-vector" x1="28" y1="106" x2="165" y2="18" marker-end="url(#primer-arrow)"/>
            <line class="primer-projection-line" x1="165" y1="18" x2="185" y2="49"/>
            <line class="primer-residual-vector" x1="185" y1="49" x2="165" y2="18" marker-end="url(#primer-arrow)"/>
            <text x="188" y="43">question q</text><text x="151" y="14">answer r</text><text x="121" y="39">residual r⊥</text>
          </svg>
          <code>r⊥ = r − proj<sub>q</sub>(r)</code>
          <p>The exact paired-question component is subtracted before comparison.</p>
        </section>
        <section class="embedding-step embedding-geometry-step">
          <span>04 · Compare directions</span>
          <svg viewBox="0 0 230 128" aria-hidden="true">
            <line class="primer-axis" x1="28" y1="106" x2="214" y2="106"/><line class="primer-axis" x1="28" y1="106" x2="28" y2="12"/>
            <line class="primer-point-ray close" x1="28" y1="106" x2="166" y2="33"/><line class="primer-point-ray close" x1="28" y1="106" x2="179" y2="48"/><line class="primer-point-ray distant" x1="28" y1="106" x2="76" y2="22"/>
            <circle class="primer-point close" cx="166" cy="33" r="7"/><circle class="primer-point close" cx="179" cy="48" r="7"/><circle class="primer-point distant" cx="76" cy="22" r="7"/>
            <text x="153" y="25">A</text><text x="183" y="48">B</text><text x="62" y="18">C</text>
            <path class="primer-angle" d="M64 87 A42 42 0 0 1 68 82"/>
          </svg>
          <p>A and B point in similar directions, so their cosine is higher. C points elsewhere.</p>
        </section>
      </div>
      <div class="embedding-primer-facts" aria-label="Observed preprocessing facts">
        <span><b>${dimensionLabel}</b><small>dimensions used in the actual analysis</small></span>
        <span><b>${formatNumber(rawQuestionCosine)}</b><small>mean raw answer–question cosine</small></span>
        <span><b>${escapeHtml(residualLabel)}</b><small>maximum absolute cosine after projection</small></span>
      </div>
      <figcaption id="embedding-primer-caption">The point diagram is a two-dimensional teaching sketch, not an observed scatterplot. The calculations use every coordinate in the ${dimensionLabel}-dimensional vectors; the two projection audit values are measured from this dataset.</figcaption>
    </figure>`;
}

export function questionSimilarityHeatmapFigure(heatmap) {
  const questionOptions = heatmap.questions
    .map((question, index) => ({ question, index }))
    .sort((first, second) => first.question.id - second.question.id)
    .map(({ question, index }) => `<option value="${index}">Q${question.id} · ${escapeHtml(question.conflict)}</option>`)
    .join("");
  const firstQuestion = heatmap.questions[0];
  const secondQuestion = heatmap.questions[1];
  const initialValue = heatmap.matrix[0][1];
  const questionCard = (question, name) => `
    <article class="question-pair-card" data-heatmap-question-card="${name}">
      <span>Q${question.id} · ${escapeHtml(question.domain)}</span>
      <strong>${escapeHtml(question.conflict)}</strong>
      <p>${escapeHtml(question.question)}</p>
    </article>`;
  return `
    <div class="question-heatmap-layout">
      <div class="question-heatmap-visual">
        <div class="question-heatmap-scroll" tabindex="0" aria-label="Scrollable 93 by 93 question-similarity heatmap">
          <canvas id="question-similarity-canvas" class="question-similarity-canvas" width="760" height="760" tabindex="0" role="img" aria-describedby="question-heatmap-description">
            A 93 by 93 matrix of average question-orthogonalized response cosine similarities. Use the pair inspector for exact values.
          </canvas>
        </div>
        <p class="heatmap-scroll-hint">Swipe horizontally to inspect the complete matrix <span aria-hidden="true">→</span></p>
        <div class="question-heatmap-scale" aria-label="Linear cosine color scale from 0.20 to 0.65"><span>0.20 lower</span><span>0.425</span><span>0.65 higher</span></div>
        <div class="question-heatmap-stats">
          <span><b>${formatNumber(heatmap.summary.minimum)}</b><small>minimum pair</small></span>
          <span><b>${formatNumber(heatmap.summary.mean)}</b><small>mean pair</small></span>
          <span><b>${formatNumber(heatmap.summary.maximum)}</b><small>maximum pair</small></span>
          <span><b>${heatmap.summary.uniquePairCount.toLocaleString()}</b><small>unique pairs</small></span>
        </div>
      </div>
      <aside class="question-heatmap-guide">
        <span class="panel-kicker">Inspect any cell</span>
        <h4>Which two scenarios receive similarly directed responses?</h4>
        <p id="question-heatmap-description">Each cell averages six separately calculated residual cosines—one per model. The response vectors themselves are never averaged.</p>
        <div class="question-heatmap-order-controls" role="group" aria-label="Question matrix display order">
          <button type="button" data-heatmap-order="similarity" aria-pressed="true">Similarity order</button>
          <button type="button" data-heatmap-order="question-id" aria-pressed="false">Question ID order</button>
        </div>
        <div class="question-pair-selectors">
          <label><span>First scenario</span><select data-heatmap-select="first">${questionOptions}</select></label>
          <label><span>Second scenario</span><select data-heatmap-select="second">${questionOptions}</select></label>
        </div>
        <output class="question-pair-value"><strong data-heatmap-value>${formatNumber(initialValue)}</strong><span>mean model-wise residual cosine</span></output>
        <div class="sr-only" data-heatmap-status aria-live="polite"></div>
        <div class="question-pair-details">
          ${questionCard(firstQuestion, "first")}
          ${questionCard(secondQuestion, "second")}
        </div>
        <p class="question-heatmap-ordering">Rows and columns use the same average-linkage ordering to place similar patterns together. This changes only display order; it does not define or test clusters.</p>
      </aside>
    </div>`;
}

function interpolateColor(start, end, amount) {
  const values = start.map((value, index) => Math.round(value + (end[index] - value) * amount));
  return `rgb(${values.join(", ")})`;
}

export function setupQuestionSimilarityHeatmap(heatmap) {
  const canvas = document.querySelector("#question-similarity-canvas");
  if (!canvas) return;
  const context = canvas.getContext("2d");
  const root = document.documentElement;
  const matrixLeft = 82;
  const matrixTop = 62;
  const cellSize = 7;
  const tickIndices = [0, 15, 30, 45, 60, 75, heatmap.questions.length - 1];
  const firstSelect = document.querySelector('[data-heatmap-select="first"]');
  const secondSelect = document.querySelector('[data-heatmap-select="second"]');
  const orderButtons = [...document.querySelectorAll("[data-heatmap-order]")];
  const clusteredOrder = heatmap.questions.map((_, index) => index);
  const questionIdOrder = [...clusteredOrder].sort((first, second) => heatmap.questions[first].id - heatmap.questions[second].id);
  let displayOrder = clusteredOrder;
  let selectedRow = 0;
  let selectedColumn = 1;

  firstSelect.value = String(selectedRow);
  secondSelect.value = String(selectedColumn);

  const draw = () => {
    const dark = root.dataset.theme === "dark";
    const neutral = dark ? [16, 40, 45] : [244, 241, 233];
    const positive = dark ? [102, 202, 187] : [13, 109, 101];
    const diagonal = dark ? [43, 56, 57] : [222, 217, 206];
    const labelColor = dark ? "rgba(236, 243, 238, .68)" : "rgba(16, 41, 45, .7)";
    const outline = dark ? "#fffdf8" : "#10292d";
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.imageSmoothingEnabled = false;
    displayOrder.forEach((matrixRowIndex, rowIndex) => {
      displayOrder.forEach((matrixColumnIndex, columnIndex) => {
        const value = heatmap.matrix[matrixRowIndex][matrixColumnIndex];
        const colorAmount = Math.max(0, Math.min(1, (value - 0.20) / 0.45));
        context.fillStyle = matrixRowIndex === matrixColumnIndex
          ? `rgb(${diagonal.join(", ")})`
          : interpolateColor(neutral, positive, colorAmount);
        context.fillRect(
          matrixLeft + columnIndex * cellSize,
          matrixTop + rowIndex * cellSize,
          cellSize + 0.25,
          cellSize + 0.25,
        );
      });
    });
    context.strokeStyle = outline;
    context.lineWidth = 2;
    const selectedDisplayRow = displayOrder.indexOf(selectedRow);
    const selectedDisplayColumn = displayOrder.indexOf(selectedColumn);
    for (const [rowIndex, columnIndex] of [[selectedDisplayRow, selectedDisplayColumn], [selectedDisplayColumn, selectedDisplayRow]]) {
      context.strokeRect(
        matrixLeft + columnIndex * cellSize + 0.5,
        matrixTop + rowIndex * cellSize + 0.5,
        cellSize - 1,
        cellSize - 1,
      );
    }
    context.fillStyle = labelColor;
    context.font = "10px Avenir Next, Segoe UI, sans-serif";
    tickIndices.forEach((index) => {
      const question = heatmap.questions[displayOrder[index]];
      const position = index * cellSize + cellSize / 2;
      context.textAlign = "right";
      context.textBaseline = "middle";
      context.fillText(`Q${question.id}`, matrixLeft - 8, matrixTop + position);
      context.save();
      context.translate(matrixLeft + position, matrixTop - 8);
      context.rotate(-Math.PI / 3);
      context.textAlign = "left";
      context.fillText(`Q${question.id}`, 0, 0);
      context.restore();
    });
  };

  const updateQuestionCard = (name, question) => {
    const card = document.querySelector(`[data-heatmap-question-card="${name}"]`);
    card.querySelector("span").textContent = `Q${question.id} · ${question.domain}`;
    card.querySelector("strong").textContent = question.conflict;
    card.querySelector("p").textContent = question.question;
  };

  const updateSelection = (rowIndex, columnIndex, announce = false) => {
    selectedRow = rowIndex;
    selectedColumn = columnIndex;
    firstSelect.value = String(rowIndex);
    secondSelect.value = String(columnIndex);
    document.querySelector("[data-heatmap-value]").textContent = formatNumber(heatmap.matrix[rowIndex][columnIndex]);
    updateQuestionCard("first", heatmap.questions[rowIndex]);
    updateQuestionCard("second", heatmap.questions[columnIndex]);
    if (announce) {
      document.querySelector("[data-heatmap-status]").textContent = `Q${heatmap.questions[rowIndex].id} and Q${heatmap.questions[columnIndex].id}: mean residual cosine ${formatNumber(heatmap.matrix[rowIndex][columnIndex])}`;
    }
    draw();
  };

  const inspectPointer = (event) => {
    const bounds = canvas.getBoundingClientRect();
    const x = (event.clientX - bounds.left) * (canvas.width / bounds.width);
    const y = (event.clientY - bounds.top) * (canvas.height / bounds.height);
    const displayColumnIndex = Math.floor((x - matrixLeft) / cellSize);
    const displayRowIndex = Math.floor((y - matrixTop) / cellSize);
    if (displayRowIndex < 0 || displayColumnIndex < 0 || displayRowIndex >= heatmap.questions.length || displayColumnIndex >= heatmap.questions.length) return;
    const rowIndex = displayOrder[displayRowIndex];
    const columnIndex = displayOrder[displayColumnIndex];
    if (rowIndex !== selectedRow || columnIndex !== selectedColumn) updateSelection(rowIndex, columnIndex, event.type === "pointerdown");
  };

  firstSelect.addEventListener("change", () => updateSelection(Number(firstSelect.value), selectedColumn, true));
  secondSelect.addEventListener("change", () => updateSelection(selectedRow, Number(secondSelect.value), true));
  orderButtons.forEach((button) => button.addEventListener("click", () => {
    displayOrder = button.dataset.heatmapOrder === "question-id" ? questionIdOrder : clusteredOrder;
    orderButtons.forEach((candidate) => candidate.setAttribute("aria-pressed", String(candidate === button)));
    draw();
  }));
  canvas.addEventListener("pointermove", inspectPointer);
  canvas.addEventListener("pointerdown", inspectPointer);
  canvas.addEventListener("keydown", (event) => {
    const rowPosition = displayOrder.indexOf(selectedRow);
    const columnPosition = displayOrder.indexOf(selectedColumn);
    const movement = {
      ArrowUp: [-1, 0],
      ArrowDown: [1, 0],
      ArrowLeft: [0, -1],
      ArrowRight: [0, 1],
    }[event.key];
    if (!movement) return;
    event.preventDefault();
    const nextRow = Math.max(0, Math.min(heatmap.questions.length - 1, rowPosition + movement[0]));
    const nextColumn = Math.max(0, Math.min(heatmap.questions.length - 1, columnPosition + movement[1]));
    updateSelection(displayOrder[nextRow], displayOrder[nextColumn], true);
  });
  new MutationObserver(draw).observe(root, { attributes: true, attributeFilter: ["data-theme"] });
  updateSelection(selectedRow, selectedColumn);
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
  const symbol = options.symbol || "ρ";
  const statisticLabel = options.statisticLabel || "Partial Spearman rho";
  const intervalLabel = options.intervalLabel || "node-bootstrap 95% CI";
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
    body += `<circle class="chart-point" cx="${point}" cy="${y}" r="7"><title>${escapeHtml(modelLabel(row.model))}: ${escapeHtml(symbol)} ${formatNumber(row.rho)}; ${escapeHtml(intervalLabel)} [${formatNumber(row.low)}, ${formatNumber(row.high)}]; Holm p ${formatNumber(row.holm, 4)}</title></circle>`;
    body += `<text class="chart-value" x="${width - right + 10}" y="${y + 5}">${formatNumber(row.rho)}</text>`;
  });
  const label = options.label || "Held-out model correlations with node-bootstrap intervals";
  return chartFrame({
    body, width, height, label, className: "forest-chart",
    headers: options.tableHeaders || ["Model", statisticLabel, "95% CI low", "95% CI high", "Holm p"],
    rows: options.tableRows || rows.map((row) => [modelLabel(row.model), formatNumber(row.rho), formatNumber(row.low), formatNumber(row.high), formatNumber(row.holm, 4)]),
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

export function rSquaredComparisonPlot(rows, options = {}) {
  const width = 820;
  const rowHeight = 58;
  const top = 50;
  const bottom = 44;
  const left = 174;
  const right = 92;
  const min = 0;
  const max = options.max ?? 0.8;
  const baselineLegend = options.baselineLegend || "measured question/source covariates";
  const addedLegend = options.addedLegend || "+ other-model consensus";
  const baselineTooltip = options.baselineTooltip || "controls-only";
  const fullTooltip = options.fullTooltip || "controls + consensus";
  const betaHeader = options.betaHeader || "Standardized β";
  const average = (key) => rows.reduce((total, row) => total + row[key], 0) / rows.length;
  const displayRows = [
    {
      label: "Mean",
      topicOnlyR2: average("topicOnlyR2"),
      fullR2: average("fullR2"),
      incrementalR2: average("incrementalR2"),
      highlight: true,
    },
    ...rows.map((row) => ({ ...row, label: modelLabel(row.model) })),
  ];
  const height = top + bottom + displayRows.length * rowHeight;
  let body = axisMarkup({ min, max, left, right, top, bottom, width, height, digits: 2 });
  body += `<g class="chart-legend r2-legend"><circle class="r2-topic-point" cx="${left}" cy="18" r="5"/><text x="${left + 11}" y="22">${escapeHtml(baselineLegend)}</text><circle class="r2-full-point" cx="${left + 250}" cy="18" r="6"/><text x="${left + 262}" y="22">${escapeHtml(addedLegend)}</text></g>`;
  displayRows.forEach((row, index) => {
    const y = top + rowHeight * index + rowHeight / 2;
    const topicX = scale(row.topicOnlyR2, min, max, left, width - right);
    const fullX = scale(row.fullR2, min, max, left, width - right);
    if (row.highlight) body += `<rect class="r2-mean-band" x="${left - 162}" y="${y - 24}" width="${width - right - left + 244}" height="48" rx="9"/>`;
    body += `<text class="chart-label${row.highlight ? " strong" : ""}" x="${left - 14}" y="${y + 5}" text-anchor="end">${escapeHtml(row.label)}</text>`;
    body += `<line class="r2-gain-line${row.highlight ? " highlight" : ""}" x1="${topicX}" x2="${fullX}" y1="${y}" y2="${y}" />`;
    body += `<circle class="r2-topic-point" cx="${topicX}" cy="${y}" r="6"><title>${escapeHtml(row.label)} ${escapeHtml(baselineTooltip)} R²: ${formatNumber(row.topicOnlyR2)}</title></circle>`;
    body += `<circle class="r2-full-point" cx="${fullX}" cy="${y}" r="7"><title>${escapeHtml(row.label)} ${escapeHtml(fullTooltip)} R²: ${formatNumber(row.fullR2)}; ΔR² ${formatNumber(row.incrementalR2)}${Number.isFinite(row.incrementalLow) ? `; ΔR² node-bootstrap CI [${formatNumber(row.incrementalLow)}, ${formatNumber(row.incrementalHigh)}]` : ""}</title></circle>`;
    body += `<text class="r2-end-value" x="${topicX}" y="${y - 12}" text-anchor="middle">${formatNumber(row.topicOnlyR2)}</text>`;
    body += `<text class="r2-end-value full" x="${fullX}" y="${y - 12}" text-anchor="middle">${formatNumber(row.fullR2)}</text>`;
    body += `<text class="chart-value" x="${width - right + 12}" y="${y + 5}">Δ ${formatNumber(row.incrementalR2)}</text>`;
  });
  const label = options.label || "Controls-only versus controls-plus-consensus R-squared by held-out model";
  return chartFrame({
    body, width, height, label, className: "r2-comparison-chart",
    headers: ["Model", `${baselineLegend} R²`, `${addedLegend} R²`, "Incremental R²", "ΔR² CI low", "ΔR² CI high", betaHeader, "β CI low", "β CI high", "Exploratory Holm p (β)"],
    rows: displayRows.map((row) => [
      row.label,
      formatNumber(row.topicOnlyR2),
      formatNumber(row.fullR2),
      formatNumber(row.incrementalR2),
      row.highlight ? "—" : formatNumber(row.incrementalLow),
      row.highlight ? "—" : formatNumber(row.incrementalHigh),
      row.highlight ? formatNumber(average("beta")) : formatNumber(row.beta),
      row.highlight ? "—" : formatNumber(row.betaLow),
      row.highlight ? "—" : formatNumber(row.betaHigh),
      row.highlight ? "—" : formatNumber(row.holm, 4),
    ]),
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
    const labelLines = wrapLabel(row.label, options.wrapLabelsAt);
    const firstLineY = y + 5 - ((labelLines.length - 1) * 7);
    body += `<text class="chart-label" x="${left - 14}" y="${firstLineY}" text-anchor="end">${labelLines.map((line, lineIndex) => `<tspan x="${left - 14}" dy="${lineIndex === 0 ? 0 : 14}">${escapeHtml(line)}</tspan>`).join("")}</text>`;
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
