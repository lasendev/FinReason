"use strict";

const OUTPUT_ROOT = "../workspace/outputs";
const SOURCES = {
  scorecard: `${OUTPUT_ROOT}/tables/section12_final_scorecard.csv`,
  heldout: `${OUTPUT_ROOT}/tables/section10_heldout_metrics.csv`,
  tradeoffs: `${OUTPUT_ROOT}/tables/section10_tradeoffs.csv`,
  bfsMetrics: `${OUTPUT_ROOT}/tables/section5_bfs_training_metrics.csv`,
  bfsCoverage: `${OUTPUT_ROOT}/tables/section5_bfs_component_coverage.csv`,
  hardNegatives: `${OUTPUT_ROOT}/tables/section5_bfs_hard_negatives.csv`,
  section9Metrics: `${OUTPUT_ROOT}/tables/section9_training_metrics.csv`,
  section9Report: `${OUTPUT_ROOT}/audit_trails/section9_baseline_report.json`,
  testingReport: `${OUTPUT_ROOT}/audit_trails/section11_testing_report.json`,
  finalReport: `${OUTPUT_ROOT}/audit_trails/section12_final_report.json`,
  demoCases: `${OUTPUT_ROOT}/tables/section12_demo_cases.csv`,
};

function parseCsv(source) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;

  for (let index = 0; index < source.length; index += 1) {
    const character = source[index];
    if (quoted) {
      if (character === '"' && source[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (character === '"') {
        quoted = false;
      } else {
        field += character;
      }
    } else if (character === '"') {
      quoted = true;
    } else if (character === ",") {
      row.push(field);
      field = "";
    } else if (character === "\n") {
      row.push(field.replace(/\r$/, ""));
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }

  if (field || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  if (quoted) throw new Error("CSV contains an unterminated quoted field");
  if (rows.length < 2) throw new Error("CSV contains no data rows");

  const headers = rows[0];
  return rows.slice(1).map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])),
  );
}

async function fetchSource(path, type) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path} returned HTTP ${response.status}`);
  return type === "json" ? response.json() : parseCsv(await response.text());
}

async function loadResults() {
  const entries = await Promise.all(
    Object.entries(SOURCES).map(async ([name, path]) => {
      const type = path.endsWith(".json") ? "json" : "csv";
      return [name, await fetchSource(path, type)];
    }),
  );
  return Object.fromEntries(entries);
}

function requiredRow(rows, description, predicate) {
  const row = rows.find(predicate);
  if (!row) throw new Error(`Missing ${description} in generated outputs`);
  return row;
}

function number(value, description) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) throw new Error(`Invalid ${description}: ${value}`);
  return parsed;
}

const percent = (value) => `${(number(value, "percentage") * 100).toFixed(2)}%`;
const decimal = (value) => number(value, "decimal").toFixed(2);
const integer = (value) => Math.round(number(value, "integer")).toLocaleString("en-US");
const seconds = (value) => `${number(value, "runtime").toFixed(2)} s`;
const pp = (value) => `${value < 0 ? "−" : "+"}${Math.abs(value * 100).toFixed(2)} pp`;

function setText(id, value) {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Dashboard element #${id} is missing`);
  element.textContent = value;
}

function setBar(id, value) {
  const element = document.getElementById(id);
  element.style.setProperty("--bar-value", `${Math.max(0, Math.min(100, value * 100))}%`);
}

function prettyConstant(value) {
  return String(value)
    .toLowerCase()
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function renderModelChart(rows, selectedVariant) {
  const chart = document.getElementById("model-chart");
  chart.replaceChildren();
  const ranked = [...rows].sort((left, right) => number(right.f1) - number(left.f1));
  chart.setAttribute(
    "aria-label",
    ranked.map((row) => `${row.variant_label}: ${percent(row.f1)} F1`).join("; "),
  );

  ranked.forEach((row) => {
    const wrapper = document.createElement("div");
    wrapper.className = `bar-row${row.variant === selectedVariant ? " selected" : ""}`;

    const label = document.createElement("div");
    label.className = "bar-label";
    const name = document.createElement("strong");
    name.textContent = row.variant_label;
    label.append(name);
    if (row.variant === selectedVariant) {
      const badge = document.createElement("span");
      badge.textContent = "Selected";
      label.append(badge);
    }

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.style.setProperty("--bar-value", percent(row.f1));
    track.append(fill);

    const value = document.createElement("div");
    value.className = "bar-value";
    value.textContent = percent(row.f1);
    wrapper.append(label, track, value);
    chart.append(wrapper);
  });
}

function renderModelTable(rows, selectedVariant) {
  const body = document.getElementById("model-table-body");
  body.replaceChildren();
  rows.forEach((row) => {
    const tableRow = document.createElement("tr");
    if (row.variant === selectedVariant) tableRow.className = "selected-row";

    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = row.variant_label;
    if (row.variant === selectedVariant) {
      const badge = document.createElement("span");
      badge.textContent = "Selected";
      heading.append(" ", badge);
    }
    tableRow.append(heading);

    ["precision", "recall", "f1", "false_positive_rate", "scenario_detection_rate"].forEach(
      (field) => {
        const cell = document.createElement("td");
        cell.textContent = percent(row[field]);
        tableRow.append(cell);
      },
    );
    const burden = document.createElement("td");
    burden.textContent = decimal(row.alert_burden_per_1000);
    tableRow.append(burden);
    body.append(tableRow);
  });
}

function renderStability(validation, test) {
  ["precision", "recall", "f1"].forEach((metric) => {
    const validationValue = number(validation[metric], `${metric} validation`);
    const testValue = number(test[metric], `${metric} test`);
    setText(`${metric}-validation`, percent(validationValue));
    setText(`${metric}-test`, percent(testValue));
    setText(`${metric}-delta`, pp(testValue - validationValue));
    setBar(`${metric}-validation-bar`, validationValue);
    setBar(`${metric}-test-bar`, testValue);
  });
  setText("f1-change", `${(Math.abs(number(test.f1) - number(validation.f1)) * 100).toFixed(2)} percentage points`);
}

function renderDemoCases(cases) {
  const container = document.getElementById("demo-cases");
  container.replaceChildren();
  cases.forEach((caseData) => {
    const article = document.createElement("article");
    const successful = caseData.outcome === "TRUE_POSITIVE";
    article.className = `case-card ${successful ? "case-success" : "case-warning"}`;

    const label = document.createElement("span");
    label.textContent = `${prettyConstant(caseData.outcome)} · ${prettyConstant(caseData.variant)}`;
    const title = document.createElement("h3");
    title.textContent = prettyConstant(caseData.scenario_type || caseData.case_role);
    const summary = document.createElement("p");
    summary.textContent = caseData.summary;
    const identifier = document.createElement("code");
    identifier.textContent = caseData.scenario_id || caseData.alert_or_scenario_id;
    article.append(label, title, summary, identifier);
    container.append(article);
  });
}

function renderResults(data) {
  const selectedVariant = data.finalReport.best_test_f1_variant;
  const testRows = data.scorecard.filter(
    (row) => row.source === data.finalReport.headline_source
      && row.split === data.finalReport.headline_split
      && row.label_column === data.finalReport.headline_label,
  );
  const selected = requiredRow(testRows, "selected scorecard row", (row) => row.variant === selectedVariant);
  const graphOnly = requiredRow(testRows, "graph-only scorecard row", (row) => row.variant === "GRAPH_ONLY");
  const union = requiredRow(testRows, "combined-union scorecard row", (row) => row.variant === "COMBINED_UNION");
  const validation = requiredRow(
    data.tradeoffs,
    "selected validation trade-off row",
    (row) => row.variant === selectedVariant && row.split === "validation" && row.label_column === "is_laundering",
  );
  const exactTest = requiredRow(
    data.heldout,
    "selected transaction test metrics",
    (row) => row.variant === selectedVariant && row.split === "test"
      && row.label_column === "is_laundering" && row.evaluation_unit === "transaction",
  );

  setText("hero-f1", `${percent(selected.f1)} F1 score`);
  setText("project-status", `Research prototype: ${prettyConstant(data.finalReport.status)}`);
  setText("selected-approach", selected.variant_label);
  setText("evaluation-population", `${integer(selected.population)} transactions`);
  setText("positive-labels", `${integer(selected.expected_positives)} transactions`);
  setText("decision-status", prettyConstant(data.finalReport.share_status));
  setText("headline-precision", percent(selected.precision));
  setText("headline-recall", percent(selected.recall));
  setText("headline-f1", percent(selected.f1));
  setText("headline-fpr", percent(selected.false_positive_rate));
  setText(
    "false-positive-summary",
    `${integer(exactTest.false_positives)} false positives among ${integer(number(exactTest.true_negatives) + number(exactTest.false_positives))} negative labels`,
  );

  renderModelChart(testRows, selectedVariant);
  renderModelTable(testRows, selectedVariant);
  setText("predicted-positives", integer(selected.predicted_positives));
  setText("operational-precision", percent(selected.precision));
  setText("recall-improvement", `${((number(selected.recall) - number(graphOnly.recall)) * 100).toFixed(2)} percentage points`);
  setText("scenario-detection", percent(selected.scenario_detection_rate));
  setText("alert-burden", decimal(selected.alert_burden_per_1000));
  setText("pipeline-runtime", seconds(selected.pipeline_runtime_seconds));
  setText("traceability", percent(selected.alert_traceability_rate));
  setText(
    "union-tradeoff",
    `the ${union.variant_label.toLowerCase()} found more positives, but its ${percent(union.false_positive_rate)} false-positive rate and larger review queue reduced its F1 to ${percent(union.f1)}.`,
  );
  renderStability(validation, selected);

  const bfs = requiredRow(
    data.bfsMetrics,
    "Section 5 laundering transaction metrics",
    (row) => row.evaluation_unit === "transaction" && row.label_column === "is_laundering",
  );
  setText("bfs-precision", percent(bfs.precision));
  setText("bfs-recall", percent(bfs.recall));
  setText("bfs-f1", percent(bfs.f1));
  setText("bfs-false-positives", integer(bfs.false_positives));
  ["R014", "R015"].forEach((rule) => {
    const row = requiredRow(data.bfsCoverage, `${rule} coverage`, (item) => item.rule_id === rule);
    setText(
      `${rule.toLowerCase()}-summary`,
      `${rule} component precision reached ${percent(row.precision)}, recall ${percent(row.recall)}, and F1 ${percent(row.f1)}.`,
    );
  });
  const suppressed = data.hardNegatives.filter((row) => row.variant === "contextual");
  setText(
    "hard-negative-summary",
    suppressed.map((row) => `${integer(row.scenario_count)} ${row.pattern}`).join(" and ")
      + " hard negatives were tested; contextual hits were zero.",
  );

  const rolling = data.section9Report.rolling_state;
  const windowLabel = `${integer(data.section9Report.selected_rolling_window_hours)}-hour`;
  setText("rolling-title-window", windowLabel);
  setText("rolling-window", `${integer(data.section9Report.selected_rolling_window_hours)} h`);
  setText("peak-events", integer(rolling.peak_active_events));
  setText("peak-queue", integer(rolling.peak_account_queue));
  setText("rolling-complexity", String(rolling.complexity).split(" time")[0]);
  const trainingMetric = (variant) => requiredRow(
    data.section9Metrics,
    `${variant} training metric`,
    (row) => row.variant === variant && row.evaluation_unit === "transaction" && row.label_column === "is_laundering",
  );
  setText("legacy-union-f1", percent(trainingMetric("LEGACY_UNION").f1));
  setText("enhanced-union-f1", percent(trainingMetric("ENHANCED_UNION").f1));

  const testing = data.testingReport;
  setText("test-count", `${integer(testing.tests_executed)} / ${integer(testing.tests_executed)}`);
  setText("evidence-traceability", percent(selected.alert_traceability_rate));
  setText("benchmark-runtime", seconds(testing.scalability_max_pipeline_seconds));
  setText("benchmark-rows", integer(testing.scalability_max_rows));
  setText("benchmark-memory", `${decimal(testing.scalability_peak_python_allocation_mib)} MiB`);
  setText("integration-checks", integer(testing.integration_checks_passed));
  setText("determinism-checks", integer(testing.determinism_checks_passed));
  setText("error-reconciliations", integer(testing.error_reconciliations_passed));
  setText("artifact-count", integer(data.finalReport.manifest_entry_count));
  setText(
    "demo-case-intro",
    `${integer(data.demoCases.length)} deterministic cases show the system's strengths and remaining blind spots.`,
  );
  renderDemoCases(data.demoCases);
  setText("limitation-population", integer(selected.population));
  setText(
    "supported-conclusion",
    `On the held-out synthetic test set, ${selected.variant_label} achieved ${percent(selected.precision)} precision, ${percent(selected.recall)} recall and ${percent(selected.f1)} F1, with a ${percent(selected.false_positive_rate)} false-positive rate and ${percent(selected.alert_traceability_rate)} evidence traceability.`,
  );

  const generated = new Date(testing.created_utc);
  const timestamp = Number.isNaN(generated.getTime()) ? "the latest notebook run" : generated.toLocaleString();
  const status = document.querySelector("[data-results-status]");
  status.classList.add("results-status-ready");
  status.textContent = `Live notebook outputs loaded · generated ${timestamp}`;
}

function showLoadError(error) {
  console.error(error);
  const status = document.querySelector("[data-results-status]");
  status.classList.add("results-status-error");
  status.textContent = "Evaluation results could not be loaded. Run python scripts/run_project.py, then serve this repository with python -m http.server 8000.";
}

const menuButton = document.querySelector(".menu-toggle");
const navigation = document.querySelector(".site-navigation");
const navigationLinks = [...document.querySelectorAll('.site-navigation a[href^="#"]')];

function closeMenu() {
  document.body.classList.remove("menu-open");
  menuButton?.setAttribute("aria-expanded", "false");
}

menuButton?.addEventListener("click", () => {
  const isOpen = document.body.classList.toggle("menu-open");
  menuButton.setAttribute("aria-expanded", String(isOpen));
});
navigation?.addEventListener("click", (event) => {
  if (event.target.closest("a")) closeMenu();
});
window.addEventListener("resize", () => {
  if (window.innerWidth > 800) closeMenu();
});
document.querySelector("[data-print]")?.addEventListener("click", () => window.print());

const sections = navigationLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);
if ("IntersectionObserver" in window) {
  const observer = new IntersectionObserver((entries) => {
    const visible = entries
      .filter((entry) => entry.isIntersecting)
      .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];
    if (!visible) return;
    navigationLinks.forEach((link) => {
      if (link.getAttribute("href") === `#${visible.target.id}`) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    });
  }, { rootMargin: "-20% 0px -65%", threshold: [0, 0.25, 0.6] });
  sections.forEach((section) => observer.observe(section));
}

loadResults().then(renderResults).catch(showLoadError);
