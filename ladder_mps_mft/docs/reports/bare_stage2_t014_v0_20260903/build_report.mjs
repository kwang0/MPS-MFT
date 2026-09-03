import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const reportDir = path.dirname(fileURLToPath(import.meta.url));

function coerce(value) {
  if (value === "true") return true;
  if (value === "false") return false;
  if (/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) {
    const numeric = Number(value);
    return Number.isFinite(numeric) ? numeric : value;
  }
  return value;
}

function readTsv(name) {
  const text = fs.readFileSync(path.join(reportDir, name), "utf8").trim();
  const lines = text.split(/\r?\n/);
  const columns = lines[0].split("\t");
  return lines.slice(1).map((line) => {
    const values = line.split("\t");
    return Object.fromEntries(columns.map((column, index) => [column, coerce(values[index] ?? "")]));
  });
}

function readSummary() {
  const result = {};
  for (const line of fs.readFileSync(path.join(reportDir, "analysis_summary.txt"), "utf8").trim().split(/\r?\n/)) {
    const [key, value] = line.split("=", 2);
    result[key] = Number(value);
  }
  return result;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function f(value, digits = 3) {
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return "n/a";
  return Number(value).toPrecision(digits);
}

function pct(value, digits = 1) {
  return `${(100 * Number(value)).toFixed(digits)}%`;
}

function geometryLabel(value) {
  return {
    cubic_frustrated: "Cubic frustrated",
    cubic_unfrustrated: "Cubic unfrustrated",
    square: "Square",
  }[value] ?? value;
}

function basisLabel(value) {
  return value
    .replace("spin_odd_m", "spin odd m=")
    .replace("charge_even_m", "charge even m=")
    .replace("pair_leg_s_q0", "leg-s q=0")
    .replace("pair_rung_s_q0", "rung-s q=0")
    .replace("pair_onsite_s_q0", "onsite-s q=0");
}

const summary = readSummary();
const bareFields = readTsv("bare_field_summary.tsv");
const bareProfiles = readTsv("bare_field_profiles.tsv");
const betaClasses = readTsv("beta_bond_class_summary.tsv");
const chargeModes = readTsv("bare_charge_mode_overlap.tsv");
const currentSeeds = readTsv("current_seed_summary.tsv");
const currentSeedProfiles = readTsv("current_seed_profiles.tsv");
const firstSteps = readTsv("first_step_comparison.tsv");
const scfEndpoints = readTsv("scf_endpoint_summary.tsv");
const scfEndpointPairs = readTsv("scf_endpoint_pairwise.tsv");
const eigenvalues = readTsv("eigenvalue_summary.tsv");
const compositions = readTsv("eigenmode_composition.tsv");
const modeSeedOverlaps = readTsv("dominant_mode_seed_overlap.tsv");
const leakage = readTsv("projected_leakage.tsv");
const resources = readTsv("resource_efficiency.tsv");

const squareField = bareFields.find((row) => row.geometry === "square");
const cubicField = bareFields.find((row) => row.geometry === "cubic_unfrustrated");
const currentSeedNorm = currentSeeds[0].total_norm;
const fullScaleRatio = squareField.total_norm / currentSeedNorm;
const modulationScaleRatio = squareField.modulation_norm / currentSeedNorm;
const reservationNodeHours = 18.65625;
const pairShare = summary.pair_measured_node_hours / summary.total_measured_node_hours;

const topComposition = new Map();
for (const row of compositions) {
  const key = `${row.geometry}|${row.block}|${row.rank}`;
  const current = topComposition.get(key);
  if (!current || row.weight > current.weight) topComposition.set(key, row);
}

const normalModes = eigenvalues
  .filter((row) => row.block === "normal" && row.rank <= 3)
  .map((row) => {
    const component = topComposition.get(`${row.geometry}|${row.block}|${row.rank}`);
    return {
      ...row,
      geometry_label: geometryLabel(row.geometry),
      leading_component: component?.basis_label ?? "",
      mode_label: `${geometryLabel(row.geometry)} r${row.rank} · ${basisLabel(component?.basis_label ?? "")}`,
    };
  });

const pairModes = eigenvalues
  .filter((row) => row.block === "pair" && row.rank === 1)
  .map((row) => ({
    ...row,
    geometry_label: geometryLabel(row.geometry),
    mode_label: `${geometryLabel(row.geometry)} · leading pair mode`,
  }));

const bestModeOverlaps = [];
for (const geometry of ["cubic_frustrated", "cubic_unfrustrated", "square"]) {
  const result = { geometry, geometry_label: geometryLabel(geometry) };
  for (const block of ["normal", "pair"]) {
    const candidates = modeSeedOverlaps
      .filter((row) => row.geometry === geometry && row.block === block)
      .sort((a, b) => b.absolute_cosine - a.absolute_cosine);
    result[`${block}_overlap`] = candidates[0].absolute_cosine;
    result[`${block}_seed`] = candidates[0].seed;
    result[`${block}_eigenvalue`] = candidates[0].eigenvalue_real;
  }
  bestModeOverlaps.push(result);
}

const fieldScales = bareFields.map((row) => ({
  ...row,
  geometry_label: geometryLabel(row.geometry),
  current_seed_norm: currentSeedNorm,
}));

const fieldScalesTidy = fieldScales.flatMap((row) => [
  { ...row, component: "beta", field_norm: row.beta_norm },
  { ...row, component: "Hartree", field_norm: row.mu_cdw_norm },
  { ...row, component: "nonuniform remainder", field_norm: row.modulation_norm },
  { ...row, component: "current seed total", field_norm: row.current_seed_norm },
]);

const m4Profile = new Map(
  currentSeedProfiles
    .filter((row) => row.seed === "stripe, m=4")
    .map((row) => [row.rung, row.charge_even]),
);
const m5Profile = new Map(
  currentSeedProfiles
    .filter((row) => row.seed === "stripe, m=5")
    .map((row) => [row.rung, row.charge_even]),
);
const squareProfiles = bareProfiles
  .filter((row) => row.geometry === "square")
  .map((row) => ({
    rung: row.rung,
    bare_charge_even_centered: row.charge_even_centered,
    m4_charge_seed: m4Profile.get(row.rung),
    m5_charge_seed: m5Profile.get(row.rung),
    bare_beta_nn_centered: row.beta_nn_centered,
  }));
const squareProfilesTidy = squareProfiles.flatMap((row) => [
  {
    rung: row.rung,
    series: "bare F(0), centered",
    line_style: "solid",
    hartree_field: row.bare_charge_even_centered,
    bare_beta_nn_centered: row.bare_beta_nn_centered,
  },
  {
    rung: row.rung,
    series: "current m=4 charge part",
    line_style: "dashed",
    hartree_field: row.m4_charge_seed,
    bare_beta_nn_centered: row.bare_beta_nn_centered,
  },
  {
    rung: row.rung,
    series: "current m=5 charge part",
    line_style: "dotted",
    hartree_field: row.m5_charge_seed,
    bare_beta_nn_centered: row.bare_beta_nn_centered,
  },
]);

const bestModeOverlapsTidy = bestModeOverlaps.flatMap((row) => [
  {
    ...row,
    block_label: "Normal mode",
    absolute_cosine: row.normal_overlap,
    best_seed: row.normal_seed,
    eigenvalue: row.normal_eigenvalue,
  },
  {
    ...row,
    block_label: "Pair mode",
    absolute_cosine: row.pair_overlap,
    best_seed: row.pair_seed,
    eigenvalue: row.pair_eigenvalue,
  },
]);

const seedComparison = currentSeeds.map((seed) => {
  const step = firstSteps.find((row) => row.branch === seed.branch);
  return {
    ...seed,
    ...step,
    bare_modulation_overlap: seed.bare_modulation_cosine,
    charge_profile_overlap: Number.isFinite(seed.charge_even_profile_cosine)
      ? seed.charge_even_profile_cosine
      : null,
  };
});

const endpointComponentsTidy = scfEndpoints.flatMap((row) => [
  { ...row, component: "alpha", component_norm: row.alpha_norm },
  { ...row, component: "beta", component_norm: row.beta_norm },
  { ...row, component: "Hartree", component_norm: row.mu_cdw_norm },
]);
const endpointEnergySpread =
  Math.max(...scfEndpoints.map((row) => row.corrected_energy)) -
  Math.min(...scfEndpoints.map((row) => row.corrected_energy));
const endpointNormalDistanceMax = Math.max(...scfEndpoints.map((row) => row.normal_distance_from_bare));
const endpointAlphaMin = Math.min(...scfEndpoints.map((row) => row.alpha_norm));
const endpointAlphaMax = Math.max(...scfEndpoints.map((row) => row.alpha_norm));

const topBareChargeModes = chargeModes
  .filter((row) => row.geometry === "square")
  .sort((a, b) => b.absolute_profile_cosine - a.absolute_profile_cosine)
  .slice(0, 8)
  .map((row) => ({ ...row, mode_label: `m=${row.mode_number}` }));

const fieldTable = bareFields.map((row) => ({
  ...row,
  geometry_label: geometryLabel(row.geometry),
  full_vs_current_seed: row.total_norm / currentSeedNorm,
  modulation_vs_current_seed: row.modulation_norm / currentSeedNorm,
}));

const eigenTable = eigenvalues.map((row) => ({
  ...row,
  geometry_label: geometryLabel(row.geometry),
  critical_tp_display: Number.isFinite(row.critical_tp) ? row.critical_tp : "∞",
}));

const rankOneComposition = compositions
  .filter((row) => row.rank === 1 && row.weight >= 5e-4)
  .map((row) => ({
    ...row,
    geometry_label: geometryLabel(row.geometry),
    basis_label_display: basisLabel(row.basis_label),
  }));

const topLeakage = [...leakage]
  .sort((a, b) => b.leakage_relative - a.leakage_relative)
  .slice(0, 12)
  .map((row) => ({
    ...row,
    geometry_label: geometryLabel(row.geometry),
    basis_label_display: basisLabel(row.basis_label),
  }));

const betaTable = betaClasses
  .filter((row) => row.rms > 1e-10 && row.offset <= 2)
  .map((row) => ({ ...row, geometry_label: geometryLabel(row.geometry) }));

const resourceSummary = ["normal", "pair"].map((category) => {
  const rows = resources.filter((row) => row.category === category);
  const elapsedTotal = rows.reduce((total, row) => total + row.elapsed_hours, 0);
  const cpuTotal = rows.reduce((total, row) => total + row.total_cpu_hours, 0);
  return {
    category,
    category_label: category === "normal" ? "Number-conserving" : "Parity-only pairing",
    jobs: rows.length,
    node_hours: rows.reduce((total, row) => total + row.node_hours, 0),
    median_elapsed_hours: median(rows.map((row) => row.elapsed_hours)),
    min_elapsed_hours: Math.min(...rows.map((row) => row.elapsed_hours)),
    max_elapsed_hours: Math.max(...rows.map((row) => row.elapsed_hours)),
    average_busy_cores: cpuTotal / elapsedTotal,
    step_cpu_efficiency: cpuTotal / elapsedTotal / 8,
    billed_cpu_efficiency:
      cpuTotal /
      rows.reduce((total, row) => total + row.elapsed_hours * row.allocated_logical_cpus, 0),
    max_rss_gib: Math.max(...rows.map((row) => row.max_rss_gib)),
    requested_memory_gib: Math.max(...rows.map((row) => row.requested_memory_gib)),
    max_memory_request_utilization: Math.max(...rows.map((row) => row.memory_request_utilization)),
  };
});
resourceSummary.push({
  category: "overhead",
  category_label: "Preparation + assembly",
  jobs: 2,
  node_hours: summary.overhead_measured_node_hours,
  median_elapsed_hours: null,
  min_elapsed_hours: null,
  max_elapsed_hours: null,
  average_busy_cores: null,
  step_cpu_efficiency: null,
  billed_cpu_efficiency: null,
  max_rss_gib: null,
  requested_memory_gib: 8,
  max_memory_request_utilization: null,
});

const headline = [{
  cubic_pair_gain: pairModes.find((row) => row.geometry === "cubic_frustrated").abs_eigenvalue,
  cubic_pair_critical_tp: pairModes.find((row) => row.geometry === "cubic_frustrated").critical_tp,
  square_pair_gain: pairModes.find((row) => row.geometry === "square").abs_eigenvalue,
  square_pair_critical_tp: pairModes.find((row) => row.geometry === "square").critical_tp,
  square_bare_norm: squareField.total_norm,
  full_seed_ratio: fullScaleRatio,
  square_modulation_norm: squareField.modulation_norm,
  modulation_seed_ratio: modulationScaleRatio,
  node_hours: summary.total_measured_node_hours,
  reservation_utilization: summary.total_measured_node_hours / reservationNodeHours,
  pair_share: pairShare,
}];

const generatedAt = new Date().toISOString();
const reportDataPath = "docs/reports/bare_stage2_t014_v0_20260903";
function tsvQuery(filename, description, tablesUsed = [filename]) {
  return {
    sql: `SELECT * FROM read_csv_auto('${reportDataPath}/${filename}', delim='\\t', header=true);`,
    description,
    executed_at: generatedAt,
    language: "DuckDB SQL over the reproducible Julia extractor outputs",
    tables_used: tablesUsed.map((name) => `${reportDataPath}/${name}`),
  };
}

const sources = [
  {
    id: "stage2_eigenvalues",
    label: "Stage 2 projected-response eigenspectrum",
    path: `${reportDataPath}/eigenvalue_summary.tsv`,
    query: {
      ...tsvQuery("eigenvalue_summary.tsv", "Load the eigenspectrum extracted from the accepted compact Stage 2 discovery artifact."),
      filters: ["V=0", "t0=1.4", "tp=0.1", "L=64", "density=0.9375", "r_range=4"],
      metric_definitions: [
        "Eigenvalues are those of the geometry-specific projected SCF Jacobian Kχ in the retained 12-direction basis.",
        "critical_tp = 0.1 / sqrt(abs(eigenvalue)), assuming the perturbative kernel scales as tp^2.",
      ],
    },
  },
  {
    id: "stage2_composition",
    label: "Stage 2 leading-mode composition",
    path: `${reportDataPath}/eigenmode_composition.tsv`,
    query: tsvQuery("eigenmode_composition.tsv", "Load retained-basis weights for the projected Stage 2 eigenmodes."),
  },
  {
    id: "stage2_leakage",
    label: "Stage 2 retained-basis leakage",
    path: `${reportDataPath}/projected_leakage.tsv`,
    query: tsvQuery("projected_leakage.tsv", "Load the projected and omitted response norms for each probe direction."),
  },
  {
    id: "stage2_baseline",
    label: "Bare-field image from the accepted Stage 2 zero-field reference",
    path: `${reportDataPath}/bare_field_summary.tsv`,
    query: {
      ...tsvQuery("bare_field_summary.tsv", "Load F_geometry(0) field scales and their background/nonuniform decomposition."),
      description: "Map the stored zero-field correlation state through each geometry kernel to obtain F_geometry(0), then decompose it into bond-class/uniform background and nonuniform residual.",
      filters: ["scientifically_accepted=true", "particle number N=120"],
      metric_definitions: [
        "Field norm = sqrt((sum(alpha^2)+sum(beta^2)+sum(mu_cdw^2))/(2L)).",
        "Nonuniform residual subtracts the center-of-mass mean separately from every beta spin/offset/leg class and subtracts the global scalar mean from mu_cdw.",
      ],
    },
  },
  {
    id: "seed_comparison",
    label: "Bare-image versus current square V=0 seed analysis",
    path: `${reportDataPath}/current_seed_summary.tsv`,
    query: {
      sql: `SELECT s.*, f.first_measured_norm, f.distance_from_bare, f.gain_along_seed FROM read_csv_auto('${reportDataPath}/current_seed_summary.tsv', delim='\\t', header=true) s LEFT JOIN read_csv_auto('${reportDataPath}/first_step_comparison.tsv', delim='\\t', header=true) f USING (branch);`,
      description: "Compare F_square(0) with fields/initial and the first measured SCF image in each of the six compact chi=200 square V=0 states.",
      executed_at: generatedAt,
      language: "DuckDB SQL over the reproducible Julia extractor outputs",
      tables_used: [
        `${reportDataPath}/current_seed_summary.tsv`,
        `${reportDataPath}/first_step_comparison.tsv`,
      ],
      filters: ["same V=0, t0=1.4, tp=0.1, L=64 point", "initial seed norm=1e-3"],
      metric_definitions: [
        "Profile overlap is the centered cosine of the leg-even, spin-even Hartree rung profile.",
        "First-image distance uses the same full-field norm as the Stage 2 basis.",
      ],
    },
  },
  {
    id: "mode_overlap",
    label: "Leading Stage 2 modes versus current SCF seed bank",
    path: `${reportDataPath}/dominant_mode_seed_overlap.tsv`,
    query: tsvQuery("dominant_mode_seed_overlap.tsv", "Load full-field cosines between each leading projected eigenmode and each current square seed."),
  },
  {
    id: "scf_endpoints",
    label: "Accepted square V=0 SCF endpoint fields",
    path: `${reportDataPath}/scf_endpoint_summary.tsv`,
    query: {
      sql: `SELECT e.*, p.right_seed, p.full_distance, p.normal_distance, p.alpha_distance FROM read_csv_auto('${reportDataPath}/scf_endpoint_summary.tsv', delim='\\t', header=true) e LEFT JOIN read_csv_auto('${reportDataPath}/scf_endpoint_pairwise.tsv', delim='\\t', header=true) p ON e.seed=p.left_seed;`,
      description: "Load the accepted final restart fields and pairwise field distances for all six current square SCF seed branches.",
      executed_at: generatedAt,
      language: "DuckDB SQL over the reproducible Julia extractor outputs",
      tables_used: [`${reportDataPath}/scf_endpoint_summary.tsv`, `${reportDataPath}/scf_endpoint_pairwise.tsv`],
      filters: ["accepted=true", "status=fixed_point"],
      metric_definitions: [
        "Normal distance omits alpha and compares only beta plus mu_cdw in the full-field metric.",
        "Corrected energy is the stored target-density-corrected variational energy.",
      ],
    },
  },
  {
    id: "seed_profiles",
    label: "Square bare and current seed rung profiles",
    path: `${reportDataPath}/bare_field_profiles.tsv`,
    query: {
      sql: `SELECT b.rung, b.charge_even_centered AS bare_charge_even_centered, s.seed, s.charge_even AS seed_charge_even FROM read_csv_auto('${reportDataPath}/bare_field_profiles.tsv', delim='\\t', header=true) b LEFT JOIN read_csv_auto('${reportDataPath}/current_seed_profiles.tsv', delim='\\t', header=true) s USING (rung) WHERE b.geometry='square';`,
      description: "Join the centered square bare Hartree profile to the m=4 and m=5 charge components of the current SCF seeds.",
      executed_at: generatedAt,
      language: "DuckDB SQL over the reproducible Julia extractor outputs",
      tables_used: [`${reportDataPath}/bare_field_profiles.tsv`, `${reportDataPath}/current_seed_profiles.tsv`],
    },
  },
  {
    id: "resource_accounting",
    label: "Perlmutter Stage 2 Slurm accounting",
    path: `${reportDataPath}/resource_efficiency.tsv`,
    query: {
      ...tsvQuery("resource_efficiency.tsv", "Load job-level elapsed time, CPU use, memory use, and measured allocation charge."),
      description: "Aggregate the synced sacct allocation and step records into measured node-hours, active-core estimates, and memory utilization.",
      tables_used: [
        `${reportDataPath}/resource_efficiency.tsv`,
      ],
      metric_definitions: [
        "node-hours = ElapsedRaw hours × AllocCPUS / 256 logical CPUs per CPU node.",
        "average busy cores = TotalCPU / elapsed wall time.",
        "memory utilization = MaxRSS / requested memory.",
      ],
    },
  },
  {
    id: "stage2_integrity",
    label: "Stage 2 compact-artifact manifest",
    path: "output/bare_stage2/20260902_bare_t014_v0_stage2/stateless_results/stateless_manifest.tsv",
    query: {
      sql: "SELECT * FROM read_csv_auto('output/bare_stage2/20260902_bare_t014_v0_stage2/stateless_results/stateless_manifest.tsv', delim='\\t', header=true);",
      description: "Local read-only verification of all 22 compact artifact sizes, SHA-256 values, stateless markers, and absence of MPS tensors.",
      executed_at: generatedAt,
      language: "DuckDB SQL; hashes and HDF5 statelessness independently checked by scripts/verify_stateless_results.jl",
      tables_used: ["stateless_manifest.tsv"],
      filters: ["compact mirror only; full scratch artifacts were not locally mounted"],
    },
  },
];

const title = "Bare-ladder Stage 2 response and SCF seed analysis";

const artifact = {
  surface: "report",
  manifest: {
    version: 1,
    surface: "report",
    title,
    description: "Projected linear-response, bare-field seed, and CPU-efficiency analysis at V=0, t0=1.4.",
    generatedAt,
    sources,
    cards: [
      {
        id: "card-cubic-pair",
        dataset: "headline",
        sourceId: "stage2_eigenvalues",
        description: "Largest projected pairing eigenvalue for either cubic geometry at tp=0.1.",
        metrics: [
          { label: "Cubic |λpair|", field: "cubic_pair_gain", format: "number" },
          { label: "critical tp", field: "cubic_pair_critical_tp", format: "number" },
        ],
      },
      {
        id: "card-square-pair",
        dataset: "headline",
        sourceId: "stage2_eigenvalues",
        description: "Largest projected square-geometry pairing eigenvalue at tp=0.1.",
        metrics: [
          { label: "Square |λpair|", field: "square_pair_gain", format: "number" },
          { label: "critical tp", field: "square_pair_critical_tp", format: "number" },
        ],
      },
      {
        id: "card-bare-scale",
        dataset: "headline",
        sourceId: "stage2_baseline",
        description: "Full square bare-image norm; the comparator is the exact 1e-3 norm of every current matched seed.",
        metrics: [
          { label: "Square ||F(0)||", field: "square_bare_norm", format: "number" },
          { label: "× current seed", field: "full_seed_ratio", format: "number" },
        ],
      },
      {
        id: "card-modulation-scale",
        dataset: "headline",
        sourceId: "stage2_baseline",
        description: "Square nonuniform remainder after removing bond-class beta means and the uniform Hartree offset.",
        metrics: [
          { label: "Nonuniform ||δF(0)||", field: "square_modulation_norm", format: "number" },
          { label: "× current seed", field: "modulation_seed_ratio", format: "number" },
        ],
      },
      {
        id: "card-node-hours",
        dataset: "headline",
        sourceId: "resource_accounting",
        description: "Measured Stage 2 discovery charge, including both references, 12 probes, preparation, and assembly.",
        metrics: [
          { label: "Measured node-hours", field: "node_hours", format: "number" },
          { label: "of 18.656 reserved", field: "reservation_utilization", format: "percent" },
          { label: "pair-sector share", field: "pair_share", format: "percent" },
        ],
      },
    ],
    charts: [
      {
        id: "chart-normal-eigenvalues",
        title: "Projected normal-sector eigenvalues",
        subtitle: "Top three modes per geometry at tp=0.1; reference lines mark |λ|=1.",
        type: "horizontalBar",
        dataset: "normal_modes",
        sourceId: "stage2_eigenvalues",
        intent: "comparison",
        question: "Which normal modes are unstable and is their raw-map feedback monotone or oscillatory?",
        rationale: "Signed horizontal bars preserve the instability threshold and the negative frustrated-geometry feedback.",
        encodings: {
          x: { field: "mode_label", type: "nominal", label: "Mode" },
          y: { field: "eigenvalue_real", type: "quantitative", label: "Projected eigenvalue λ" },
          tooltip: [
            { field: "abs_eigenvalue", type: "quantitative", label: "|λ|" },
            { field: "critical_tp", type: "quantitative", label: "critical tp" },
            { field: "leading_component", type: "text", label: "largest basis component" },
          ],
        },
        palette: { kind: "diverging", midpoint: 0 },
        referenceLines: [
          { axis: "y", value: -1, label: "−1", color: "neutral", lineStyle: "dashed" },
          { axis: "y", value: 1, label: "+1", color: "neutral", lineStyle: "dashed" },
        ],
        labels: { values: "auto" },
        settings: { sort: "none", showValues: true },
        layout: "full",
      },
      {
        id: "chart-pair-eigenvalues",
        title: "Leading projected pairing eigenvalue",
        subtitle: "One leading mode per geometry at tp=0.1; |λ|>1 indicates an unstable zero-pairing map in the retained subspace.",
        type: "horizontalBar",
        dataset: "pair_modes",
        sourceId: "stage2_eigenvalues",
        intent: "comparison",
        encodings: {
          x: { field: "geometry_label", type: "nominal", label: "Geometry" },
          y: { field: "abs_eigenvalue", type: "quantitative", label: "|λpair|" },
          tooltip: [
            { field: "critical_tp", type: "quantitative", label: "critical tp" },
            { field: "residual_norm", type: "quantitative", label: "eigensystem residual" },
          ],
        },
        palette: { kind: "sequential" },
        referenceLines: [
          { axis: "y", value: 1, label: "instability threshold", color: "neutral", lineStyle: "dashed" },
        ],
        labels: { values: "all" },
        settings: { sort: "descending", showValues: true },
        layout: "full",
      },
      {
        id: "chart-mode-seed-overlap",
        title: "Best current-seed overlap with the leading response modes",
        subtitle: "Absolute full-field cosine; seed labels are available in the chart details.",
        type: "bar",
        dataset: "best_mode_overlaps_tidy",
        sourceId: "mode_overlap",
        intent: "comparison",
        question: "Does the existing targeted bank already cover the leading Stage 2 directions?",
        encodings: {
          x: { field: "geometry_label", type: "nominal", label: "Geometry" },
          y: { field: "absolute_cosine", type: "quantitative", label: "Absolute cosine" },
          color: { field: "block_label", type: "nominal", label: "Response block" },
          tooltip: [
            { field: "best_seed", type: "text", label: "best current seed" },
            { field: "eigenvalue", type: "quantitative", label: "eigenvalue" },
          ],
        },
        palette: { kind: "categorical" },
        referenceLines: [
          { axis: "y", value: 1, label: "exact alignment", color: "neutral", lineStyle: "dashed" },
        ],
        labels: { values: "all" },
        legend: { position: "bottom", sort: "spec" },
        settings: { groupMode: "grouped", showValues: true },
        layout: "full",
      },
      {
        id: "chart-bare-field-scales",
        title: "Bare-image field scales by geometry",
        subtitle: "Full-field metric per physical site; nonuniform residual is not additive with beta and Hartree norms.",
        type: "bar",
        dataset: "field_scales_tidy",
        sourceId: "stage2_baseline",
        intent: "comparison",
        encodings: {
          x: { field: "geometry_label", type: "nominal", label: "Geometry" },
          y: { field: "field_norm", type: "quantitative", label: "Field norm" },
          color: { field: "component", type: "nominal", label: "Component" },
          tooltip: [
            { field: "total_norm", type: "quantitative", label: "full bare norm" },
            { field: "modulation_fraction", type: "quantitative", label: "nonuniform fraction" },
          ],
        },
        palette: { kind: "categorical" },
        labels: { values: "auto" },
        legend: { position: "bottom", sort: "spec" },
        settings: { groupMode: "grouped", showValues: true },
        layout: "full",
      },
      {
        id: "chart-square-charge-profile",
        title: "Square bare Hartree modulation and current stripe charge seeds",
        subtitle: "Leg-even, spin-even rung fields; the bare curve has its uniform −0.004266 offset removed, while seed curves retain their exact amplitudes.",
        type: "line",
        dataset: "square_profiles_tidy",
        sourceId: "seed_profiles",
        intent: "comparison",
        encodings: {
          x: { field: "rung", type: "quantitative", label: "Rung" },
          y: { field: "hartree_field", type: "quantitative", label: "Hartree field" },
          color: { field: "series", type: "nominal", label: "Profile" },
          lineStyle: { field: "line_style", type: "nominal", label: "Line style" },
          tooltip: [
            { field: "bare_beta_nn_centered", type: "quantitative", label: "bare centered NN beta" },
          ],
        },
        palette: { kind: "categorical" },
        labels: { values: "none" },
        legend: { position: "bottom", sort: "spec" },
        settings: { showPoints: "never" },
        layout: "full",
      },
      {
        id: "chart-endpoint-components",
        title: "Accepted SCF endpoint field scales",
        subtitle: "All six current seeds acquire nearly the same beta, Hartree, and pairing norms; signs and spatial textures are summarized in the table.",
        type: "bar",
        dataset: "endpoint_components_tidy",
        sourceId: "scf_endpoints",
        intent: "comparison",
        question: "Do current seed families land at parametrically different field scales?",
        encodings: {
          x: { field: "seed", type: "nominal", label: "Initial seed" },
          y: { field: "component_norm", type: "quantitative", label: "Endpoint component norm" },
          color: { field: "component", type: "nominal", label: "Field component" },
          tooltip: [
            { field: "normal_distance_from_bare", type: "quantitative", label: "normal distance from F(0)" },
            { field: "spin_odd_rms", type: "quantitative", label: "spin-odd RMS" },
            { field: "corrected_energy", type: "quantitative", label: "corrected energy" },
          ],
        },
        palette: { kind: "categorical" },
        labels: { values: "auto" },
        legend: { position: "bottom", sort: "spec" },
        settings: { groupMode: "grouped", showValues: false },
        layout: "full",
      },
      {
        id: "chart-resource-charge",
        title: "Measured Stage 2 node-hour charge",
        subtitle: "Elapsed allocation charge on a 256-logical-CPU node; total 3.874 node-hours.",
        type: "horizontalBar",
        dataset: "resource_summary",
        sourceId: "resource_accounting",
        intent: "composition",
        encodings: {
          x: { field: "category_label", type: "nominal", label: "Workload" },
          y: { field: "node_hours", type: "quantitative", label: "Node-hours" },
          tooltip: [
            { field: "jobs", type: "quantitative", label: "jobs" },
            { field: "median_elapsed_hours", type: "quantitative", label: "median elapsed hours" },
            { field: "max_rss_gib", type: "quantitative", label: "maximum RSS GiB" },
          ],
        },
        palette: { kind: "sequential" },
        labels: { values: "all" },
        settings: { sort: "descending", showValues: true },
        layout: "full",
      },
    ],
    tables: [
      {
        id: "table-field-summary",
        title: "Bare-image field values",
        subtitle: "Exact summary from the accepted zero-field correlation state; field norms use ||x||/sqrt(2L).",
        dataset: "field_table",
        sourceId: "stage2_baseline",
        defaultSort: { field: "total_norm", direction: "desc" },
        density: "dense",
        layout: "full",
        columns: [
          { field: "geometry_label", label: "Geometry", type: "text" },
          { field: "total_norm", label: "Full norm", format: "number" },
          { field: "beta_norm", label: "beta norm", format: "number" },
          { field: "mu_cdw_norm", label: "Hartree norm", format: "number" },
          { field: "modulation_norm", label: "Nonuniform norm", format: "number" },
          { field: "mu_uniform_value", label: "Uniform mu_cdw/site", format: "number" },
          { field: "charge_even_modulation_rms", label: "Charge-even RMS", format: "number" },
          { field: "beta_nn_mean", label: "NN beta mean", format: "number" },
          { field: "full_vs_current_seed", label: "Full / 1e-3 seed", format: "number" },
          { field: "modulation_vs_current_seed", label: "Nonuniform / seed", format: "number" },
        ],
      },
      {
        id: "table-seed-comparison",
        title: "Current square seed bank versus the bare image",
        subtitle: "All current seeds have total norm 1e-3 and beta=0; first-image results come from the synced chi=200 six-seed campaign.",
        dataset: "seed_comparison",
        sourceId: "seed_comparison",
        defaultSort: { field: "seed", direction: "asc" },
        density: "dense",
        layout: "full",
        columns: [
          { field: "seed", label: "Current seed", type: "text" },
          { field: "alpha_norm", label: "alpha norm", format: "number" },
          { field: "mu_cdw_norm", label: "Hartree norm", format: "number" },
          { field: "bare_modulation_overlap", label: "cos(bare residual, seed)", format: "number" },
          { field: "charge_profile_overlap", label: "charge-profile cosine", format: "number" },
          { field: "first_measured_norm", label: "first-image norm", format: "number" },
          { field: "distance_from_stage2_bare", label: "first-image distance", format: "number" },
          { field: "response_gain_along_seed", label: "first gain along seed", format: "number" },
          { field: "first_mu_evaluations", label: "first mu evals", format: "number" },
          { field: "first_iteration_wall_hours", label: "first iter hours", format: "number" },
        ],
      },
      {
        id: "table-scf-endpoints",
        title: "Accepted SCF endpoints reached from the current seed bank",
        subtitle: "Final restart fields from six chi=200 square branches; all are accepted period-one fixed points.",
        dataset: "scf_endpoints",
        sourceId: "scf_endpoints",
        defaultSort: { field: "seed", direction: "asc" },
        density: "dense",
        layout: "full",
        columns: [
          { field: "seed", label: "Initial seed", type: "text" },
          { field: "iterations", label: "Iterations", format: "number" },
          { field: "total_norm", label: "Total norm", format: "number" },
          { field: "alpha_norm", label: "alpha norm", format: "number" },
          { field: "beta_norm", label: "beta norm", format: "number" },
          { field: "mu_cdw_norm", label: "Hartree norm", format: "number" },
          { field: "normal_distance_from_bare", label: "Normal distance from F(0)", format: "number" },
          { field: "spin_odd_rms", label: "Spin-odd RMS", format: "number" },
          { field: "cosine_with_bare", label: "cos(endpoint, F(0))", format: "number" },
          { field: "corrected_energy", label: "Corrected energy", format: "number" },
          { field: "fixed_point_rel_residual", label: "FP relative residual", format: "number" },
        ],
      },
      {
        id: "table-eigenvalues",
        title: "Complete retained-basis eigenspectrum",
        subtitle: "All projected normal and pair eigenvalues; this is discovery-scale evidence pending h/2 validation.",
        dataset: "eigen_table",
        sourceId: "stage2_eigenvalues",
        defaultSort: { field: "abs_eigenvalue", direction: "desc" },
        density: "dense",
        layout: "full",
        columns: [
          { field: "geometry_label", label: "Geometry", type: "text" },
          { field: "block", label: "Block", type: "text" },
          { field: "rank", label: "Rank", format: "number" },
          { field: "eigenvalue_real", label: "Re λ", format: "number" },
          { field: "abs_eigenvalue", label: "|λ|", format: "number" },
          { field: "critical_tp_display", label: "critical tp", type: "text" },
          { field: "recurrence_character", label: "Raw-map character", type: "text" },
          { field: "residual_norm", label: "Eigen residual", format: "number" },
        ],
      },
      {
        id: "table-mode-composition",
        title: "Leading-mode compositions",
        subtitle: "Components carrying at least 0.05% weight in each rank-one right eigenvector.",
        dataset: "rank_one_composition",
        sourceId: "stage2_composition",
        defaultSort: { field: "weight", direction: "desc" },
        density: "dense",
        layout: "full",
        columns: [
          { field: "geometry_label", label: "Geometry", type: "text" },
          { field: "block", label: "Block", type: "text" },
          { field: "basis_label_display", label: "Basis component", type: "text" },
          { field: "coefficient_real", label: "Coefficient", format: "number" },
          { field: "weight", label: "Weight", format: "percent" },
        ],
      },
      {
        id: "table-resource-summary",
        title: "CPU and memory efficiency",
        subtitle: "TotalCPU and MaxRSS are from the Julia srun step; billed efficiency uses the full Slurm AllocCPUS value.",
        dataset: "resource_summary",
        sourceId: "resource_accounting",
        defaultSort: { field: "node_hours", direction: "desc" },
        density: "dense",
        layout: "full",
        columns: [
          { field: "category_label", label: "Workload", type: "text" },
          { field: "jobs", label: "Jobs", format: "number" },
          { field: "node_hours", label: "Node-hours", format: "number" },
          { field: "median_elapsed_hours", label: "Median wall h", format: "number" },
          { field: "average_busy_cores", label: "Avg busy cores", format: "number" },
          { field: "step_cpu_efficiency", label: "Of 8 step CPUs", format: "percent" },
          { field: "billed_cpu_efficiency", label: "Of billed CPUs", format: "percent" },
          { field: "max_rss_gib", label: "Max RSS GiB", format: "number" },
          { field: "requested_memory_gib", label: "Requested GiB", format: "number" },
          { field: "max_memory_request_utilization", label: "Memory utilization", format: "percent" },
        ],
      },
      {
        id: "table-leakage",
        title: "Largest retained-basis leakage",
        subtitle: "Columns with the largest response outside the 12-direction discovery basis.",
        dataset: "top_leakage",
        sourceId: "stage2_leakage",
        defaultSort: { field: "leakage_relative", direction: "desc" },
        density: "dense",
        layout: "full",
        columns: [
          { field: "geometry_label", label: "Geometry", type: "text" },
          { field: "basis_label_display", label: "Input direction", type: "text" },
          { field: "leakage_relative", label: "Leakage", format: "percent" },
          { field: "beta_fraction", label: "beta fraction", format: "percent" },
        ],
      },
    ],
    blocks: [
      { id: "title", type: "markdown", body: `# ${title}`, layout: "full" },
      {
        id: "technical-summary",
        type: "markdown",
        layout: "full",
        body: `## Technical Summary\n\nThe Stage 2 discovery run is internally consistent and physically informative, but its eigenvalues remain **pilot estimates**, not production stability boundaries. All 12 probes and both zero-field references passed their configured DMRG and density gates; the raw susceptibility has 0.121% reciprocity error and only 0.332% normal–pair cross-block norm. The retained map nevertheless leaks as much as 83.3% of a response—mostly into beta fields absent from the input basis—and the planned h/2 linearity checks have not yet run.\n\nAt tp=0.1, the dominant projected pairing eigenvalue is 24.47 for both cubic kernels and 4.669 for square. The leading cubic pair vector is the expected d-wave-like mixture, while square is 99.87% leg pairing. The strongest normal mode is the spin-odd m=59 direction used by the existing m=4 stripe seed: its projected eigenvalue is −1.188 for cubic frustrated, +3.564 for cubic unfrustrated, and +1.188 for square. Thus the current targeted bank already covers the leading directions well; a broad randomized search is not the next efficient step.\n\nThe square bare image F(0) has zero alpha and norm ${f(squareField.total_norm, 4)}, versus 1.0e−3 for every current seed. Most of that is a symmetry-preserving beta background. After subtracting bond-class beta means and the uniform Hartree offset, the nonuniform remainder is ${f(squareField.modulation_norm, 4)}. Its charge-even profile overlaps the current m=4 charge harmonic by 0.725, but it contains no resolved spin or leg-odd source. It can reinforce the boundary-pinned charge/bond texture; it does **not** by itself identify a different SDW basin.\n\nThis is also borne out nonlinearly: all six current chi=200 starts are accepted fixed points with total field norms 0.03012–0.03016, beta norms 0.02876–0.02879, and alpha norms ${f(endpointAlphaMin, 4)}–${f(endpointAlphaMax, 4)}. Their normal-sector distances from F(0) are at most ${f(endpointNormalDistanceMax, 4)}. Residual spin amplitudes differ, so the endpoints are not numerically identical, but the bare beta/mu background is common to them rather than a new basin label.\n\nRecommendation: test **bare background + controlled symmetry-breaking increment**, not the bare image alone. Retain the current m=4 stripe direction, retain d-wave for cubic, add a pure leg-pairing control for square, and keep the original zero-background starts as basin controls.`,
      },
      { id: "headline-metrics", type: "metric-strip", cardIds: ["card-cubic-pair", "card-square-pair", "card-bare-scale", "card-modulation-scale", "card-node-hours"], layout: "full" },
      { id: "findings-heading", type: "markdown", body: "## Key Findings and Visual Evidence", layout: "full" },
      {
        id: "response-finding",
        type: "markdown",
        sourceId: "stage2_eigenvalues",
        layout: "full",
        body: `### The dominant directions are already represented in the current bank\n\nThe rank-one normal vector is 97.9% spin-odd m=59 with a 2.08% m=63 admixture. The pure m=4 stripe seed has absolute cosine 0.967 with that vector in square and cubic unfrustrated geometry (0.966 in cubic frustrated). The cubic pair vector is 66.6% leg and 33.4% rung with opposite signs, giving 0.9996 overlap with the current d-wave seed. Square instead has 99.87% leg weight, so the current d-wave seed reaches it with cosine 0.814; a pure leg-s seed is the clean missing control.`,
      },
      { id: "normal-chart", type: "chart", chartId: "chart-normal-eigenvalues", layout: "full" },
      { id: "pair-chart", type: "chart", chartId: "chart-pair-eigenvalues", layout: "full" },
      { id: "overlap-chart", type: "chart", chartId: "chart-mode-seed-overlap", layout: "full" },
      { id: "composition-table", type: "table", tableId: "table-mode-composition", layout: "full" },
      {
        id: "bare-seed-finding",
        type: "markdown",
        sourceId: "seed_comparison",
        layout: "full",
        body: `### The bare beta/mu fields are a shared normal background, with a real m=4 charge bias\n\nFor square, beta has norm ${f(squareField.beta_norm, 4)} and nearest-neighbor mean ${f(squareField.beta_nn_mean, 4)}; mu_cdw has a uniform per-entry value ${f(squareField.mu_uniform_value, 4)} plus a charge-even modulation with RMS ${f(squareField.charge_even_modulation_rms, 4)} and maximum deviation ${f(squareField.charge_even_modulation_max, 4)}. Spin-even, spin-odd, and charge-odd components are at numerical-noise level. The centered charge profile is broad because of open boundaries, but m=8 (q/pi=0.1270) is its strongest current-template component: cosine 0.725 versus 0.047 for the m=10 charge harmonic in the m=5 stripe seed.\n\nIn the *full* field metric, the bare residual projects onto only 0.329 copies of the total m=4 stripe seed because that seed is deliberately spin dominated and the bare reference has no spin field. This is why the beta/mu image should be treated as a background rather than renamed as a stripe seed.`,
      },
      { id: "field-chart", type: "chart", chartId: "chart-bare-field-scales", layout: "full" },
      { id: "field-table", type: "table", tableId: "table-field-summary", layout: "full" },
      { id: "profile-chart", type: "chart", chartId: "chart-square-charge-profile", layout: "full" },
      { id: "seed-table", type: "table", tableId: "table-seed-comparison", layout: "full" },
      {
        id: "first-map-finding",
        type: "markdown",
        sourceId: "seed_comparison",
        layout: "full",
        body: `### Starting from F(0) mostly skips a step; it does not explore a new direction\n\nThe current driver begins with an unmixed raw-map probe. Across all six square branches, the first measured field already has norm 0.03083–0.03144, very close to the Stage 2 bare-image norm 0.03067. Its distance from F(0) is 0.00396–0.00496 and is dominated by the induced alpha component in every branch. The first-step gain along the current d-wave seed is 3.74 and along the m=4 stripe seed is 1.09, consistent in scale with the Stage 2 square eigenvalues 4.67 and 1.19 despite the chi=200 versus chi=1200 difference.\n\nA bare-only seed has alpha=0 and no spin source. In an exactly number-conserving solve it cannot leave that subspace, and even in the unrestricted solver it relies on numerical symmetry leakage rather than a controlled probe. Adding F(0) to the existing perturbations is therefore the scientifically useful A/B test.`,
      },
      {
        id: "endpoint-finding",
        type: "markdown",
        sourceId: "scf_endpoints",
        layout: "full",
        body: `### The existing nonlinear endpoints confirm that beta/mu are a shared background, not a separate discovered basin\n\nAll six current branches are accepted period-one fixed points. Their endpoint component scales are remarkably narrow: alpha ${f(endpointAlphaMin, 5)}–${f(endpointAlphaMax, 5)}, beta 0.028765–0.028792, Hartree 0.006335–0.006481, and total norm 0.030125–0.030156. Every endpoint has cosine 0.9749–0.9761 with the bare F(0) image, while the beta+mu distance from F(0) is only 0.00213–0.00262. The target-density-corrected energy spread is ${f(endpointEnergySpread, 4)} in total (${f(endpointEnergySpread / 128, 3)} per physical site), too small to interpret as robust basin ordering at chi=200.\n\nSome pairwise full-field distances are about 0.01264 because the real pairing field has converged with the opposite global sign; that is the same superconducting state up to its U(1) phase, not a distinct basin. There is, however, a meaningful remaining normal-sector distinction: spin-odd RMS ranges from about 3e−7 in the pure pairing controls to 1.07e−3 in the m=4 stripe control. F(0) alone remains symmetry preserving; with an explicit pairing increment it is closest to the pair-only lineage, while **F(0)+epsilon v** lets us test the same stripe or pair directions cleanly. It does not reveal an additional beta/mu basin that the present raw-map startup misses.`,
      },
      { id: "endpoint-chart", type: "chart", chartId: "chart-endpoint-components", layout: "full" },
      { id: "endpoint-table", type: "table", tableId: "table-scf-endpoints", layout: "full" },
      {
        id: "efficiency-finding",
        type: "markdown",
        sourceId: "resource_accounting",
        layout: "full",
        body: `### The run cost ${f(summary.total_measured_node_hours, 4)} node-hours; memory requests, not Julia core allocation, are the clearest efficiency lever\n\nThe number-conserving reference plus nine probes used ${f(summary.normal_measured_node_hours, 4)} node-hours. The parity-only reference plus three probes used ${f(summary.pair_measured_node_hours, 4)} node-hours (${pct(pairShare)} of total). Normal jobs took 1.01–1.99 hours and used at most 5.25 GiB; pairing jobs took 2.87–5.53 hours and used at most 9.83 GiB. Average active CPU was about 2.45 cores for normal work and 2.11 for pairing work out of eight logical CPUs assigned to each Julia step.\n\nThe 48/64 GiB requests used at most 10.9%/15.4% of their reservations and inflated billed AllocCPUS to 26/36 logical CPUs. Before a larger response campaign, benchmark one normal solve at 16 GiB and one parity-only solve at 24 GiB with the same four Julia threads. Those settings retain roughly 3.0× and 2.4× the observed peak RSS; they should be treated as calibration candidates, not silently adopted.`,
      },
      { id: "resource-chart", type: "chart", chartId: "chart-resource-charge", layout: "full" },
      { id: "resource-table", type: "table", tableId: "table-resource-summary", layout: "full" },
      {
        id: "scope-data-metrics",
        type: "markdown",
        layout: "full",
        body: `## Scope, Data, and Metric Definitions\n\n**Point and basis.** L=64, U=8, V=0, t0=1.4, tp=0.1, density 0.9375, range-four mean fields. The retained discovery basis contains nine normal directions and three independent q=0 pair directions. Geometry kernels are applied after the ladder response, using the backbone hole-pair binding −0.1465102212.\n\n**Bare image.** “Bare seed” means F_geometry(0): the geometry-specific mean-field map evaluated on the accepted number-conserving zero-field ladder correlations. It is not an SCF fixed point. Its alpha array is exactly zero by number conservation.\n\n**Field scale.** Norms are sqrt((sum alpha² + sum beta² + sum mu_cdw²)/(2L)); symmetric bond entries therefore appear exactly as they do in the SCF state. The nonuniform residual removes the center-of-mass mean separately for each beta spin/offset/leg class and removes the scalar mean of mu_cdw. This conservative subtraction distinguishes a physical normal self-energy background from finite-open-ladder texture.\n\n**Stability metric.** The displayed eigenvalues are eigenvalues of the projected map derivative Kχ at zero field. |λ|>1 means raw Picard instability inside the retained subspace; λ<−1 predicts an alternating/period-two tendency, while λ>1 predicts monotone growth. It is not a Hessian eigenvalue or an ordered-state energy comparison.`,
      },
      {
        id: "methodology",
        type: "markdown",
        layout: "full",
        body: `## Methodology\n\n1. Verified all 22 compact Stage 2 artifacts locally against the synced stateless manifest; no MPS tensors are present in the compact mirror.\n2. Reconstructed F_geometry(0) from the stored normal-reference pair, exchange, and density correlations with the exact project mean-field kernel.\n3. Read the projected Jacobians and right eigensystems directly from the immutable discovery artifact and decomposed the leading eigenvectors in the retained orthonormal basis.\n4. Read fields/initial, the first measured image, and the accepted final restart field from each synced square V=0 chi=200 compact state, then compared them in the same field metric.\n5. Recomputed elapsed allocation charge from sacct as elapsed hours × AllocCPUS/256 and obtained active-core and memory estimates from TotalCPU and MaxRSS.\n\nThe reproducible extraction is in analyze_stage2.jl; every table shown here is generated from its TSV outputs.`,
      },
      {
        id: "limitations",
        type: "markdown",
        layout: "full",
        body: `## Limitations, Uncertainty, and Robustness\n\n- **Linearity is not yet measured.** Discovery used h=1e−4 only. The generated three-mode h/2 validation plan has not been executed, so the especially large cubic pairing gain could contain finite-field curvature.\n- **The basis is not closed.** Maximum projected leakage is 83.3%; charge probes produce beta-dominated output that the present input basis cannot feed back. The retained eigenvalues are therefore subspace eigenvalues, not converged dominant eigenvalues of the full SCF map.\n- **Bond dimension is active.** Every DMRG solve reached chi=1200 with maximum discarded weight about 2.5e−7. Sweep-energy stability and reciprocity are good, but neither replaces a chi check for susceptibility values.\n- **Open boundaries seed charge texture.** The m=8 match in F(0) is a finite-L, boundary-pinned one-point field signature. It is not a connected structure factor, spontaneous order parameter, or proof of a distinct thermodynamic basin.\n- **Reference symmetry differs from unrestricted SCF.** The normal reference fixes N=120, while the comparison SCF campaign is unrestricted and only chi=200. First-image agreement is a useful cross-check, not a controlled bond-dimension extrapolation.\n- **Full scratch artifacts were not reverified locally.** The compact mirror passed all 22 size/hash/stateless checks; the recorded 6.739 GB full artifacts remain Perlmutter provenance only.`,
      },
      { id: "leakage-table", type: "table", tableId: "table-leakage", layout: "full" },
      { id: "eigen-table", type: "table", tableId: "table-eigenvalues", layout: "full" },
      {
        id: "recommendations",
        type: "markdown",
        layout: "full",
        body: `## Recommended Next Steps\n\n1. **Run the existing three-direction h and h/2 validation** before treating any quoted eigenvalue as quantitative: cubic-unfrustrated normal rank 1, cubic-frustrated pair rank 1, and square pair rank 1.\n2. **Pilot paired SCF starts at this point.** For each selected direction compare the current 1e−3 seed with F_geometry(0) plus the same 1e−3 symmetry-breaking increment. Do not normalize the combined field back to 1e−3; that would erase the physical background. Keep the present starts as controls.\n3. **Use a point-specific scalar-mu predictor.** The current first solve spends 11–16 mu evaluations moving from 0.55 to 1.672–1.677; the next field image settles near 1.656. For a square F(0)-background pilot, initialize the scalar chemical potential near 1.6563 and let the density solver verify it. If the uniform mu_cdw offset is gauge-removed, compensate it in the scalar chemical potential consistently.\n4. **Keep m=4 and d-wave; add pure square leg pairing.** The current m=4 stripe already covers the leading normal mode. Cubic d-wave is essentially exact for the leading pair mode. A square leg-s start removes the 18.6% angular mismatch of the current d-wave template.\n5. **Expand the normal basis with beta directions after validation.** The first targeted complement should be built from the measured beta leakage of the charge probes; a broad random search is premature while an identified 83% residual subspace remains untreated.\n6. **Right-size resources through two calibration jobs.** Test 16 GiB normal and 24 GiB pairing requests with the current four Julia threads, preserving the present science settings and comparing elapsed time, MaxRSS, and response coordinates before changing defaults.`,
      },
      {
        id: "further-questions",
        type: "markdown",
        layout: "full",
        body: `## Further Questions\n\n- Does F(0)+epsilon v reduce the first two density-search/SCF transients without changing the accepted endpoint reached from epsilon v alone?\n- Does the bare m=8 charge texture strengthen or weaken after subtracting a bulk-window beta background instead of the full bond-class mean?\n- Do h/2 and h/4 preserve the cubic pair gain and the sign of the frustrated normal mode?\n- Once beta leakage directions are admitted, does the leading normal spectrum remain spin-odd m=59, or does a bond/charge hybrid overtake it?\n- Are the square rank-one pair mode and the common paired SCF endpoint stable with chi and L?`,
      },
    ],
  },
  snapshot: {
    version: 1,
    generatedAt,
    status: "ready",
    datasets: {
      headline,
      normal_modes: normalModes,
      pair_modes: pairModes,
      best_mode_overlaps: bestModeOverlaps,
      best_mode_overlaps_tidy: bestModeOverlapsTidy,
      field_scales: fieldScales,
      field_scales_tidy: fieldScalesTidy,
      square_profiles: squareProfiles,
      square_profiles_tidy: squareProfilesTidy,
      seed_comparison: seedComparison,
      scf_endpoints: scfEndpoints,
      scf_endpoint_pairs: scfEndpointPairs,
      endpoint_components_tidy: endpointComponentsTidy,
      field_table: fieldTable,
      eigen_table: eigenTable,
      rank_one_composition: rankOneComposition,
      top_leakage: topLeakage,
      beta_bond_classes: betaTable,
      top_bare_charge_modes: topBareChargeModes,
      resource_summary: resourceSummary,
    },
  },
  sources,
};

fs.writeFileSync(path.join(reportDir, "artifact.json"), `${JSON.stringify(artifact, null, 2)}\n`);
console.log(`artifact written to ${path.join(reportDir, "artifact.json")}`);
