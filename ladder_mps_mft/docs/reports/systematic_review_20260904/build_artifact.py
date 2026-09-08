"""Build the canonical review artifact and a compact reproducibility notebook."""
from pathlib import Path
import csv, json, re, sqlite3
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
STAMP = datetime.now(timezone.utc).isoformat()

def source(sid, label, path, description):
    return dict(id=sid, label=label, path=path, query=dict(
        description=description, tables_used=[path], executed_at=STAMP,
        metric_definitions=[description]))

sources = [
    source("inventory", "Local terminal-state inventory and recurrence screen",
        "docs/reports/systematic_review_20260904/campaign_inventory.csv",
        "Counts of local state.h5 artifact paths, not independent seeds. Existing audit_scf_numerics.py screens added oscillation and slow-mode gates; passing is not recertification. Reproduced by extract_review.py."),
    source("states", "Six current square V=0 endpoints",
        "docs/reports/systematic_review_20260904/state_inventory.csv",
        "Exact current campaign 20260902_phase1_square_t014_v000_seed_chi200_loose_cuda130, L=64, Ns=128, U=8, t0=1.4, V=0, density=0.9375, chi=200. Relative corrected energy is (Etarget-min(Etarget))/Ns; density correction scale is abs(mu*(n-ntarget)), not the remaining error."),
    source("code", "Source contracts and review reproductions",
        "docs/reports/systematic_review_20260904/review_notebook.ipynb",
        "Four focused checks execute current source with fixtures: omitted CUDA extension in implementation hash, missing/nonfinite ranking inputs, missing inner-DMRG gate, and mu_initial in physical identity. No DMRG solve."),
    source("backbone", "Bare ladder sector convergence",
        "docs/reports/bare_stage1_t014_v0_20260902/data/backbone_convergence.csv",
        "Six fixed-sector energies combined at each chi. All final chi=1200 sectors pass stored gates; chi=800 does not pass all-sector convergence. Spin gap is E(N,Sz=1)-E(N,0); even charge gap is (E(N+2)+E(N-2)-2E(N))/2."),
    source("stage2", "Bare Stage 2 discovery evidence",
        "output/bare_stage2/20260902_bare_t014_v0_stage2/stateless_results/stage2_discovery.h5",
        "Twelve-direction discovery at h=1e-4; maximum omitted-response norm ratio 0.8326719638018206. Discovery acceptance precedes amplitude/basis validation. Full scratch artifacts not verified."),
    source("snapshot", "Dated local project snapshot", "docs/PROJECT_STATE.md",
        "September 4, 2026 local snapshot and current plan; job IDs and live membership unavailable. No live scheduler evidence was obtained."),
]

inventory = list(csv.DictReader((HERE / "campaign_inventory.csv").open(encoding="utf-8")))
states = list(csv.DictReader((HERE / "state_inventory.csv").open(encoding="utf-8")))
latest = [r for r in states if r["campaign"] == "20260902_phase1_square_t014_v000_seed_chi200_loose_cuda130"]
emin = min(float(r["corrected_per_site"]) for r in latest)
labels = {"legacy_pairing_mixed":"Smooth pairing", "pairing_dwave_m000":"Uniform d-wave seed",
    "stripe_m004":"Stripe m=4", "stripe_m005":"Stripe m=5",
    "stripe_pairing_m004":"Stripe + pairing m=4", "stripe_pairing_m005":"Stripe + pairing m=5"}
seed_rows = []
for r in latest:
    path = r["path"]
    label = next(v for k,v in labels.items() if f"square__{k}_chi200_loose" in path)
    seed_rows.append(dict(seed=label, iterations=int(r["iterations"]),
        delta_e_micro_t_per_site=(float(r["corrected_per_site"])-emin)*1e6,
        density_correction_micro_t_per_site=float(r["density_correction_scale"])*1e6,
        last_sweep_change=float(r["last_sweep_change"]),
        status=r["status"], model_fingerprint=r["model_fp"], numerical_fingerprint=r["numerical_fp"]))

status_rows = [
    dict(category="Stored accepted; passes screen", count=17, fraction=17/52, definition="Retains its stored accepted flag under the added oscillation/slow-mode screen; not recertified."),
    dict(category="Stored accepted; fails screen", count=11, fraction=11/52, definition="Needs renewed numerical eligibility review; preserve source artifact."),
    dict(category="Unaccepted; history available", count=14, fraction=14/52, definition="Already unaccepted, sufficient history for the screen."),
    dict(category="Older history unavailable", count=9, fraction=9/52, definition="V2 files lack applied-field history; all unaccepted."),
    dict(category="Frozen-field diagnostic", count=1, fraction=1/52, definition="Single frozen solve; not an SCF endpoint."),
]
screening = list(csv.DictReader((HERE / "recurrence_screen.csv").open(encoding="utf-8")))
by_sha = {r["full_artifact_sha256"]: r for r in screening}
db = sqlite3.connect(":memory:")
db.row_factory = sqlite3.Row
db.execute("CREATE TABLE review_terminal_artifacts (path TEXT, campaign TEXT, stored_accepted INTEGER, has_history INTEGER, screen_accepted INTEGER, category TEXT)")
for r in states:
    screened = by_sha.get(r["full_sha256"])
    accepted = r["accepted"] == "True"
    screen_accepted = bool(screened and screened["revised_accepted"] == "True")
    category = ("Stored accepted; passes screen" if screen_accepted else
        "Stored accepted; fails screen" if accepted else
        "Unaccepted; history available" if screened else
        "Frozen-field diagnostic" if r["status"] == "frozen_field_evaluation" else
        "Older history unavailable")
    db.execute("INSERT INTO review_terminal_artifacts VALUES (?,?,?,?,?,?)",
        (r["path"],r["campaign"],int(accepted),int(screened is not None),int(screen_accepted),category))
status_sql = """SELECT category, COUNT(*) AS count,
       COUNT(*) * 1.0 / (SELECT COUNT(*) FROM review_terminal_artifacts) AS fraction
FROM review_terminal_artifacts
GROUP BY category
ORDER BY CASE category
 WHEN 'Stored accepted; passes screen' THEN 1
 WHEN 'Stored accepted; fails screen' THEN 2
 WHEN 'Unaccepted; history available' THEN 3
 WHEN 'Older history unavailable' THEN 4 ELSE 5 END;"""
queried = [dict(r) for r in db.execute(status_sql)]
assert [(r["category"],r["count"]) for r in queried] == [(r["category"],r["count"]) for r in status_rows]
definitions = {r["category"]:r["definition"] for r in status_rows}
status_rows = [dict(r,definition=definitions[r["category"]]) for r in queried]
sources[0]["query"].update(sql=status_sql,engine="SQLite",language="SQL",
    tables_used=["review_terminal_artifacts","docs/reports/systematic_review_20260904/state_inventory.csv","docs/reports/systematic_review_20260904/recurrence_screen.csv"])
db.execute("CREATE TABLE review_current_seed_inputs (seed TEXT, iterations INTEGER, corrected_per_site REAL, mu REAL, density REAL, target_density REAL, last_sweep_change REAL, status TEXT, model_fingerprint TEXT, numerical_fingerprint TEXT)")
for r, expected in zip(latest,seed_rows):
    db.execute("INSERT INTO review_current_seed_inputs VALUES (?,?,?,?,?,?,?,?,?,?)",
        (expected["seed"],int(r["iterations"]),float(r["corrected_per_site"]),
         float(r["mu"]),float(r["density"]),float(r["target_density"]),
         float(r["last_sweep_change"]),r["status"],r["model_fp"],r["numerical_fp"]))
seed_sql = """SELECT seed, iterations,
       (corrected_per_site - MIN(corrected_per_site) OVER ()) * 1000000.0
           AS delta_e_micro_t_per_site,
       ABS(mu * (density - target_density)) * 1000000.0
           AS density_correction_micro_t_per_site,
       last_sweep_change, status, model_fingerprint, numerical_fingerprint
FROM review_current_seed_inputs
ORDER BY seed ASC;"""
seed_rows = [dict(r) for r in db.execute(seed_sql)]
sources[1]["query"].update(sql=seed_sql,language="SQL",engine="SQLite",
    transformation="extract_review.py reads stored target-density-corrected canonical energies, divides by 2L, and extracts chemical potential and densities into state_inventory.csv. build_artifact.py loads the exact dated campaign into an in-memory table; this SQL materially computes the displayed relative energy and correction magnitudes. Original HDF5 extraction and field definitions remain in extract_review.py.",
    tables_used=["review_current_seed_inputs","docs/reports/systematic_review_20260904/state_inventory.csv"])
(HERE / "report_queries.sql").write_text(status_sql+"\n\n"+seed_sql+"\n",encoding="utf-8")
db.close()
assert sum(r["count"] for r in status_rows)==len(states)==52
assert sum(int(r["screened_accepted"]) for r in inventory)==17
assert len({r["model_fingerprint"] for r in seed_rows})==1
assert len({r["numerical_fingerprint"] for r in seed_rows})==1

text = (HERE / "REVIEW.md").read_text(encoding="utf-8")
parts = re.split(r"(?m)^## ", text)
title = parts[0].strip().removeprefix("# ")
blocks = [dict(id="title",type="markdown",body="# "+title,layout="full")]
for index, part in enumerate(parts[1:]):
    section = "## " + part.strip()
    section_sources = {1:"snapshot",2:"states",3:"inventory",4:"code",5:"backbone",6:"stage2",7:"code",8:"snapshot",9:"code",10:"snapshot"}
    blocks.append(dict(id=f"section-{index+1}",type="markdown",body=section,layout="full",sourceId=section_sources[index+1]))
    if part.startswith("The current results"):
        blocks.append(dict(id="endpoint-table",type="table",tableId="endpoints",layout="full"))
    if part.startswith("Historical acceptance"):
        blocks.append(dict(id="artifact-count-chart",type="chart",chartId="artifact-status",layout="full"))
        blocks.append(dict(id="chart-interpretation",type="markdown",layout="full",
            body="The bars partition all 52 local terminal artifacts. The first category preserves a stored classification after a limited diagnostic screen; it does not count independently certified physical solutions. The six newest square V=0 endpoints are all in that category."))

artifact = dict(surface="report", manifest=dict(version=1,surface="report",title=title,
    description="Implementation review, local evidence audit, and literature-informed scientific priorities.",
    generatedAt=STAMP,sources=sources,blocks=blocks,
    charts=[dict(id="artifact-status",title="Local terminal artifact classifications",type="horizontalBar",
        subtitle="52 artifact paths; review categories are not counts of independent physical solutions.",
        dataset="status_counts",sourceId="inventory",intent="comparison",
        question="Which locally stored endpoint classifications can be used without renewed review?",
        rationale="A single horizontal count chart shows the evidence boundary without implying a chronological improvement or independent-seed sample.",
        encodings=dict(x=dict(field="category",type="nominal",label="Review category"),
            y=dict(field="count",type="quantitative",label="Artifact paths"),
            tooltip=[dict(field="definition",type="text",label="Interpretation"),dict(field="fraction",type="quantitative",label="Fraction of 52")]),
        palette=dict(kind="categorical",colors=["#2870A7"]),labels=dict(values="auto"),
        settings=dict(sort="none",showValues=True),layout="full")],
    tables=[dict(id="endpoints",title="Current square endpoints",dataset="current_seeds",sourceId="states",
        subtitle="V=0, t0=1.4, L=64, chi=200. Energy and density scales are in millionths of t per physical site; fine ordering is unresolved.",
        defaultSort=dict(field="seed",direction="asc"),density="spacious",layout="full",columns=[
            dict(field="seed",label="Initial seed",type="text"),
            dict(field="iterations",label="SCF evaluations",format="number"),
            dict(field="delta_e_micro_t_per_site",label="Corrected energy above minimum",format="number"),
            dict(field="density_correction_micro_t_per_site",label="Magnitude of applied density correction",format="number"),
        ])]),
    snapshot=dict(version=1,status="ready",generatedAt=STAMP,datasets=dict(status_counts=status_rows,current_seeds=seed_rows)),
    sources=sources)
(HERE / "artifact.json").write_text(json.dumps(artifact,indent=2,ensure_ascii=False),encoding="utf-8")

julia_file = HERE / "review_contract_checks.jl"
contract = julia_file.read_text(encoding="utf-8") if julia_file.exists() else (HERE/"contract_checks.txt").read_text(encoding="utf-8")
(HERE/"contract_checks.txt").write_text(contract,encoding="utf-8")
cells = [dict(cell_type="markdown",metadata={},source=["# Reproduce the September 4 project review\n",
    "Run from this directory. Inputs are read-only local evidence. The extractor writes CSV/JSON review outputs. No DMRG or Perlmutter access occurs.\n"]),
    dict(cell_type="code",execution_count=None,metadata={},outputs=[],source=[
        "from pathlib import Path\nimport runpy, sys\nsys.dont_write_bytecode = True\n",
        "here = Path.cwd()\nassert (here / 'extract_review.py').is_file(), 'Open this notebook in its report directory'\n",
        "runpy.run_path(str(here / 'extract_review.py'), run_name='__main__')\n"]),
    dict(cell_type="code",execution_count=None,metadata={},outputs=[],source=[
        "import csv, json\n",
        "summary = json.loads((here / 'evidence_summary.json').read_text(encoding='utf-8'))\n",
        "assert summary['state_paths'] == 52\nassert len(summary['current_terminal_hash_checks']) == 6\n",
        "assert all(r['hash_match'] and r['size_match'] for r in summary['current_terminal_hash_checks'])\n",
        "summary\n"]),
    dict(cell_type="markdown",metadata={},source=[
        "## Focused Julia source-contract reproductions\n",
        "The checks below intentionally confirm observed gaps, not a corrected implementation. The file adapter is in-memory; tensor libraries are not loaded. Six assertions passed when reviewed with Julia 1.12.7. Copy the following source into a temporary `.jl` file in this report directory and run `julia --startup-file=no <file>`, then delete it.\n",
        "```julia\n"+contract+"\n```\n"]),
    dict(cell_type="markdown",metadata={},source=[
        "## Scope and interpretation\n",
        "The existing recurrence screen is not a complete reimplementation of acceptance. Full scratch artifacts and live scheduler state were not inspected. See SOURCE_NOTES.md for source coverage, validation, and chart decisions.\n"])]
notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name="Python 3",language="python",name="python3")),cells=cells)
for i,c in enumerate(cells): c["id"]=f"review-{i}"
(HERE/"review_notebook.ipynb").write_text(json.dumps(notebook,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps(dict(title=title,blocks=len(blocks),charts=1,tables=1,bytes=(HERE/'artifact.json').stat().st_size)))
