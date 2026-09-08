# Review provenance and verification

- Review date: September 4, 2026, local Windows workspace.
- Baseline: `a744d29`, `codex/mps-mft-phase0-refactor`; initial tracked worktree clean, root `.claude/` untracked.
- Requested result: full-project implementation/status review and scientific recommendations, not source fixes or new campaigns.
- Audience: technical. Delivery mode: MCP report, with canonical saved artifact and Markdown narrative as supporting evidence. HTML fallback only if rendering fails.
- Required structure mapping: technical summary; current results and historical evidence; implementation and scientific-method findings; uncertainty/convergence; explicit scope/verification; next steps and open scientific questions. Scope/method moved after findings for an answer-first reading path. Limitations appear next to each affected claim.

## Local evidence

`extract_review.py` reads 52 Phase 1 terminal state paths and saves `state_inventory.csv`. It reuses `scripts/audit_scf_numerics.py` for the 42 states with adequate histories and writes `recurrence_screen.csv`. The screen checks added slow-mode and oscillation rules; it is not a new acceptance certificate. All ten screening omissions are explained in `read_gaps.csv`: nine older histories and one frozen solve. None is an unreadable HDF5 in this inventory.

Six current V=0 terminal artifacts were independently matched to the sizes and SHA-256 values in their compact manifests (`current_terminal_hash_checks.csv`). Full-source identifiers are recorded, not verified scratch availability. Counts are artifact-path counts; no claim of statistical independence or exhaustive basin discovery is made.

Backbone numbers were checked against `../bare_stage1_t014_v0_20260902/data/backbone_convergence.csv` and its report. Stage 2 reciprocity, leakage, field strength, and discovery acceptance were read from local `stage2_discovery.h5`; leading spectra and bare-image/SCF comparisons were checked against the September 3 TSV/summary outputs. Those older report inputs were not rerun or blanket rehashed. The E_p registry has 107 rows and no duplicate complete parameter-plus-chi keys; the specific V=0, t0=1.4 row was inspected. This is not a full registry convergence audit.

Source review covered the module interfaces and core logic in Types, Config, Geometry, MeanField, Solver, Mixing, Convergence, Variational, Selection, Provenance, Storage, Device, CUDA extension, Diagnostics, Backbone, BareStage1, and BareStage2; existing tests; campaign preparation/Slurm guards; legacy model/convergence/plotting paths; active plans and method docs. Larger source files and launchers received targeted review of relevant entry points, guards, and calculations, not exhaustive line-by-line proof.

## Focused local validation

- Actual Julia source-contract reproductions: 6 assertions in 4 test sets passed, using Julia 1.12.7 in 4.77 seconds wall time. The fixtures confirm existing gaps. Selection uses an in-memory HDF5-shaped adapter; no HDF5 writer, ITensor, CUDA, or DMRG runs in these checks. Exact source is in `contract_checks.txt` and `review_notebook.ipynb`; the executed temporary `.jl` file is removed after preservation.
- Existing Python spatial-analysis suite: 6 tests passed in 0.008 seconds of test execution.
- Evidence extractor: approximately 3.05 seconds wall time.
- No full Julia test suite, full-physics convergence run, scheduler query, transfer, live allocation action, or source fix.
- A test import changed a tracked Python bytecode file. Its original HEAD bytes were restored using read-only `git show`; generated test/audit bytecode was removed. Git's usual restore path needed an unwritable index lock, so no index operation was used. This is not a blocker or an unperformed user action.
- Most review time was spent reading code, checking evidence, and literature synthesis, not numerical execution.

## Literature and claim discipline

Primary sources were searched/opened during the review: Bollmark et al. PRX 2023 (arXiv:2207.03754), Bollmark/Köhler/Kantian PRB 2025 (arXiv:2301.08116), Shen/Zhang/Qin PRB 2023 (arXiv:2303.16487), Dolfi et al. PRB 2015 (arXiv:1509.04709), White/Affleck/Scalapino PRB 2002 (cond-mat/0111320), Xu et al. Science 2024 (arXiv:2303.08376), Jaefari/Fradkin PRB 2012 (arXiv:1111.6320), Zauner-Stauber et al. PRB 2018 (arXiv:1701.07035), and Staelens et al. 2026 preprint (arXiv:2602.21695). Links appear near the associated claims in the report.

The charge-gap correction is an interpretation of the retained/removed manifold distinction, supported by the original perturbation theory and Luther–Emery characterization; it does not assert that t_perp=0.1 is controlled. Both 2026 sources are explicitly preprints. The August 19 Köhler/Kantian preprint (arXiv:2608.18861) was also inspected, including Secs. 4.2–4.3 and 5: tradeoffs among pairing, susceptibility and phase coherence, unresolved competition, and excitation character under finite-size level crossings. Related mixD, 2D and Hubbard–Heisenberg results motivate questions and are not evidence for the present Hamiltonian. Search was targeted; no exhaustive novelty certification is claimed.

## Visual and report decisions

One native horizontal count chart partitions the 52 terminal artifact paths into five evidence classes. It has one count axis, a zero baseline, direct labels, a single blue root, and expanded definitions/fractions in the data. This is a status visualization, not a scientific phase plot. A native table retains exact current-seed energy/density scales. The six-seed table is alphabetically sorted to avoid presenting unresolved energy differences as a physical ranking. No synthetic performance or error-budget chart is constructed.

Scientific uncertainties are discussed quantitatively in prose and the exact-value table: density-correction magnitude is not the error after correction; missing finite-chi/length/range bounds cannot be plotted as estimated confidence bars. No new phase map is drawn from incomplete campaigns. Code recommendations use prose because their severity is a judgment, not a quantitative score.

The report includes all requested review areas and explicit open questions. No Sites publication is requested. Canonical artifact validation/render results are recorded in `artifact_validation.json` and `artifact_render_result.json` after the handoff attempt.

The MCP handoff returned `ok: true` but did not expose an inspectable report URL. Browser inventory contains no report preview; native application access is disabled in this session. Therefore pixel-level visual QA of that surface is unavailable, and no claim that `ok: true` proves visible rendering is made. The complete readable narrative is also saved as `REVIEW.md`. The artifact reading order, source bindings, one full-width chart, one full-width six-row table, and unchanged datasets were checked structurally. No parallel HTML report or public site was created.

The report queries are actually executed against an in-memory SQLite database populated from the saved local CSV evidence. The inventory query computes category counts and fractions; the seed query computes relative corrected energy and applied density-correction magnitudes. HDF5 extraction and the original field definitions remain in `extract_review.py`; SQL is not presented as the original data source.
