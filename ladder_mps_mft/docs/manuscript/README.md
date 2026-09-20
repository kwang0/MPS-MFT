# Introduction, background, and current-results draft

Created 15 September 2026. This is an editable manuscript starting point using
the existing annotated literature review and locally synchronized analyses
through September 19, with a September 20 common-cell trellis energy audit
(updated from the initial September 13 cutoff).

Final-results update, September 19: the [combined campaign review](../reports/campaign_review_20260918/README.md)
is now incorporated into the actual LaTeX and rebuilt PDF: complete square
and cubic grids, square continuations, finer-cut full variational energies
and matched physical spin/pairing cuts, all four positive-V square starts,
and all four one-/two-ladder trellis runs. Sections 3.11 and 3.12 now develop
the square and trellis analyses separately, including magnetic defects,
spatial-cell dependence, convergence and alternating relaxation.

September 20 clarification: evaluating paired copies in the same rectangular
cell changes their energy by only 2.464e-6 t/site; the saved striped trials
remain 0.002238–0.002308 t/site lower. Section 3.12 distinguishes this direct
trial-energy comparison from a stationary phase ranking and explains how
diagonal stripes would require charge/spin commensurability across ladders.

The rebuilt draft has 37 pages and 48 cited references. The introduction and
Section 2.4 now motivate the trellis runs through the chain--ladder materials,
pressure-induced superconductivity, charge and magnetic order, optical/model
parameter estimates (including the limits on V), and reciprocal one-/two-ladder
spatial cells. Eleven new material references supplement the initial five
stripe/coherence references; three existing material records are also cited.

- [Edit the LaTeX document](introduction_and_results.tex).
- [Read the compiled PDF](introduction_and_results.pdf).
- [Additional references](additional_references.bib).
- [Reference and evidence notes](SOURCE_NOTES.md).

## Contents

1. Draft introduction: correlated-electron motivation, stripes, pairing and
   coherence, ladders, extended interactions, and the project's question.
2. Physical background: stripe interpretations, magnetic spectroscopy,
   internal pairing symmetry versus a pair-density wave, ladder materials,
   doping and pressure, model parameters, and the trellis extension.
3. Current results: isolated-ladder gaps and correlations; response discovery;
   the revision of the early paired-basin interpretation; the two raw anchor
   comparisons; stripe wavevectors and slow texture relaxation; canonical
   energies and the bond-dimension comparison; complete grid results, fine-cut
   energy shapes, positive-V and trellis outcomes; remaining physical controls.
4. An appendix linking claims to sixteen groups of local evidence.

The document preserves provisional status and historical acceptance flags.
The prepared longer-ladder seeds are not new evidence. All four positive-V
and all four trellis final states are now analyzed; old partial logs remain
historical. There are thirteen figures, including the earlier September 12
spatial evolution, the square/cubic comparisons, full transition-cut energies,
physical spin/pairing cuts and completed square/trellis history/profile plots. Figures 2–5 compare square and cubic side by side: phase diagrams,
full energy histories, physical spin RMS and physical pairing RMS. The three
history comparisons use landscape pages and retain all square continuations.
Every comparison is referenced in the text. The PDFs and LaTeX sources are
updated together.

## Build locally

From this directory, run either:

    latexmk -pdf introduction_and_results.tex

or:

    tectonic introduction_and_results.tex

This draft uses standard LaTeX packages, natbib, and BibTeX. Its bibliography
combines [the existing bibliography](../literature/references.bib) with the
sixteen supplemental entries here. Both LaTeX documents now detect whether the
build starts in their own directory or the parent documentation/project root.
No path edits are needed when switching between those layouts. The shared
bibliography is reused without editing its entries.

## Build on Overleaf

Preserve this layout at the Overleaf project root:

```text
literature/
  literature_review.tex
  references.bib
manuscript/
  introduction_and_results.tex
  additional_references.bib
reports/
  two_basin_v000_20260912/
    profile_evolution.pdf
  campaign_review_20260918/
    square_cubic_phase_diagrams.pdf
    square_cubic_energy_grids.pdf
    square_cubic_spin_grids.pdf
    square_cubic_pairing_grids.pdf
  square_fine_cuts_20260918/
    variational_energy_cuts.pdf
    variational_energy_shape.pdf
    order_parameter_cuts.pdf
  square_positive_v_20260918/
    histories.pdf
    terminal_profiles.pdf
  trellis_progress_20260918/
    histories.pdf
    profiles.pdf
    two_ladder_relaxation.pdf
METHODS_NOTES.tex
```

Set the main document to `manuscript/introduction_and_results.tex`, then
choose **Recompile from scratch** after replacing an earlier upload. Select
`literature/literature_review.tex` instead to compile the annotated review.
Use the normal pdfLaTeX compiler with BibTeX (the documents use natbib).
The path handling follows the project-root behavior documented in
[Overleaf's multi-file guidance](https://www.overleaf.com/learn/latex/Multi-file_LaTeX_projects).

All thirteen figure PDFs are required, even though they are outside the two source
folders. The Git-tracked [Overleaf upload ZIP](overleaf_upload.zip) contains
the LaTeX sources, bibliographies, living methods notes, required figures and
these instructions, so it can be downloaded from another device.
Regenerate it after later source edits by running
`python package_overleaf.py` from this directory (Python standard library only).
The ZIP is a generated snapshot; edit the original sources, not its contents.
Appendix links to local analysis files remain local evidence pointers; the
reports they link to are not required for compilation or included in the ZIP.

The checked build used the existing repository-local Tectonic 0.17.0 executable.
Build logs, extracted-text checks, and rendered pages are under the ignored
output/notes_update_20260918 directory (the initial draft checks remain under
output/manuscript_draft). No DMRG calculation is part of this build.

The historical material-discussion build is under
`output/trellis_manuscript_20260918/`. Final-results rendering and checks are
under `output/notes_update_20260919/`. The September 19 build retains all
48 references and convergence qualifications and is checked for unresolved
references, missing figures, layout and Overleaf path coverage.
