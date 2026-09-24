# Literature review

Plain LaTeX annotations for an introduction on competing orders in coupled
repulsive Hubbard ladders. Updated 24 September 2026: 73 annotated papers,
arranged in eight groups from the general Hubbard problem to the closest MPS+MF work.
Each paper has a medium-length summary, a paragraph on relevance and limits,
persistent links, and a stable BibTeX key. An introduction outline follows.

The actual LaTeX/PDF now includes a September 19 project-evidence update:
cubic, finer square energy/order cuts, all positive-V tests and all trellis
cells/seeds, including the alternating two-ladder relaxation.
That project-evidence paragraph is historical. The material review now covers
the core insulating and doped ladder cuprates, including Notbohm's La4Sr10
neutron benchmark, SrCu2O3/Sr2Cu3O5, La6Ca8, the Sr/Ca chain--ladder family,
LaCuO2.5 and buckled CaCu2O3. Exchange/hopping conventions, cyclic exchange,
carrier partition, charge order, pressure-dependent superconductivity,
structural modulation and trellis models are compared explicitly.

Eleven existing manuscript material references moved into the shared
bibliography with their keys preserved; thirteen new entries were added.
`Scheie2025` now cites its 2026 PRB publication. The three remaining preprints
include the separately qualified 24 K uniaxial resistive-onset report.
The manuscript's numerical results and simulation settings are unchanged.

- [Read the compiled PDF](literature_review.pdf).
- [Edit the LaTeX review](literature_review.tex).
- [Reuse or extend the BibTeX bibliography](references.bib).
- [Read the search and source notes](SOURCE_NOTES.md).

The annotations are an introduction-oriented narrative review. Source notes
distinguish abstract-based summaries from papers checked in selected full-text
sections. Three entries are marked as preprints; the collection is not a claim
of exhaustive coverage or of project novelty.

## Build

Run locally from this directory with a TeX distribution that provides `latexmk`:

```powershell
latexmk -pdf literature_review.tex
```

Or use Tectonic, which runs TeX and BibTeX as needed:

```powershell
tectonic literature_review.tex
```

The supplied PDF was compiled with Tectonic 0.17.0. The source uses standard
LaTeX packages and BibTeX with `natbib`; no custom class or template is needed.
The September 24 build has 40 pages with resolved citations and no overfull
boxes or missing characters. Source checks, build logs and page renders are
under the ignored `output/ladder_materials_review_20260924/` directory.

## Incorporate into the paper

Copy or point the manuscript at `references.bib`, retain its citation keys,
and use the bibliography style required by the target journal. For example,
with `natbib`:

```latex
\citep{Bollmark2023,Bollmark2025}
\bibliographystyle{unsrtnat}
\bibliography{references}
```

The `.bib` file is the maintained source of bibliographic metadata. Add new
papers there and add a matching `\paper{key}{heading}` annotation to the review.
When a preprint is published, update the existing entry's type, journal, year,
and DOI while preserving its key and arXiv identifier. Update the review date
and source notes, then rebuild the PDF. There is no automatic metadata rewrite.
