# Literature review

Plain LaTeX annotations for an introduction on competing orders in coupled
repulsive Hubbard ladders. Updated 7 September 2026: 49 papers, arranged in
eight groups from the general Hubbard problem to the closest MPS+MF work.
Each paper has a medium-length summary, a paragraph on relevance and limits,
persistent links, and a stable BibTeX key. An introduction outline follows.

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
