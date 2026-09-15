# Introduction, background, and current-results draft

Created 15 September 2026. This is an editable manuscript starting point using
the existing annotated literature review and locally synchronized analyses
through September 13.

The compiled draft has 16 pages and 34 cited references, including five new
references from the stripe, neutron-scattering, and coherence discussion.

- [Edit the LaTeX document](introduction_and_results.tex).
- [Read the compiled PDF](introduction_and_results.pdf).
- [Additional references](additional_references.bib).
- [Reference and evidence notes](SOURCE_NOTES.md).

## Contents

1. Draft introduction: correlated-electron motivation, stripes, pairing and
   coherence, ladders, extended interactions, and the project's question.
2. Physical background: stripe interpretations, magnetic spectroscopy,
   internal pairing symmetry versus a pair-density wave, and the model.
3. Current results: isolated-ladder gaps and correlations; response discovery;
   the revision of the early paired-basin interpretation; the two raw anchor
   comparisons; stripe wavevectors and slow texture relaxation; canonical
   energies and the bond-dimension comparison; remaining physical controls.
4. An appendix linking claims to eleven groups of local evidence.

The document preserves provisional status and historical acceptance flags.
The prepared longer-ladder seeds and incomplete grid coverage are not
presented as completed physics results. The included spatial figure is reused
from the September 12 analysis without changing its data.

## Build locally

From this directory, run either:

    latexmk -pdf introduction_and_results.tex

or:

    tectonic introduction_and_results.tex

This draft uses standard LaTeX packages, natbib, and BibTeX. Its bibliography
combines [the existing bibliography](../literature/references.bib) with the
five supplemental entries here. The annotated review and its bibliography
were not edited. Keep both bibliography files and the referenced figure
available when moving the draft to another checkout.

The checked build used the existing repository-local Tectonic 0.17.0 executable.
Build logs, extracted-text checks, and rendered pages are under the ignored
output/manuscript_draft directory. No DMRG calculation is part of this build.

Validation found no unresolved citations or cross-references, overfull text
boxes, or text outside the checked margins. All 16 local evidence links and the
reused figure resolve. Every rendered page was visually inspected.
