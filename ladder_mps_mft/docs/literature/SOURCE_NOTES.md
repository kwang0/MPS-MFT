# Search and source notes

Research date: **7 September 2026**. This is a narrative, introduction-oriented
collection, not a systematic review with a reproducible database-export
denominator. The 49 papers include 46 journal publications and three entries
retained as preprints. No claim that every relevant paper was found is made.

## Selection and search scope

The repository's repulsive two-leg extended Hubbard model, its principal
`U/t = 8`, `n = 0.9375` regime, rung-hopping and nearest-neighbor-interaction
scans, transverse geometries, and MPS+MF construction determined relevance.
Project context came from the current state, architecture, method, energy,
convergence, and publication-gate documents in the parent `docs/` directory.
The review describes the project question; it does not certify the current
numerical phase assignments.

Searches combined foundational author/title lookups with topical queries for
Hubbard-ladder superconductivity, stripes and competing orders, extended
interactions and cuprate spectroscopy, weakly coupled ladders and dimensional
crossover, matrix product states plus mean field, and recent 2025-2026 work.
Reference lists of the closest MPS+MF and ladder papers helped identify earlier
work. Primary publisher records, author manuscripts on arXiv, and author-hosted
copies supplied the scientific evidence. Search-result snippets were discovery
aids, not the sole basis for the annotations.

The groups deliberately include a few foundational method papers useful for
the introduction-to-methods transition. A complete review of all tensor-network
algorithms, all cuprate materials, or every proposed Hubbard phase is outside
this collection's scope. Closely related work with a different Hamiltonian is
retained when that difference helps position the project.

## Reading depth and bibliographic checks

**A** means the summary is grounded in a primary publisher or author abstract,
with its bibliographic record checked; the complete paper was not critically
reviewed. **S** means those checks were supplemented with the selected full-text
passages listed below. Nine papers received this closer reading. S does not
mean every derivation, figure, supplement, or numerical result was independently
validated. Relevance paragraphs and the introduction outline are synthesis.

Authors, titles, journal/volume/page or article number, publication year, and
DOIs were checked against publisher deposits in Crossref and primary records.
Where available, arXiv identifiers were matched to titles and authors. The
maintained `.bib` uses the published record rather than a separate duplicate
entry for its preprint. DOI links identify the publication; manuscript links
identify accessible author versions that may differ from the final text.

| BibTeX key | Depth | Primary record / manuscript | Selected passages, when applicable |
|---|---|---|---|
| `Hubbard1963` | A | [Publication](https://doi.org/10.1098/rspa.1963.0204) | Abstract and bibliographic record |
| `Anderson1987` | A | [Publication](https://doi.org/10.1126/science.235.4793.1196) | Abstract and bibliographic record |
| `Lee2006` | A | [Publication](https://doi.org/10.1103/RevModPhys.78.17); [arXiv](https://arxiv.org/abs/cond-mat/0410445) | Abstract and bibliographic record |
| `Scalapino2012` | A | [Publication](https://doi.org/10.1103/RevModPhys.84.1383); [arXiv](https://arxiv.org/abs/1207.4093) | Abstract and bibliographic record |
| `Arovas2022` | A | [Publication](https://doi.org/10.1146/annurev-conmatphys-031620-102024); [arXiv](https://arxiv.org/abs/2103.12097) | Abstract and bibliographic record |
| `Tranquada1995` | A | [Publication](https://doi.org/10.1038/375561a0) | Abstract and bibliographic record |
| `Fradkin2015` | A | [Publication](https://doi.org/10.1103/RevModPhys.87.457); [arXiv](https://arxiv.org/abs/1407.4480) | Abstract and bibliographic record |
| `Zheng2017` | A | [Publication](https://doi.org/10.1126/science.aam7127); [arXiv](https://arxiv.org/abs/1701.00054) | Abstract and bibliographic record |
| `JiangDevereaux2019` | A | [Publication](https://doi.org/10.1126/science.aal5304); [arXiv](https://arxiv.org/abs/1806.01465) | Abstract and bibliographic record |
| `Qin2020` | A | [Publication](https://doi.org/10.1103/PhysRevX.10.031016); [arXiv](https://arxiv.org/abs/1910.08931) | Abstract and bibliographic record |
| `JiangKivelson2022` | A | [Publication](https://doi.org/10.1073/pnas.2109406119); [arXiv](https://arxiv.org/abs/2105.07048) | Abstract and bibliographic record |
| `Xu2024` | A | [Publication](https://doi.org/10.1126/science.adh7691); [arXiv](https://arxiv.org/abs/2303.08376) | Abstract and bibliographic record |
| `Agterberg2020` | A | [Publication](https://doi.org/10.1146/annurev-conmatphys-031119-050711); [arXiv](https://arxiv.org/abs/1904.09687) | Abstract and bibliographic record |
| `Uehara1996` | A | [Publication](https://doi.org/10.1143/JPSJ.65.2764) | Abstract and bibliographic record |
| `Nagata1998` | A | [Publication](https://doi.org/10.1103/PhysRevLett.81.1090) | Abstract and bibliographic record |
| `Abbamonte2004` | A | [Publication](https://doi.org/10.1038/nature02925); [arXiv](https://arxiv.org/abs/cond-mat/0501087) | Abstract and bibliographic record |
| `Hirthe2023` | A | [Publication](https://doi.org/10.1038/s41586-022-05437-y); [arXiv](https://arxiv.org/abs/2203.10027) | Abstract and bibliographic record |
| `Chen2021` | A | [Publication](https://doi.org/10.1126/science.abf5174); [arXiv](https://arxiv.org/abs/2106.14272) | Abstract and bibliographic record |
| `Padma2025` | S | [Publication](https://doi.org/10.1103/PhysRevX.15.021049); [arXiv](https://arxiv.org/abs/2501.10287); [Read text](https://arxiv.org/html/2501.10287v1) | Main-text spectroscopy, model comparison, and model-parameter passages |
| `Scheie2025` | S | [arXiv](https://arxiv.org/abs/2501.10296); [Read text](https://arxiv.org/pdf/2501.10296v1) | Main text, pp. 1-5: magnetic response, model, attraction, and boundary pinning |
| `MerminWagner1966` | A | [Publication](https://doi.org/10.1103/PhysRevLett.17.1133) | Abstract and bibliographic record |
| `LutherEmery1974` | A | [Publication](https://doi.org/10.1103/PhysRevLett.33.589) | Abstract and bibliographic record |
| `Dagotto1992` | A | [Publication](https://doi.org/10.1103/PhysRevB.45.5744) | Abstract and bibliographic record |
| `DagottoRice1996` | A | [Publication](https://doi.org/10.1126/science.271.5249.618); [arXiv](https://arxiv.org/abs/cond-mat/9509181) | Abstract and bibliographic record |
| `Noack1994` | A | [Publication](https://doi.org/10.1103/PhysRevLett.73.882); [arXiv](https://arxiv.org/abs/cond-mat/9401013) | Abstract and bibliographic record |
| `BalentsFisher1996` | A | [Publication](https://doi.org/10.1103/PhysRevB.53.12133); [arXiv](https://arxiv.org/abs/cond-mat/9503045) | Abstract and bibliographic record |
| `Noack1997` | A | [Publication](https://doi.org/10.1103/PhysRevB.56.7162); [arXiv](https://arxiv.org/abs/cond-mat/9612165) | Abstract and bibliographic record |
| `Lin1998` | A | [Publication](https://doi.org/10.1103/PhysRevB.58.1794); [arXiv](https://arxiv.org/abs/cond-mat/9801285) | Abstract and bibliographic record |
| `WhiteAffleckScalapino2002` | A | [Publication](https://doi.org/10.1103/PhysRevB.65.165122); [arXiv](https://arxiv.org/abs/cond-mat/0111320) | Abstract and bibliographic record |
| `Dolfi2015` | S | [Publication](https://doi.org/10.1103/PhysRevB.92.195139); [arXiv](https://arxiv.org/abs/1509.04709); [Read text](https://arxiv.org/pdf/1509.04709) | Introduction, model and method, correlation-exponent and extrapolation discussion |
| `Shen2023` | S | [Publication](https://doi.org/10.1103/PhysRevB.108.165113); [arXiv](https://arxiv.org/abs/2303.16487); [Read text](https://arxiv.org/pdf/2303.16487) | Model, boundary/reference-bond dependence, and low-doping correlation analysis |
| `Zhou2023` | S | [Publication](https://doi.org/10.1103/PhysRevB.108.195136); [arXiv](https://arxiv.org/abs/2303.14723); [Read text](https://harvest.aps.org/v2/journals/articles/10.1103/PhysRevB.108.195136/fulltext) | Introduction, Hamiltonian, and intersite-interaction results |
| `Peng2023` | A | [Publication](https://doi.org/10.1103/PhysRevB.107.L201102); [arXiv](https://arxiv.org/abs/2206.03486) | Abstract and bibliographic record |
| `Huang2025` | S | [arXiv](https://arxiv.org/abs/2509.24415); [Read text](https://arxiv.org/html/2509.24415v2) | Sections I-II and selected III results: density, interaction topology, and correlation criteria |
| `Kivelson2003` | A | [Publication](https://doi.org/10.1103/RevModPhys.75.1201); [arXiv](https://arxiv.org/abs/cond-mat/0210683) | Abstract and bibliographic record |
| `Emery1997` | A | [Publication](https://doi.org/10.1103/PhysRevB.56.6120); [arXiv](https://arxiv.org/abs/cond-mat/9610094) | Abstract and bibliographic record |
| `KishineYonemitsu1998` | A | [Publication](https://doi.org/10.1143/JPSJ.67.1714); [arXiv](https://arxiv.org/abs/cond-mat/9802185) | Abstract and bibliographic record |
| `GiamarchiTsvelik1999` | A | [Publication](https://doi.org/10.1103/PhysRevB.59.11398); [arXiv](https://arxiv.org/abs/cond-mat/9810219) | Abstract and bibliographic record |
| `Arrigoni2004` | A | [Publication](https://doi.org/10.1103/PhysRevB.69.214519); [arXiv](https://arxiv.org/abs/cond-mat/0309572) | Abstract and bibliographic record |
| `Kantian2019` | A | [Publication](https://doi.org/10.1103/PhysRevB.100.075138); [arXiv](https://arxiv.org/abs/1903.12184) | Abstract and bibliographic record |
| `White1992` | A | [Publication](https://doi.org/10.1103/PhysRevLett.69.2863) | Abstract and bibliographic record |
| `Schollwock2011` | A | [Publication](https://doi.org/10.1016/j.aop.2010.09.012); [arXiv](https://arxiv.org/abs/1008.3477) | Abstract and bibliographic record |
| `CalabreseCardy2004` | A | [Publication](https://doi.org/10.1088/1742-5468/2004/06/P06002); [arXiv](https://arxiv.org/abs/hep-th/0405152) | Abstract and bibliographic record |
| `Pollmann2009` | A | [Publication](https://doi.org/10.1103/PhysRevLett.102.255701); [arXiv](https://arxiv.org/abs/0812.2903) | Abstract and bibliographic record |
| `Fishman2022` | A | [Publication](https://doi.org/10.21468/SciPostPhysCodeb.4); [arXiv](https://arxiv.org/abs/2007.14822) | Abstract and bibliographic record |
| `Bollmark2020` | A | [Publication](https://doi.org/10.1103/PhysRevB.102.195145); [arXiv](https://arxiv.org/abs/2005.02364) | Abstract and bibliographic record |
| `Bollmark2023` | S | [Publication](https://doi.org/10.1103/PhysRevX.13.011039); [arXiv](https://arxiv.org/abs/2207.03754); [Read text](https://arxiv.org/pdf/2207.03754) | Fermionic construction, Section VI repulsive-ladder application, and Appendix E |
| `Bollmark2025` | S | [Publication](https://doi.org/10.1103/PhysRevB.111.125141); [arXiv](https://arxiv.org/abs/2301.08116); [Read text](https://arxiv.org/pdf/2301.08116) | Multichannel construction and Section III A alternating CDW partners |
| `Kohler2026` | S | [arXiv](https://arxiv.org/abs/2608.18861); [Read text](https://arxiv.org/html/2608.18861v1) | Introduction, mixed-dimensional model, gap/charge diagnostics, and ordering-scale discussion |

## Record-specific notes

- `JiangKivelson2022`: the PNAS issue year is 2022; online publication was in
  December 2021. The citation uses the issue year.
- `Xu2024`: the Science publication is from 2024; its arXiv identifier begins
  with 2303 because the preprint appeared in 2023.
- `Fishman2022`: the author list follows the publication's author names,
  including Steven R. White and E. Miles Stoudenmire.
- `Scheie2025`, `Huang2025`, and `Kohler2026`: the checked arXiv records supplied
  no journal reference. The entries are explicitly marked as preprints, not
  assigned an inferred journal or DOI. Their status is a dated check, not a
  guarantee that no later publication exists. Recheck before submission.
- The closer reading of `Huang2025` used v2; that of `Kohler2026` used v1,
  submitted 19 August 2026. Dates in rendered author manuscripts are not used
  to replace verified arXiv submission or journal publication years.

The bibliography contains factual citation metadata and links, and the review
contains original summaries. Full source texts and copied abstracts are not
included in the deliverable. Keep citation keys stable as records are updated.
