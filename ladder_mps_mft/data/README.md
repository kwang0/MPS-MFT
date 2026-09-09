# Pair-binding registry

`E_p_values.csv` is a value-for-value copy of the legacy repository file
`../E_p_values.csv`. The copy normalizes CRLF line endings to LF so that it is a
clean text artifact in this isolated project; no fields, rows, or numerical
values were changed.

Source raw SHA-256 (CRLF):

`dbb04ba2317b014267aed8bf2671a44611803692a35873023b9e1ed9089abba5`

Isolated-project SHA-256 (LF):

`2209bd2ca3c1ad02c0e542d1a9d63ecf90fdfa49120ad9cc3af599a5b4bc1f0e`

Every run records the isolated-project registry hash. Lookup is exact in
`(L,U,V,t0,density)`; the solver never interpolates missing values.

## Two-basin reference correlations

`two_basin_references.h5` is the small, versioned input for
`slurm/submit_square_two_basin.sh`. It contains the selected chi=200 stripe
correlations at `(t0,V)=(1.0,0.0)` and uniform d-wave correlations at `(1.4,-0.4)`,
with source paths and SHA-256 provenance. It contains no MPS. The preparer checks
the bundle and both source hashes before constructing 95%/5% mixtures using
each target Hamiltonian's couplings.

Bundle SHA-256:
`e01a1ea7d6be813110870d26377db0df529d1584816c946af78584e1e1fbddc1`.

This is a byte-identical copy of the locally extracted reference bundle under
`output/seed_previews/20260908_square_two_basin/`. `prepare_two_basin_references.py`
records its extraction from the immutable source results. The narrow ignore-rule
exception versions this input; simulation outputs remain excluded from Git.
