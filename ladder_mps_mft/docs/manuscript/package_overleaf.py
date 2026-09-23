"""Package the maintained LaTeX sources for Overleaf, preserving relative paths."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def main():
    manuscript = Path(__file__).resolve().parent
    docs = manuscript.parent
    names = (
        "METHODS_NOTES.tex",
        "literature/literature_review.tex",
        "literature/references.bib",
        "manuscript/introduction_and_results.tex",
        "manuscript/additional_references.bib",
        "reports/two_basin_v000_20260912/profile_evolution.pdf",
        "reports/campaign_review_20260918/square_cubic_phase_diagrams.pdf",
        "reports/campaign_review_20260918/square_cubic_energy_grids.pdf",
        "reports/campaign_review_20260918/square_cubic_spin_grids.pdf",
        "reports/campaign_review_20260918/square_cubic_pairing_grids.pdf",
        "reports/square_fine_cuts_20260918/variational_energy_cuts.pdf",
        "reports/square_fine_cuts_20260918/variational_energy_shape.pdf",
        "reports/square_fine_cuts_20260918/order_parameter_cuts.pdf",
        "reports/square_positive_v_20260918/histories.pdf",
        "reports/square_positive_v_20260918/terminal_profiles.pdf",
        "reports/trellis_progress_20260918/histories.pdf",
        "reports/trellis_progress_20260918/profiles.pdf",
        "reports/trellis_progress_20260918/two_ladder_relaxation.pdf",
        "reports/pair_correlations_20260922/pair_decay_comparison.pdf",
        "reports/pair_correlations_20260922/matched_pair_comparisons.pdf",
        "reports/pair_correlations_20260922/stripe_pairing_profiles.pdf",
        "reports/pair_correlations_20260922/square_AB_complete.pdf",
        "reports/pair_correlations_20260922/square_AB_pair_correlations.pdf",
        "reports/stripe_pairing_grid_20260923/paired_stripe_weights.pdf",
        "reports/stripe_pairing_grid_20260923/paired_stripe_distance_correlations.pdf",
        "manuscript/README.md",
    )
    # Read everything first, so a missing source cannot truncate a valid ZIP.
    sources = {name: (docs / name).read_bytes() for name in names}
    archive = manuscript / "overleaf_upload.zip"
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as bundle:
        for name, content in sources.items():
            # Fixed metadata avoids binary changes caused only by file timestamps.
            entry = ZipInfo(name, date_time=(2026, 9, 15, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.create_system = 3
            entry.external_attr = 0o100644 << 16
            bundle.writestr(entry, content)
    with ZipFile(archive) as bundle:
        assert bundle.testzip() is None, "ZIP integrity check failed"
        assert set(bundle.namelist()) == set(sources), "Unexpected ZIP contents"
        for name, content in sources.items():
            assert bundle.read(name) == content, "Packaged file differs: " + name
    print(f"Created and verified {archive} ({len(sources)} files)")


if __name__ == "__main__":
    main()
