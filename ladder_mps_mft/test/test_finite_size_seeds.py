"""Focused geometry tests for the seed extension; no DMRG or source artifacts."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_phase1_finite_size_seeds import extend_profile


class FiniteSizeSeedTests(unittest.TestCase):
    def test_periodic_stripe_preserves_wavevector_and_sublattice_for_every_bond(self):
        for offset in range(-4, 5):
            start, stop = max(0, -offset), min(64, 64-offset)
            sc = np.arange(start, stop) + 1 + offset/2
            waveform = lambda c: np.column_stack((
                np.cos(2*np.pi*c/32) * (-1.)**np.floor(c),
                np.sin(2*np.pi*c/16),
            ))
            for L in (96, 128):
                tc = np.arange(max(0, -offset), min(L, L-offset)) + 1 + offset/2
                result = extend_profile(waveform(sc), sc, tc, L, "stripe")
                np.testing.assert_allclose(result, waveform(tc), atol=8e-15, rtol=0)

    def test_pairing_plateau_retains_staggered_components(self):
        sc = np.arange(1, 65)
        source = np.column_stack((np.full(64, 0.004), 0.02 * (-1.)**sc))
        for L in (96, 128):
            tc = np.arange(1, L+1)
            expected = np.column_stack((np.full(L, 0.004), 0.02 * (-1.)**tc))
            np.testing.assert_allclose(extend_profile(source, sc, tc, L, "pairing"), expected, atol=1e-17, rtol=0)

    def test_asymmetric_edges_are_preserved_and_blends_do_not_overshoot(self):
        sc = np.arange(1, 65)
        source = (sc + 3*np.sin(sc))[:, None]
        for lineage in ("pairing", "stripe"):
            for L in (96, 128):
                result = extend_profile(source, sc, np.arange(1, L+1), L, lineage)
                np.testing.assert_array_equal(result[:32], source[:32])
                np.testing.assert_array_equal(result[-32:], source[-32:])
                self.assertGreaterEqual(result.min(), source.min())
                self.assertLessEqual(result.max(), source.max())

    def test_unsupported_length_is_not_silently_stretched(self):
        with self.assertRaises(ValueError):
            extend_profile(np.ones((64, 1)), np.arange(1, 65), np.arange(1, 105), 104, "stripe")


if __name__ == "__main__":
    unittest.main()
