import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_spatial_phase_defects.py"
SPEC = importlib.util.spec_from_file_location("audit_spatial_phase_defects", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class SpatialPhaseDefectTests(unittest.TestCase):
    def test_h5py_arrays_are_restored_to_julia_dimension_order(self):
        storage_order = np.zeros((9, 2, 2, 64, 64))
        self.assertEqual(AUDIT._julia_array(storage_order).shape, (64, 64, 2, 2, 9))

    def test_spectrum_recovers_known_wavevector(self):
        length = 64
        q_expected = 5.0 * math.pi / 16.0
        x = np.arange(length)
        values = np.cos(q_expected * x + 0.37)
        _, _, metrics = AUDIT.spatial_spectrum(values, subtract_mean=True, fraction=0.75)
        self.assertLess(abs(metrics["q"] - q_expected), 0.06)
        self.assertGreater(metrics["peak_share"], 0.08)
        self.assertGreater(metrics["peak_band_share"], metrics["peak_share"])

    def test_second_peak_ignores_adjacent_leakage_bins(self):
        length = 64
        x = np.arange(length)
        values = np.cos(0.3 * math.pi * x) + 0.6 * np.cos(0.75 * math.pi * x)
        _, _, metrics = AUDIT.spatial_spectrum(values, subtract_mean=True, fraction=0.75)
        self.assertGreater(metrics["second_peak_ratio"], 0.15)

    def test_phase_slip_requires_amplitude_zero_and_phase_jump(self):
        length = 64
        phase = np.zeros(length)
        phase[32:] = math.pi
        amplitude = np.ones(length)
        amplitude[31:34] = (0.2, 0.05, 0.2)
        envelope = amplitude * np.exp(1j * phase)
        candidates = AUDIT.phase_slip_candidates(envelope)
        self.assertTrue(any(abs(item["rung"] - 33) <= 2 for item in candidates))

        smooth = np.ones(length, dtype=complex)
        self.assertEqual(AUDIT.phase_slip_candidates(smooth), [])

    def test_residual_center_of_mass_localizes_pair_change(self):
        length = 12
        time_count = 1
        alpha = np.zeros((length, length, 2, 2, time_count))
        beta = np.zeros((2, length, length, 2, 2, time_count))
        mu = np.zeros((2, 2 * length, time_count))
        measured_alpha = alpha.copy()
        measured_alpha[4, 6, 0, 0, 0] = 2.0
        applied = {"alpha": alpha, "beta": beta, "mu_cdw": mu}
        measured = {"alpha": measured_alpha, "beta": beta.copy(), "mu_cdw": mu.copy()}
        residual = AUDIT.residual_by_rung(applied, measured)["normalized_total"][0]
        self.assertEqual(int(np.argmax(residual)) + 1, 6)
        self.assertAlmostEqual(float(np.sum(residual)), 1.0)

    def test_same_phase_residual_removes_exact_period_two_contrast(self):
        length = 6
        time_count = 6
        alpha = np.zeros((length, length, 2, 2, time_count))
        beta = np.zeros((2, length, length, 2, 2, time_count))
        mu = np.zeros((2, 2 * length, time_count))
        mu[:, :, 1::2] = 1.0
        fields = {"alpha": alpha, "beta": beta, "mu_cdw": mu}
        same_phase = AUDIT.same_phase_residual_by_rung(fields, 2)
        self.assertTrue(np.allclose(same_phase["global_relative"][2:], 0.0))
        raw = AUDIT.residual_by_rung(
            {name: values[..., :-1] for name, values in fields.items()},
            {name: values[..., 1:] for name, values in fields.items()},
        )
        self.assertTrue(np.all(np.sum(raw["total_mass"], axis=1) > 0.0))


if __name__ == "__main__":
    unittest.main()
