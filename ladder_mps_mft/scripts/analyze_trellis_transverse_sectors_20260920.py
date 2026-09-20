"""Resolve spatial A/B charge sectors without assigning iteration parity to space.

Read-only input: the four hashed endpoints in the existing trellis report.
K_perp = pi here means odd under translation by one LADDER, not leg-odd
within a ladder. Physical longitudinal origins are A=0 and B=-1/2 rung.
The returned weights describe finite, nonstationary profiles, not a phase
classification or a transverse susceptibility.
"""

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "docs/reports/trellis_progress_20260918"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fourier(profiles, rungs, q, origin):
    values = profiles[:, rungs]
    values = values - values.mean(axis=1, keepdims=True)
    x = rungs + 1 + origin
    return values @ np.exp(-1j * q * x) / len(rungs)


def sector(a, b):
    even, odd = (a + b) / 2, (a - b) / 2
    denominator = np.abs(even) ** 2 + np.abs(odd) ** 2
    np.testing.assert_allclose(denominator, (np.abs(a) ** 2 + np.abs(b) ** 2) / 2,
                               rtol=1e-13, atol=1e-18)
    return even, odd, np.abs(odd) ** 2 / denominator


def main():
    source_report = OUT / "analysis.json"
    previous = json.loads(source_report.read_text())
    results = []
    q = 2 * np.pi * 4 / 64
    for row in previous["runs"]:
        path = PROJECT / row["source"]
        assert sha(path) == row["compact_sha256"]
        result = {k: row[k] for k in ("source", "compact_sha256", "job_id", "cell", "family",
                                      "accepted", "status", "cell_sweeps")}
        charge = {}
        result["within_ladder_charge"] = {}
        with h5py.File(path, "r") as f:
            assert int(f["model/L"][()]) == 64
            assert float(f["model/tau0"][()]) == float(f["model/tau1"][()]) == 0.1
            for label, ladder in f["ladders"].items():
                h = ladder["history"]
                c = h["correlations"]
                up, down = c["density_up"][()], c["density_down"][()]
                assert up.shape == down.shape == (60, 128)
                legs = (up + down).reshape(60, 64, 2)
                charge[label] = legs.mean(axis=2)
                odd_leg = (legs[:, :, 0] - legs[:, :, 1]) / 2
                central = slice(16, 48)
                result["within_ladder_charge"][label] = dict(
                    definition="charge_even=(n_leg0+n_leg1)/2; charge_odd=(n_leg0-n_leg1)/2; central rungs 17-48",
                    even_modulation_std_final=float(np.std(charge[label][-1, central])),
                    odd_rms_final=float(np.sqrt(np.mean(odd_leg[-1, central]**2))),
                    odd_modulation_std_final=float(np.std(odd_leg[-1, central])),
                    odd_max_abs_final=float(np.max(np.abs(odd_leg[-1, central]))))
                np.testing.assert_allclose(charge[label].mean(axis=1), h["density"][()],
                                           rtol=0, atol=1e-14)
                np.testing.assert_array_equal(c["density_up"][-1], ladder["correlations/density_up"][()])
                np.testing.assert_array_equal(c["density_down"][-1], ladder["correlations/density_down"][()])
        if row["cell"] == "two_ladder":
            windows = {}
            for label, rungs in (("full_rungs_1_64", np.arange(64)),
                                  ("central_rungs_17_48", np.arange(16, 48))):
                a = fourier(charge["A"], rungs, q, 0)
                b = fourier(charge["B"], rungs, q, -0.5)
                even, odd, fraction = sector(a, b)
                # Correcting B's origin multiplies its raw-index coefficient by exp(+iq/2).
                np.testing.assert_allclose(b, fourier(charge["B"], rungs, q, 0) * np.exp(1j*q/2),
                                           rtol=1e-13, atol=1e-15)
                phase = np.angle(b * a.conj(), deg=True)
                windows[label] = dict(
                    qx_radians=float(q), nominal_wavelength_rungs=16,
                    final=dict(A_amplitude=float(abs(a[-1])), B_amplitude=float(abs(b[-1])),
                               even_amplitude=float(abs(even[-1])), odd_amplitude=float(abs(odd[-1])),
                               odd_weight_fraction=float(fraction[-1]), B_minus_A_phase_degrees=float(phase[-1])),
                    last10=dict(odd_weight_fraction_min=float(min(fraction[-10:])),
                                odd_weight_fraction_max=float(max(fraction[-10:])),
                                B_minus_A_phase_degrees_min=float(min(phase[-10:])),
                                B_minus_A_phase_degrees_max=float(max(phase[-10:]))),
                    history=dict(iteration=list(range(1, 61)), odd_weight_fraction=fraction.tolist(),
                                 B_minus_A_phase_degrees=phase.tolist()))
            result["charge_sectors"] = windows
        assert sha(path) == row["compact_sha256"]
        results.append(result)

    # Algebraic checks of the spatial decomposition at arbitrary nonzero phase.
    synthetic = np.array([1 + 2j, -0.3 + 0.2j])
    np.testing.assert_allclose(sector(synthetic, synthetic)[2], 0, atol=1e-15)
    np.testing.assert_allclose(sector(synthetic, -synthetic)[2], 1, atol=1e-15)
    output = dict(date="2026-09-20", source_report=source_report.relative_to(PROJECT).as_posix(),
                  source_report_sha256=sha(source_report),
                  definition="n_l(q)=mean_x[(n_l(x)-mean(n_l))*exp(-iq*x)], physical x_A=i, x_B=i-1/2; n_even/odd=(n_A +/- n_B)/2",
                  limitation="A/B ladder parity only; both endpoints are unaccepted and nonstationary. A finite OBC Fourier projection is not an exact translation eigenstate.",
                  runs=results)
    (OUT / "transverse_sectors_20260920.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in results:
        print(row["cell"], row["family"], "within_ladder_charge", json.dumps(row["within_ladder_charge"]))
        if "charge_sectors" in row:
            print(row["family"], json.dumps({w: v["final"] for w, v in row["charge_sectors"].items()}))
    print("Verified four source hashes before/after, density normalization, endpoint alignment, origin phase and parity identities.")


if __name__ == "__main__":
    main()
