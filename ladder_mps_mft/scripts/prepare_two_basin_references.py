"""Extract the two user-selected correlation templates without modifying sources."""
import json
from pathlib import Path
import h5py
import numpy as np
from compile_square_grid import REPO, REPORT, sha

OUTPUT = REPO / "ladder_mps_mft/output/seed_previews/20260908_square_two_basin/references.h5"
HASHES = {
    "stripe": "ae6a3bfe76ca8f06f2396fd731b18bca8539e0b7ee68df016cc9156fdceeb074",
    "pairing": "8a1cf2d64d2fbe0eb59521192b829cab43e19a4d7ac026519ea847f6ac0778b8",
}


def main():
    selection = json.loads((REPORT / "selection.json").read_text())
    sources = {
        "stripe": next(r for r in selection["selected"] if r["t0"] == 1 and r["V"] == 0),
        "pairing": next(r for r in selection["candidates"] if r["t0"] == 1.4 and r["V"] == -.4
                        and r["branch"] == "square__pairing_dwave_m000_chi200_loose"),
    }
    payload = {}
    for family, row in sources.items():
        source = REPO / row["source"]
        assert sha(source) == HASHES[family] == row["source_sha256"]
        with h5py.File(source) as f:
            if family == "stripe":
                assert bool(f["completed"][()]) and not bool(f["period2_cycle_detected"][()])
                arrays = dict(zip(("pair", "exchange_down", "exchange_up"),
                                  [np.asarray(f[k + "_list"][-1]) for k in ("C_pair", "C_exc_dn", "C_exc_up")]))
                arrays["density_down"] = np.diag(arrays["exchange_down"]).copy()
                arrays["density_up"] = np.diag(arrays["exchange_up"]).copy()
            else:
                assert bool(f["accepted"][()]) and f["status"][()].decode() == "fixed_point"
                arrays = {k: np.asarray(f["correlations/" + k]) for k in
                          ("pair", "exchange_down", "exchange_up", "density_down", "density_up")}
            for key, values in arrays.items():
                assert np.isfinite(values).all()
                if np.iscomplexobj(values):
                    assert np.max(np.abs(values.imag)) < 1e-12
                payload[family + "/" + key] = np.asarray(values.real, dtype=np.float64)
        payload[family + "/source_sha256"] = np.bytes_(HASHES[family])
        payload[family + "/source_path"] = np.bytes_(row["source"])
        payload[family + "/t0"] = row["t0"]
        payload[family + "/V"] = row["V"]
    payload["artifact_kind"] = np.bytes_("two_basin_correlation_templates")
    payload["L"] = 64
    payload["chi"] = 200
    payload["density"] = .9375
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        with h5py.File(OUTPUT) as f:
            for key, values in payload.items():
                np.testing.assert_array_equal(f[key][()], values)
    else:
        with h5py.File(OUTPUT, "x") as f:
            for key, values in payload.items():
                f.create_dataset(key, data=values, track_times=False)
    print(json.dumps(dict(path=str(OUTPUT), sha256=sha(OUTPUT), source_hashes=HASHES), indent=2))


if __name__ == "__main__":
    main()
