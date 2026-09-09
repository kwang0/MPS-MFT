"""Package a dated local source snapshot and reference correlations for handoff."""
import hashlib
import json
from pathlib import Path
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "output/source_bundles/two_basin_raw_20260908_eps005_iter80.zip"
REFERENCE = PROJECT / "output/seed_previews/20260908_square_two_basin/references.h5"


def digest(data): return hashlib.sha256(data).hexdigest()


def main():
    files = {name: PROJECT / name for name in ("Project.toml", "Manifest.toml", "AGENTS.md")}
    for directory in ("src", "ext", "scripts", "configs", "slurm", "data", "gpu", "test"):
        for path in (PROJECT / directory).rglob("*"):
            if path.is_file() and path.suffix in (".jl", ".py", ".toml", ".csv", ".sh"):
                relative = path.relative_to(PROJECT)
                if any(part.startswith(".") or part == "__pycache__" for part in relative.parts): continue
                if path.name.startswith("_verify_"): continue
                if path.name == "LocalPreferences.toml": continue
                files[relative.as_posix()] = path
    for name in ("README.md", "PROJECT_STATE.md", "ARCHITECTURE.md", "CONVERGENCE.md", "SEEDING.md", "RUN_LOG.md", "plans/ACTIVE.md",
                 "VARIATIONAL_FUNCTIONAL.md", "PHASE1_NUMERICAL_ERROR_BUDGET.md"):
        files["docs/" + name] = PROJECT / "docs" / name
    for name in ("README.md", "seed_profiles.png", "seed_profiles.pdf", "seed_amplitudes.csv"):
        relative = "docs/reports/two_basin_raw_20260908/" + name
        files[relative] = PROJECT / relative
    files["references.h5"] = REFERENCE
    contents = {name: path.read_bytes() for name, path in sorted(files.items())}
    # Linux shell scripts in the handoff must not inherit Windows line endings.
    for name in contents:
        if name.endswith(".sh"): contents[name] = contents[name].replace(b"\r\n", b"\n")
    manifest = {name: dict(bytes=len(data), sha256=digest(data)) for name, data in contents.items()}
    contents["SOURCE_MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    staging = OUT.with_suffix(".zip.partial")
    with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in contents.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 8, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    with zipfile.ZipFile(staging) as z:
        assert z.testzip() is None
        for name, metadata in manifest.items():
            assert digest(z.read(name)) == metadata["sha256"]
    checksum = digest(staging.read_bytes())
    if OUT.exists() and digest(OUT.read_bytes()) != checksum:
        raise FileExistsError("The dated bundle already exists with different contents; use a new snapshot name.")
    staging.replace(OUT)
    OUT.with_suffix(".sha256").write_text(checksum + "  " + OUT.name + "\n", encoding="ascii")
    OUT.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(path=str(OUT), sha256=checksum, files=len(manifest), bytes=OUT.stat().st_size), indent=2))


if __name__ == "__main__": main()
