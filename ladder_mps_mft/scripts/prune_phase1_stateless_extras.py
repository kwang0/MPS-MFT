#!/usr/bin/env python3
"""Safely prune redundant checkpoint/orbit files from Phase 1 stateless mirrors.

The authoritative full MPS artifacts are never modified. By default this tool
is a read-only planner. Applying a plan requires an explicit recovery boundary:
``--require-full`` verifies every referenced scratch artifact, while
``--local-only`` declares that the disposable local mirror may be regenerated
from its recorded full source later.

Final ``state.h5`` files, diagnostics, summaries, manifests, configs, and logs
are retained. Only compact ``checkpoint_best.h5``, ``checkpoint_latest.h5``,
and ``orbit_period_*_iter_*.h5`` files with a verified sibling ``state.h5`` are
eligible for removal.
"""

import argparse
import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Set, Tuple


MANIFEST_NAME = "stateless_manifest.tsv"
EXPECTED_HEADER = (
    "relative_path",
    "kind",
    "full_path",
    "full_sha256",
    "full_bytes",
    "compact_path",
    "compact_sha256",
    "compact_bytes",
    "omitted_paths",
)
CHECKPOINT_NAMES = {"checkpoint_best.h5", "checkpoint_latest.h5"}
ORBIT_NAME = re.compile(r"^orbit_period_[0-9]+_iter_[0-9]+\.h5$")


class ManifestRow:
    def __init__(self, values: Tuple[str, ...]) -> None:
        self.values = values

    @property
    def relative_path(self) -> str:
        return self.values[0]

    @property
    def kind(self) -> str:
        return self.values[1]

    @property
    def full_path(self) -> str:
        return self.values[2]

    @property
    def full_sha256(self) -> str:
        return self.values[3].lower()

    @property
    def full_bytes(self) -> int:
        return int(self.values[4])

    @property
    def compact_sha256(self) -> str:
        return self.values[6].lower()

    @property
    def compact_bytes(self) -> int:
        return int(self.values[7])


class PrunePlan:
    def __init__(
        self,
        manifest: Path,
        rows: Tuple[ManifestRow, ...],
        removable: Tuple[ManifestRow, ...],
    ) -> None:
        self.manifest = manifest
        self.rows = rows
        self.removable = removable

    @property
    def retained(self) -> Tuple[ManifestRow, ...]:
        removed = {row.relative_path for row in self.removable}
        return tuple(row for row in self.rows if row.relative_path not in removed)

    @property
    def current_bytes(self) -> int:
        return sum(row.compact_bytes for row in self.rows)

    @property
    def removed_bytes(self) -> int:
        return sum(row.compact_bytes for row in self.removable)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def local_path(manifest: Path, relative_path: str) -> Path:
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        raise ValueError(f"unsafe relative path in {manifest}: {relative_path!r}")
    candidate = manifest.parent.joinpath(*pure.parts)
    resolved_parent = candidate.parent.resolve(strict=True)
    root = manifest.parent.resolve(strict=True)
    if not is_relative_to(resolved_parent, root):
        raise ValueError(f"manifest artifact escapes its root: {relative_path}")
    if candidate.is_symlink():
        raise ValueError(f"refusing symlinked compact artifact: {candidate}")
    return candidate


def read_manifest(manifest: Path) -> Tuple[ManifestRow, ...]:
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        try:
            header = tuple(next(reader))
        except StopIteration as error:
            raise ValueError(f"empty stateless manifest: {manifest}") from error
        if header != EXPECTED_HEADER:
            raise ValueError(f"unexpected stateless manifest header: {manifest}")
        rows = []  # type: List[ManifestRow]
        seen = set()  # type: Set[str]
        for line_number, values in enumerate(reader, start=2):
            if len(values) != len(EXPECTED_HEADER):
                raise ValueError(f"malformed row {line_number} in {manifest}")
            row = ManifestRow(tuple(values))
            if row.relative_path in seen:
                raise ValueError(f"duplicate relative path in {manifest}: {row.relative_path}")
            seen.add(row.relative_path)
            rows.append(row)
    if not rows:
        raise ValueError(f"manifest has no artifacts: {manifest}")
    return tuple(rows)


def verify_compact(manifest: Path, row: ManifestRow) -> Path:
    path = local_path(manifest, row.relative_path)
    if not path.is_file():
        raise ValueError(f"compact artifact is missing: {path}")
    if path.stat().st_size != row.compact_bytes:
        raise ValueError(f"compact artifact size mismatch: {path}")
    if sha256_file(path) != row.compact_sha256:
        raise ValueError(f"compact artifact SHA-256 mismatch: {path}")
    return path


def verify_full(row: ManifestRow) -> None:
    path = Path(row.full_path)
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"full artifact is unavailable: {path}")
    if path.is_symlink():
        path = path.resolve(strict=True)
    if path.stat().st_size != row.full_bytes:
        raise ValueError(f"full artifact size mismatch: {path}")
    if sha256_file(path) != row.full_sha256:
        raise ValueError(f"full artifact SHA-256 mismatch: {path}")


def is_prunable(row: ManifestRow) -> bool:
    name = PurePosixPath(row.relative_path).name
    return row.kind == "stateless_hdf5" and (
        name in CHECKPOINT_NAMES or ORBIT_NAME.fullmatch(name) is not None
    )


def build_plan(manifest: Path, require_full: bool) -> PrunePlan:
    rows = read_manifest(manifest)
    by_path = {row.relative_path: row for row in rows}
    for row in rows:
        verify_compact(manifest, row)
        if require_full:
            verify_full(row)

    removable = []  # type: List[ManifestRow]
    for row in rows:
        if not is_prunable(row):
            continue
        state_path = str(PurePosixPath(row.relative_path).with_name("state.h5"))
        state = by_path.get(state_path)
        if state is None or state.kind != "stateless_hdf5":
            raise ValueError(
                f"refusing to prune {row.relative_path}: verified sibling state.h5 is absent"
            )
        removable.append(row)
    return PrunePlan(manifest, rows, tuple(removable))


def write_manifest(path: Path, rows: Iterable[ManifestRow]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
            writer.writerow(EXPECTED_HEADER)
            for row in rows:
                writer.writerow(row.values)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_julia_verifier(project_root: Path, manifest: Path, julia: str) -> None:
    verifier = project_root / "scripts" / "verify_stateless_results.jl"
    command = [
        julia,
        "--startup-file=no",
        f"--project={project_root}",
        str(verifier),
        str(manifest.parent),
    ]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise ValueError(f"Julia stateless verifier failed for {manifest.parent}")


def apply_plan(plan: PrunePlan, timestamp: str, require_full: bool) -> None:
    if not plan.removable:
        return
    backup = plan.manifest.with_name(
        f"stateless_manifest.before-prune-{timestamp}.tsv"
    )
    if backup.exists():
        raise ValueError(f"manifest backup already exists: {backup}")
    shutil.copy2(plan.manifest, backup)

    # Install the smaller manifest first. If a later unlink fails, the extra
    # untracked compact file is harmless and a subsequent run can report it.
    write_manifest(plan.manifest, plan.retained)
    for row in plan.removable:
        local_path(plan.manifest, row.relative_path).unlink()

    validated = build_plan(plan.manifest, require_full=require_full)
    if validated.removable:
        raise RuntimeError(f"pruned manifest still contains removable rows: {plan.manifest}")


def resolve_run(project_root: Path, value: str) -> Path:
    allowed_root = (project_root / "output" / "phase1_gpu").resolve(strict=True)
    supplied = Path(value)
    if not supplied.is_absolute():
        direct = (Path.cwd() / supplied)
        supplied = direct if direct.exists() else allowed_root / supplied
    run = supplied.resolve(strict=True)
    if not run.is_dir() or not is_relative_to(run, allowed_root):
        raise ValueError(f"run must be a directory below {allowed_root}: {value}")
    return run


def format_mib(value: int) -> str:
    return f"{value / (1024 ** 2):.3f} MiB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs",
        nargs="+",
        help="Phase 1 run IDs or directories below output/phase1_gpu",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="apply the verified plan; omission is a read-only dry run",
    )
    recovery = parser.add_mutually_exclusive_group()
    recovery.add_argument(
        "--require-full",
        action="store_true",
        help="verify every recorded full artifact before applying (use on Perlmutter)",
    )
    recovery.add_argument(
        "--local-only",
        action="store_true",
        help="allow apply to a disposable local mirror without mounted scratch",
    )
    parser.add_argument(
        "--julia",
        default="julia",
        help="Julia executable used for mandatory HDF5/MPS checks during --apply",
    )
    args = parser.parse_args()
    if args.apply and not (args.require_full or args.local_only):
        parser.error("--apply requires either --require-full or --local-only")

    project_root = Path(__file__).resolve().parent.parent
    runs = [resolve_run(project_root, value) for value in args.runs]
    manifests = []  # type: List[Path]
    for run in runs:
        found = sorted(run.rglob(MANIFEST_NAME))
        if not found:
            raise ValueError(f"no {MANIFEST_NAME} found below {run}")
        manifests.extend(found)
    if len(set(manifests)) != len(manifests):
        raise ValueError("the supplied runs overlap and select a manifest more than once")

    plans = [build_plan(path, require_full=args.require_full) for path in manifests]
    current = sum(plan.current_bytes for plan in plans)
    removed = sum(plan.removed_bytes for plan in plans)
    retained = current - removed
    print(f"mode={'apply' if args.apply else 'dry-run'}")
    print(f"recovery_boundary={'full-verified' if args.require_full else 'local-only' if args.local_only else 'plan-only'}")
    for plan in plans:
        print(
            "manifest="
            f"{plan.manifest} artifacts={len(plan.rows)} "
            f"remove={len(plan.removable)} savings={format_mib(plan.removed_bytes)}"
        )
    print(f"current_compact_payload={format_mib(current)}")
    print(f"projected_compact_payload={format_mib(retained)}")
    print(f"projected_savings={format_mib(removed)}")

    if not args.apply:
        print("no files changed; add --apply with an explicit recovery boundary to prune")
        return 0

    for plan in plans:
        run_julia_verifier(project_root, plan.manifest, args.julia)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for plan in plans:
        apply_plan(plan, timestamp, require_full=args.require_full)
        run_julia_verifier(project_root, plan.manifest, args.julia)
    print("prune_complete=true")
    print("full_artifacts_modified=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
