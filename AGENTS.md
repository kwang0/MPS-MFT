# Repository operating rules

## Local and Perlmutter host boundary

1. This Codex workspace is local Windows. Use PowerShell-compatible commands
   locally and do not assume Bash is installed.
2. Perlmutter is a separate Linux system. Codex must never authenticate to or
   log in to NERSC/Perlmutter from this workspace. Do not open authentication
   flows or run `ssh` against NERSC hosts.
3. The user always performs every transfer between this repository and
   Perlmutter and every live scheduler or allocation action. Codex must not run
   `scp`, `sftp`, `rsync`, Globus transfers, `sbatch`, `squeue`, `sacct`,
   `scancel`, or equivalent operations against Perlmutter.
4. Codex may prepare and validate code, configs, launchers, source bundles, and
   analysis locally. Commands intended for Perlmutter must be clearly labeled
   as handoff commands for the user to run; do not execute them from here.
5. Analyze Perlmutter results only after the user has synchronized the relevant
   logs and artifacts into the local workspace.

## User-managed Perlmutter checkout

- The ladder subproject checkout on Perlmutter is
  `$CFS/m4863/MPS-MFT/ladder_mps_mft`.
- Perlmutter handoff commands for the ladder workflow should use
  `cd "$CFS/m4863/MPS-MFT/ladder_mps_mft"` as their starting directory.
- This path is reference information for commands the user runs; it does not
  authorize Codex to connect to Perlmutter or perform synchronization or
  scheduler actions.

## Scope and validation discipline

1. Use the smallest workflow that directly completes the user's request. If a
   requested script, analysis, or implementation already exists, inspect and
   reuse it; do not silently turn an incremental request into a redesign or
   production-readiness audit.
2. Before materially expanding scope, state what additional work is proposed,
   why it is necessary, and its likely wall-clock cost. Obtain the user's
   approval before proceeding unless the extra work is required for immediate
   correctness or safety. When possible, report out-of-scope improvements as
   recommendations instead of implementing them.
3. Prefer focused validation while developing. Run an expensive full suite at
   most once before handoff and only when relevant source changes justify it.
   Documentation, report, configuration, or resource-request edits do not by
   themselves justify rerunning unrelated DMRG tests.
4. Do not repeat an unchanged expensive test merely for reassurance. If a
   late change affects only a narrow component, rerun that component's focused
   checks. Before starting a local command expected to take more than five
   minutes, tell the user what will run and the expected duration.
5. Optimize for prompt user feedback as well as correctness. Distinguish time
   spent executing code from time spent reviewing or expanding the work, and
   surface unexpected delays or scope growth while they are happening.
