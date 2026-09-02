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
