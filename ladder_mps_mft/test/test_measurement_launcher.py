"""Local fake-scheduler tests. No Perlmutter access or real Slurm calls."""
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = 'C:/Program Files/Git/bin/bash.exe' if os.name == 'nt' else shutil.which('bash')

@unittest.skipUnless(BASH and Path(BASH).is_file(), 'Bash unavailable')
class MeasurementLauncher(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ('slurm', 'src', 'scripts', 'bin'):
            (self.root/name).mkdir()
        for name in ('phase1_gpu.sh', 'measure_latest_campaigns.sh', 'retry_latest_correlations.sh'):
            shutil.copyfile(ROOT/'slurm'/name, self.root/'slurm'/name)
        self.write('slurm/phase0_calibrate_cpu.sh', '#!/bin/bash\n[[ "$1" == plan ]]\n')
        self.write('Project.toml', '')
        self.write('Manifest.toml', '')
        self.write('src/test.jl', '# frozen source\n')
        self.write('bin/julia', '''#!/bin/bash
set -euo pipefail
while (( $# )); do
  if [[ "$1" == prepare ]]; then
    out="$3"; mkdir -p "$out"
    printf 'index\\tcampaign\\tlabel\\tconfig_path\\tconfig_sha256\\tcompact_path\\tcompact_sha256\\tsource_path\\tsource_sha256\\tmodel_fingerprint\\tstatus\\taccepted\\titeration\\tsamples\\n' >"$out/manifest.tsv"
    for i in $(seq 1 56); do
      n=1; (( i <= 54 )) || n=2
      campaign=campaign; [[ "${TEST_MANIFEST_MISMATCH:-0}" != 1 ]] || campaign=changed
      printf '%s\\t%s\\tlabel%s\\tconfig\\tcfgsha\\tcompact\\tcmpsha\\tstate\\tsha\\tfingerprint\\tmaximum_iterations\\tfalse\\t60\\t%s\\n' "$i" "$campaign" "$i" "$n" >>"$out/manifest.tsv"
    done
    exit 0
  fi
  shift
done
''')
        self.write('bin/sbatch', '''#!/bin/bash
set -euo pipefail
n=0; [[ ! -f "$TEST_ROOT/count" ]] || n=$(<"$TEST_ROOT/count")
n=$((n+1)); printf '%s' "$n" >"$TEST_ROOT/count"
printf '%s\\n' "$*" >>"$TEST_ROOT/submitted"
printf '%s\\n' "$((9000+n))"
''')
        self.write('bin/module', '#!/bin/bash\n[[ "$*" == "load julia" ]]\n')
        self.write('bin/srun', '''#!/bin/bash
set -euo pipefail
printf '%s\\n' "$@" >"$TEST_ROOT/worker_arguments"
printf '%s %s %s %s\\n' "$JULIA_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$MKL_NUM_THREADS" "$OMP_NUM_THREADS" >"$TEST_ROOT/worker_threads"
exit "${TEST_SRUN_EXIT:-0}"
''')
        self.write('bin/sacct', '''#!/bin/bash
set -euo pipefail
while (( $# )); do
  if [[ "$1" == -j ]]; then
    printf '%s\\n' "$2" >>"$TEST_ROOT/accounting_queries"
    printf '%s|%s|1|2026-09-18T00:00:00|2026-09-18T00:00:01\\n' "$2" "${TEST_SACCT_STATE:-FAILED}"
    exit 0
  fi
  shift
done
exit 1
''')
        self.env = dict(os.environ,
            PHASE1_PROJECT_DIR=self.root.as_posix(),
            PHASE1_JULIA=(self.root/'bin/julia').as_posix(),
            PHASE1_BUDGET_ROOT=(self.root/'budget').as_posix(),
            PHASE1_ADDITIONAL_NODE_HOUR_CAP='400',
            DIAGNOSTICS_ROOT=(self.root/'measurements').as_posix(),
            TEST_ROOT=self.root.as_posix())

    def write(self, name, content):
        path = self.root/name
        path.write_text(content, newline='\n')
        path.chmod(0o755)

    def run_launcher(self, action='submit', **overrides):
        env = dict(self.env, **overrides)
        command = 'test_root="$(cd "$TEST_ROOT" && pwd)"; export PATH="$test_root/bin:/usr/bin:$PATH"; exec bash "$test_root/slurm/measure_latest_campaigns.sh" "$1" fixture_measurements'
        return subprocess.run([BASH, '-c', command, '_', action], env=env,
                              text=True, capture_output=True, timeout=60)

    def run_retry(self, action='plan', **overrides):
        command = 'test_root="$(cd "$TEST_ROOT" && pwd)"; export PATH="$test_root/bin:/usr/bin:$PATH"; exec bash "$test_root/slurm/retry_latest_correlations.sh" "$1" fixture_measurements fixture_retry'
        return subprocess.run([BASH, '-c', command, '_', action],
                              env=dict(self.env, **overrides), text=True,
                              capture_output=True, timeout=60)

    def prepare_failed_campaign(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        parent = self.root/'measurements/fixture_measurements'
        with (parent/'jobs.tsv').open() as stream:
            jobs = list(csv.DictReader(stream, delimiter='\t'))
        for job in jobs:
            job_id = job['job_id']
            (parent/f"logs/{job['label']}-{job_id}.out").write_text(
                f'/var/spool/slurmd/job{job_id}/slurm_script: line 5: '
                f'/var/spool/slurmd/job{job_id}/phase1_gpu.sh: No such file or directory\n',
                newline='\n')
        return parent

    def run_spooled_worker(self, **overrides):
        command = '''test_root="$(cd "$TEST_ROOT" && pwd)"
export PATH="$test_root/bin:/usr/bin:$PATH"
exec bash "$test_root/spool/slurm_script" _run "$test_root/measurements/fixture_measurements" 1'''
        return subprocess.run([BASH, '-c', command], env=dict(self.env, **overrides),
                              text=True, capture_output=True, timeout=30)

    def test_worker_runs_from_spool_without_sibling_scripts(self):
        self.prepare_failed_campaign()
        spool = self.root/'spool'
        spool.mkdir()
        shutil.copyfile(self.root/'measurements/fixture_measurements/source/slurm/measure_latest_campaigns.sh', spool/'slurm_script')
        self.assertFalse((spool/'phase1_gpu.sh').exists())
        result = self.run_spooled_worker()
        self.assertEqual(result.returncode, 0, result.stderr)
        arguments = (self.root/'worker_arguments').read_text()
        self.assertIn('--project=', arguments)
        self.assertIn('/measurements/fixture_measurements/source', arguments)
        self.assertIn('/source/scripts/measure_latest_campaigns.jl', arguments)
        self.assertIn('run\n', arguments)
        self.assertEqual((self.root/'worker_threads').read_text().strip(), '4 1 1 1')
        self.assertEqual(self.run_spooled_worker(TEST_SRUN_EXIT='17').returncode, 17)
        (self.root/'worker_arguments').unlink()
        with (self.root/'measurements/fixture_measurements/source/src/test.jl').open('a') as stream:
            stream.write('# changed\n')
        self.assertNotEqual(self.run_spooled_worker().returncode, 0)
        self.assertFalse((self.root/'worker_arguments').exists())

    def test_retry_preserves_parent_and_running_square_reservations(self):
        parent = self.prepare_failed_campaign()
        before = {p.relative_to(parent): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in parent.rglob('*') if p.is_file()}
        ledger = self.root/'budget/additional_node_hours.tsv'
        with ledger.open('a', newline='') as stream:
            stream.write('2026-09-20T00:00:00Z\tongoing_square_AB\tbranch\tA\t1\tgpu\t4\t99999\t16:00:00\toriginal\n')
        result = self.run_retry()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.root/'accounting_queries').exists())
        self.assertFalse((self.root/'measurements/fixture_retry').exists())
        result = self.run_retry('submit')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root/'count').read_text(), '112')
        queried = (self.root/'accounting_queries').read_text().splitlines()
        self.assertEqual(len(queried), 56)
        self.assertNotIn('99999', queried)
        new = self.root/'measurements/fixture_retry'
        self.assertEqual((new/'manifest.tsv').read_bytes(), (parent/'manifest.tsv').read_bytes())
        self.assertTrue((new/'retry_parent_manifest.sha256').is_file())
        with ledger.open() as stream:
            reservations = list(csv.DictReader(stream, delimiter='\t'))
        self.assertEqual(len(reservations), 113)
        self.assertEqual(sum(r['campaign']=='ongoing_square_AB' for r in reservations), 1)
        with (self.root/'budget/additional_node_hours_reconciliations.tsv').open() as stream:
            released = list(csv.DictReader(stream, delimiter='\t'))
        self.assertEqual(len(released), 56)
        self.assertTrue(all(r['campaign']=='fixture_measurements' for r in released))
        result = self.run_retry('submit')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root/'count').read_text(), '112')
        after = {p.relative_to(parent): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in parent.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_retry_requires_terminal_accounting(self):
        self.prepare_failed_campaign()
        result = self.run_retry('submit', TEST_SACCT_STATE='RUNNING')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('lacks reconciled failure accounting', result.stderr)
        self.assertEqual((self.root/'count').read_text(), '56')
        self.assertFalse((self.root/'measurements/fixture_retry').exists())

    def test_retry_blocks_changed_inventory_before_submission(self):
        self.prepare_failed_campaign()
        result = self.run_retry('submit', TEST_MANIFEST_MISMATCH='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('retry inventory differs', result.stderr)
        self.assertEqual((self.root/'count').read_text(), '56')

    def test_retry_retains_shared_project_cap(self):
        self.prepare_failed_campaign()
        ledger = self.root/'budget/additional_node_hours.tsv'
        with ledger.open('a', newline='') as stream:
            stream.write('2026-09-20T00:00:00Z\tongoing_square_AB\tbranch\tA\t1\tgpu\t395\t99999\t16:00:00\toriginal\n')
        result = self.run_retry('submit')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exceed cap', result.stderr)
        self.assertEqual((self.root/'count').read_text(), '56')

    def test_retry_rejects_outputs_or_unrecognized_failure(self):
        parent = self.prepare_failed_campaign()
        (parent/'results').mkdir()
        result = self.run_retry()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('parent has measurement outputs', result.stderr)
        (parent/'results').rmdir()
        (parent/'logs/1-9001.out').write_text('some different failure\n')
        result = self.run_retry()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('does not match the verified startup failure', result.stderr)
        self.assertEqual((self.root/'count').read_text(), '56')

    def test_budget_snapshot_and_duplicate_guard(self):
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        ledger = self.root/'budget/additional_node_hours.tsv'
        with ledger.open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        self.assertEqual(len(rows), 56)
        self.assertAlmostEqual(sum(float(r['reserved_node_hours']) for r in rows), 8.15625)
        self.assertEqual(sum(r['requested_time'] == '04:00:00' for r in rows), 2)
        self.assertTrue(all(r['pool']=='cpu' for r in rows))
        submitted = (self.root/'submitted').read_text()
        self.assertNotIn('--gpus', submitted)
        self.assertIn('--constraint=cpu', submitted)
        self.assertIn('/source/slurm/measure_latest_campaigns.sh', submitted)
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root/'count').read_text(), '56')
        source = self.root/'measurements/fixture_measurements/source/src/test.jl'
        source.write_text('# changed\n')
        self.assertNotEqual(self.run_launcher().returncode, 0)
        self.assertEqual((self.root/'count').read_text(), '56')

    def test_project_cap_blocks_before_submission(self):
        result = self.run_launcher(PHASE1_ADDITIONAL_NODE_HOUR_CAP='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exceed cap', result.stderr)
        self.assertFalse((self.root/'count').exists())

    def test_syntax(self):
        for name in ('measure_latest_campaigns.sh', 'retry_latest_correlations.sh', 'phase1_gpu.sh'):
            result = subprocess.run([BASH, '-n', (ROOT/'slurm'/name).as_posix()], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == '__main__':
    unittest.main()
