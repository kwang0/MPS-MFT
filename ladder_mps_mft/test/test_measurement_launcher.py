"""Local fake-scheduler tests. No Perlmutter access or real Slurm calls."""
import csv
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
        for name in ('phase1_gpu.sh', 'measure_latest_campaigns.sh'):
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
      printf '%s\\tcampaign\\tlabel%s\\tconfig\\tcfgsha\\tcompact\\tcmpsha\\tstate\\tsha\\tfingerprint\\tmaximum_iterations\\tfalse\\t60\\t%s\\n' "$i" "$i" "$n" >>"$out/manifest.tsv"
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
        for name in ('measure_latest_campaigns.sh', 'phase1_gpu.sh'):
            result = subprocess.run([BASH, '-n', (ROOT/'slurm'/name).as_posix()], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

if __name__ == '__main__':
    unittest.main()
