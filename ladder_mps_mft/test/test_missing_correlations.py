"""Selective timeout retry tests using local fake Slurm and measurement status."""
import csv
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import unittest

import test_measurement_launcher as fixtures

MISSING = (6, 11, 31, 34, 46, 51, 53)


@unittest.skipUnless(fixtures.BASH and Path(fixtures.BASH).is_file(), 'Bash unavailable')
class MissingCorrelations(unittest.TestCase):
    write = fixtures.MeasurementLauncher.write
    run_launcher = fixtures.MeasurementLauncher.run_launcher

    def setUp(self):
        fixtures.MeasurementLauncher.setUp(self)
        shutil.copyfile(fixtures.ROOT/'slurm/complete_missing_correlations.sh',
                        self.root/'slurm/complete_missing_correlations.sh')
        fake_julia = (self.root/'bin/julia').read_text()
        fake_julia = fake_julia.replace('  shift\ndone', '''  if [[ "$1" == status ]]; then
    if [[ "$2" == */fixture_measurements ]]; then
      for i in $(seq 1 56); do
        state=MEASURED
        case "$i" in 6|11|31|34|46|51|53) state=MISSING;; esac
        [[ "$i" != "${TEST_EXTRA_MISSING:-}" ]] || state=MISSING
        [[ "$i" != "${TEST_COMPLETED_ROW:-}" ]] || state=MEASURED
        printf '%s\\t%s\\tcampaign/label%s\\n' "$i" "$state" "$i"
      done
      echo complete=49/56
    else
      n=0
      for i in 6 11 31 34 46 51 53; do
        n=$((n+1)); printf '%s\\tMEASURED\\tcampaign/label%s\\n' "$n" "$i"
      done
      echo complete=7/7
    fi
    exit 0
  fi
  shift
done''')
        self.write('bin/julia', fake_julia)
        accounting = (self.root/'bin/sacct').read_text().replace(
            '    printf \'%s|%s|1|', '''    state=COMPLETED
    case "$2" in 9006|9011|9031|9034|9046|9051|9053) state="${TEST_SACCT_STATE:-TIMEOUT}";; esac
    printf '%s|%s|1|''').replace('"${TEST_SACCT_STATE:-FAILED}"', '"$state"')
        self.write('bin/sacct', accounting)
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.parent = self.root/'measurements/fixture_measurements'
        self.new = self.root/'measurements/fixture_retry2'
        self.write('config.toml', '# immutable config\n')
        self.write('state.h5', 'immutable MPS fixture\n')
        with (self.parent/'manifest.tsv').open() as stream:
            rows = list(csv.DictReader(stream, delimiter='\t'))
        for row in rows:
            row['config_path'] = (self.root/'config.toml').as_posix()
            row['config_sha256'] = hashlib.sha256((self.root/'config.toml').read_bytes()).hexdigest()
            row['source_path'] = (self.root/'state.h5').as_posix()
        with (self.parent/'manifest.tsv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        digest = hashlib.sha256((self.parent/'manifest.tsv').read_bytes()).hexdigest()
        (self.parent/'manifest.sha256').write_text(
            f'{digest}  {(self.parent/"manifest.tsv").as_posix()}\n', newline='\n')
        # Preserve successful outputs and arbitrary partial files alike.
        for row in rows:
            output = self.parent/'results'/row['campaign']/row['label']
            output.mkdir(parents=True)
            (output/'diagnostics.h5').write_bytes(b'partial' if int(row['index']) in MISSING else b'completed')

    def run_missing(self, action='plan', **overrides):
        command = '''test_root="$(cd "$TEST_ROOT" && pwd)"
export PATH="$test_root/bin:/usr/bin:$PATH"
exec bash "$test_root/slurm/complete_missing_correlations.sh" "$1" fixture_measurements fixture_retry2'''
        return subprocess.run([fixtures.BASH, '-c', command, '_', action],
                              env=dict(self.env, **overrides), text=True,
                              capture_output=True, timeout=60)

    def parent_hashes(self):
        return {p.relative_to(self.parent): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in self.parent.rglob('*') if p.is_file()}

    def test_selective_submission_status_and_immutable_parent(self):
        before = self.parent_hashes()
        ledger = self.root/'budget/additional_node_hours.tsv'
        with ledger.open('a', newline='') as stream:
            stream.write('2026-09-21T00:00:00Z\tongoing_square_AB\tbranch\tA\t1\tgpu\t4\t99999\t16:00:00\toriginal\n')
        result = self.run_missing()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('1.968750000', result.stdout)
        self.assertFalse(self.new.exists())
        self.assertFalse((self.root/'accounting_queries').exists())
        result = self.run_missing('submit')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root/'count').read_text(), '63')
        with (self.new/'manifest.tsv').open() as stream:
            subset = list(csv.DictReader(stream, delimiter='\t'))
        with (self.parent/'manifest.tsv').open() as stream:
            original = list(csv.DictReader(stream, delimiter='\t'))
        for i, parent_index in enumerate(MISSING):
            self.assertEqual(subset[i], dict(original[parent_index-1], index=str(i+1)))
        with (self.new/'retry_rows.tsv').open() as stream:
            mapping = list(csv.DictReader(stream, delimiter='\t'))
        self.assertEqual([int(r['parent_index']) for r in mapping], list(MISSING))
        self.assertEqual([int(r['parent_job_id']) for r in mapping], [9000+i for i in MISSING])
        with (self.new/'jobs.tsv').open() as stream:
            jobs = list(csv.DictReader(stream, delimiter='\t'))
        self.assertEqual(len(jobs), 7)
        self.assertTrue(all(r['requested_time']=='04:00:00' for r in jobs))
        self.assertAlmostEqual(sum(float(r['reserved_node_hours']) for r in jobs), 1.96875)
        submitted = (self.root/'submitted').read_text().splitlines()[-7:]
        self.assertTrue(all('--time=04:00:00' in s and '--mem=32768M' in s and '--cpus-per-task=8' in s for s in submitted))
        self.assertTrue(all('/fixture_retry2/source/slurm/measure_latest_campaigns.sh' in s for s in submitted))
        self.assertFalse((self.new/'results').exists())
        self.assertNotIn('99999', (self.root/'accounting_queries').read_text().splitlines())
        self.assertEqual(before, self.parent_hashes())
        result = self.run_missing('submit')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root/'count').read_text(), '63')
        result = self.run_missing('status', TEST_COMPLETED_ROW='6')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('combined_complete=56/56', result.stdout)
        # Tampering is rejected before an already submitted run can resume.
        with (self.new/'source/src/test.jl').open('a') as stream:
            stream.write('# changed\n')
        self.assertNotEqual(self.run_missing('submit').returncode, 0)
        self.assertEqual((self.root/'count').read_text(), '63')

    def test_running_job_changed_status_and_budget_block_submissions(self):
        result = self.run_missing(TEST_EXTRA_MISSING='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exactly the seven', result.stderr)
        result = self.run_missing(TEST_COMPLETED_ROW='6')
        self.assertNotEqual(result.returncode, 0)
        result = self.run_missing('submit', TEST_SACCT_STATE='RUNNING')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('lacks reconciled TIMEOUT', result.stderr)
        self.assertFalse(self.new.exists())
        ledger = self.root/'budget/additional_node_hours.tsv'
        with ledger.open('a', newline='') as stream:
            stream.write('2026-09-21T00:00:00Z\tongoing_square_AB\tbranch\tA\t1\tgpu\t399\t99999\t16:00:00\toriginal\n')
        result = self.run_missing('submit')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('exceed cap', result.stderr)
        self.assertEqual((self.root/'count').read_text(), '56')
        self.assertFalse(self.new.exists())

    def test_missing_source_and_changed_configuration_stop_preparation(self):
        (self.root/'state.h5').unlink()
        result = self.run_missing('submit')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('full MPS is unavailable', result.stderr)
        self.assertFalse(self.new.exists())
        self.write('state.h5', 'immutable MPS fixture\n')
        self.write('config.toml', '# changed\n')
        result = self.run_missing('submit')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.new.exists())
        self.assertEqual((self.root/'count').read_text(), '56')


if __name__ == '__main__':
    unittest.main()
