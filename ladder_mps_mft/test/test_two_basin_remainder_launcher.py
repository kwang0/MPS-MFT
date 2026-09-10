"""Exercise only a fake launcher; never invoke Slurm or access Perlmutter."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = str(Path('C:/Program Files/Git/bin/bash.exe')) if os.name == 'nt' else shutil.which('bash')


@unittest.skipUnless(BASH and Path(BASH).is_file(), 'local Bash unavailable')
class RemainderLauncherTest(unittest.TestCase):
    def test_shared_environment_and_exact_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'original checkout' / 'ladder_mps_mft'
            new = root / 'new checkout' / 'ladder_mps_mft'
            (new / 'slurm').mkdir(parents=True)
            shutil.copyfile(ROOT / 'slurm/submit_square_two_basin_remainder.sh', new / 'slurm/submit.sh')
            anchor = original / 'output/phase1_gpu/20260908_square_two_basin_95_5_80_anchors'
            anchor.mkdir(parents=True)
            environment = "\n".join([
                'PHASE1_PROJECT_DIR=old_source', 'PHASE1_REPO_ROOT=old_repo',
                'PHASE1_RUN_ROOT=shared_runs', 'PHASE1_SCRATCH_ROOT=shared_scratch',
                'PHASE1_BUDGET_ROOT=shared_budget', 'PHASE1_LEDGER_PATH=shared_ledger',
                'PHASE1_RECONCILIATION_PATH=shared_reconciliation', 'PHASE1_ACCOUNT=existing_account',
                'PHASE1_GPU_TIME=12:00:00', 'PHASE1_ADDITIONAL_NODE_HOUR_CAP=400', ''])
            (anchor / 'run.env').write_text(environment, encoding='utf-8')
            fake = '''#!/bin/bash
set -euo pipefail
printf '%s|%s|%s|%s|%s|%s|%s|%s|%s\\n' "$*" "$PHASE1_PROJECT_DIR" "$PHASE1_TWO_BASIN_CONFIG" "$PHASE1_RUN_ROOT" "$PHASE1_SCRATCH_ROOT" "$PHASE1_LEDGER_PATH" "$PHASE1_ACCOUNT" "$PHASE1_GPU_TIME" "$PHASE1_ADDITIONAL_NODE_HOUR_CAP" >> "$TEST_RECEIPT"
'''
            (new / 'slurm/phase1_gpu.sh').write_text(fake, encoding='utf-8', newline='\n')
            receipt = root / 'calls.txt'
            env = dict(os.environ, TWO_BASIN_ORIGINAL_PROJECT=original.as_posix(), TEST_RECEIPT=receipt.as_posix())
            result = subprocess.run([BASH, (new / 'slurm/submit.sh').as_posix()], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = [line.split('|') for line in receipt.read_text().splitlines()]
            self.assertEqual(len(calls), 3)
            self.assertTrue(calls[0][0].startswith('prepare-square-two-basin-raw '))
            self.assertTrue(calls[0][0].endswith('20260910_square_two_basin_95_5_40_remainder remainder'))
            self.assertEqual(calls[1][0], 'reconcile')
            self.assertEqual(calls[2][0], 'submit 20260910_square_two_basin_95_5_40_remainder')
            for call in calls:
                self.assertTrue(call[1].endswith('/new checkout/ladder_mps_mft'))
                self.assertTrue(call[2].endswith('/new checkout/ladder_mps_mft/configs/phase1_gpu_square_two_basin_chi200_raw40.toml'))
                self.assertEqual(call[3:], ['shared_runs', 'shared_scratch', 'shared_ledger', 'existing_account', '12:00:00', '400'])
            self.assertEqual((anchor / 'run.env').read_text(), environment)
            # Reject running the new source in the checkout used by pending jobs.
            (original / 'slurm').mkdir()
            shutil.copyfile(new / 'slurm/submit.sh', original / 'slurm/submit.sh')
            blocked = subprocess.run([BASH, (original / 'slurm/submit.sh').as_posix()], env=env, capture_output=True, text=True)
            self.assertNotEqual(blocked.returncode, 0)
            self.assertIn('separate checkout', blocked.stderr)
            self.assertEqual(len(receipt.read_text().splitlines()), 3)


if __name__ == '__main__':
    unittest.main()
