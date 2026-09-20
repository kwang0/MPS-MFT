"""Local syntax and fake-launcher checks. No Slurm command or remote access."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = Path('C:/Program Files/Git/bin/bash.exe') if os.name == 'nt' else Path(shutil.which('bash') or '/unavailable')

@unittest.skipUnless(BASH.is_file(), 'local Bash unavailable')
class SquareCellLauncher(unittest.TestCase):
    def test_syntax(self):
        for name in ('phase1_gpu.sh', 'submit_square_two_ladder.sh'):
            result = subprocess.run([str(BASH), '-n', (ROOT/'slurm'/name).as_posix()], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_shared_environment_and_failure_stops_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root/'current checkout'/'ladder_mps_mft'
            scripts = project/'slurm'
            scripts.mkdir(parents=True)
            anchor = project/'output/phase1_gpu/20260908_square_two_basin_95_5_80_anchors'
            anchor.mkdir(parents=True)
            old = '\n'.join(['PHASE1_RUN_SCRIPT_VERSION=1.19.0', 'PHASE1_RUN_SCRATCH_DIR=old',
                'PHASE1_ACCOUNT=saved_account', 'PHASE1_RUN_ROOT=shared_runs',
                'PHASE1_SCRATCH_ROOT=shared_scratch', 'PHASE1_LEDGER_PATH=shared_ledger',
                'PHASE1_RECONCILIATION_PATH=shared_reconciliation',
                'PHASE1_ADDITIONAL_NODE_HOUR_CAP=400', 'PHASE1_MAX_SEGMENTS=4', ''])
            (anchor/'run.env').write_text(old)
            for name in ('submit_square_two_ladder.sh', 'two_basin_submission_environment.sh'):
                shutil.copyfile(ROOT/'slurm'/name, scripts/name)
            (scripts/'phase1_gpu.sh').write_text('''#!/bin/bash
set -euo pipefail
[[ ! -v PHASE1_RUN_SCRIPT_VERSION && ! -v PHASE1_RUN_SCRATCH_DIR ]] || exit 17
printf '%s|%s|%s|%s|%s|%s|%s|%s|%s\\n' "$*" "$PHASE1_PROJECT_DIR" "$PHASE1_SQUARE_TWO_LADDER_CONFIG" "$PHASE1_LEDGER_PATH" "$PHASE1_ACCOUNT" "$PHASE1_MAX_SEGMENTS" "$PHASE1_GPU_TIME" "$PHASE1_ADDITIONAL_NODE_HOUR_CAP" "$PHASE1_RECONCILIATION_PATH" >> "$TEST_RECEIPT"
if [[ "${TEST_FAIL_PREPARE:-0}" == 1 && "$1" == prepare-* ]]; then exit 9; fi
''', newline='\n')
            receipt = root/'receipt.txt'
            env = dict(os.environ, TWO_BASIN_ORIGINAL_PROJECT=project.as_posix(), TEST_RECEIPT=receipt.as_posix())
            for fail in (False,True):
                receipt.write_text('')
                env['TEST_FAIL_PREPARE'] = '1' if fail else '0'
                result = subprocess.run([str(BASH),(scripts/'submit_square_two_ladder.sh').as_posix()], env=env, capture_output=True,text=True)
                calls = [line.split('|') for line in receipt.read_text().splitlines()]
                self.assertEqual(len(calls),1 if fail else 3)
                self.assertEqual(result.returncode != 0,fail,result.stderr)
                self.assertTrue(calls[0][0].startswith('prepare-square-two-ladder '))
                self.assertTrue(calls[0][0].endswith(' 20260920_square_two_ladder_two_basin_60'))
                for call in calls:
                    self.assertTrue(call[1].endswith('/current checkout/ladder_mps_mft'))
                    self.assertTrue(call[2].endswith('/configs/phase1_gpu_square_two_ladder_chi200_raw60.toml'))
                    self.assertEqual(call[3:],['shared_ledger','saved_account','1','16:00:00','400','shared_reconciliation'])
                if not fail:
                    self.assertEqual(calls[1][0],'reconcile')
                    self.assertEqual(calls[2][0],'submit 20260920_square_two_ladder_two_basin_60')
            self.assertEqual((anchor/'run.env').read_text(),old)

    def test_production_guard_requires_four_branches_and_cell_contract(self):
        source = (ROOT/'slurm/phase1_gpu.sh').read_text()
        start = source.index('validate_initialized_run() {')
        function = source[start:source.index('\n}\n',start)+3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('configs','full/results'):
                (root/name).mkdir(parents=True)
            for name in ('run.env','jobs.tsv','gpu-Manifest.toml'):
                (root/name).write_text('')
            (root/'gpu-Manifest.toml.sha256').write_text(hashlib.sha256(b'').hexdigest()+'  '+(root/'gpu-Manifest.toml').as_posix()+'\n',newline='\n')
            (root/'campaign_kind.txt').write_text('square_two_ladder\n',newline='\n')
            script = root/'validate.sh'
            script.write_text('''#!/bin/bash
set -euo pipefail
export PATH="/usr/bin:$PATH"
PHASE1_SCRIPT_VERSION=1.24.0
die() { echo "error: $*" >&2; exit 1; }
full_run_directory_from_control() { printf '%s/full\\n' "$1"; }
'''+function+'\nvalidate_initialized_run "$1"\n',newline='\n')
            for count,cell,success in ((3,True,False),(4,False,False),(4,True,True)):
                (root/'branch_count.txt').write_text(str(count)+'\n',newline='\n')
                (root/'manifest.tsv').write_text('label\tconfig\n'+''.join(f'b{i}\tunused\n' for i in range(count)),newline='\n')
                for i in range(count):
                    (root/f'configs/b{i}.segment-001.toml').write_text('')
                (root/'seed_contract.toml').write_text('branches = 4\n'+('spatial_cell = "two_ladder"\n' if cell else ''),newline='\n')
                result = subprocess.run([str(BASH),script.as_posix(),root.as_posix()],capture_output=True,text=True)
                self.assertEqual(result.returncode==0,success,result.stderr)

if __name__=='__main__':
    unittest.main()
