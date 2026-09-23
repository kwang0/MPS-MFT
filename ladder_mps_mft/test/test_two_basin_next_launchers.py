"""Local fake-launcher checks; never invoke Slurm or access Perlmutter."""
import os
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASH = str(Path('C:/Program Files/Git/bin/bash.exe')) if os.name == 'nt' else shutil.which('bash')


@unittest.skipUnless(BASH and Path(BASH).is_file(), 'local Bash unavailable')
class NextCampaignLaunchers(unittest.TestCase):
    def test_actual_prepared_run_guard_rejects_compact_preview(self):
        # Exercise the production validation function alone, without the CLI's
        # scheduler entry points. The fixture contains no scientific state.
        source = (ROOT / 'slurm/phase1_gpu.sh').read_text()
        start = source.index('validate_initialized_run() {')
        function = source[start:source.index('\n}\n', start) + 3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('configs', 'full/results'):
                (root / name).mkdir(parents=True)
            for name in ('run.env', 'jobs.tsv', 'gpu-Manifest.toml'):
                (root / name).write_text('')
            (root / 'gpu-Manifest.toml.sha256').write_text(
                hashlib.sha256(b'').hexdigest() + '  ' + (root / 'gpu-Manifest.toml').as_posix() + '\n', newline='\n')
            (root / 'configs/one.segment-001.toml').write_text('')
            (root / 'manifest.tsv').write_text('label\tconfig\none\tunused\n', newline='\n')
            (root / 'branch_count.txt').write_text('1\n', newline='\n')
            (root / 'campaign_kind.txt').write_text('square_two_basin_finish\n', newline='\n')
            contract = root / 'continuation_contract.toml'
            script = root / 'validate.sh'
            script.write_text('''#!/bin/bash
set -euo pipefail
export PATH="/usr/bin:$PATH"
PHASE1_SCRIPT_VERSION=1.20.0
die() { echo "error: $*" >&2; exit 1; }
full_run_directory_from_control() { printf '%s/full\\n' "$1"; }
''' + function + '\nvalidate_initialized_run "$1"\n', newline='\n')
            for verified in (False, True):
                contract.write_text('full_sources_verified = ' + str(verified).lower() + '\n', newline='\n')
                result = subprocess.run([BASH, script.as_posix(), root.as_posix()], capture_output=True, text=True)
                if verified:
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('compact previews cannot be submitted', result.stderr)

    def test_syntax(self):
        for name in ('phase1_gpu.sh', 'two_basin_submission_environment.sh',
                     'submit_cubic_unfrustrated_two_basin.sh', 'submit_square_two_basin_finish.sh',
                     'submit_square_two_basin_fine_cuts.sh', 'submit_square_positive_v.sh', 'submit_trellis_comparison.sh',
                     'submit_trellis_vm1_comparison.sh', 'submit_square_tp_scan.sh',
                     'submit_trellis_intertwined.sh'):
            result = subprocess.run([BASH, '-n', (ROOT / 'slurm' / name).as_posix()],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_scan_guard_and_trellis_version_compatibility(self):
        source = (ROOT / 'slurm/phase1_gpu.sh').read_text()
        start = source.index('validate_initialized_run() {')
        function = source[start:source.index('\n}\n', start) + 3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'validate.sh'
            script.write_text('''#!/bin/bash
set -euo pipefail
export PATH="/usr/bin:$PATH"
PHASE1_SCRIPT_VERSION=1.26.0
PHASE1_RUN_SCRIPT_VERSION="$2"
die() { echo "error: $*" >&2; exit 1; }
full_run_directory_from_control() { printf '%s/full\\n' "$1"; }
''' + function + '\nvalidate_initialized_run "$1"\n', newline='\n')
            cases = [('square_tp_scan', version, 8) for version in ('1.25.0','1.26.0')] + [
                ('trellis_comparison', version, 4) for version in ('1.23.0','1.24.0','1.25.0','1.26.0')] + [
                ('trellis_intertwined', '1.26.0', 4)]
            for kind, version, count in cases:
                with self.subTest(kind=kind, version=version):
                    run = root / (kind + version)
                    (run/'configs').mkdir(parents=True)
                    (run/'full/results').mkdir(parents=True)
                    for name in ('run.env','jobs.tsv','gpu-Manifest.toml'):
                        (run/name).write_text('')
                    (run/'gpu-Manifest.toml.sha256').write_text(
                        hashlib.sha256(b'').hexdigest()+'  '+(run/'gpu-Manifest.toml').as_posix()+'\n', newline='\n')
                    (run/'campaign_kind.txt').write_text(kind+'\n', newline='\n')
                    (run/'branch_count.txt').write_text(str(count)+'\n', newline='\n')
                    (run/'manifest.tsv').write_text('label\tconfig\n'+''.join(
                        f'branch{i}\tunused\n' for i in range(count)), newline='\n')
                    for i in range(count):
                        (run/f'configs/branch{i}.segment-001.toml').write_text('')
                    contract = run/'seed_contract.toml'
                    contract_text = (f'branches = {count}\nstage = "tp_scan"\ninterpolated_ep = false\n'
                                     'trellis_cell = "two_ladder"\nfamily = "intertwined_lambda16"\n')
                    contract.write_text(contract_text, newline='\n')
                    command = [BASH,script.as_posix(),run.as_posix(),version]
                    result = subprocess.run(command,capture_output=True,text=True)
                    self.assertEqual(result.returncode,0,result.stderr)
                    if kind == 'square_tp_scan':
                        contract.write_text('branches = 8\nstage = "tp_scan"\ninterpolated_ep = true\n', newline='\n')
                        result = subprocess.run(command,capture_output=True,text=True)
                        self.assertNotEqual(result.returncode,0)
                        self.assertIn('requires exact E_p',result.stderr)
                    if kind == 'trellis_intertwined':
                        for before, after, error in (
                            ('branches = 4', 'branches = 8', 'require four branches'),
                            ('two_ladder', 'one_ladder', 'require two ladders'),
                            ('intertwined_lambda16', 'pairing', 'seed family changed'),
                            ('interpolated_ep = false', 'interpolated_ep = true', 'require exact E_p'),
                        ):
                            contract.write_text(contract_text.replace(before, after), newline='\n')
                            result = subprocess.run(command,capture_output=True,text=True)
                            self.assertNotEqual(result.returncode,0)
                            self.assertIn(error,result.stderr)
                        contract.write_text(contract_text, newline='\n')
                        # A matching branch_count cannot conceal extra manifest rows.
                        (run/'manifest.tsv').write_text('label\tconfig\n'+''.join(
                            f'branch{i}\tunused\n' for i in range(5)), newline='\n')
                        (run/'branch_count.txt').write_text('5\n', newline='\n')
                        result = subprocess.run(command,capture_output=True,text=True)
                        self.assertNotEqual(result.returncode,0)
                        self.assertIn('exactly four branches',result.stderr)

    def test_fine_cuts_isolate_source_and_share_accounting(self):
        self.check_isolated_wrapper('submit_square_two_basin_fine_cuts.sh',
            'PHASE1_TWO_BASIN_FINE_CUTS_CONFIG', 'prepare-square-two-basin-fine-cuts',
            '20260915_square_two_basin_fine_cuts_95_5_60',
            'phase1_gpu_square_two_basin_fine_cuts_chi200_raw60.toml')

    def test_positive_v_isolate_source_and_share_accounting(self):
        self.check_isolated_wrapper('submit_square_positive_v.sh',
            'PHASE1_POSITIVE_V_CONFIG', 'prepare-square-positive-v',
            '20260915_square_t012_vp02_four_seeds_60',
            'phase1_gpu_square_positive_v_chi200_raw60.toml')

    def test_trellis_isolate_source_and_share_accounting(self):
        self.check_isolated_wrapper('submit_trellis_comparison.sh',
            'PHASE1_TRELLIS_CONFIG', 'prepare-trellis-comparison',
            '20260916_trellis_two_basin_comparison_60',
            'phase1_gpu_trellis_chi200_raw60.toml')

    def test_square_tp_scan_isolates_trellis_source_and_latest_pointer(self):
        self.check_isolated_wrapper('submit_square_tp_scan.sh',
            'PHASE1_SQUARE_TP_SCAN_CONFIG', 'prepare-square-tp-scan',
            '20260922_square_t014_tp_scan_95_5_60',
            'phase1_gpu_square_tp_scan_chi200_raw60.toml',
            wall='16:00:00', run_subdir='/square_tp_scan')

    def test_trellis_intertwined_isolates_both_active_campaigns(self):
        self.check_isolated_wrapper('submit_trellis_intertwined.sh',
            'PHASE1_TRELLIS_INTERTWINED_CONFIG', 'prepare-trellis-intertwined',
            '20260923_trellis_two_ladder_intertwined_lambda16_60',
            'phase1_gpu_trellis_intertwined_chi200_raw60.toml',
            wall='16:00:00', run_subdir='/trellis_intertwined')

    def check_isolated_wrapper(self, wrapper, config_var, prepare, run, config,
                               wall='12:00:00', run_subdir=''):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / 'original' / 'ladder_mps_mft'
            new = root / 'fine cuts' / 'ladder_mps_mft'
            anchor = original / 'output/phase1_gpu/20260908_square_two_basin_95_5_80_anchors'
            anchor.mkdir(parents=True)
            environment = '\n'.join(['PHASE1_RUN_SCRIPT_VERSION=1.19.0',
                'PHASE1_RUN_SCRATCH_DIR=old', 'PHASE1_LEDGER_PATH=shared_ledger',
                'PHASE1_RECONCILIATION_PATH=shared_reconciliation',
                'PHASE1_RUN_ROOT=shared_runs', 'PHASE1_SCRATCH_ROOT=shared_scratch',
                'PHASE1_BUDGET_ROOT=shared_budget', 'PHASE1_ADDITIONAL_NODE_HOUR_CAP=400',
                'PHASE1_ACCOUNT=existing_account', 'PHASE1_MAX_SEGMENTS=4', ''])
            (anchor/'run.env').write_text(environment)
            receipt=root/'receipt.txt'
            fake='''#!/bin/bash
set -euo pipefail
[[ ! -v PHASE1_RUN_SCRIPT_VERSION && ! -v PHASE1_RUN_SCRATCH_DIR ]] || exit 17
printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\\n' "$*" "$PHASE1_PROJECT_DIR" "$CONFIG_VARIABLE" "$PHASE1_LEDGER_PATH" "$PHASE1_RECONCILIATION_PATH" "$PHASE1_ACCOUNT" "$PHASE1_MAX_SEGMENTS" "$PHASE1_GPU_TIME" "$PHASE1_RUN_ROOT" "$PHASE1_SCRATCH_ROOT" "$PHASE1_BUDGET_ROOT" "$PHASE1_ADDITIONAL_NODE_HOUR_CAP" >> "$TEST_RECEIPT"
if [[ "${TEST_FAIL_PREPARE:-0}" == 1 && "$1" == prepare-* ]]; then exit 9; fi
'''
            fake=fake.replace('CONFIG_VARIABLE',config_var)
            for project in (original,new):
                (project/'slurm').mkdir(parents=True)
                for name in ('two_basin_submission_environment.sh',wrapper):
                    shutil.copyfile(ROOT/'slurm'/name,project/'slurm'/name)
                (project/'slurm/phase1_gpu.sh').write_text(fake,newline='\n')
            env=dict(os.environ,TWO_BASIN_ORIGINAL_PROJECT=original.as_posix(),TEST_RECEIPT=receipt.as_posix())
            blocked=subprocess.run([BASH,(original/'slurm'/wrapper).as_posix()],env=env,capture_output=True,text=True)
            self.assertNotEqual(blocked.returncode,0)
            self.assertIn('separate checkout',blocked.stderr)
            self.assertFalse(receipt.exists())
            if wrapper == 'submit_trellis_intertwined.sh':
                # Also refuse a separately located checkout used by the tp scan.
                blocked=subprocess.run([BASH,(new/'slurm'/wrapper).as_posix()],
                    env=dict(env,TWO_BASIN_SQUARE_TP_PROJECT=new.as_posix()),capture_output=True,text=True)
                self.assertNotEqual(blocked.returncode,0)
                self.assertIn('separate checkout',blocked.stderr)
                self.assertFalse(receipt.exists())
            result=subprocess.run([BASH,(new/'slurm'/wrapper).as_posix()],env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            calls=[line.split('|') for line in receipt.read_text().splitlines()]
            self.assertEqual(len(calls),3)
            self.assertTrue(calls[0][0].startswith(prepare+' '))
            self.assertTrue(calls[0][0].endswith(' '+run))
            self.assertEqual(calls[1][0],'reconcile')
            self.assertEqual(calls[2][0],'submit '+run)
            for call in calls:
                self.assertTrue(call[1].endswith('/fine cuts/ladder_mps_mft'))
                self.assertTrue(call[2].endswith('/configs/'+config))
                self.assertEqual(call[3:],['shared_ledger','shared_reconciliation','existing_account','1',wall,
                    'shared_runs'+run_subdir,'shared_scratch','shared_budget','400'])
            self.assertEqual((anchor/'run.env').read_text(),environment)
            receipt.write_text('')
            failed=subprocess.run([BASH,(new/'slurm'/wrapper).as_posix(),'custom_run'],
                env=dict(env,TEST_FAIL_PREPARE='1'),capture_output=True,text=True)
            self.assertEqual(failed.returncode,9)
            self.assertEqual(len(receipt.read_text().splitlines()),1)
            self.assertIn('custom_run',receipt.read_text())

    def test_shared_accounting_current_checkout_and_scope(self):
        cases = (
            ('submit_cubic_unfrustrated_two_basin.sh', 'prepare-cubic-unfrustrated-two-basin-raw',
             '20260915_cubic_unfrustrated_two_basin_95_5_60', '12:00:00',
             'phase1_gpu_cubic_unfrustrated_two_basin_chi200_raw60.toml'),
            ('submit_square_two_basin_finish.sh', 'prepare-square-two-basin-finish',
             '20260915_square_t014_v000_two_basin_finish20', '08:00:00',
             'phase1_gpu_square_two_basin_finish20.toml'),
            ('submit_trellis_vm1_comparison.sh', 'prepare-trellis-comparison',
             '20260922_trellis_vm1_two_basin_comparison_60', '16:00:00',
             'phase1_gpu_trellis_vm1_chi200_raw60.toml'),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / 'current checkout' / 'ladder_mps_mft'
            slurm = project / 'slurm'; slurm.mkdir(parents=True)
            anchor = project / 'output/phase1_gpu/20260908_square_two_basin_95_5_80_anchors'
            anchor.mkdir(parents=True)
            original_environment = '\n'.join([
                'PHASE1_RUN_SCRIPT_VERSION=1.19.0', 'PHASE1_RUN_SCRATCH_DIR=old_run_only',
                'PHASE1_PROJECT_DIR=obsolete_source', 'PHASE1_REPO_ROOT=obsolete_repo',
                'PHASE1_RUN_ROOT=shared_runs', 'PHASE1_SCRATCH_ROOT=shared_scratch',
                'PHASE1_BUDGET_ROOT=shared_budget', 'PHASE1_LEDGER_PATH=shared_ledger',
                'PHASE1_RECONCILIATION_PATH=shared_reconciliation', 'PHASE1_ACCOUNT=existing_account',
                'PHASE1_GPU_TIME=12:00:00', 'PHASE1_ADDITIONAL_NODE_HOUR_CAP=400',
                'PHASE1_MAX_SEGMENTS=3', ''])
            (anchor / 'run.env').write_text(original_environment, encoding='utf-8')
            for name in ['two_basin_submission_environment.sh'] + [c[0] for c in cases]:
                shutil.copyfile(ROOT / 'slurm' / name, slurm / name)
            fake = '''#!/bin/bash
set -euo pipefail
[[ ! -v PHASE1_RUN_SCRIPT_VERSION && ! -v PHASE1_RUN_SCRATCH_DIR ]] || exit 17
printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\\n' "$*" "$PHASE1_PROJECT_DIR" "$PHASE1_RUN_ROOT" "$PHASE1_SCRATCH_ROOT" "$PHASE1_BUDGET_ROOT" "$PHASE1_LEDGER_PATH" "$PHASE1_RECONCILIATION_PATH" "$PHASE1_ACCOUNT" "$PHASE1_GPU_TIME" "$PHASE1_ADDITIONAL_NODE_HOUR_CAP" "$PHASE1_MAX_SEGMENTS" "${PHASE1_CUBIC_TWO_BASIN_CONFIG:-}" "${PHASE1_SQUARE_TWO_BASIN_FINISH_CONFIG:-}" "${PHASE1_TRELLIS_CONFIG:-}" >> "$TEST_RECEIPT"
if [[ "${TEST_FAIL_PREPARE:-0}" == 1 && "$1" == prepare-* ]]; then exit 9; fi
'''
            (slurm / 'phase1_gpu.sh').write_text(fake, encoding='utf-8', newline='\n')
            receipt = root / 'calls.txt'
            env = dict(os.environ, TWO_BASIN_ORIGINAL_PROJECT=project.as_posix(),
                       TEST_RECEIPT=receipt.as_posix())
            for script, prepare, run_id, wall, config in cases:
                with self.subTest(script=script):
                    receipt.write_text('')
                    result = subprocess.run([BASH, (slurm / script).as_posix()], env=env,
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    calls = [line.split('|') for line in receipt.read_text().splitlines()]
                    self.assertEqual(len(calls), 3)
                    self.assertTrue(calls[0][0].startswith(prepare + ' '))
                    self.assertTrue(calls[0][0].endswith(' ' + run_id))
                    if 'square' in script:
                        self.assertEqual(calls[0][0], prepare + ' shared_runs/20260908_square_two_basin_95_5_80_anchors ' + run_id)
                    self.assertEqual(calls[1][0], 'reconcile')
                    self.assertEqual(calls[2][0], 'submit ' + run_id)
                    for call in calls:
                        self.assertTrue(call[1].endswith('/current checkout/ladder_mps_mft'))
                        self.assertEqual(call[2:11], ['shared_runs', 'shared_scratch', 'shared_budget',
                            'shared_ledger', 'shared_reconciliation', 'existing_account', wall, '400', '1'])
                        self.assertTrue(any(value.endswith('/configs/' + config) for value in call[11:]))
                    self.assertEqual((anchor / 'run.env').read_text(), original_environment)
                    receipt.write_text('')
                    failed = subprocess.run([BASH, (slurm / script).as_posix(), 'custom_run'],
                        env=dict(env, TEST_FAIL_PREPARE='1'), capture_output=True, text=True)
                    self.assertEqual(failed.returncode, 9)
                    self.assertEqual(len(receipt.read_text().splitlines()), 1)
                    self.assertIn('custom_run', receipt.read_text())


if __name__ == '__main__':
    unittest.main()
