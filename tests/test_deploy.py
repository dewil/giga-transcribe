import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('deploy', Path(__file__).resolve().parents[1] / 'deploy.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.git('init', '-q')
        self.git('config', 'user.email', 'test@example.invalid')
        self.git('config', 'user.name', 'Test')
        (self.repo / 'tests').mkdir()
        (self.repo / 'tests/test_smoke.py').write_text('import unittest\nclass Smoke(unittest.TestCase):\n def test_ok(self): self.assertTrue(True)\n')
        (self.repo / 'transcribe_meeting.py').write_text('import os\nprint(os.environ["GIGA_TRANSCRIBE_HOME"])\n')
        self.commit()
        self.home = self.root / "runtime ' with spaces"
        (self.home / 'models').mkdir(parents=True)
        (self.home / 'venv/bin').mkdir(parents=True)
        (self.home / 'venv/bin/python').symlink_to(sys.executable)

    def git(self, *args):
        return deploy.git(self.repo, *args)

    def commit(self):
        self.git('add', '.')
        self.git('commit', '-qm', 'fixture')

    def test_committed_only_and_launcher_quotes(self):
        expected = (self.repo / 'transcribe_meeting.py').read_bytes()
        (self.repo / 'transcribe_meeting.py').write_text('raise RuntimeError("dirty")')
        (self.repo / 'untracked.py').write_text('dirty')
        launcher = self.root / "bin ' space/meeting"
        commit = deploy.deploy(self.repo, self.home, launcher)
        self.assertEqual((self.home / 'transcribe_meeting.py').read_bytes(), expected)
        self.assertEqual(json.loads((self.home / 'REVISION').read_text())['commit'], commit)
        self.assertEqual(subprocess.check_output([str(launcher)], text=True).strip(), str(self.home))
        self.assertFalse((self.home / 'untracked.py').exists())
        (self.home / 'transcribe_meeting.py').write_text('local change')
        with self.assertRaisesRegex(ValueError, 'local changes'):
            deploy.deploy(self.repo, self.home, launcher, adopt_existing=True)

    def test_first_adoption_requires_flag_and_keeps_backup(self):
        runtime = self.home / 'transcribe_meeting.py'
        runtime.write_text('old')
        with self.assertRaisesRegex(ValueError, 'adopt-existing'):
            deploy.deploy(self.repo, self.home)
        deploy.deploy(self.repo, self.home, adopt_existing=True)
        self.assertEqual(next(self.home.glob('deploy-backup-*/0')).read_text(), 'old')

    def test_archive_rejects_export_ignore(self):
        (self.repo / '.gitattributes').write_text('transcribe_meeting.py export-ignore\n')
        self.commit()
        with self.assertRaisesRegex(ValueError, 'file list'):
            deploy.archive_commit(self.repo, 'HEAD', self.root / 'archive')

    def test_archive_rejects_symlinks_and_lfs(self):
        (self.repo / 'link').symlink_to('transcribe_meeting.py')
        self.commit()
        with self.assertRaisesRegex(ValueError, 'symlink'):
            deploy.archive_commit(self.repo, 'HEAD', self.root / 'archive')
        (self.repo / 'link').unlink()
        (self.repo / 'pointer').write_text('version https://git-lfs.github.com/spec/v1\n')
        self.commit()
        with self.assertRaisesRegex(ValueError, 'LFS'):
            deploy.archive_commit(self.repo, 'HEAD', self.root / 'archive')

    def test_failed_test_preserves_runtime(self):
        deploy.deploy(self.repo, self.home)
        old = (self.home / 'REVISION').read_bytes()
        (self.repo / 'tests/test_smoke.py').write_text('raise RuntimeError("failed test")')
        self.commit()
        with self.assertRaises(subprocess.CalledProcessError):
            deploy.deploy(self.repo, self.home)
        self.assertEqual((self.home / 'REVISION').read_bytes(), old)
