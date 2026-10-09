#!/usr/bin/env python3
"""Deploy a verified Git commit without installing dependencies."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import tarfile
import tempfile


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args])


def archive_commit(repo, ref, destination):
    commit = git(repo, 'rev-parse', '--verify', '--end-of-options', ref + '^{commit}').decode().strip()
    entries = git(repo, 'ls-tree', '-rz', commit).split(b'\0')
    expected = set()
    for entry in filter(None, entries):
        metadata, name = entry.split(b'\t', 1)
        mode, kind, _ = metadata.split()
        if mode not in (b'100644', b'100755') or kind != b'blob':
            raise ValueError('Unsupported Git entry (symlink/submodule): ' + os.fsdecode(name))
        expected.add(os.fsdecode(name))
    data = git(repo, 'archive', '--format=tar', commit)
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        files = [m for m in archive.getmembers() if not m.isdir()]
        names = [m.name for m in files]
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError('Archive differs from commit file list (check export-ignore)')
        for member in files:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive entry: ' + member.name)
            content = archive.extractfile(member).read()
            if content.startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError('LFS pointers are unsupported: ' + member.name)
            # Reject export-subst and other transformations too.
            original = git(repo, 'show', commit + ':' + member.name)
            if content != original:
                raise ValueError('Archive transformed committed content: ' + member.name)
            target = destination / member.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            target.chmod(member.mode & 0o777)
    return commit


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path, data, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def deploy(repo, home, launcher=None, ref='HEAD', adopt_existing=False):
    home = home.absolute()
    runtime = home / 'transcribe_meeting.py'
    revision = home / 'REVISION'
    python = home / 'venv/bin/python'
    if not python.is_file() or not (home / 'models').is_dir():
        raise ValueError('Prepare home/venv/bin/python and home/models before deploy')
    if runtime.exists():
        if revision.exists():
            recorded = json.loads(revision.read_text())
            if digest(runtime) != recorded['sha256']:
                raise ValueError('Runtime has local changes; preserve and review them before deployment')
        elif not adopt_existing:
            raise ValueError('Untracked runtime: preserve it, then use --adopt-existing for first adoption')
    paths = [runtime, revision] + ([launcher.absolute()] if launcher else [])
    if len(set(paths)) != len(paths):
        raise ValueError('Launcher must be separate from runtime and REVISION')
    for path in paths:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise ValueError('Target must be a regular file: ' + str(path))
    with tempfile.TemporaryDirectory(prefix='giga-deploy-') as temporary:
        source = Path(temporary)
        commit = archive_commit(repo, ref, source)
        if not (source / 'tests').is_dir() or not list((source / 'tests').glob('test*.py')):
            raise ValueError('Committed tests are required')
        env = dict(os.environ, GIGA_TRANSCRIBE_HOME=str(home))
        env.pop('PYTHONPATH', None)
        subprocess.run([str(python), '-m', 'unittest', 'discover', '-s', 'tests', '-v'], cwd=source, env=env, check=True)
        subprocess.run([str(python), str(source / runtime.name), '--help'], cwd=source, env=env, check=True)
        new_script = (source / runtime.name).read_bytes()
        new_revision = (json.dumps({'commit': commit, 'sha256': hashlib.sha256(new_script).hexdigest()}, indent=2) + '\n').encode()
        backup = Path(tempfile.mkdtemp(prefix='deploy-backup-', dir=home))
        previous = {}
        for index, path in enumerate(paths):
            previous[path] = (path.read_bytes(), path.stat().st_mode & 0o777) if path.exists() else None
            if previous[path]:
                shutil.copy2(path, backup / str(index))
        (backup / 'manifest.json').write_text(json.dumps({str(i): str(p) for i, p in enumerate(paths)}, indent=2) + '\n')
        try:
            atomic_write(runtime, new_script)
            atomic_write(revision, new_revision)
            if launcher:
                text = '#!/bin/sh\nexport GIGA_TRANSCRIBE_HOME=' + shlex.quote(str(home)) + '\nexec ' + shlex.quote(str(python)) + ' ' + shlex.quote(str(runtime)) + ' "$@"\n'
                atomic_write(launcher.absolute(), text.encode(), 0o755)
        except BaseException:
            for path, old in previous.items():
                if old is None:
                    path.unlink(missing_ok=True)
                else:
                    atomic_write(path, *old)
            raise
    print('Deployed ' + commit + '; backup: ' + str(backup))
    return commit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--home', required=True, type=Path)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--launcher', type=Path)
    target.add_argument('--no-launcher', action='store_true')
    parser.add_argument('--ref', default='HEAD')
    parser.add_argument('--adopt-existing', action='store_true')
    args = parser.parse_args()
    try:
        deploy(Path(__file__).resolve().parent, args.home, args.launcher, args.ref, args.adopt_existing)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(1, 'Deploy failed: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
