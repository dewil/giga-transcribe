"""Regression contracts for the preserved installed runtime; no model inference."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import transcribe_meeting as tm


class ResourceGuards(unittest.TestCase):
    def test_thread_budget_and_explicit_override(self):
        with patch.object(tm, 'available_cpus', return_value=6):
            self.assertEqual(3, tm.resolve_threads(None, False))
            self.assertEqual(6, tm.resolve_threads(None, True))
            self.assertEqual(6, tm.resolve_threads(0, False))
            self.assertEqual(2, tm.resolve_threads(2, True))
            with self.assertRaises(SystemExit):
                tm.resolve_threads(-1, False)
        for cpus, expected in [(1, 1), (2, 1), (32, 4)]:
            with patch.object(tm, 'available_cpus', return_value=cpus):
                self.assertEqual(expected, tm.default_threads())

    def test_affinity_overrides_host_cpu_count(self):
        with patch.object(os, 'sched_getaffinity', return_value={2, 4}), patch.object(os, 'cpu_count', return_value=64):
            self.assertEqual(2, tm.available_cpus())
        with patch.object(os, 'sched_getaffinity', side_effect=OSError), patch.object(os, 'cpu_count', return_value=None):
            self.assertEqual(1, tm.available_cpus())

    def test_scope_preserves_memory_limit_in_fast_mode(self):
        for background in (True, False):
            with self.subTest(background=background), patch.dict(os.environ, {}, clear=True), patch.object(tm.shutil, 'which', return_value='/bin/systemd-run'), patch.object(tm.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, b'', b'')) as probe, patch.object(os, 'execvpe') as execute:
                tm.reexec_in_scope('4G', background=background)
                cmd = execute.call_args.args[1]
                self.assertIn('MemoryMax=4G', cmd)
                self.assertIn('MemorySwapMax=0', cmd)
                self.assertEqual(background, 'CPUWeight=20' in cmd)
                self.assertEqual(background, 'IOWeight=20' in cmd)
                self.assertEqual('1', execute.call_args.args[2]['TRANSCRIBE_SCOPE'])
                self.assertEqual(15, probe.call_args.kwargs['timeout'])

    def test_failed_scope_probe_warns_and_continues(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(tm.shutil, 'which', return_value='/bin/systemd-run'), patch.object(tm.subprocess, 'run', side_effect=subprocess.TimeoutExpired('systemd-run', 15)), patch.object(os, 'execvpe') as execute, contextlib.redirect_stderr(io.StringIO()) as stderr:
            tm.reexec_in_scope('4G')
            execute.assert_not_called()
            self.assertIn('без лимита памяти', stderr.getvalue())


class MetadataGuards(unittest.TestCase):
    def test_non_object_lock_metadata_does_not_crash_waiter(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory, 'lock')
            for content in ('[]', 'null', '123', '{broken'):
                lock.write_text(content)
                with patch.object(tm, 'LOCK_FILE', str(lock)):
                    self.assertEqual({}, tm.read_lock_info())

    def test_unavailable_memory_measurement_is_visible(self):
        with patch.object(tm, 'mem_available_mb', return_value=None), contextlib.redirect_stderr(io.StringIO()) as stderr:
            tm.wait_for_memory(2500, False, 0)
        self.assertIn('предохранитель по памяти не работает', stderr.getvalue())


class InputGuards(unittest.TestCase):
    def test_media_commands_have_timeout_and_report_expiry(self):
        for function, args in [(tm.extract_wav, ('in.wav', 'out.wav')), (tm.extract_channel_wav, ('in.wav', 0, 'out.wav')), (tm.probe_channels, ('in.wav',))]:
            with self.subTest(function=function.__name__), patch.object(tm.shutil, 'which', return_value='/bin/tool'), patch.object(tm.subprocess, 'run', side_effect=subprocess.TimeoutExpired('tool', 3600)) as run, contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit):
                    function(*args)
                self.assertEqual(3600, run.call_args.kwargs['timeout'])
                self.assertIn('не ответил', stderr.getvalue())

    def test_bad_probe_does_not_silently_turn_stereo_into_mono(self):
        with patch.object(tm.shutil, 'which', return_value='/bin/ffprobe'), patch.object(tm.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, '', 'invalid audio')):
            with self.assertRaises(SystemExit):
                tm.probe_channels('broken.wav')
        with patch.object(tm.shutil, 'which', return_value=None):
            with self.assertRaises(SystemExit):
                tm.probe_channels('stereo.wav')

    def test_track_identifiers_cannot_merge_people(self):
        for inputs, names in [(['a.wav', 'b.wav'], 'A, '), (['a.wav', 'b.wav'], 'A,A'), (['/one/a.wav', '/two/a.wav'], None)]:
            with self.subTest(inputs=inputs, names=names), self.assertRaises(SystemExit):
                tm.plan_tracks(inputs, False, names)

    def test_output_rejection_precedes_lock_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            for filename, output in [('recording.wav', 'recording.wav'), ('recording.wav', 'output.txt'), ('notes.md', 'notes.md')]:
                source = Path(directory, filename)
                source.write_bytes(b'original input')
                argv = ['transcribe-meeting', str(source), '-o', str(Path(directory, output)), '--no-limit']
                with self.subTest(filename=filename, output=output), patch.object(sys, 'argv', argv), patch.dict(os.environ), patch.object(tm, 'lower_priority', return_value=19, create=True), patch.object(tm, 'acquire_lock', side_effect=AssertionError('must reject before acquiring lock')) as lock, self.assertRaises(SystemExit):
                    tm.main()
                lock.assert_not_called()
                self.assertEqual(b'original input', source.read_bytes())


class DiarizationAndBleed(unittest.TestCase):
    def test_singleton_silhouette_is_zero_not_perfect(self):
        self.assertEqual(0.0, tm._silhouette(np.eye(3), np.arange(3)))
        # Two identical vectors score 1 each; singleton scores 0: mean = 2/3.
        self.assertAlmostEqual(2 / 3, tm._silhouette(np.array([[1., 0.], [1., 0.], [0., 1.]]), np.array([0, 0, 1])))

    def test_two_segments_do_not_force_two_speakers(self):
        with patch.object(tm, '_diar_models_ok', return_value=True), patch.object(tm, '_segment', return_value=[(0, 1), (2, 3)]), patch.object(tm, '_embed_segments', return_value=(np.eye(2), [(0, 1), (2, 3)])), patch.object(tm, '_skmeans', side_effect=AssertionError('k must be less than number of segments')):
            result = tm.run_diar(np.zeros(10), 0, 'campplus', 1)
        self.assertEqual(1, len({row[2] for row in result}))

    def test_bleed_requires_both_similar_text_and_quiet_audio(self):
        for second_text, levels in [('другой ответ', [1., .01]), ('привет коллеги', [1., .9])]:
            blocks = [(0, 2, 'A', 'привет коллеги'), (0, 2, 'B', second_text)]
            self.assertEqual(blocks, tm.drop_bleed(blocks, levels.__getitem__))
        blocks = [(0, 2, 'A', 'привет коллеги'), (0, 2, 'B', 'привет коллеги')]
        self.assertEqual([blocks[0]], tm.drop_bleed(blocks, [1., .01].__getitem__))
