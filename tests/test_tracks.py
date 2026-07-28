"""Тесты многодорожечного сведения: группировка, похожесть текста, фильтр протечек.

Гоняются без моделей и без аудио - все проверяемое здесь чистые функции:
    python3 -m unittest discover -s tests -v
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import transcribe_meeting as tm


def words(*triples):
    return [[a, b, t] for a, b, t in triples]


class GroupTrack(unittest.TestCase):
    def test_splits_on_pause_and_labels_speaker(self):
        blocks = tm.group_track(
            words((0.0, 0.5, "привет"), (0.6, 1.0, "как"), (5.0, 5.4, "дела")),
            "Иван", gap=1.2)
        self.assertEqual([(0.0, 1.0, "Иван", "привет как"), (5.0, 5.4, "Иван", "дела")], blocks)

    def test_keeps_one_block_when_pauses_are_short(self):
        blocks = tm.group_track(words((0.0, 0.4, "раз"), (0.9, 1.2, "два")), "Петр", gap=1.2)
        self.assertEqual(1, len(blocks))
        self.assertEqual("раз два", blocks[0][3])

    def test_empty_input_gives_no_blocks(self):
        self.assertEqual([], tm.group_track([], "Иван"))


class TextSimilarity(unittest.TestCase):
    def test_identical_text(self):
        self.assertEqual(1.0, tm.text_similarity("да, согласен", "да согласен"))

    def test_disjoint_text(self):
        self.assertEqual(0.0, tm.text_similarity("привет всем", "бюджет утвержден"))

    def test_short_inside_long(self):
        # знаменатель - более короткая реплика, поэтому вложенность дает 1.0
        self.assertEqual(1.0, tm.text_similarity("да", "да, конечно, давай так"))

    def test_empty_is_not_similar(self):
        self.assertEqual(0.0, tm.text_similarity("", "что-то"))


class DropBleed(unittest.TestCase):
    @staticmethod
    def loud(levels):
        return lambda i: levels[i]

    def test_drops_quiet_duplicate_of_same_phrase(self):
        blocks = [(0.0, 2.0, "Иван", "бюджет утвердили в среду"),
                  (0.1, 1.9, "Петр", "бюджет утвердили в среду")]
        kept = tm.drop_bleed(blocks, self.loud([0.20, 0.02]))
        self.assertEqual([blocks[0]], kept)

    def test_keeps_genuine_simultaneous_speech(self):
        # оба говорят одновременно, но разные слова и оба громко - это перебивание, не протечка
        blocks = [(0.0, 2.0, "Иван", "предлагаю перенести встречу"),
                  (0.3, 2.1, "Петр", "нет подождите я не согласен")]
        kept = tm.drop_bleed(blocks, self.loud([0.20, 0.18]))
        self.assertEqual(blocks, kept)

    def test_drops_much_quieter_even_when_text_garbled(self):
        # протечка обычно распознается искаженно, поэтому текст не совпадет - спасает громкость
        blocks = [(0.0, 2.0, "Иван", "квартальный отчет готов"),
                  (0.1, 1.8, "Петр", "картальны отчт готв")]
        kept = tm.drop_bleed(blocks, self.loud([0.30, 0.05]))
        self.assertEqual([blocks[0]], kept)

    def test_keeps_blocks_that_do_not_overlap(self):
        blocks = [(0.0, 1.0, "Иван", "одно и то же"),
                  (5.0, 6.0, "Петр", "одно и то же")]
        self.assertEqual(blocks, tm.drop_bleed(blocks, self.loud([0.2, 0.01])))

    def test_ignores_short_touch_of_two_blocks(self):
        # пересечение 0.1 c при длине 2 c - это стык реплик, а не протечка
        blocks = [(0.0, 2.0, "Иван", "давайте начнем"),
                  (1.9, 3.9, "Петр", "давайте начнем")]
        self.assertEqual(blocks, tm.drop_bleed(blocks, self.loud([0.2, 0.01])))

    def test_disabled_by_caller_keeps_everything(self):
        # min_overlap=2 недостижим - ни одна пара не пройдет порог
        blocks = [(0.0, 2.0, "Иван", "одно и то же"), (0.0, 2.0, "Петр", "одно и то же")]
        self.assertEqual(blocks, tm.drop_bleed(blocks, self.loud([0.2, 0.01]), min_overlap=2))


class PlanTracks(unittest.TestCase):
    def test_names_default_to_file_stems(self):
        got = tm.plan_tracks(["/tmp/ivan.wav", "/tmp/petr.m4a"], False, None)
        self.assertEqual([("ivan", "/tmp/ivan.wav", None), ("petr", "/tmp/petr.m4a", None)], got)

    def test_explicit_names_override_stems(self):
        got = tm.plan_tracks(["/tmp/a.wav", "/tmp/b.wav"], False, "Иван, Петр")
        self.assertEqual(["Иван", "Петр"], [n for n, _, _ in got])

    def test_name_count_mismatch_is_fatal(self):
        with self.assertRaises(SystemExit):
            tm.plan_tracks(["/tmp/a.wav", "/tmp/b.wav"], False, "Иван")

    def test_split_channels_needs_single_file(self):
        with self.assertRaises(SystemExit):
            tm.plan_tracks(["/tmp/a.wav", "/tmp/b.wav"], True, None)


class WriteMd(unittest.TestCase):
    """Путь "слова -> реплики -> markdown" целиком, без моделей: имя дорожки должно доехать до строки."""

    def test_multitrack_lines_are_labeled_by_track_name(self):
        blocks = (tm.group_track(words((0.0, 1.0, "привет"), (1.2, 1.6, "коллеги")), "Иван")
                  + tm.group_track(words((10.0, 10.5, "да")), "Петр"))
        lines = [(s0, name, tx) for s0, _, name, tx in sorted(blocks)]
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "o.md")
            tm.write_md(out, "Тест", "ivan.wav, petr.wav", 12.0, lines,
                        "GigaAM (ASR), спикер = дорожка", "**Дорожек:** 2 (Иван, Петр)")
            with open(out, encoding="utf-8") as fh:
                md = fh.read()
        self.assertIn("**[00:00] Иван:** привет коллеги", md)
        self.assertIn("**[00:10] Петр:** да", md)
        self.assertIn("**Дорожек:** 2 (Иван, Петр)", md)
        self.assertNotIn("Спикер 1", md)          # в многодорожечном режиме обезличенных нет

    def test_single_track_without_speakers_has_bare_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "o.md")
            tm.write_md(out, "Тест", "call.mp3", 5.0, [(0.0, None, "текст без спикера")],
                        "GigaAM (ASR)")
            with open(out, encoding="utf-8") as fh:
                md = fh.read()
        self.assertIn("**[00:00]** текст без спикера", md)


if __name__ == "__main__":
    unittest.main()
