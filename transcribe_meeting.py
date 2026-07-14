#!/usr/bin/env python3
"""
transcribe-meeting - локальная транскрипция аудио/видео встречи в markdown.

Движок (отобран сравнением с облачным сервисом и Whisper, см. docs/benchmarks.md):
  ASR         - GigaAM v3 e2e_rnnt (русский, локально; бьет локальный Whisper, паритет с облаком по тексту)
  Диаризация  - sherpa-onnx (pyannote-segmentation-3.0 + campplus эмбеддинг), опционально, token-free

Запускается python-ом из venv проекта (launcher в ~/.local/bin это настраивает).
Пути к venv/моделям берутся из GIGA_TRANSCRIBE_HOME (дефолт: ~/.local/share/giga-transcribe).

Потолок диаризации: из одной смешанной записи звонка надежно разделяет ~2-3 голоса; 4+ похожих
голосов сливаются при любых настройках (нужны пофайловые дорожки участников). Спикеры приблизительные.

Устойчивость: все промежуточные файлы - в своем temp-каталоге на прогон (параллель-безопасно),
md пишется атомарно, при остановке (Ctrl-C / kill) temp-файлы чистятся.

Примеры:
  transcribe-meeting "встреча.mp4"
  transcribe-meeting rec.webm -o out.md --speakers 3
  transcribe-meeting audio.wav --no-diar --title "Недельный синк"
"""
import argparse, os, sys, time, tempfile, subprocess, datetime, shutil, signal, atexit

HOME_DIR = os.environ.get("GIGA_TRANSCRIBE_HOME", os.path.expanduser("~/.local/share/giga-transcribe"))
DIAR_DIR = os.path.join(HOME_DIR, "models")
SEG_MODEL = os.path.join(DIAR_DIR, "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")
EMB = {"campplus": os.path.join(DIAR_DIR, "embedding.onnx"),
       "titanet":  os.path.join(DIAR_DIR, "titanet.onnx")}
ASR_MODEL = "v3_e2e_rnnt"
CHUNK_S = 24            # лимит GigaAM .transcribe - 25 сек
SR = 16000

# --- ресурсы прогона, чистятся при выходе/сигнале ---
_tmpdir = None
_partial_md = None


def _cleanup():
    global _tmpdir, _partial_md
    if _tmpdir and os.path.isdir(_tmpdir):
        shutil.rmtree(_tmpdir, ignore_errors=True)
        _tmpdir = None
    if _partial_md and os.path.exists(_partial_md):
        try:
            os.remove(_partial_md)
        except OSError:
            pass
        _partial_md = None


def _on_signal(signum, frame):
    _cleanup()
    print(f"[прервано] сигнал {signum}, временные файлы очищены", file=sys.stderr)
    sys.exit(130)


atexit.register(_cleanup)
signal.signal(signal.SIGINT, _on_signal)
signal.signal(signal.SIGTERM, _on_signal)


def die(msg, code=1):
    print(f"[ошибка] {msg}", file=sys.stderr)
    sys.exit(code)


def mmss(t):
    t = int(t)
    return f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}" if t >= 3600 else f"{t//60:02d}:{t%60:02d}"


def extract_wav(src, dst):
    if not shutil.which("ffmpeg"):
        die("ffmpeg не найден в PATH (поставь его, напр. `brew install ffmpeg`)")
    r = subprocess.run(["ffmpeg", "-y", "-i", src, "-ar", str(SR), "-ac", "1", dst],
                       capture_output=True)
    if r.returncode != 0 or not os.path.exists(dst):
        die("ffmpeg не смог извлечь аудио:\n" + r.stderr.decode(errors="ignore")[-500:])


def run_asr(audio, sr, tmpdir):
    import gigaam, soundfile as sf
    model = gigaam.load_model(ASR_MODEL, device="cpu")
    words, ch = [], CHUNK_S * sr
    for i in range(0, len(audio), ch):
        p = os.path.join(tmpdir, f"chunk_{i}.wav")   # tmpdir уникален -> параллель-безопасно
        sf.write(p, audio[i:i+ch], sr)
        off = i / sr
        try:
            for w in model.transcribe(p, word_timestamps=True).words:
                words.append([w.start + off, w.end + off, w.text])
        finally:
            os.path.exists(p) and os.remove(p)
    return words


def run_diar(audio, n_speakers, emb_key):
    import sherpa_onnx
    emb = EMB.get(emb_key)
    if not (os.path.exists(SEG_MODEL) and emb and os.path.exists(emb)):
        die(f"нет моделей диаризации в {DIAR_DIR} (запусти install.sh)")
    clustering = (sherpa_onnx.FastClusteringConfig(num_clusters=n_speakers)
                  if n_speakers and n_speakers > 0
                  else sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=0.5))
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG_MODEL)),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb),
        clustering=clustering, min_duration_on=0.2, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    return [(r.start, r.end, r.speaker) for r in sd.process(audio).sort_by_start_time()]


def group_by_speaker(words, segs):
    def spk(a, b):
        best, ov = None, 0.0
        for s0, s1, s in segs:
            o = max(0, min(b, s1) - max(a, s0))
            if o > ov:
                ov, best = o, s
        return best if best is not None else 0
    lines, cur, buf, st = [], None, [], 0
    for a, b, tx in words:
        s = spk(a, b)
        if s != cur:
            if buf:
                lines.append((st, cur, " ".join(buf)))
            cur, buf, st = s, [tx], a
        else:
            buf.append(tx)
    if buf:
        lines.append((st, cur, " ".join(buf)))
    return lines


def group_by_pause(words, gap=1.2):
    lines, buf, st, prev = [], [], 0, None
    for a, b, tx in words:
        if prev is not None and a - prev > gap and buf:
            lines.append((st, None, " ".join(buf)))
            buf, st = [], a
        if not buf:
            st = a
        buf.append(tx)
        prev = b
    if buf:
        lines.append((st, None, " ".join(buf)))
    return lines


def write_md(path, title, src, dur, lines, diarized, emb_key):
    global _partial_md
    today = datetime.date.today().isoformat()
    engine = f"GigaAM {ASR_MODEL} (ASR)" + (f" + sherpa-onnx/{emb_key} (диаризация)" if diarized else "")
    spk_ids = sorted({s for _, s, _ in lines if s is not None})
    head = [f"# {title}", "",
            f"**Файл:** {os.path.basename(src)}  ",
            f"**Длительность:** {mmss(dur)}  ",
            f"**Обработано:** {today}, локально  ",
            f"**Движок:** {engine}  "]
    if diarized:
        head.append(f"**Спикеров выделено:** {len(spk_ids)} "
                    f"(диаризация приблизительная - из смешанной записи надежно 2-3 голоса; "
                    f"сверить и проставить имена вручную)  ")
    head += ["", "## Транскрипт", ""]
    body = []
    for st, s, tx in lines:
        label = f"Спикер {s+1}" if s is not None else None
        prefix = f"**[{mmss(st)}] {label}:** " if label else f"**[{mmss(st)}]** "
        body.append(prefix + tx.strip())
    # атомарно: пишем в .tmp рядом с целью (та же ФС) и переименовываем
    _partial_md = path + ".tmp"
    with open(_partial_md, "w") as f:
        f.write("\n".join(head) + "\n" + "\n\n".join(body) + "\n")
    os.replace(_partial_md, path)
    _partial_md = None


def main():
    global _tmpdir
    ap = argparse.ArgumentParser(description="Локальная транскрипция встречи (GigaAM + sherpa) в markdown.")
    ap.add_argument("input", help="аудио или видео файл (mp4/webm/wav/m4a/...)")
    ap.add_argument("-o", "--output", help="путь к .md (по умолчанию рядом с входным файлом)")
    ap.add_argument("--speakers", type=int, default=0, help="число участников (точнее диаризация); 0 = авто")
    ap.add_argument("--no-diar", action="store_true", help="без диаризации, чистый текст по паузам")
    ap.add_argument("--embedding", choices=list(EMB), default="campplus", help="эмбеддинг диаризации")
    ap.add_argument("--title", help="заголовок в md (по умолчанию - имя файла)")
    args = ap.parse_args()

    src = os.path.abspath(os.path.expanduser(args.input))
    if not os.path.exists(src):
        die(f"файл не найден: {src}")
    out = os.path.abspath(os.path.expanduser(args.output)) if args.output else os.path.splitext(src)[0] + ".md"
    title = args.title or os.path.splitext(os.path.basename(src))[0]

    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass

    import soundfile as sf
    _tmpdir = tempfile.mkdtemp(prefix="transcribe_")
    try:
        print("[1/4] извлекаю аудио...", file=sys.stderr)
        wav = os.path.join(_tmpdir, "audio.wav")
        extract_wav(src, wav)
        audio, sr = sf.read(wav, dtype="float32")
        dur = len(audio) / sr

        print(f"[2/4] распознавание речи ({dur/60:.1f} мин)...", file=sys.stderr)
        t = time.time()
        words = run_asr(audio, sr, _tmpdir)
        print(f"      слов={len(words)}, {time.time()-t:.0f}с", file=sys.stderr)

        if args.no_diar:
            lines, diarized = group_by_pause(words), False
        else:
            print("[3/4] диаризация...", file=sys.stderr)
            t = time.time()
            segs = run_diar(audio, args.speakers, args.embedding)
            print(f"      {time.time()-t:.0f}с", file=sys.stderr)
            lines, diarized = group_by_speaker(words, segs), True

        print("[4/4] пишу markdown...", file=sys.stderr)
        write_md(out, title, src, dur, lines, diarized, args.embedding)
        print(out)   # stdout = путь к результату (для пайпов)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
