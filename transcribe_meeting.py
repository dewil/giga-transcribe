#!/usr/bin/env python3
"""
transcribe-meeting - локальная транскрипция аудио/видео встречи в markdown.

Движок (отобран сравнением с облачным сервисом и Whisper, см. docs/benchmarks.md):
  ASR         - GigaAM v3 e2e_rnnt (русский, локально; бьет локальный Whisper, паритет с облаком по тексту)
  Диаризация  - sherpa-onnx (pyannote-segmentation-3.0 + campplus эмбеддинг), опционально, token-free

Запускается python-ом из venv проекта (launcher в ~/.local/bin это настраивает).
Пути к venv/моделям берутся из GIGA_TRANSCRIBE_HOME (дефолт: ~/.local/share/giga-transcribe).

Диаризация: sherpa сегментирует речь, дальше свой per-segment эмбеддинг + k-means (при
известном --speakers N - k=N; иначе число оценивается по silhouette). На длинных содержательных
репликах разметка надежна (~95% против облака), короткие backchannel ("угу", быстрые вопросы) могут
уходить не тому спикеру - это неустранимый хвост смешанной записи. Спикеры обезличены (Спикер N).

Устойчивость: все промежуточные файлы - в своем temp-каталоге на прогон (параллель-безопасно),
md пишется атомарно, при остановке (Ctrl-C / kill) temp-файлы чистятся.

Примеры:
  transcribe-meeting "встреча.mp4"
  transcribe-meeting rec.webm -o out.md --speakers 3
  transcribe-meeting audio.wav --no-diar --title "Недельный синк"
"""
import argparse, os, sys, time, tempfile, subprocess, datetime, shutil, signal, atexit, fcntl, json

HOME_DIR = os.environ.get("GIGA_TRANSCRIBE_HOME", os.path.expanduser("~/.local/share/giga-transcribe"))
LOCK_FILE = os.path.join(HOME_DIR, ".transcribe.lock")
DIAR_DIR = os.path.join(HOME_DIR, "models")
SEG_MODEL = os.path.join(DIAR_DIR, "sherpa-onnx-pyannote-segmentation-3-0", "model.onnx")
EMB = {"campplus": os.path.join(DIAR_DIR, "embedding.onnx"),
       "titanet":  os.path.join(DIAR_DIR, "titanet.onnx")}
ASR_MODEL = "v3_e2e_rnnt"
CHUNK_S = 24            # лимит GigaAM .transcribe - 25 сек
SR = 16000
DIAR_MIN_ON = 0.5      # мин. длина речевого сегмента (баланс: короткие реплики vs шум эмбеддинга)
DIAR_KMAX = 6          # верхняя граница числа спикеров при авто-оценке (silhouette)

# --- ресурсы прогона, чистятся при выходе/сигнале ---
_tmpdir = None
_partial_md = None
_lock_fd = None


# --- предохранители общей машины (см. docs/runbook-shared-machine.md) ---


def detect_project(path):
    """Чей это прогон - имя папки проекта, из которой пришел файл.

    Ищем вверх по дереву маркер рабочей папки. Нужно только для человеческого
    сообщения ждущему: "занято проектом X" понятнее, чем "ресурс занят".
    """
    cur = os.path.abspath(path)
    while True:
        parent = os.path.dirname(cur)
        if os.path.isdir(parent):
            for marker in ("CLAUDE.md", ".claude", ".git"):
                if os.path.exists(os.path.join(parent, marker)):
                    return os.path.basename(parent)
        if parent == cur:
            return os.path.basename(os.getcwd())
        cur = parent


def read_lock_info():
    try:
        with open(LOCK_FILE, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def describe_holder(info):
    if not info:
        return "занято другим прогоном (подробностей нет)"
    parts = []
    if info.get("project"):
        parts.append(f'проект "{info["project"]}"')
    if info.get("file"):
        parts.append(f"файл {info['file']}")
    if info.get("started"):
        try:
            started = datetime.datetime.fromisoformat(info["started"])
            mins = int((datetime.datetime.now() - started).total_seconds() // 60)
            parts.append(f"идет {mins} мин (с {started.strftime('%H:%M')})")
        except ValueError:
            pass
    if info.get("pid"):
        parts.append(f"pid {info['pid']}")
    return ", ".join(parts)


def acquire_lock(source, wait, timeout):
    """Движок один на машину, прогоны идут по очереди.

    Почему именно взаимное исключение, а не просто уникальные имена файлов:
      - каждый процесс держит порядка полутора гигабайт (модель плюс torch);
      - каждый ставит число потоков по числу ядер, и два прогона дерутся за CPU,
        оба идут медленнее, чем шли бы по очереди;
      - gigaam качает веса НЕ атомарно (пишет прямо в целевой файл и считает
        существующий готовым), поэтому одновременный первый запуск с новой
        моделью оставляет битый кэш.

    Лок берется ДО тяжелых импортов: ждущий висит на десятках мегабайт, а не
    на полутора гигабайтах, поэтому очередь любой длины стоит по памяти как
    один прогон. Ядро снимает лок при завершении процесса, в том числе аварийном.
    """
    global _lock_fd
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    _lock_fd = open(LOCK_FILE, "a+", encoding="utf-8")  # "a+", не "w": не затираем данные держателя
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        holder = describe_holder(read_lock_info())
        if not wait:
            die(f"занято: на машине уже идет транскрибация - {holder}.\n"
                "Запусти позже или убери --no-wait, чтобы встать в очередь.")
        print(f"Жду очереди: сейчас {holder}", flush=True)
        print(f"(движок один на машину, прогоны идут последовательно; "
              f"жду не дольше {timeout // 60} мин)", flush=True)
        deadline = time.time() + timeout
        while True:
            try:
                fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() > deadline:
                    die(f"не дождался за {timeout // 60} мин: все еще "
                        f"{describe_holder(read_lock_info())}.\n"
                        "Если тот прогон завис - сними его вручную и запусти снова.")
                time.sleep(5)
    _lock_fd.seek(0)
    _lock_fd.truncate()
    json.dump({
        "pid": os.getpid(),
        "project": detect_project(source),
        "file": os.path.basename(source),
        "model": ASR_MODEL,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }, _lock_fd, ensure_ascii=False)
    _lock_fd.flush()


def mem_available_mb():
    """Свободная память в МБ; None - если система не дает ее дешево узнать."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        return None          # macOS и прочие без /proc - проверку пропускаем
    return None


def wait_for_memory(need_mb, wait, timeout):
    """Лок свободен - память может быть все равно занята (браузер, агенты, БД).

    Дешевле подождать, чем словить OOM: жертву при нехватке памяти выбирает
    ядро, а не мы, и ей становится не обязательно транскрибация.
    """
    free = mem_available_mb()
    if free is None or free >= need_mb:
        return
    if not wait:
        die(f"мало памяти: свободно {free} МБ, нужно от {need_mb} МБ. "
            "Освободи память или запусти позже.")
    print(f"Жду память: свободно {free} МБ, нужно от {need_mb} МБ", flush=True)
    deadline = time.time() + timeout
    while (mem_available_mb() or need_mb) < need_mb:
        if time.time() > deadline:
            die(f"память так и не освободилась за {timeout // 60} мин "
                f"(сейчас {mem_available_mb()} МБ). Прерываюсь, чтобы не ронять машину.")
        time.sleep(10)


def reexec_in_scope(memory_max):
    """Перезапуск себя в cgroup с лимитом памяти (systemd-run --user --scope).

    Без лимита превышение памяти обрабатывает ГЛОБАЛЬНЫЙ OOM-killer, и жертву он
    выбирает по всей системе - падает база, докер, что угодно, а не транскрибация.
    Внутри scope OOM срабатывает локально: худший исход - умер наш прогон, машина жива.

    Своп режем (MemorySwapMax=0): уход инференса в своп кладет отзывчивость машины
    не хуже нехватки памяти. Нет systemd (macOS, контейнер) - молча работаем как есть.
    """
    if os.environ.get("TRANSCRIBE_SCOPE") or not shutil.which("systemd-run"):
        return
    # перезапускаем через интерпретатор, а не сам файл: скрипт запускают и как
    # "python transcribe_meeting.py" (бита +x нет), и как установленную команду
    cmd = ["systemd-run", "--user", "--scope", "--quiet",
           "-p", f"MemoryMax={memory_max}", "-p", "MemorySwapMax=0",
           "--", sys.executable, os.path.abspath(sys.argv[0]), *sys.argv[1:]]
    env = dict(os.environ, TRANSCRIBE_SCOPE="1")
    try:
        os.execvpe(cmd[0], cmd, env)
    except OSError as exc:                       # нет systemd-user - работаем как есть
        print(f"Предупреждение: не удалось ограничить память через systemd-run ({exc}); "
              "иду без лимита", file=sys.stderr, flush=True)


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


def run_asr(audio, sr, tmpdir, threads):
    import gigaam, soundfile as sf
    if threads > 0:
        import torch
        torch.set_num_threads(threads)
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


# --- Диаризация: sherpa только сегментирует речь, кластеризуем сами. ---
# Встроенная FastClustering у sherpa вырождается на длинных файлах (все в 1 спикера
# либо сотни микрокластеров из-за дрейфа эмбеддингов). Поэтому: берем сегменты речи,
# считаем эмбеддинг каждого, и кластеризуем сферическим k-means (взвешенным по длине)
# с известным N. Проверено против облака: на длинных репликах ~95% совпадение.

def _diar_models_ok(emb):
    return os.path.exists(SEG_MODEL) and emb and os.path.exists(emb)


def _segment(audio, threads):
    """Временные границы речи (лейблы sherpa игнорируем - кластеризуем сами)."""
    import sherpa_onnx
    nt = threads if threads > 0 else 1
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG_MODEL),
            num_threads=nt),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=EMB["campplus"], num_threads=nt),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=2),
        min_duration_on=DIAR_MIN_ON, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)
    return [(r.start, r.end) for r in sd.process(audio).sort_by_start_time()]


def _embed_segments(audio, segs, emb_path, threads):
    """Эмбеддинг каждого сегмента (по всему его аудио). Возвращает (матрица, оставленные сегменты)."""
    import numpy as np, sherpa_onnx
    nt = threads if threads > 0 else 1
    ext = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb_path, num_threads=nt))
    X, kept = [], []
    for s0, s1 in segs:
        a = audio[int(s0*SR):int(s1*SR)]
        if len(a) < int(0.25*SR):     # слишком коротко для надежного эмбеддинга
            continue
        st = ext.create_stream(); st.accept_waveform(SR, a); st.input_finished()
        X.append(np.asarray(ext.compute(st), dtype="float32")); kept.append((s0, s1))
    if not X:
        return None, []
    return np.vstack(X), kept


def _skmeans(Xn, w, k, iters=100):
    """Сферический k-means (косинус) на L2-нормированных Xn, веса w (длительности)."""
    import numpy as np
    n = len(Xn)
    k = max(1, min(k, n))
    if k == 1:
        return np.zeros(n, dtype=int)
    # инициализация: жадно самые непохожие центры среди длинных сегментов
    li = np.argsort(-w)[:max(k*10, 40)]
    idx = [int(li[int(np.argmax(w[li]))])]
    for _ in range(k-1):
        sims = (Xn @ Xn[idx].T).max(1)
        idx.append(int(np.argmin(sims)))
    C = Xn[idx].copy()
    lab = np.full(n, -1)
    for _ in range(iters):
        new = (Xn @ C.T).argmax(1)
        if (new == lab).all():
            break
        lab = new
        for c in range(k):
            m = lab == c
            if m.any():
                v = (Xn[m] * w[m, None]).sum(0)
                C[c] = v / (np.linalg.norm(v) + 1e-9)
    return lab


def _silhouette(Xn, lab):
    import numpy as np
    D = 1.0 - Xn @ Xn.T
    np.fill_diagonal(D, 0.0)
    labs = np.unique(lab)
    if len(labs) < 2:
        return -1.0
    s = np.zeros(len(lab))
    for i in range(len(lab)):
        same = lab == lab[i]; same[i] = False
        a = D[i, same].mean() if same.any() else 0.0
        others = [D[i, lab == c].mean() for c in labs if c != lab[i]]
        b = min(others) if others else 0.0
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(s.mean())


def run_diar(audio, n_speakers, emb_key, threads):
    import numpy as np
    emb = EMB.get(emb_key)
    if not _diar_models_ok(emb):
        die(f"нет моделей диаризации в {DIAR_DIR} (запусти install.sh)")
    segs = _segment(audio, threads)
    if not segs:
        return []
    X, kept = _embed_segments(audio, segs, emb, threads)
    if X is None:
        return []
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    w = np.array([s1 - s0 for s0, s1 in kept], dtype="float32")
    if n_speakers and n_speakers > 0:
        lab = _skmeans(Xn, w, n_speakers)
    else:
        # авто-оценка числа спикеров по silhouette (порог у sherpa ненадежен)
        best_s, lab = -2.0, np.zeros(len(Xn), dtype=int)
        for k in range(2, min(DIAR_KMAX, len(Xn)) + 1):
            l = _skmeans(Xn, w, k)
            if len(np.unique(l)) < k:
                continue
            s = _silhouette(Xn, l)
            if s > best_s:
                best_s, lab = s, l
    # перенумеровать метки в непрерывные 0..m-1 по времени первого появления
    remap, nxt = {}, 0
    for c in lab:
        if c not in remap:
            remap[c] = nxt; nxt += 1
    return [(kept[i][0], kept[i][1], remap[int(lab[i])]) for i in range(len(kept))]


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
                    f"(разметка надежна на длинных репликах, короткие могут путаться; "
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
    ap.add_argument("--threads", type=int, default=0, help="число CPU-потоков (0 = авто/дефолт библиотек)")
    ap.add_argument("--no-wait", action="store_true",
                    help="не ждать очереди и памяти, а сразу выйти, если занято")
    ap.add_argument("--wait-timeout", type=int, default=7200,
                    help="сколько секунд ждать очереди/памяти (по умолчанию 2 ч)")
    ap.add_argument("--min-free-mb", type=int, default=2500,
                    help="не стартовать, пока свободной памяти меньше этого (МБ; только Linux)")
    ap.add_argument("--memory-max", default="4G",
                    help="лимит памяти cgroup для прогона (systemd-run --scope; только Linux)")
    ap.add_argument("--no-limit", action="store_true",
                    help="не заворачивать прогон в cgroup с лимитом памяти")
    args = ap.parse_args()

    # до всего остального: перезапуск себя под лимитом памяти, чтобы промах
    # убивал только транскрибацию, а не случайный процесс на машине
    if not args.no_limit:
        reexec_in_scope(args.memory_max)

    if args.threads and args.threads > 0:
        for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            os.environ[_v] = str(args.threads)

    src = os.path.abspath(os.path.expanduser(args.input))
    if not os.path.exists(src):
        die(f"файл не найден: {src}")
    out = os.path.abspath(os.path.expanduser(args.output)) if args.output else os.path.splitext(src)[0] + ".md"
    title = args.title or os.path.splitext(os.path.basename(src))[0]

    # очередь и память - ДО тяжелых импортов ниже: ждущий процесс должен стоить
    # десятки мегабайт, а не полтора гигабайта загруженной модели
    acquire_lock(src, wait=not args.no_wait, timeout=args.wait_timeout)
    wait_for_memory(args.min_free_mb, wait=not args.no_wait, timeout=args.wait_timeout)

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
        words = run_asr(audio, sr, _tmpdir, args.threads)
        print(f"      слов={len(words)}, {time.time()-t:.0f}с", file=sys.stderr)

        if args.no_diar:
            lines, diarized = group_by_pause(words), False
        else:
            print("[3/4] диаризация...", file=sys.stderr)
            t = time.time()
            segs = run_diar(audio, args.speakers, args.embedding, args.threads)
            print(f"      {time.time()-t:.0f}с", file=sys.stderr)
            lines, diarized = group_by_speaker(words, segs), True

        print("[4/4] пишу markdown...", file=sys.stderr)
        write_md(out, title, src, dur, lines, diarized, args.embedding)
        print(out)   # stdout = путь к результату (для пайпов)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
