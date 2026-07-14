#!/usr/bin/env bash
# Установщик giga-transcribe.
# Создает самодостаточный каталог (venv + GigaAM + модели диаризации), затем связывает:
#   - launcher CLI в ~/.local/bin/transcribe-meeting
#   - обертку Quick Action в ~/.local/bin/transcribe-quickaction
#   - (macOS) Быстрое действие Finder в ~/Library/Services
#
# Каталог установки переопределяется через GIGA_TRANSCRIBE_HOME (дефолт: ~/.local/share/giga-transcribe).
# Флаг --titanet - доп. качает более крупный эмбеддинг titanet (сильнее, ~100МБ, медленнее).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOME_DIR="${GIGA_TRANSCRIBE_HOME:-$HOME/.local/share/giga-transcribe}"
MODELS="$HOME_DIR/models"
BIN="$HOME/.local/bin"
WITH_TITANET=0
[ "${1:-}" = "--titanet" ] && WITH_TITANET=1

echo "==> каталог установки: $HOME_DIR"
mkdir -p "$HOME_DIR" "$MODELS" "$BIN"

# --- предпосылки ---
command -v python3 >/dev/null 2>&1 || { echo "ОШИБКА: python3 не найден"; exit 1; }
command -v git     >/dev/null 2>&1 || { echo "ОШИБКА: git не найден"; exit 1; }
command -v ffmpeg  >/dev/null 2>&1 || echo "ВНИМАНИЕ: ffmpeg не найден - поставь (brew install ffmpeg / apt install ffmpeg)"

# --- venv ---
if [ ! -x "$HOME_DIR/venv/bin/python" ]; then
  echo "==> создаю venv"
  python3 -m venv "$HOME_DIR/venv"
fi
PY="$HOME_DIR/venv/bin/python"
"$PY" -m pip install -U pip >/dev/null

# --- GigaAM (из git; на PyPI устаревшая версия без v3) ---
if [ ! -d "$HOME_DIR/GigaAM" ]; then
  echo "==> клонирую GigaAM"
  git clone --depth 1 https://github.com/salute-developers/GigaAM.git "$HOME_DIR/GigaAM"
fi
echo "==> ставлю GigaAM + зависимости (torch, sherpa-onnx, ...)"
"$PY" -m pip install -e "$HOME_DIR/GigaAM[torch]"
"$PY" -m pip install sherpa-onnx certifi soundfile

# --- модели диаризации (token-free ONNX из релизов sherpa-onnx) ---
if [ ! -f "$MODELS/sherpa-onnx-pyannote-segmentation-3-0/model.onnx" ]; then
  echo "==> качаю модель сегментации"
  curl -fsSL -o "$MODELS/seg.tar.bz2" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
  tar xjf "$MODELS/seg.tar.bz2" -C "$MODELS"
  rm -f "$MODELS/seg.tar.bz2"
fi
if [ ! -f "$MODELS/embedding.onnx" ]; then
  echo "==> качаю speaker-эмбеддинг (campplus)"
  curl -fsSL -o "$MODELS/embedding.onnx" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
fi
if [ "$WITH_TITANET" = 1 ] && [ ! -f "$MODELS/titanet.onnx" ]; then
  echo "==> качаю эмбеддинг titanet (опционально)"
  curl -fsSL -o "$MODELS/titanet.onnx" \
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_large.onnx"
fi

# --- скрипты + launcher ---
echo "==> ставлю скрипты"
cp "$REPO/transcribe_meeting.py" "$HOME_DIR/transcribe_meeting.py"
cp "$REPO/bin/transcribe-quickaction" "$BIN/transcribe-quickaction"
chmod +x "$BIN/transcribe-quickaction"

cat > "$BIN/transcribe-meeting" <<EOF
#!/bin/sh
export GIGA_TRANSCRIBE_HOME="$HOME_DIR"
exec "$HOME_DIR/venv/bin/python" "$HOME_DIR/transcribe_meeting.py" "\$@"
EOF
chmod +x "$BIN/transcribe-meeting"

# --- macOS Быстрое действие ---
if [ "$(uname)" = "Darwin" ]; then
  echo "==> ставлю Быстрое действие Finder"
  SVC="$HOME/Library/Services/Транскрибировать (GigaAM).workflow"
  rm -rf "$SVC"
  mkdir -p "$SVC/Contents"
  cp "$REPO/quick-action/Транскрибировать (GigaAM).workflow/Contents/"* "$SVC/Contents/"
  /System/Library/CoreServices/pbs -flush 2>/dev/null || true
  echo "    (правый клик по файлу -> Службы -> \"Транскрибировать (GigaAM)\"; если нет - killall Finder)"
fi

case ":$PATH:" in *":$BIN:"*) ;; *) echo "ПРИМЕЧАНИЕ: добавь $BIN в PATH";; esac

echo
echo "Готово. Пробуй:  transcribe-meeting \"запись.mp4\" --speakers 3"
