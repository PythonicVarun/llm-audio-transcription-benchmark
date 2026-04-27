#!/usr/bin/env bash
set -euo pipefail

DATASET_DIR="${1:-dataset}"
MODEL="${2:-base}"
LANGUAGE="${3:-auto}"

if ! command -v whisper >/dev/null 2>&1; then
  echo "Error: whisper CLI not found. Install with: pip install -U openai-whisper" >&2
  exit 1
fi

if [[ ! -d "$DATASET_DIR" ]]; then
  echo "Error: dataset directory not found: $DATASET_DIR" >&2
  exit 1
fi

echo "Scanning: $DATASET_DIR"
echo "Model: $MODEL | Language: $LANGUAGE"

find "$DATASET_DIR" -type f \( \
  -iname "*.wav"  -o \
  -iname "*.mp3"  -o \
  -iname "*.flac" -o \
  -iname "*.m4a"  -o \
  -iname "*.ogg"  -o \
  -iname "*.opus" -o \
  -iname "*.aac"  -o \
  -iname "*.wma"  -o \
  -iname "*.mp4"  -o \
  -iname "*.mkv" \
\) -print0 | while IFS= read -r -d '' audio_file; do
  dir="$(dirname "$audio_file")"
  name="$(basename "$audio_file")"
  stem="${name%.*}"
  out_file="${dir}/${stem}.whisper.txt"

  if [[ -f "$out_file" ]]; then
    echo "Skip (exists): $out_file"
    continue
  fi

  tmp_dir="$(mktemp -d)"
  whisper_args=(
    --model "$MODEL"
    --task transcribe
    --output_format txt
    --output_dir "$tmp_dir"
    --fp16 False
  )

  if [[ "${LANGUAGE,,}" != "auto" ]]; then
    whisper_args+=(--language "$LANGUAGE")
  fi

  echo "Transcribing: $audio_file"
  whisper "$audio_file" "${whisper_args[@]}"

  generated="${tmp_dir}/${stem}.txt"
  if [[ ! -f "$generated" ]]; then
    generated="$(find "$tmp_dir" -maxdepth 1 -type f -name "*.txt" -print -quit || true)"
  fi

  if [[ -z "${generated:-}" || ! -f "$generated" ]]; then
    echo "Failed: no transcript produced for $audio_file" >&2
    rm -rf "$tmp_dir"
    continue
  fi

  mv "$generated" "$out_file"
  rm -rf "$tmp_dir"
  echo "Saved: $out_file"
done

echo "Done."
