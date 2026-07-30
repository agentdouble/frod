#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ocr_worker_dir="${FROD_OCR_WORKER_DIR:-$project_dir/../ocr-worker}"
ocr_url="${FROD_OCR_URL:-http://127.0.0.1:8007}"

if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' \
    "uv est requis pour demarrer le serveur OCR." \
    "Installation : https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if [[ ! -d "$ocr_worker_dir" ]]; then
  printf 'Worker OCR introuvable : %s\n' "$ocr_worker_dir"
  printf '%s\n' \
    "Definissez FROD_OCR_WORKER_DIR vers le depot qui contient le module glmocr."
  exit 1
fi

if [[ ! -f "$ocr_worker_dir/pyproject.toml" ]]; then
  printf 'Configuration Python introuvable dans : %s\n' "$ocr_worker_dir"
  exit 1
fi

cd "$ocr_worker_dir"
unset VIRTUAL_ENV

printf '%s\n' "Installation et verification des dependances OCR..."
uv sync --group model

printf 'Demarrage du serveur OCR pour Frod (%s)...\n' "$ocr_url"
printf '%s\n' \
  "Dans un autre terminal, lancez Frod avec :" \
  "FROD_OCR_URL=$ocr_url ./start.sh"

export FROD_OCR_URL="$ocr_url"
export PYTHONPATH=src
exec uv run --with torch --with torchvision python -m glmocr.server
