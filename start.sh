#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

if ! command -v uv >/dev/null 2>&1; then
  printf '%s\n' \
    "uv est requis pour demarrer Frod." \
    "Installation : https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

printf '%s\n' "Installation et verification des dependances..."
uv sync --extra ai --locked

model_url="https://huggingface.co/AbyssLumine/GAPL/resolve/main/checkpoint.pt"
model_sha256="ffbcb5eb526f0df0fd197d7266bdd0325b66813e95010f1285685acf2d267235"
model_path="${FROD_GAPL_WEIGHTS:-$project_dir/models/gapl/checkpoint.pt}"

file_sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    uv run python -c \
      'import hashlib, sys; print(hashlib.file_digest(open(sys.argv[1], "rb"), "sha256").hexdigest())' \
      "$1"
  fi
}

model_ready=false
if [[ -f "$model_path" ]] && [[ "$(file_sha256 "$model_path")" == "$model_sha256" ]]; then
  model_ready=true
fi

if [[ "$model_ready" != true ]]; then
  printf '%s\n' "Telechargement du modele GAPL (premier demarrage uniquement)..."
  mkdir -p "$(dirname -- "$model_path")"
  partial_path="${model_path}.part"
  trap 'rm -f -- "$partial_path"' EXIT
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --progress-bar "$model_url" --output "$partial_path"
  elif command -v wget >/dev/null 2>&1; then
    wget --show-progress --output-document="$partial_path" "$model_url"
  else
    printf '%s\n' "curl ou wget est requis pour telecharger le modele."
    exit 1
  fi
  if [[ "$(file_sha256 "$partial_path")" != "$model_sha256" ]]; then
    printf '%s\n' "Le modele telecharge est invalide (SHA-256 incorrect)."
    exit 1
  fi
  mv -f -- "$partial_path" "$model_path"
  trap - EXIT
fi

requested_port="${FROD_PORT:-8501}"
port="$(
  uv run python - "$requested_port" <<'PY'
import socket
import sys

start = int(sys.argv[1])
for candidate in range(start, start + 20):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", candidate))
        except OSError:
            continue
    print(candidate)
    break
else:
    raise SystemExit("Aucun port local disponible.")
PY
)"

export FROD_GAPL_WEIGHTS="$model_path"
printf '\nFrod est disponible sur http://localhost:%s\n\n' "$port"
exec uv run streamlit run app/streamlit_app.py \
  --server.address 127.0.0.1 \
  --server.port "$port" \
  --server.headless true \
  --browser.gatherUsageStats false
