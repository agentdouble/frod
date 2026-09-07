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

config_path="${FROD_CONFIG:-$project_dir/config.yaml}"
IFS=$'\t' read -r \
  configured_host \
  configured_port \
  configured_max_upload \
  gapl_enabled \
  gapl_path \
  < <(
    uv run python - "$config_path" <<'PY'
import sys

from fraude_detector.run_config import load_run_config

config = load_run_config(sys.argv[1])
print(
    "\t".join(
        (
            config.application.host,
            str(config.application.port),
            str(config.application.max_upload_size_mb),
            str(config.gapl.enabled).lower(),
            str(config.gapl.weights_path),
        )
    )
)
PY
  )

gapl_url="https://huggingface.co/AbyssLumine/GAPL/resolve/main/checkpoint.pt"
gapl_sha256="ffbcb5eb526f0df0fd197d7266bdd0325b66813e95010f1285685acf2d267235"

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

download_file() {
  local url="$1"
  local output="$2"
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --progress-bar "$url" --output "$output"
  elif command -v wget >/dev/null 2>&1; then
    wget --show-progress --output-document="$output" "$url"
  else
    printf '%s\n' "curl ou wget est requis pour telecharger les modeles."
    exit 1
  fi
}

if [[ "$gapl_enabled" == true ]]; then
  gapl_ready=false
  if [[ -f "$gapl_path" ]] && [[ "$(file_sha256 "$gapl_path")" == "$gapl_sha256" ]]; then
    gapl_ready=true
  fi

  if [[ "$gapl_ready" != true ]]; then
    printf '%s\n' "Telechargement du modele GAPL (premier demarrage uniquement)..."
    mkdir -p "$(dirname -- "$gapl_path")"
    partial_path="${gapl_path}.part"
    trap 'rm -f -- "$partial_path"' EXIT
    download_file "$gapl_url" "$partial_path"
    if [[ "$(file_sha256 "$partial_path")" != "$gapl_sha256" ]]; then
      printf '%s\n' "Le modele telecharge est invalide (SHA-256 incorrect)."
      exit 1
    fi
    mv -f -- "$partial_path" "$gapl_path"
    trap - EXIT
  fi
fi

requested_port="$configured_port"
port="$(
  uv run python - "$configured_host" "$requested_port" <<'PY'
import socket
import sys

host = sys.argv[1]
start = int(sys.argv[2])
for candidate in range(start, start + 20):
    with socket.socket() as sock:
        try:
            sock.bind((host, candidate))
        except OSError:
            continue
    print(candidate)
    break
else:
    raise SystemExit("Aucun port local disponible.")
PY
)"

export FROD_CONFIG="$config_path"
export FROD_GAPL_WEIGHTS="$gapl_path"
printf '\nApplication disponible sur http://%s:%s\n\n' "$configured_host" "$port"
exec uv run streamlit run app/streamlit_app.py \
  --server.address "$configured_host" \
  --server.port "$port" \
  --server.maxUploadSize "$configured_max_upload" \
  --server.headless true \
  --browser.gatherUsageStats false
