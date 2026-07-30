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
  trufor_enabled \
  trufor_path \
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
            str(config.trufor.enabled).lower(),
            str(config.trufor.weights_path),
        )
    )
)
PY
  )

gapl_url="https://huggingface.co/AbyssLumine/GAPL/resolve/main/checkpoint.pt"
gapl_sha256="ffbcb5eb526f0df0fd197d7266bdd0325b66813e95010f1285685acf2d267235"

trufor_url="https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip"
trufor_archive_md5="7bee48f3476c75616c3c5721ab256ff8"
trufor_sha256="ac1d90e329a72e0d66e8665e123a19e94bfae3209c3ef8a4f9ca3b91578c7844"

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

file_md5() {
  if command -v md5sum >/dev/null 2>&1; then
    md5sum "$1" | awk '{print $1}'
  elif command -v md5 >/dev/null 2>&1; then
    md5 -q "$1"
  else
    uv run python -c \
      'import hashlib, sys; print(hashlib.file_digest(open(sys.argv[1], "rb"), "md5").hexdigest())' \
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

if [[ "$trufor_enabled" == true ]]; then
  trufor_ready=false
  if [[ -f "$trufor_path" ]] && [[ "$(file_sha256 "$trufor_path")" == "$trufor_sha256" ]]; then
    trufor_ready=true
  fi

  if [[ "$trufor_ready" != true ]]; then
    printf '%s\n' "Telechargement du modele TruFor (premier demarrage uniquement)..."
    mkdir -p "$(dirname -- "$trufor_path")"
    trufor_archive="${trufor_path}.zip.part"
    trufor_partial="${trufor_path}.part"
    trap 'rm -f -- "$trufor_archive" "$trufor_partial"' EXIT
    download_file "$trufor_url" "$trufor_archive"
    if [[ "$(file_md5 "$trufor_archive")" != "$trufor_archive_md5" ]]; then
      printf '%s\n' "L'archive TruFor telechargee est invalide (MD5 incorrect)."
      exit 1
    fi
    uv run python - "$trufor_archive" "$trufor_partial" <<'PY'
import shutil
import sys
from pathlib import Path
from zipfile import ZipFile

archive = Path(sys.argv[1])
destination = Path(sys.argv[2])
with ZipFile(archive) as zipped:
    matches = [
        name for name in zipped.namelist()
        if name.rstrip("/").endswith("/trufor.pth.tar")
        or name.rstrip("/") == "trufor.pth.tar"
    ]
    if len(matches) != 1:
        raise SystemExit("Checkpoint TruFor absent ou ambigu dans l'archive.")
    with zipped.open(matches[0]) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)
PY
    if [[ "$(file_sha256 "$trufor_partial")" != "$trufor_sha256" ]]; then
      printf '%s\n' "Le checkpoint TruFor extrait est invalide (SHA-256 incorrect)."
      exit 1
    fi
    mv -f -- "$trufor_partial" "$trufor_path"
    rm -f -- "$trufor_archive"
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
export FROD_TRUFOR_WEIGHTS="$trufor_path"
printf '\nApplication disponible sur http://%s:%s\n\n' "$configured_host" "$port"
exec uv run streamlit run app/streamlit_app.py \
  --server.address "$configured_host" \
  --server.port "$port" \
  --server.maxUploadSize "$configured_max_upload" \
  --server.headless true \
  --browser.gatherUsageStats false
