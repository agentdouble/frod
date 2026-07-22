#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"

exec uv run python -m fraude_detector.config_cli --config "$project_dir/config.yaml"
