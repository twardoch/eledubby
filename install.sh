#!/usr/bin/env bash
# install.sh — Install eledubby locally
# eledubby is a voice dubbing tool using ElevenLabs API for speech-to-speech conversion
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Installing eledubby..."
uv pip install -e . 2>/dev/null || pip install -e . 2>/dev/null || echo "Install failed"
echo "Done."
