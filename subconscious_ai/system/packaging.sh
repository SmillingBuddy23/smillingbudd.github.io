#!/usr/bin/env bash
set -euo pipefail
python -m pip install pyinstaller
pyinstaller --onefile main.py --name subconscious_ai
