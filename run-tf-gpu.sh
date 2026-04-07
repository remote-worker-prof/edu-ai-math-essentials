#!/usr/bin/env bash
set -euo pipefail
cd /data/projects/edu-ai-math-essentials
~/.local/bin/tf-gpu-run /data/projects/edu-ai-math-essentials "$@"
