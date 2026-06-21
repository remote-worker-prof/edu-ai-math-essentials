#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${WORKSPACE_ROOT}/.." && pwd)"

MODE="resume"
if [[ "${1:-}" == "--clean" ]]; then
  MODE="clean"
elif [[ "${1:-}" == "--resume" || -z "${1:-}" ]]; then
  MODE="resume"
else
  echo "Usage: bash ${SCRIPT_DIR}/run_all.sh [--clean|--resume]"
  exit 2
fi

log() {
  echo "[run_all] $1"
}

clean_outputs() {
  log "clean mode: removing generated outputs"
  rm -rf "${WORKSPACE_ROOT}/03-topic-context/00_index"
  rm -rf "${WORKSPACE_ROOT}/03-topic-context/01_extracted"
  rm -rf "${WORKSPACE_ROOT}/03-topic-context/02_topics"
  rm -rf "${WORKSPACE_ROOT}/03-topic-context/03_topics"
  rm -f "${WORKSPACE_ROOT}/03-topic-context/TOPIC_INDEX.md"
  rm -f "${WORKSPACE_ROOT}/03-topic-context/topic_index.json"
  rm -f "${WORKSPACE_ROOT}/03-topic-context/MODULE_INDEX.md"
  rm -f "${WORKSPACE_ROOT}/03-topic-context/module_index.json"
  rm -rf "${WORKSPACE_ROOT}/04_modules"
  if [[ -d "${WORKSPACE_ROOT}/05-tests-v1v2" ]]; then
    find "${WORKSPACE_ROOT}/05-tests-v1v2" -mindepth 1 -maxdepth 1 ! -name "_backup" -exec rm -rf {} +
  fi
  rm -rf "${WORKSPACE_ROOT}/99_memory"

  mkdir -p "${WORKSPACE_ROOT}/03-topic-context"
  mkdir -p "${WORKSPACE_ROOT}/04_modules"
  mkdir -p "${WORKSPACE_ROOT}/05-tests-v1v2"
  mkdir -p "${WORKSPACE_ROOT}/05-tests-v1v2/_backup"
  mkdir -p "${WORKSPACE_ROOT}/99_memory"
}

if [[ "${MODE}" == "clean" ]]; then
  clean_outputs
else
  log "resume mode: keeping existing generated artifacts"
fi

run_step() {
  local script_name="$1"
  log "running ${script_name}"
  python3 "${SCRIPT_DIR}/${script_name}" \
    --project-root "${PROJECT_ROOT}" \
    --workspace-root "90-assessment-ai-math"
}

run_step "01_inventory_materials.py"
run_step "02_extract_content.py"
run_step "03_topic_mapping.py"
run_step "04_build_topic_context.py"
run_step "05_prepare_memory_payload.py"
run_step "06_build_integrated_modules.py"
run_step "07_generate_module_tests.py"
run_step "08_validate_generated_tests.py"

log "pipeline completed successfully (${MODE})"
