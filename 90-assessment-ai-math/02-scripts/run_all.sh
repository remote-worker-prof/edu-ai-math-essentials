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

build_student_release() {
  local tests_root="${WORKSPACE_ROOT}/05-tests-v1v2"
  local release_dir="${tests_root}/student_release"
  log "building student release package"
  rm -rf "${release_dir}"
  mkdir -p "${release_dir}"

  cp "${tests_root}/test_variant_1.md" "${release_dir}/test_variant_1.md"
  cp "${tests_root}/test_variant_2.md" "${release_dir}/test_variant_2.md"
  cp "${tests_root}/answer_template_1.md" "${release_dir}/answer_template_1.md"
  cp "${tests_root}/answer_template_2.md" "${release_dir}/answer_template_2.md"

  cat > "${release_dir}/README_student.md" <<'EOF'
# Пакет материалов для студентов

Состав пакета:
- `test_variant_1.md`
- `test_variant_2.md`
- `answer_template_1.md`
- `answer_template_2.md`

Инструкция:
1. Выберите вариант теста и соответствующий шаблон ответов.
2. Заполните поля ФИО, номер группы и email.
3. Для закрытых вопросов укажите номера выбранных вариантов.
4. Для открытых вопросов дайте краткий структурированный ответ.

В пакет не входят ключи ответов и технические внутренние файлы.
EOF
}

run_step "01_inventory_materials.py"
run_step "02_extract_content.py"
run_step "03_topic_mapping.py"
run_step "04_build_topic_context.py"
run_step "05_prepare_memory_payload.py"
run_step "06_build_integrated_modules.py"
run_step "07_generate_module_tests.py"
run_step "08_validate_generated_tests.py"
build_student_release

log "pipeline completed successfully (${MODE})"
