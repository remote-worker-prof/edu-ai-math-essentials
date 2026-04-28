"""Массово нормализует учебные notebook-и без изменения учебных постановок.

Скрипт делает безопасные текстовые преобразования в source ячеек:
- переводит docstring-секции `Args/Returns/Raises` в русские секции;
- русифицирует ограниченный набор англоязычных учебных комментариев;
- приводит простые однострочные `np.array([...], dtype=...)` к многострочному виду;
- синхронизирует source `runs_*`-ноутбуков с каноническими solution-ноутбуками,
  сохраняя outputs/metadata целевых файлов.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
THEMES_DIR = ROOT / "themes"

SECTION_REPLACEMENTS = (
    ("Args:", "Аргументы:"),
    ("Returns:", "Возвращает:"),
    ("Raises:", "Исключения:"),
)

COMMENT_REPLACEMENTS = (
    ("# Mini-check", "# Мини-проверка"),
    (
        "# Reuse-block from `03-Transformer / ЛР01` is given ready above.",
        "# Блок переиспользования из `03-Transformer / ЛР01` уже подготовлен выше.",
    ),
    ("# Step ", "# Шаг "),
    ("# Hint:", "# Подсказка:"),
)

SIMPLE_ARRAY_PATTERN = re.compile(
    r"(?P<indent>[ \t]*)"
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*np\.array\(\["
    r"(?P<body>[^\[\]\n]+)"
    r"\],\s*dtype=(?P<dtype>[A-Za-z0-9_\.]+)\)"
)

RUN_SYNC_MAP: dict[str, list[str]] = {
    "themes/04-Autoregression/lab/solutions/02_decoder_only_tiny_shakespeare_gpu_solution.ipynb": [
        "themes/04-Autoregression/lab/solutions/runs_acceptance/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_acceptance/02_decoder_only_tiny_shakespeare_gpu_solution.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_g635_demo/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_g635_demo/02_decoder_only_tiny_shakespeare_gpu_solution.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_no_gpu_expected_fail/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_quick/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_quick/02_decoder_only_tiny_shakespeare_gpu_solution.attempt2.gpu_60m_boost.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_quick/02_decoder_only_tiny_shakespeare_gpu_solution.profile_gpu60m_budget5.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_quick_tf/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_quick_tf/02_decoder_only_tiny_shakespeare_gpu_solution.attempt2.gpu_60m_boost.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_quick_tf/02_decoder_only_tiny_shakespeare_gpu_solution.profile_gpu60m_budget5_tf.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_quick_tf/02_decoder_only_tiny_shakespeare_gpu_solution.profile_gpu60m_budget5_tf2.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_quick_tf/02_decoder_only_tiny_shakespeare_gpu_solution.profile_gpu60m_budget5_tf3.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_quick_tf/02_decoder_only_tiny_shakespeare_gpu_solution.profile_gpu60m_budget5_tf4.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_stability_a/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_tf221_local_gpu_demo/02_decoder_only_tiny_shakespeare_gpu_solution.attempt1.gpu_60m.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_tf221_local_gpu_demo/02_decoder_only_tiny_shakespeare_gpu_solution.attempt2.gpu_60m_boost.executed.ipynb",
        "themes/04-Autoregression/lab/solutions/runs_tf221_local_gpu_demo/02_decoder_only_tiny_shakespeare_gpu_solution.executed.ipynb",
    ],
    "themes/05-Full-Transformer/lab/solutions/01_full_transformer_wikitext2_solution.ipynb": [
        "themes/05-Full-Transformer/lab/solutions/runs_gpu_full/01_full_transformer_wikitext2_solution.gpu_full_allow_errors.executed.ipynb",
        "themes/05-Full-Transformer/lab/solutions/runs_gpu_full_tuned/01_full_transformer_wikitext2_solution.gpu_full_tuned.executed.ipynb",
        "themes/05-Full-Transformer/lab/solutions/runs_gpu_full_tuned2/01_full_transformer_wikitext2_solution.gpu_full_tuned2.executed.ipynb",
        "themes/05-Full-Transformer/lab/solutions/runs_gpu_full_tuned3/01_full_transformer_wikitext2_solution.gpu_full_tuned3.executed.ipynb",
    ],
}

BEGINNER_LAYER_TITLE = "## Beginner-слой: как читать эту тетрадь с нуля"
FLOW_MARKER_LINE = (
    "Поток изучения: контракт -> теория -> ручной пример -> TODO -> проверки -> диагностика."
)
FOUNDATIONS_README = ROOT / "themes" / "00-Foundations" / "README.md"
THEME_PREVIOUS_STEP = {
    "00-Foundations": "Стартовая точка курса: с нуля к формам, маскам и базовым метрикам.",
    "01-RNN": "После Foundations: формы последовательностей и базовый словарь token/mask уже знакомы.",
    "02-Attention": "После 01-RNN / ЛР03: вы уже умеете читать seq2seq и decoder shift.",
    "03-Transformer": "После 02-Attention: понятны query/key/value и карта attention-соответствий.",
    "04-Autoregression": "После 03-Transformer: понятны self-attention и роль positional/масок.",
    "05-Full-Transformer": "После 04-Autoregression: закреплены causal mask, leakage checks и perplexity gates.",
}


def iter_notebooks() -> list[Path]:
    """Возвращает отсортированный список всех notebook-ов в themes."""

    return sorted(THEMES_DIR.rglob("*.ipynb"))


def is_run_notebook(relative_path: str) -> bool:
    """Возвращает True для executed run-ноутбуков."""

    return "/runs_" in relative_path


def as_relative_link(source_notebook: Path, target_file: Path) -> str:
    """Строит относительную markdown-ссылку от notebook к целевому файлу."""

    link = os.path.relpath(target_file, source_notebook.parent)
    return Path(link).as_posix()


def normalize_simple_array(match: re.Match[str]) -> str:
    """Разворачивает простую однострочную 1D np.array в многострочный блок."""

    indent = match.group("indent")
    name = match.group("name")
    body = match.group("body")
    dtype = match.group("dtype")

    values = [item.strip() for item in body.split(",") if item.strip()]
    if len(values) <= 1:
        return match.group(0)

    rows = "\n".join(f"{indent}        {value}," for value in values)
    return (
        f"{indent}{name} = np.array(\n"
        f"{indent}    [\n"
        f"{rows}\n"
        f"{indent}    ],\n"
        f"{indent}    dtype={dtype},\n"
        f"{indent})"
    )


def normalize_source(source: str) -> str:
    """Применяет безопасные текстовые преобразования к source ячейки."""

    updated = source
    for old, new in SECTION_REPLACEMENTS:
        updated = updated.replace(old, new)
    for old, new in COMMENT_REPLACEMENTS:
        updated = updated.replace(old, new)
    updated = SIMPLE_ARRAY_PATTERN.sub(normalize_simple_array, updated)
    return updated


def build_beginner_scaffold(path: Path) -> str | None:
    """Строит markdown-врезку beginner-first для non-run notebook-а."""

    relative_path = path.relative_to(ROOT).as_posix()
    if is_run_notebook(relative_path):
        return None

    parts = Path(relative_path).parts
    if len(parts) < 2 or parts[0] != "themes":
        return None

    theme_name = parts[1]
    theory_path = ROOT / "themes" / theme_name / "theory" / "theory.md"
    if not theory_path.exists():
        return None

    theory_link = as_relative_link(path, theory_path)
    foundations_link = as_relative_link(path, FOUNDATIONS_README)
    previous_step = THEME_PREVIOUS_STEP.get(
        theme_name,
        "Продолжайте в том же ритме: сначала контракт данных и маски, затем формализация.",
    )

    return (
        f"{BEGINNER_LAYER_TITLE}\n\n"
        "### Кому читать\n"
        "- Если вы стартуете с нуля и хотите понимать, зачем каждый блок идёт именно в таком порядке.\n"
        "- Если формулы пока тяжёлые, а опора нужна через интуицию и ручной мини-пример.\n\n"
        "### Что изменилось после прошлого шага\n"
        f"- {previous_step}\n\n"
        "### Теоретический ориентир\n"
        f"- Теория этой темы: [{theory_link}]({theory_link})\n"
        f"- Общий вход курса: [{foundations_link}]({foundations_link})\n\n"
        f"{FLOW_MARKER_LINE}\n"
    )


def inject_beginner_scaffold(path: Path, notebook: dict) -> bool:
    """Добавляет beginner-врезку в notebook, если её ещё нет."""

    scaffold = build_beginner_scaffold(path)
    if scaffold is None:
        return False

    markdown_blob = "\n".join(
        cell_source(cell).lower()
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    if BEGINNER_LAYER_TITLE.lower() in markdown_blob:
        return False

    insert_index = 0
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "markdown":
            insert_index = index + 1
            break

    notebook["cells"].insert(
        insert_index,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": scaffold,
        },
    )
    return True


def read_notebook(path: Path) -> dict:
    """Читает notebook как JSON-объект."""

    notebook = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise SystemExit(f"{path.relative_to(ROOT)} has invalid notebook JSON structure.")
    return notebook


def cell_source(cell: dict) -> str:
    """Нормализует source ячейки к строке."""

    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def write_notebook(path: Path, notebook: dict) -> None:
    """Записывает notebook обратно на диск в формате v4."""

    serialized = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    path.write_text(serialized, encoding="utf-8")


def normalize_notebooks(paths: Iterable[Path]) -> tuple[int, list[str]]:
    """Нормализует указанные notebook-и и возвращает число измененных файлов."""

    changed_count = 0
    changed_paths: list[str] = []

    for path in paths:
        notebook = read_notebook(path)
        relative_path = path.relative_to(ROOT).as_posix()

        changed = False
        for cell in notebook["cells"]:
            source = cell_source(cell)
            normalized = normalize_source(source)
            if normalized != source:
                cell["source"] = normalized
                changed = True

        if inject_beginner_scaffold(path, notebook):
            changed = True

        if changed:
            write_notebook(path, notebook)
            changed_count += 1
            changed_paths.append(relative_path)

    return changed_count, changed_paths


def sync_run_sources() -> tuple[int, list[str]]:
    """Синхронизирует source run-ноутбуков с каноническими solution-файлами."""

    changed = 0
    changed_paths: list[str] = []

    for canonical_rel, targets in RUN_SYNC_MAP.items():
        canonical_path = ROOT / canonical_rel
        canonical = read_notebook(canonical_path)

        for target_rel in targets:
            target_path = ROOT / target_rel
            target = read_notebook(target_path)

            canonical_cells = canonical["cells"]
            target_cells = target["cells"]
            rebuilt_cells: list[dict] = []
            target_index = 0
            target_changed = False

            for canonical_index, src_cell in enumerate(canonical_cells):
                src_type = src_cell.get("cell_type")

                if target_index < len(target_cells):
                    dst_cell = target_cells[target_index]
                    dst_type = dst_cell.get("cell_type")
                else:
                    dst_cell = None
                    dst_type = None

                if dst_cell is not None and dst_type == src_type:
                    src_source = cell_source(src_cell)
                    dst_source = cell_source(dst_cell)
                    if dst_source != src_source:
                        dst_cell["source"] = src_source
                        target_changed = True
                    rebuilt_cells.append(dst_cell)
                    target_index += 1
                    continue

                if src_type == "markdown":
                    rebuilt_cells.append(
                        {
                            "cell_type": "markdown",
                            "metadata": {},
                            "source": cell_source(src_cell),
                        }
                    )
                    target_changed = True
                    continue

                raise SystemExit(
                    "Run sync failed: incompatible cell layout for "
                    f"{target_rel} at canonical cell {canonical_index}."
                )

            if target_index != len(target_cells):
                raise SystemExit(
                    "Run sync failed: target has extra trailing cells for "
                    f"{target_rel} vs {canonical_rel}."
                )

            if target.get("cells") != rebuilt_cells:
                target["cells"] = rebuilt_cells
                target_changed = True

            if target_changed:
                write_notebook(target_path, target)
                changed += 1
                changed_paths.append(target_rel)

    return changed, changed_paths


def parse_args() -> argparse.Namespace:
    """Разбирает аргументы командной строки."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-run-sync",
        action="store_true",
        help="Не синхронизировать source run-ноутбуков с каноническими solution.",
    )
    return parser.parse_args()


def main() -> None:
    """Запускает нормализацию notebook-ов и выводит краткий отчет."""

    args = parse_args()
    notebook_paths = iter_notebooks()

    normalized_count, normalized_paths = normalize_notebooks(notebook_paths)
    print(f"normalized-notebooks: {normalized_count}")
    for path in normalized_paths:
        print(f"normalized: {path}")

    if args.no_run_sync:
        return

    synced_count, synced_paths = sync_run_sources()
    print(f"synced-run-notebooks: {synced_count}")
    for path in synced_paths:
        print(f"synced: {path}")


if __name__ == "__main__":
    main()
