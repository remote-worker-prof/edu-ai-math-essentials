"""Проверяет структурные контракты notebook-ов EAIME."""

from __future__ import annotations

import json
from pathlib import Path
import re

from notebook_contract_data import (
    EXPECTED_NOTEBOOKS,
    ROOT,
    RUN_SYNC_MAP,
    STARTER_SOLUTION_PAIRS,
)


TODO_MARKER = "TODO"
UNRESOLVED_PATTERNS = (
    re.compile(r"NotImplementedError\(\s*['\"]TODO"),
    re.compile(r"=\s*\.\.\."),
)


def read_notebook(path: Path) -> dict:
    """Читает notebook и валидирует базовую JSON-схему."""

    notebook = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(notebook, dict):
        raise SystemExit(f"{path.relative_to(ROOT)} is not a JSON object.")
    if "cells" not in notebook or not isinstance(notebook["cells"], list):
        raise SystemExit(f"{path.relative_to(ROOT)} has invalid 'cells' field.")
    return notebook


def cell_source(cell: dict) -> str:
    """Нормализует source ячейки к строке."""

    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def notebook_text(notebook: dict) -> str:
    """Возвращает объединенный source всех ячеек."""

    return "\n".join(cell_source(cell) for cell in notebook["cells"])


def code_text(notebook: dict) -> str:
    """Возвращает объединенный source только code-ячеек."""

    return "\n".join(
        cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def normalized_todo_headings(notebook: dict) -> list[str]:
    """Собирает только TODO-заголовки, нормализованные для pair-сравнения."""

    headings: list[str] = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        for raw_line in cell_source(cell).splitlines():
            line = raw_line.strip()
            if not line.startswith("## "):
                continue
            if "todo" not in line.lower():
                continue
            normalized = line.replace("(решение)", "").replace("(solution)", "")
            normalized = re.sub(r"\s+", " ", normalized).strip()
            headings.append(normalized)
    return headings


def has_checklist_heading(notebook: dict) -> bool:
    """Проверяет наличие чек-листа в markdown-заголовках."""

    for cell in notebook["cells"]:
        if cell.get("cell_type") != "markdown":
            continue
        for raw_line in cell_source(cell).splitlines():
            line = raw_line.strip().lower()
            if line.startswith("## ") and "чек-лист" in line:
                return True
    return False


def checklist_required_for_pair(starter_rel: str) -> bool:
    """Возвращает, обязателен ли checklist-заголовок для пары notebook-ов."""

    _ = starter_rel
    return True


def check_inventory(errors: list[str]) -> None:
    """Проверяет, что набор notebook-ов совпадает с зафиксированным контрактом."""

    actual = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "themes").rglob("*.ipynb")
    }
    missing = sorted(EXPECTED_NOTEBOOKS - actual)
    unexpected = sorted(actual - EXPECTED_NOTEBOOKS)

    if missing:
        errors.append(f"Notebook inventory mismatch, missing: {missing}")
    if unexpected:
        errors.append(f"Notebook inventory mismatch, unexpected: {unexpected}")


def check_starter_solution_alignment(errors: list[str]) -> None:
    """Проверяет контракты starter/solution пар."""

    for starter_rel, solution_rel in STARTER_SOLUTION_PAIRS:
        starter_path = ROOT / starter_rel
        solution_path = ROOT / solution_rel
        starter = read_notebook(starter_path)
        solution = read_notebook(solution_path)

        starter_text = notebook_text(starter)
        solution_code = code_text(solution)

        if TODO_MARKER not in starter_text:
            errors.append(f"{starter_rel}: starter notebook must contain TODO markers.")

        for pattern in UNRESOLVED_PATTERNS:
            if pattern.search(solution_code):
                errors.append(
                    f"{solution_rel}: unresolved placeholder matched pattern {pattern.pattern!r}."
                )

        starter_headings = normalized_todo_headings(starter)
        solution_headings = normalized_todo_headings(solution)
        if starter_headings and solution_headings and starter_headings != solution_headings:
            errors.append(
                f"{starter_rel} vs {solution_rel}: TODO heading alignment mismatch."
            )
        if checklist_required_for_pair(starter_rel):
            if not has_checklist_heading(starter):
                errors.append(f"{starter_rel}: starter notebook must contain checklist heading.")
            if not has_checklist_heading(solution):
                errors.append(f"{solution_rel}: solution notebook must contain checklist heading.")

        print(f"pair-contract-ok: {starter_rel} <-> {solution_rel}")


def check_run_source_alignment(errors: list[str]) -> None:
    """Проверяет source-alignment run-ноутбуков с каноническими solution."""

    for canonical_rel, targets in RUN_SYNC_MAP.items():
        canonical = read_notebook(ROOT / canonical_rel)
        canonical_sources = [cell_source(cell) for cell in canonical["cells"]]
        canonical_types = [cell.get("cell_type") for cell in canonical["cells"]]

        for target_rel in sorted(targets):
            target = read_notebook(ROOT / target_rel)
            target_sources = [cell_source(cell) for cell in target["cells"]]
            target_types = [cell.get("cell_type") for cell in target["cells"]]

            if canonical_types != target_types:
                errors.append(
                    f"{target_rel}: cell type layout differs from canonical {canonical_rel}."
                )
                continue
            if canonical_sources != target_sources:
                errors.append(
                    f"{target_rel}: source cells are not synchronized with canonical {canonical_rel}."
                )
                continue

            print(f"run-sync-ok: {target_rel} == {canonical_rel}")


def main() -> None:
    """Запускает все структурные проверки notebook-контрактов."""

    errors: list[str] = []
    check_inventory(errors)
    check_starter_solution_alignment(errors)
    check_run_source_alignment(errors)

    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
