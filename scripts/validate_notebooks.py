"""Проверяет, что все notebook-и в themes читаются и проходят schema-validation."""

from __future__ import annotations

import json
from pathlib import Path

from notebook_contract_data import ROOT


THEMES_DIR = ROOT / "themes"
INDENT_PREFIX = "    "
MIN_INDENTED_MARKDOWN_LINES = 3
MARKDOWNISH_PREFIXES = ("- ", "* ", "+ ", "#")
MATH_CORRUPTION_FRAGMENT = "\t" + "ext{"


def iter_notebooks() -> list[Path]:
    """Собирает все notebook-файлы внутри themes."""

    return sorted(THEMES_DIR.rglob("*.ipynb"))


def cell_source(cell: dict) -> str:
    """Нормализует source ячейки к строке."""

    raw = cell.get("source", "")
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        return "".join(str(part) for part in raw)
    return str(raw)


def read_notebook_json(path: Path) -> dict:
    """Читает notebook как JSON и проверяет минимальную структуру."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"{path.relative_to(ROOT)} is not a JSON object.")
    if "cells" not in payload or not isinstance(payload["cells"], list):
        raise SystemExit(f"{path.relative_to(ROOT)}: missing list field 'cells'.")
    if "nbformat" not in payload:
        raise SystemExit(f"{path.relative_to(ROOT)}: missing field 'nbformat'.")
    return payload


def has_suspicious_markdown_indentation(source: str) -> bool:
    """Ловит markdown-блоки, которые случайно превратятся в code block."""

    non_empty_lines = [line for line in source.splitlines() if line.strip()]
    if len(non_empty_lines) < MIN_INDENTED_MARKDOWN_LINES:
        return False

    indented_lines = [line for line in non_empty_lines if line.startswith(INDENT_PREFIX)]
    if len(indented_lines) < MIN_INDENTED_MARKDOWN_LINES:
        return False

    if len(indented_lines) * 2 < len(non_empty_lines):
        return False

    suspicious_lines = 0
    for line in indented_lines:
        stripped = line.lstrip()
        if (
            stripped.startswith(MARKDOWNISH_PREFIXES)
            or stripped == "$$"
            or "$" in stripped
            or "\\" in stripped
        ):
            suspicious_lines += 1
    return suspicious_lines >= MIN_INDENTED_MARKDOWN_LINES


def has_literal_tab_text_corruption(source: str) -> bool:
    """Ищет поломку `\\text{...}` -> tab + `ext{...}`."""

    return MATH_CORRUPTION_FRAGMENT in source


def validate_markdown_content(path: Path, notebook: dict) -> list[str]:
    """Проверяет markdown-ячейки на типовые поломки форматирования."""

    errors: list[str] = []
    relative_path = path.relative_to(ROOT)
    for cell_index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "markdown":
            continue
        source = cell_source(cell)
        if has_suspicious_markdown_indentation(source):
            errors.append(
                f"{relative_path} markdown cell {cell_index}: suspicious leading indentation "
                "would render explanatory markdown as a code block."
            )
        if has_literal_tab_text_corruption(source):
            errors.append(
                f"{relative_path} markdown cell {cell_index}: literal tab before 'ext{{' "
                "suggests a broken '\\text{{...}}' escape."
            )
    return errors


def main() -> None:
    """Запускает валидацию notebook-ов и печатает короткий прогресс."""

    notebooks = iter_notebooks()
    if not notebooks:
        raise SystemExit("No notebooks found in themes/ to validate.")

    errors: list[str] = []
    for path in notebooks:
        notebook = read_notebook_json(path)
        errors.extend(validate_markdown_content(path, notebook))
        print(f"validated: {path.relative_to(ROOT)}")

    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
