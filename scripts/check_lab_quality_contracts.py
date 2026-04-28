"""Проверяет тематические quality-контракты учебных материалов EAIME."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re

from notebook_contract_data import EXPECTED_NOTEBOOKS, ROOT


FORBIDDEN_ENGLISH_DOCSTRING_SECTIONS = ("Args:", "Returns:", "Raises:")
FORBIDDEN_COMMENT_FRAGMENTS = (
    "# Mini-check",
    "# Reuse-block from",
    "# Step ",
    "# Hint:",
)

README_MARKERS = {
    "themes/00-Foundations/README.md": (
        "единый мини-глоссарий 00-05",
        "готов/не готов к следующей теме",
    ),
    "themes/01-RNN/lab/README.md": (
        "shape bridge",
        "(batch, time, features)",
    ),
    "themes/02-Attention/lab/README.md": (
        "query",
        "key",
        "value",
        "attention_scores",
    ),
    "themes/03-Transformer/lab/README.md": (
        "self-attention",
        "positional",
    ),
    "themes/04-Autoregression/lab/README.md": (
        "cpu",
        "gpu",
        "leakage",
        "causal mask",
    ),
    "themes/05-Full-Transformer/lab/README.md": (
        "encoder_input",
        "decoder_input",
        "decoder_target",
        "cross-attention",
    ),
}


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


def notebook_sources(notebook: dict) -> tuple[str, str]:
    """Возвращает объединенный markdown/code source для notebook."""

    markdown = "\n".join(
        cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "markdown"
    )
    code = "\n".join(
        cell_source(cell)
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    return markdown, code


def parse_ast(source: str) -> ast.Module | None:
    """Пытается разобрать code source в AST без падения на синтаксических ячейках."""

    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def required_markers_for_notebook(relative_path: str) -> tuple[str, ...]:
    """Возвращает список обязательных маркеров для учебной темы."""

    if relative_path.startswith("themes/01-RNN/lab/"):
        if "simple_rnn" in relative_path:
            return ("(batch, time, features)", "чек-лист")
        if "lstm_many_to_many" in relative_path:
            return ("token_accuracy", "чек-лист")
        return ("decoder_input", "decoder_target", "чек-лист")
    if relative_path.startswith("themes/02-Attention/lab/"):
        return ("query", "key", "value", "attention_scores", "чек-лист")
    if relative_path.startswith("themes/03-Transformer/lab/"):
        return ("padding_mask", "attention_scores", "positional", "чек-лист")
    if relative_path.startswith("themes/04-Autoregression/lab/"):
        if "01_decoder_only_causal_toy" in relative_path:
            return ("causal_mask", "perplexity", "чек-лист")
        if "02_decoder_only_tiny_shakespeare_gpu" in relative_path:
            return ("gpu_preflight", "causal_mask", "perplexity", "baseline", "generation")
        return ("causal_mask", "perplexity", "baseline", "генерац")
    if relative_path.startswith("themes/05-Full-Transformer/lab/"):
        return (
            "encoder_input",
            "decoder_input",
            "decoder_target",
            "causal",
            "cross",
            "чек-лист",
        )
    return ()


def check_docstrings(tree: ast.Module, relative_path: str, errors: list[str]) -> None:
    """Проверяет русские секции в docstring функций."""

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        docstring = ast.get_docstring(node) or ""
        if not docstring:
            continue

        has_en = any(section in docstring for section in FORBIDDEN_ENGLISH_DOCSTRING_SECTIONS)
        has_ru = any(section in docstring for section in ("Аргументы:", "Возвращает:", "Исключения:"))

        for forbidden in FORBIDDEN_ENGLISH_DOCSTRING_SECTIONS:
            if forbidden in docstring:
                errors.append(
                    f"{relative_path}: function {node.name!r} still uses {forbidden!r}."
                )

        if not (has_en or has_ru):
            continue

        if "Аргументы:" not in docstring:
            errors.append(
                f"{relative_path}: function {node.name!r} misses 'Аргументы:' in structured docstring."
            )
        has_explicit_return = any(
            isinstance(subnode, ast.Return) and subnode.value is not None
            for subnode in ast.walk(node)
        )
        if has_explicit_return and "Возвращает:" not in docstring:
            errors.append(
                f"{relative_path}: function {node.name!r} misses 'Возвращает:' in structured docstring."
            )

        has_raise = any(isinstance(subnode, ast.Raise) for subnode in ast.walk(node))
        if has_raise and "Исключения:" not in docstring:
            errors.append(f"{relative_path}: function {node.name!r} misses 'Исключения:'.")


def check_notebook(relative_path: str, errors: list[str]) -> None:
    """Проверяет один notebook по quality-контракту."""

    path = ROOT / relative_path
    notebook = read_notebook(path)
    markdown, code = notebook_sources(notebook)
    combined = (markdown + "\n" + code).lower()

    for forbidden in FORBIDDEN_ENGLISH_DOCSTRING_SECTIONS:
        if forbidden in code:
            errors.append(f"{relative_path}: still contains forbidden section {forbidden!r}.")

    for fragment in FORBIDDEN_COMMENT_FRAGMENTS:
        if fragment in code:
            errors.append(f"{relative_path}: contains forbidden English comment fragment {fragment!r}.")

    for marker in required_markers_for_notebook(relative_path):
        if marker.lower() not in combined:
            errors.append(f"{relative_path}: missing required marker {marker!r}.")

    tree = parse_ast(code)
    if tree is not None:
        check_docstrings(tree, relative_path, errors)

    print(f"quality-ok: {relative_path}")


def check_readmes(errors: list[str]) -> None:
    """Проверяет маркеры качества в ключевых README тем."""

    for relative_path, markers in README_MARKERS.items():
        path = ROOT / relative_path
        if not path.exists():
            errors.append(f"{relative_path}: README file is missing.")
            continue

        text = path.read_text(encoding="utf-8").lower()
        for marker in markers:
            if marker.lower() not in text:
                errors.append(f"{relative_path}: missing README marker {marker!r}.")

        print(f"readme-quality-ok: {relative_path}")


def main() -> None:
    """Запускает quality-проверки notebook-ов и README."""

    errors: list[str] = []
    for relative_path in sorted(EXPECTED_NOTEBOOKS):
        check_notebook(relative_path, errors)
    check_readmes(errors)

    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
