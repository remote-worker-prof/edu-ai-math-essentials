"""Статически проверяет runtime/GPU контракты тяжелых notebook-ов 04/05."""

from __future__ import annotations

import json
from pathlib import Path
import re

from notebook_contract_data import (
    ROOT,
    RUNTIME_GPU_NOTEBOOKS,
    RUN_EXECUTION_CONTRACTS,
    RUN_SYNC_MAP,
)


FORBIDDEN_FALLBACK_FRAGMENTS = (
    "fallback to cpu",
    "fall back to cpu",
    "hidden cpu fallback",
    "use cpu instead",
)

NOTEBOOK_MARKERS = {
    "themes/04-Autoregression/lab/02_decoder_only_tiny_shakespeare_gpu.ipynb": (
        "gpu_preflight",
        "COURSE_RUNTIME_MODE",
        "local-gpu",
        "causal_mask",
        "perplexity",
        "baseline",
        "generation",
    ),
    "themes/04-Autoregression/lab/solutions/02_decoder_only_tiny_shakespeare_gpu_solution.ipynb": (
        "gpu_preflight",
        "COURSE_RUNTIME_MODE",
        "local-gpu",
        "causal_mask",
        "perplexity",
        "baseline",
        "generation",
    ),
    "themes/05-Full-Transformer/lab/01_full_transformer_tiny_shakespeare.ipynb": (
        "gpu_preflight",
        "COURSE_RUNTIME_PROFILE",
        "GPU-friendly",
        "encoder_input",
        "decoder_input",
        "decoder_target",
        "perplexity",
        "baseline",
        "mean_match_ratio",
    ),
    "themes/05-Full-Transformer/lab/solutions/01_full_transformer_wikitext2_solution.ipynb": (
        "gpu_preflight",
        "COURSE_RUNTIME_PROFILE",
        "GPU-friendly",
        "encoder_input",
        "decoder_input",
        "decoder_target",
        "perplexity",
        "baseline",
        "mean_match_ratio",
    ),
}


def read_notebook(relative_path: str) -> dict:
    """Читает notebook как JSON-объект."""

    path = ROOT / relative_path
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise SystemExit(f"{relative_path}: invalid notebook JSON structure.")
    return notebook


def cell_source(cell: dict) -> str:
    """Нормализует source ячейки к строке."""

    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(part) for part in source)
    return str(source)


def read_source(relative_path: str) -> str:
    """Читает notebook и возвращает объединенный source всех ячеек."""

    notebook = read_notebook(relative_path)
    sources: list[str] = []
    for cell in notebook["cells"]:
        sources.append(cell_source(cell))
    return "\n".join(sources)


def output_to_text(output: dict) -> str:
    """Извлекает текстовую часть из одного output-записи code-ячейки."""

    chunks: list[str] = []

    text = output.get("text", "")
    if isinstance(text, list):
        chunks.append("".join(str(part) for part in text))
    elif text:
        chunks.append(str(text))

    data = output.get("data", {})
    if isinstance(data, dict):
        text_plain = data.get("text/plain", "")
        if isinstance(text_plain, list):
            chunks.append("".join(str(part) for part in text_plain))
        elif text_plain:
            chunks.append(str(text_plain))

    return "\n".join(chunks)


def cell_outputs_text(cell: dict) -> str:
    """Объединяет текст всех outputs конкретной code-ячейки."""

    outputs = cell.get("outputs", [])
    if not isinstance(outputs, list):
        return ""
    return "\n".join(output_to_text(output) for output in outputs if isinstance(output, dict))


def resolve_run_execution_contract(relative_path: str) -> dict | None:
    """Находит контракт семейства run-ноутбука по префиксу пути."""

    for contract in RUN_EXECUTION_CONTRACTS:
        if relative_path.startswith(contract["prefix"]):
            return contract
    return None


def parse_summary_payloads(output_blob: str, summary_marker: str) -> list[dict]:
    """Извлекает JSON payload summary-маркера из текстовых outputs."""

    payloads: list[dict] = []
    marker_pattern = re.compile(re.escape(summary_marker) + r"(\{.*\})")
    for line in output_blob.splitlines():
        match = marker_pattern.search(line)
        if not match:
            continue
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payloads.append(parsed)
    return payloads


def check_runtime_notebook(relative_path: str, errors: list[str]) -> None:
    """Проверяет маркеры runtime-контракта для одного тяжелого notebook."""

    source = read_source(relative_path)
    lowered = source.lower()

    for marker in NOTEBOOK_MARKERS[relative_path]:
        if marker not in source:
            errors.append(f"{relative_path}: missing runtime marker {marker!r}.")

    for fragment in FORBIDDEN_FALLBACK_FRAGMENTS:
        if fragment in lowered:
            errors.append(
                f"{relative_path}: forbidden fallback fragment detected {fragment!r}."
            )

    if "raise RuntimeError" not in source:
        errors.append(f"{relative_path}: expected explicit RuntimeError guard for bad GPU setup.")

    print(f"runtime-contract-ok: {relative_path}")


def check_run_execution_contract(relative_path: str, errors: list[str]) -> None:
    """Проверяет executed-состояние, summary и metadata для run-ноутбука."""

    contract = resolve_run_execution_contract(relative_path)
    if contract is None:
        errors.append(f"{relative_path}: no run-execution contract found for this path.")
        return

    notebook = read_notebook(relative_path)
    metadata = notebook.get("metadata", {})
    if not isinstance(metadata, dict):
        errors.append(f"{relative_path}: top-level metadata must be an object.")
    else:
        for key in ("kernelspec", "language_info"):
            if key not in metadata:
                errors.append(f"{relative_path}: missing top-level metadata key {key!r}.")

    code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
    if len(code_cells) != int(contract["code_cells"]):
        errors.append(
            f"{relative_path}: expected {contract['code_cells']} code cells, got {len(code_cells)}."
        )

    missing_exec = [
        index for index, cell in enumerate(code_cells) if cell.get("execution_count") is None
    ]
    if missing_exec:
        errors.append(
            f"{relative_path}: execution_count is missing for code cell indexes {missing_exec}."
        )

    output_cells = [cell for cell in code_cells if cell.get("outputs")]
    if len(output_cells) != int(contract["output_cells"]):
        errors.append(
            f"{relative_path}: expected {contract['output_cells']} code cells with outputs, "
            f"got {len(output_cells)}."
        )

    output_blob = "\n".join(cell_outputs_text(cell) for cell in code_cells)
    summary_marker = str(contract["summary_marker"])
    if summary_marker not in output_blob:
        errors.append(f"{relative_path}: missing run summary marker {summary_marker!r} in outputs.")
        return

    payloads = parse_summary_payloads(output_blob, summary_marker)
    if not payloads:
        errors.append(
            f"{relative_path}: summary marker {summary_marker!r} found, but JSON payload is missing."
        )
        return

    summary = payloads[-1]
    required_keys = set(contract["summary_required_keys"])
    missing_keys = sorted(required_keys - set(summary))
    if missing_keys:
        errors.append(
            f"{relative_path}: summary payload misses required keys {missing_keys}."
        )
        return

    print(f"runtime-run-execution-ok: {relative_path}")


def check_run_sync_contract(errors: list[str]) -> None:
    """Проверяет, что run-ноутбуки синхронизированы с каноническими source."""

    for canonical_rel, targets in RUN_SYNC_MAP.items():
        canonical_notebook = read_notebook(canonical_rel)
        canonical_types = [cell.get("cell_type") for cell in canonical_notebook["cells"]]
        canonical_sources = [cell_source(cell) for cell in canonical_notebook["cells"]]
        for target_rel in sorted(targets):
            target_notebook = read_notebook(target_rel)
            target_types = [cell.get("cell_type") for cell in target_notebook["cells"]]
            target_sources = [cell_source(cell) for cell in target_notebook["cells"]]

            if target_types != canonical_types:
                errors.append(
                    f"{target_rel}: cell type layout differs from canonical runtime notebook {canonical_rel}."
                )
                continue
            if target_sources != canonical_sources:
                errors.append(
                    f"{target_rel}: source differs from canonical runtime notebook {canonical_rel}."
                )
                continue

            before_errors = len(errors)
            check_run_execution_contract(target_rel, errors)
            if len(errors) == before_errors:
                print(f"runtime-run-sync-ok: {target_rel}")


def main() -> None:
    """Запускает статические runtime/GPU checks без тяжелого обучения."""
    errors: list[str] = []

    for relative_path in RUNTIME_GPU_NOTEBOOKS:
        check_runtime_notebook(relative_path, errors)

    check_run_sync_contract(errors)

    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
