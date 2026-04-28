"""Статически проверяет runtime/GPU контракты тяжелых notebook-ов 04/05."""

from __future__ import annotations

import json
from pathlib import Path

from notebook_contract_data import ROOT, RUNTIME_GPU_NOTEBOOKS, RUN_SYNC_MAP


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


def read_source(relative_path: str) -> str:
    """Читает notebook и возвращает объединенный source всех ячеек."""

    path = ROOT / relative_path
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(notebook, dict) or not isinstance(notebook.get("cells"), list):
        raise SystemExit(f"{relative_path}: invalid notebook JSON structure.")
    sources: list[str] = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        if isinstance(source, list):
            sources.append("".join(str(part) for part in source))
        else:
            sources.append(str(source))
    return "\n".join(sources)


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


def check_run_sync_contract(errors: list[str]) -> None:
    """Проверяет, что run-ноутбуки синхронизированы с каноническими source."""

    for canonical_rel, targets in RUN_SYNC_MAP.items():
        canonical_source = read_source(canonical_rel)
        for target_rel in sorted(targets):
            target_source = read_source(target_rel)
            if target_source != canonical_source:
                errors.append(
                    f"{target_rel}: source differs from canonical runtime notebook {canonical_rel}."
                )
            else:
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
