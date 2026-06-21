#!/usr/bin/env python3
"""Extract text and structure from inventoried source materials."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def log(message: str) -> None:
    print(f"[02_extract][INFO] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text and structure for each material.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--workspace-root", default="90-assessment-ai-math", help="Assessment workspace root.")
    parser.add_argument(
        "--manifest",
        default="",
        help="Path to materials_manifest.json (defaults to workspace index path).",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "materials" in data:
        return list(data["materials"])
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported manifest format")


def markdown_structure(text: str) -> dict[str, object]:
    headings: list[dict[str, object]] = []
    for line_idx, line in enumerate(text.splitlines(), start=1):
        m = re.match(r"^(#{1,6})\s+(.*\S)\s*$", line)
        if not m:
            continue
        headings.append({"line": line_idx, "level": len(m.group(1)), "title": m.group(2)})
    return {
        "format": "md",
        "line_count": len(text.splitlines()),
        "char_count": len(text),
        "headings": headings,
    }


def read_cell_source(cell: dict[str, object]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(str(x) for x in source)
    return str(source)


def notebook_structure(nb: dict[str, object]) -> tuple[str, dict[str, object]]:
    cells = nb.get("cells", [])
    if not isinstance(cells, list):
        cells = []

    raw_chunks: list[str] = []
    cell_items: list[dict[str, object]] = []

    for idx, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict):
            continue
        ctype = str(cell.get("cell_type", "unknown"))
        source = read_cell_source(cell)

        raw_chunks.append(f"[cell {idx:03d} | {ctype}]\n{source.strip()}\n")
        cell_items.append(
            {
                "index": idx,
                "cell_type": ctype,
                "char_count": len(source),
                "line_count": len(source.splitlines()),
            }
        )

    raw = "\n".join(raw_chunks).strip() + "\n"
    structure = {
        "format": "ipynb",
        "cell_count": len(cell_items),
        "cells": cell_items,
    }
    return raw, structure


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def extract_one(project_root: Path, extracted_root: Path, material: dict[str, object]) -> None:
    material_id = str(material["material_id"])
    source_path = str(material["source_path"])
    ext_kind = str(material["ext_kind"]).lower()

    in_path = project_root / source_path
    out_dir = extracted_root / material_id
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / "raw.txt"
    struct_path = out_dir / "structure.json"
    status_path = out_dir / "extract_status.json"

    status: dict[str, object] = {
        "material_id": material_id,
        "source_path": source_path,
        "status": "ok",
        "reason": "",
    }

    try:
        if ext_kind == "md":
            text = in_path.read_text(encoding="utf-8", errors="replace")
            raw_path.write_text(text, encoding="utf-8")
            write_json(struct_path, markdown_structure(text))
        elif ext_kind == "ipynb":
            nb = json.loads(in_path.read_text(encoding="utf-8", errors="replace"))
            raw, structure = notebook_structure(nb)
            raw_path.write_text(raw, encoding="utf-8")
            write_json(struct_path, structure)
        else:
            status["status"] = "skip"
            status["reason"] = f"unsupported_ext:{ext_kind}"
            raw_path.write_text("", encoding="utf-8")
            write_json(
                struct_path,
                {
                    "format": ext_kind,
                    "supported": False,
                    "note": "unsupported file format for extraction",
                },
            )
    except Exception as exc:  # noqa: BLE001
        status["status"] = "error"
        status["reason"] = f"{type(exc).__name__}: {exc}"
        raw_path.write_text("", encoding="utf-8")
        write_json(
            struct_path,
            {
                "supported": False,
                "note": "extraction failed",
            },
        )

    write_json(status_path, status)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    workspace_root = (project_root / args.workspace_root).resolve()

    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else workspace_root / "03-topic-context" / "00_index" / "materials_manifest.json"
    )
    extracted_root = workspace_root / "03-topic-context" / "01_extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)

    materials = load_manifest(manifest_path)
    log(f"materials to extract: {len(materials)}")

    for material in materials:
        extract_one(project_root, extracted_root, material)

    ok = 0
    skip = 0
    err = 0
    for material in materials:
        status_path = extracted_root / str(material["material_id"]) / "extract_status.json"
        data = json.loads(status_path.read_text(encoding="utf-8"))
        state = data.get("status")
        if state == "ok":
            ok += 1
        elif state == "skip":
            skip += 1
        else:
            err += 1

    log(f"done: ok={ok}, skip={skip}, error={err}")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
