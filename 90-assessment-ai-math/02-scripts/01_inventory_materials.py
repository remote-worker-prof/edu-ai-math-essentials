#!/usr/bin/env python3
"""Builds a machine-readable inventory of assessment source materials."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

THEME_DIRS = [
    "themes/00-Foundations",
    "themes/01-RNN",
    "themes/02-Attention",
    "themes/03-Transformer",
    "themes/04-Autoregression",
    "themes/05-Full-Transformer",
]

PATTERNS = [
    "theory/*.md",
    "lab/*.ipynb",
    "lab/guides/*.md",
    "lab/solutions/*.ipynb",
    "examples/*.ipynb",
    "showcases/*.ipynb",
    "showcases/cards/*.md",
]

EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".serena",
    "_build",
    "__pycache__",
    "logs",
    "tmp",
}

CSV_FIELDS = [
    "material_id",
    "source_path",
    "ext_kind",
    "size_bytes",
    "sha256",
    "is_duplicate",
    "duplicate_of",
    "detected_lang_hint",
    "material_role",
]


def log(message: str) -> None:
    print(f"[01_inventory][INFO] {message}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def detect_lang_hint(path: Path, ext_kind: str) -> str:
    text_like = {"md", "ipynb", "txt", "json", "yaml", "yml", "py", "sh", "csv"}
    if ext_kind not in text_like:
        return "unknown"

    try:
        sample = path.read_bytes()[:16384].decode("utf-8", errors="ignore")
    except OSError:
        return "unknown"

    cyr = sum(1 for ch in sample if "а" <= ch.lower() <= "я")
    lat = sum(1 for ch in sample if "a" <= ch.lower() <= "z")

    if cyr > 0 and lat == 0:
        return "ru"
    if lat > 0 and cyr == 0:
        return "en"
    if cyr > 0 and lat > 0:
        return "mixed"
    return "unknown"


def classify_role(rel_path: str) -> str:
    if rel_path.startswith("00-initial/"):
        return "legacy_context"
    if "/theory/" in rel_path and rel_path.endswith(".md"):
        return "theory"
    if "/lab/guides/" in rel_path and rel_path.endswith(".md"):
        return "lab_guide"
    if "/lab/solutions/" in rel_path and rel_path.endswith(".ipynb"):
        return "lab_solution"
    if "/lab/" in rel_path and rel_path.endswith(".ipynb"):
        return "lab"
    if "/examples/" in rel_path and rel_path.endswith(".ipynb"):
        return "example"
    if "/showcases/cards/" in rel_path and rel_path.endswith(".md"):
        return "showcase_card"
    if "/showcases/" in rel_path and rel_path.endswith(".ipynb"):
        return "showcase"
    return "other"


def is_excluded(path: Path) -> bool:
    return any(part in EXCLUDED_PARTS for part in path.parts)


def collect_theme_files(project_root: Path) -> set[Path]:
    found: set[Path] = set()
    for theme_dir in THEME_DIRS:
        base = project_root / theme_dir
        if not base.exists():
            log(f"skipping missing directory: {theme_dir}")
            continue
        for pattern in PATTERNS:
            for path in base.glob(pattern):
                if path.is_file() and not is_excluded(path.relative_to(project_root)):
                    found.add(path)
    return found


def collect_legacy_files(project_root: Path) -> set[Path]:
    legacy_root = project_root / "00-initial"
    if not legacy_root.exists():
        return set()

    found: set[Path] = set()
    for path in legacy_root.rglob("*"):
        if path.is_file() and not is_excluded(path.relative_to(project_root)):
            found.add(path)
    return found


def rel_paths(paths: Iterable[Path], root: Path) -> list[str]:
    return sorted(path.relative_to(root).as_posix() for path in paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create inventory of assessment materials.")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root directory (default: current directory).",
    )
    parser.add_argument(
        "--workspace-root",
        default="90-assessment-ai-math",
        help="Assessment workspace root.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    workspace_root = (project_root / args.workspace_root).resolve()

    out_dir = workspace_root / "03-topic-context" / "00_index"
    out_dir.mkdir(parents=True, exist_ok=True)

    log("collecting source files")
    files = collect_theme_files(project_root)
    files.update(collect_legacy_files(project_root))
    paths = rel_paths(files, project_root)

    log(f"materials collected: {len(paths)}")

    records: list[dict[str, object]] = []
    first_by_hash: dict[str, str] = {}

    for idx, rel_path in enumerate(paths, start=1):
        abs_path = project_root / rel_path
        ext_kind = abs_path.suffix.lower().lstrip(".") or "noext"
        material_id = f"mat_{idx:04d}"
        digest = sha256_file(abs_path)

        duplicate_of = first_by_hash.get(digest, "")
        is_duplicate = bool(duplicate_of)
        if not is_duplicate:
            first_by_hash[digest] = material_id

        row = {
            "material_id": material_id,
            "source_path": rel_path,
            "ext_kind": ext_kind,
            "size_bytes": abs_path.stat().st_size,
            "sha256": digest,
            "is_duplicate": is_duplicate,
            "duplicate_of": duplicate_of,
            "detected_lang_hint": detect_lang_hint(abs_path, ext_kind),
            "material_role": classify_role(rel_path),
        }
        records.append(row)

    manifest_json = out_dir / "materials_manifest.json"
    manifest_csv = out_dir / "materials_manifest.csv"

    payload = {
        "materials": records,
        "stats": {
            "total_materials": len(records),
            "duplicates": sum(1 for x in records if x["is_duplicate"]),
            "unique_hashes": len(first_by_hash),
        },
    }

    manifest_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    with manifest_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    log(f"wrote {manifest_json.relative_to(project_root)}")
    log(f"wrote {manifest_csv.relative_to(project_root)}")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
