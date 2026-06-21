#!/usr/bin/env python3
"""Build per-topic context bundles and global topic indexes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "как",
    "для",
    "что",
    "это",
    "или",
    "при",
    "под",
    "над",
    "без",
    "после",
    "before",
    "after",
    "cell",
    "code",
    "markdown",
}


def log(message: str) -> None:
    print(f"[04_topic_context][INFO] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build topic context and indexes.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--workspace-root", default="90-assessment-ai-math", help="Assessment workspace root.")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    materials = data["materials"] if isinstance(data, dict) else data
    return {str(row["material_id"]): row for row in materials}


def load_topic_catalog(path: Path) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    catalog = json.loads(path.read_text(encoding="utf-8"))
    ordered_topics = list(catalog["topics"])
    by_id = {str(topic["topic_id"]): topic for topic in ordered_topics}
    return ordered_topics, by_id


def load_map_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_raw(extracted_root: Path, material_id: str) -> str:
    p = extracted_root / material_id / "raw.txt"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def top_terms(text: str, limit: int = 15) -> list[str]:
    tokens = re.findall(r"[A-Za-zА-Яа-я_\-]{4,}", text.lower())
    filtered = [tok for tok in tokens if tok not in STOPWORDS]
    freq = Counter(filtered)
    return [term for term, _ in freq.most_common(limit)]


def write_topic_files(
    topic_dir: Path,
    topic: dict[str, object],
    rows: list[dict[str, str]],
    manifest_by_id: dict[str, dict[str, object]],
    extracted_root: Path,
) -> dict[str, object]:
    topic_dir.mkdir(parents=True, exist_ok=True)

    raw_concat_parts: list[str] = []
    role_counter: Counter[str] = Counter()

    for row in rows:
        material_id = row["material_id"]
        source_path = row["source_path"]
        role = row.get("material_role", "")
        role_counter[role] += 1

        raw_text = read_raw(extracted_root, material_id)
        raw_concat_parts.append(f"===== {material_id} | {source_path} =====\n{raw_text}\n")

    all_text = "\n".join(raw_concat_parts)
    keywords = top_terms(all_text, limit=20)

    summary_lines = [
        f"# Topic Context: {topic['title']}",
        "",
        f"- topic_id: `{topic['topic_id']}`",
        f"- module_slug: `{topic['module_slug']}`",
        f"- assigned materials: **{len(rows)}**",
        f"- theme prefix: `{topic['theme_prefix']}`",
        "",
        "## Material Roles",
        "",
    ]
    for role, count in sorted(role_counter.items()):
        summary_lines.append(f"- `{role}`: {count}")

    summary_lines.extend(["", "## Salient Terms", "", ", ".join(keywords[:15]) or "n/a", ""])

    test_focus_lines = [
        f"# Test Focus: {topic['title']}",
        "",
        "## Suggested Focus Areas",
        "",
    ]

    seed_terms = list(topic.get("keywords", []))[:5]
    if seed_terms:
        for term in seed_terms:
            test_focus_lines.append(f"- Проверять уверенное понимание: `{term}`.")
    else:
        test_focus_lines.append("- Проверять базовое понимание и терминологию раздела.")

    if keywords:
        test_focus_lines.append(f"- Опираться на термины из источников: {', '.join(keywords[:8])}.")
    test_focus_lines.append("- Формировать вопросы без раскрытия преподавательских ключей.")
    test_focus_lines.append("")

    summary_path = topic_dir / "topic_summary.md"
    full_materials_path = topic_dir / "full_materials.txt"
    sources_path = topic_dir / "sources.csv"
    focus_path = topic_dir / "test_focus.md"

    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    full_materials_path.write_text(all_text, encoding="utf-8")

    with sources_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["material_id", "source_path", "material_role", "score", "assignment_reason", "sha256"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            material_id = row["material_id"]
            material_meta = manifest_by_id.get(material_id, {})
            writer.writerow(
                {
                    "material_id": material_id,
                    "source_path": row.get("source_path", ""),
                    "material_role": row.get("material_role", ""),
                    "score": row.get("score", ""),
                    "assignment_reason": row.get("assignment_reason", ""),
                    "sha256": material_meta.get("sha256", ""),
                }
            )

    focus_path.write_text("\n".join(test_focus_lines), encoding="utf-8")

    return {
        "topic_id": topic["topic_id"],
        "title": topic["title"],
        "module_slug": topic["module_slug"],
        "material_count": len(rows),
        "material_ids": [row["material_id"] for row in rows],
        "paths": {
            "topic_summary": summary_path.as_posix(),
            "full_materials": full_materials_path.as_posix(),
            "sources": sources_path.as_posix(),
            "test_focus": focus_path.as_posix(),
        },
    }


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    workspace_root = (project_root / args.workspace_root).resolve()

    manifest_path = workspace_root / "03-topic-context" / "00_index" / "materials_manifest.json"
    map_csv = workspace_root / "03-topic-context" / "02_topics" / "material_topic_map.csv"
    catalog_path = workspace_root / "03-topic-context" / "02_topics" / "topic_catalog.json"

    topics_root = workspace_root / "03-topic-context" / "03_topics"
    topics_root.mkdir(parents=True, exist_ok=True)

    manifest_by_id = load_manifest(manifest_path)
    ordered_topics, topic_catalog = load_topic_catalog(catalog_path)
    map_rows = load_map_rows(map_csv)
    extracted_root = workspace_root / "03-topic-context" / "01_extracted"

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in map_rows:
        if row.get("assigned", "").strip().lower() != "true":
            continue
        topic_id = row.get("topic_id", "")
        if not topic_id:
            continue
        grouped[topic_id].append(row)

    topic_index_entries: list[dict[str, object]] = []
    for topic in ordered_topics:
        topic_id = str(topic["topic_id"])
        rows = sorted(grouped.get(topic_id, []), key=lambda x: x["material_id"])
        topic_dir = topics_root / topic_id
        entry = write_topic_files(topic_dir, topic, rows, manifest_by_id, extracted_root)
        topic_index_entries.append(entry)

    topic_index_json = {
        "topics": topic_index_entries,
        "traceability": {
            "topic_to_material": {
                entry["topic_id"]: entry["material_ids"] for entry in topic_index_entries
            }
        },
    }

    topic_index_json_path = workspace_root / "03-topic-context" / "topic_index.json"
    topic_index_json_path.write_text(json.dumps(topic_index_json, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = ["# TOPIC_INDEX", "", "| Topic | Title | Materials | Module Slug |", "|---|---|---:|---|"]
    for entry in topic_index_entries:
        md_lines.append(
            f"| `{entry['topic_id']}` | {entry['title']} | {entry['material_count']} | `{entry['module_slug']}` |"
        )
    md_lines.append("")

    topic_index_md_path = workspace_root / "03-topic-context" / "TOPIC_INDEX.md"
    topic_index_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    log(f"wrote {topic_index_md_path.relative_to(project_root)}")
    log(f"wrote {topic_index_json_path.relative_to(project_root)}")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
