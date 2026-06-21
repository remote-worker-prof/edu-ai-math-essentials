#!/usr/bin/env python3
"""Create integrated assessment modules from topic context."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

SECTION_TITLES = {
    "foundations": "Раздел 1. Foundations",
    "rnn": "Раздел 2. RNN",
    "attention": "Раздел 3. Attention",
    "transformer_encoder": "Раздел 4. Transformer Encoder",
    "autoregression": "Раздел 5. Autoregression",
    "full_transformer": "Раздел 6. Full Transformer",
}

QUESTION_BLUEPRINT = ["single", "single", "multiple", "multiple", "open"]


def log(message: str) -> None:
    print(f"[06_modules][INFO] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build integrated assessment modules.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--workspace-root", default="90-assessment-ai-math", help="Assessment workspace root.")
    return parser.parse_args()


def load_topic_index(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sources_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    workspace_root = (project_root / args.workspace_root).resolve()

    topic_index_path = workspace_root / "03-topic-context" / "topic_index.json"
    topic_index = load_topic_index(topic_index_path)
    topics = topic_index.get("topics", [])

    modules_root = workspace_root / "04_modules"
    modules_root.mkdir(parents=True, exist_ok=True)

    module_entries: list[dict[str, object]] = []

    for idx, topic in enumerate(topics, start=1):
        topic_id = str(topic.get("topic_id", ""))
        module_slug = str(topic.get("module_slug", f"module-{idx:02d}"))
        title = str(topic.get("title", topic_id))
        section_title = SECTION_TITLES.get(topic_id, f"Раздел {idx}. {title}")

        module_id = f"module_{idx:02d}"
        module_dir = modules_root / module_slug
        module_dir.mkdir(parents=True, exist_ok=True)

        topic_sources_path = Path(str(topic.get("paths", {}).get("sources", "")))
        if not topic_sources_path.is_absolute():
            topic_sources_path = project_root / topic_sources_path
        sources = load_sources_csv(topic_sources_path)

        module_sources = [
            {
                "material_id": row.get("material_id", ""),
                "source_path": row.get("source_path", ""),
                "material_role": row.get("material_role", ""),
                "score": row.get("score", ""),
                "assignment_reason": row.get("assignment_reason", ""),
            }
            for row in sources
        ]

        summary_lines = [
            f"# Module: {section_title}",
            "",
            f"- module_id: `{module_id}`",
            f"- module_slug: `{module_slug}`",
            f"- topic_id: `{topic_id}`",
            f"- source materials: **{len(module_sources)}**",
            "",
            "## Assessment Blueprint",
            "",
            "- 5 вопросов на раздел: 2 single, 2 multiple, 1 open.",
            "- Сквозная нумерация задается на уровне варианта теста.",
            "",
        ]

        test_focus_lines = [
            f"# Test Focus: {section_title}",
            "",
            "- Проверять базовую теорию и практические следствия по разделу.",
            "- Сохранять академический, нейтральный стиль формулировок.",
            "- Не включать преподавательские ключи в студенческие материалы.",
            "",
        ]

        traceability = {
            "module_id": module_id,
            "module_slug": module_slug,
            "topic_id": topic_id,
            "topic_title": title,
            "section_title": section_title,
            "materials": module_sources,
            "question_blueprint": [
                {
                    "position_in_section": i + 1,
                    "question_type": q_type,
                }
                for i, q_type in enumerate(QUESTION_BLUEPRINT)
            ],
        }

        summary_path = module_dir / "module_summary.md"
        sources_path = module_dir / "module_sources.csv"
        traceability_path = module_dir / "module_traceability.json"
        focus_path = module_dir / "test_focus.md"

        summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
        focus_path.write_text("\n".join(test_focus_lines), encoding="utf-8")
        traceability_path.write_text(json.dumps(traceability, ensure_ascii=False, indent=2), encoding="utf-8")

        with sources_path.open("w", newline="", encoding="utf-8") as f:
            fields = ["material_id", "source_path", "material_role", "score", "assignment_reason"]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(module_sources)

        module_entries.append(
            {
                "module_id": module_id,
                "module_slug": module_slug,
                "module_title": section_title,
                "topic_id": topic_id,
                "topic_title": title,
                "material_ids": [row["material_id"] for row in module_sources],
                "materials": module_sources,
                "question_blueprint": QUESTION_BLUEPRINT,
                "paths": {
                    "module_summary": summary_path.as_posix(),
                    "module_sources": sources_path.as_posix(),
                    "module_traceability": traceability_path.as_posix(),
                    "test_focus": focus_path.as_posix(),
                },
            }
        )

    module_index = {
        "modules": module_entries,
        "stats": {
            "module_count": len(module_entries),
            "total_material_links": sum(len(m["material_ids"]) for m in module_entries),
        },
    }

    module_index_json_path = workspace_root / "03-topic-context" / "module_index.json"
    module_index_md_path = workspace_root / "03-topic-context" / "MODULE_INDEX.md"

    module_index_json_path.write_text(json.dumps(module_index, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = ["# MODULE_INDEX", "", "| Module | Topic | Materials |", "|---|---|---:|"]
    for module in module_entries:
        md_lines.append(
            f"| `{module['module_slug']}` | {module['module_title']} | {len(module['material_ids'])} |"
        )
    md_lines.append("")
    module_index_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    log(f"wrote {module_index_md_path.relative_to(project_root)}")
    log(f"wrote {module_index_json_path.relative_to(project_root)}")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
