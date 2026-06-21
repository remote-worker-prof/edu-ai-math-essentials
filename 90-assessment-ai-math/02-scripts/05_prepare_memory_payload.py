#!/usr/bin/env python3
"""Prepare Serena memory payload documents from topic artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def log(message: str) -> None:
    print(f"[05_memory][INFO] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build memory payload markdown files.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--workspace-root", default="90-assessment-ai-math", help="Assessment workspace root.")
    return parser.parse_args()


def fenced_json(payload: object) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    workspace_root = (project_root / args.workspace_root).resolve()

    topic_index_path = workspace_root / "03-topic-context" / "topic_index.json"
    topic_index = json.loads(topic_index_path.read_text(encoding="utf-8"))
    topics = topic_index.get("topics", [])

    out_dir = workspace_root / "99_memory"
    out_dir.mkdir(parents=True, exist_ok=True)

    topic_index_payload = {
        "memory_name": "discipline/ai_math_essentials/topic_index",
        "topics": [
            {
                "topic_id": t.get("topic_id"),
                "title": t.get("title"),
                "module_slug": t.get("module_slug"),
                "material_count": t.get("material_count"),
            }
            for t in topics
        ],
    }

    topic_cards_payload = {
        "memory_name": "discipline/ai_math_essentials/topic_cards",
        "cards": [
            {
                "topic_id": t.get("topic_id"),
                "title": t.get("title"),
                "focus_hint": f"Оценивать понимание раздела {t.get('title')}",
                "materials": t.get("material_ids", []),
            }
            for t in topics
        ],
    }

    traceability_payload = {
        "memory_name": "discipline/ai_math_essentials/source_traceability",
        "topic_to_material": topic_index.get("traceability", {}).get("topic_to_material", {}),
    }

    memory_md = [
        "# Serena Topic Memory Payload",
        "",
        "## Target Memory",
        "",
        "`discipline/ai_math_essentials/topic_index`",
        "",
        "## Payload",
        "",
        fenced_json(topic_index_payload),
        "",
    ]

    cards_md = [
        "# Serena Topic Cards Payload",
        "",
        "## Target Memory",
        "",
        "`discipline/ai_math_essentials/topic_cards`",
        "",
        "## Payload",
        "",
        fenced_json(topic_cards_payload),
        "",
    ]

    trace_md = [
        "# Serena Source Traceability Payload",
        "",
        "## Target Memory",
        "",
        "`discipline/ai_math_essentials/source_traceability`",
        "",
        "## Payload",
        "",
        fenced_json(traceability_payload),
        "",
    ]

    p1 = out_dir / "serena_topic_memory.md"
    p2 = out_dir / "serena_topic_cards.md"
    p3 = out_dir / "serena_source_traceability.md"

    p1.write_text("\n".join(memory_md), encoding="utf-8")
    p2.write_text("\n".join(cards_md), encoding="utf-8")
    p3.write_text("\n".join(trace_md), encoding="utf-8")

    log(f"wrote {p1.relative_to(project_root)}")
    log(f"wrote {p2.relative_to(project_root)}")
    log(f"wrote {p3.relative_to(project_root)}")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
