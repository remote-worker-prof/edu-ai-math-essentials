#!/usr/bin/env python3
"""Assign materials to canonical topics using rule-based mapping."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

TOPICS = [
    {
        "topic_id": "foundations",
        "title": "Foundations",
        "theme_prefix": "themes/00-Foundations",
        "module_slug": "01-foundations",
        "keywords": [
            "token",
            "tokens",
            "padding",
            "mask",
            "masking",
            "тензор",
            "tensor",
            "shape",
            "метрик",
            "accuracy",
        ],
    },
    {
        "topic_id": "rnn",
        "title": "RNN",
        "theme_prefix": "themes/01-RNN",
        "module_slug": "02-rnn",
        "keywords": [
            "rnn",
            "lstm",
            "gru",
            "seq2seq",
            "hidden state",
            "recurrent",
            "рекуррент",
            "последовательн",
        ],
    },
    {
        "topic_id": "attention",
        "title": "Attention",
        "theme_prefix": "themes/02-Attention",
        "module_slug": "03-attention",
        "keywords": [
            "attention",
            "alignment",
            "context vector",
            "attention map",
            "внимани",
            "выравниван",
        ],
    },
    {
        "topic_id": "transformer_encoder",
        "title": "Transformer Encoder",
        "theme_prefix": "themes/03-Transformer",
        "module_slug": "04-transformer-encoder",
        "keywords": [
            "transformer",
            "self-attention",
            "encoder",
            "positional encoding",
            "multi-head",
            "трансформер",
            "позицион",
        ],
    },
    {
        "topic_id": "autoregression",
        "title": "Autoregression",
        "theme_prefix": "themes/04-Autoregression",
        "module_slug": "05-autoregression",
        "keywords": [
            "autoregression",
            "decoder-only",
            "causal mask",
            "next token",
            "language model",
            "авторегресс",
            "causal",
        ],
    },
    {
        "topic_id": "full_transformer",
        "title": "Full Transformer",
        "theme_prefix": "themes/05-Full-Transformer",
        "module_slug": "06-full-transformer",
        "keywords": [
            "encoder-decoder",
            "teacher forcing",
            "inference",
            "training loop",
            "transformer",
            "диагностик",
            "инференс",
        ],
    },
]


def log(message: str) -> None:
    print(f"[03_mapping][INFO] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map materials to canonical topics.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--workspace-root", default="90-assessment-ai-math", help="Assessment workspace root.")
    parser.add_argument("--threshold", type=float, default=0.55, help="Assignment score threshold.")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "materials" in data:
        return list(data["materials"])
    if isinstance(data, list):
        return data
    raise ValueError("Unsupported manifest format")


def load_raw_text(extracted_root: Path, material_id: str) -> str:
    raw_path = extracted_root / material_id / "raw.txt"
    if not raw_path.exists():
        return ""
    return raw_path.read_text(encoding="utf-8", errors="replace").lower()


def score_topic(source_path: str, text: str, topic: dict[str, object]) -> tuple[float, int, bool]:
    path_hint = source_path.startswith(str(topic["theme_prefix"]))
    keyword_hits = 0
    for kw in topic["keywords"]:  # type: ignore[index]
        keyword_hits += text.count(str(kw).lower())

    path_score = 0.8 if path_hint else 0.0
    keyword_score = min(0.6, keyword_hits * 0.04)
    return path_score + keyword_score, keyword_hits, path_hint


def assign_topic(source_path: str, text: str, threshold: float) -> dict[str, object]:
    best_topic = None
    best_score = -1.0
    best_hits = 0
    used_path_hint = False

    for topic in TOPICS:
        score, hits, path_hint = score_topic(source_path, text, topic)
        if score > best_score:
            best_topic = topic
            best_score = score
            best_hits = hits
            used_path_hint = path_hint

    assert best_topic is not None

    assigned = best_score >= threshold
    assignment_reason = "below_threshold"
    if assigned and used_path_hint:
        assignment_reason = "path_hint+keywords"
    elif assigned:
        assignment_reason = "keywords"

    return {
        "topic_id": best_topic["topic_id"] if assigned else "",
        "topic_title": best_topic["title"] if assigned else "",
        "score": round(best_score, 4),
        "assigned": assigned,
        "assignment_reason": assignment_reason,
        "keyword_hits": best_hits,
    }


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    workspace_root = (project_root / args.workspace_root).resolve()

    manifest_path = workspace_root / "03-topic-context" / "00_index" / "materials_manifest.json"
    extracted_root = workspace_root / "03-topic-context" / "01_extracted"
    out_dir = workspace_root / "03-topic-context" / "02_topics"
    out_dir.mkdir(parents=True, exist_ok=True)

    materials = load_manifest(manifest_path)
    log(f"materials loaded: {len(materials)}")

    rows: list[dict[str, object]] = []
    stats_by_topic: dict[str, int] = defaultdict(int)

    for material in materials:
        material_id = str(material["material_id"])
        source_path = str(material["source_path"])
        text = load_raw_text(extracted_root, material_id)
        decision = assign_topic(source_path, text, args.threshold)

        row = {
            "material_id": material_id,
            "source_path": source_path,
            "material_role": str(material.get("material_role", "")),
            "topic_id": decision["topic_id"],
            "topic_title": decision["topic_title"],
            "score": decision["score"],
            "assigned": decision["assigned"],
            "assignment_reason": decision["assignment_reason"],
            "keyword_hits": decision["keyword_hits"],
        }
        rows.append(row)

        if decision["assigned"]:
            stats_by_topic[str(decision["topic_id"])] += 1

    rows.sort(key=lambda x: str(x["material_id"]))

    csv_path = out_dir / "material_topic_map.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "material_id",
            "source_path",
            "material_role",
            "topic_id",
            "topic_title",
            "score",
            "assigned",
            "assignment_reason",
            "keyword_hits",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    catalog = {
        "threshold": args.threshold,
        "topics": [
            {
                "topic_id": topic["topic_id"],
                "title": topic["title"],
                "theme_prefix": topic["theme_prefix"],
                "module_slug": topic["module_slug"],
                "keywords": topic["keywords"],
                "assigned_materials": stats_by_topic.get(topic["topic_id"], 0),
            }
            for topic in TOPICS
        ],
        "stats": {
            "total_materials": len(rows),
            "assigned": sum(1 for x in rows if x["assigned"]),
            "unassigned": sum(1 for x in rows if not x["assigned"]),
        },
    }

    catalog_path = out_dir / "topic_catalog.json"
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"wrote {csv_path.relative_to(project_root)}")
    log(f"wrote {catalog_path.relative_to(project_root)}")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
