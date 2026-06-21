#!/usr/bin/env python3
"""Validate generated assessment files against structural and language contracts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

QUESTION_RE = re.compile(r"^### Q(\d+) - (.+)$", re.MULTILINE)
SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)

ALLOWED_LATIN_TOKENS = {
    "rnn",
    "lstm",
    "gru",
    "transformer",
    "seq2seq",
    "email",
    "utc",
    "ffn",
}

FORBIDDEN_ANGLO_PATTERNS = [
    r"\blearning rate\b",
    r"\bhidden state\b",
    r"\btrain/validation\b",
    r"\bself-attention\b",
    r"\bdecoder-only\b",
    r"\bencoder-decoder\b",
    r"\bnext-token\b",
    r"\binference\b",
    r"\battention\b",
    r"\bpadding\b",
    r"\bmasking\b",
    r"\balignment\b",
]


class ValidationError(Exception):
    """Raised when generated tests violate a required contract."""


def log(message: str) -> None:
    print(f"[08_validate][INFO] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated tests and keys.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--workspace-root", default="90-assessment-ai-math", help="Assessment workspace root.")
    parser.add_argument(
        "--legacy-format",
        action="store_true",
        help="Validate previous legacy format (A-D options without explicit type blocks).",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def get_sections(text: str) -> list[tuple[str, int, int]]:
    matches = list(SECTION_RE.finditer(text))
    sections: list[tuple[str, int, int]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(1).strip(), start, end))
    return sections


def get_questions(block: str) -> list[tuple[int, str, int, int]]:
    matches = list(QUESTION_RE.finditer(block))
    questions: list[tuple[int, str, int, int]] = []
    for i, m in enumerate(matches):
        qid = int(m.group(1))
        title = m.group(2).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        questions.append((qid, title, start, end))
    return questions


def parse_type_label(block: str) -> str:
    m = re.search(r"^Тип:\s*(.+)$", block, flags=re.MULTILINE)
    if not m:
        return "unknown"

    label = m.group(1).strip().lower()
    if "один вариант" in label:
        return "single"
    if "несколько вариантов" in label:
        return "multiple"
    if "открытый вопрос" in label:
        return "open"
    return "unknown"


def validate_file_completeness(out_dir: Path) -> None:
    expected = [
        "test_variant_1.md",
        "test_variant_2.md",
        "answer_template_1.md",
        "answer_template_2.md",
        "answer_key_1.md",
        "answer_key_2.md",
        "README.md",
    ]
    missing = [name for name in expected if not (out_dir / name).exists()]
    if missing:
        raise ValidationError(f"missing generated files: {', '.join(missing)}")


def validate_test_file_v2(path: Path, variant: int) -> dict[int, str]:
    text = read_text(path)
    heading = f"# Вариант {variant}. Тест по дисциплине «Математические основы ИИ»"
    if not text.startswith(heading):
        raise ValidationError(f"{path.name}: invalid H1 heading")

    if "Инструкция:" not in text:
        raise ValidationError(f"{path.name}: missing instruction block")
    if "1. Для закрытых вопросов" not in text or "2. Для открытых вопросов" not in text:
        raise ValidationError(f"{path.name}: invalid instruction wording")

    q_separators = re.findall(r"^### Q\d+ - .+\n\n---\n\n", text, flags=re.MULTILINE)
    if len(q_separators) != 30:
        raise ValidationError(f"{path.name}: each question must have separator after H3")

    sections = get_sections(text)
    if len(sections) != 6:
        raise ValidationError(f"{path.name}: expected 6 sections, got {len(sections)}")

    qtype_by_id: dict[int, str] = {}
    found_ids: list[int] = []

    for _title, start, end in sections:
        block = text[start:end]
        questions = get_questions(block)
        if len(questions) != 5:
            raise ValidationError(f"{path.name}: each section must have 5 questions")

        mix = {"single": 0, "multiple": 0, "open": 0}
        for qid, _qtitle, qstart, qend in questions:
            qblock = block[qstart:qend]

            if "Тип:" not in qblock:
                raise ValidationError(f"{path.name}: Q{qid} missing type block")
            if "Формулировка вопроса:" not in qblock:
                raise ValidationError(f"{path.name}: Q{qid} missing question wording block")

            qtype = parse_type_label(qblock)
            if qtype == "unknown":
                raise ValidationError(f"{path.name}: Q{qid} has unknown type label")
            mix[qtype] += 1

            if qtype in {"single", "multiple"}:
                if "Варианты ответа:" not in qblock:
                    raise ValidationError(f"{path.name}: Q{qid} missing options header")

                if qtype == "single" and "Выберите один вариант ответа." not in qblock:
                    raise ValidationError(f"{path.name}: Q{qid} missing single-choice instruction")
                if qtype == "multiple" and "Выберите все верные варианты ответа." not in qblock:
                    raise ValidationError(f"{path.name}: Q{qid} missing multiple-choice instruction")

                option_lines = re.findall(r"^(\d+)\.\s+.+$", qblock, flags=re.MULTILINE)
                if option_lines != ["1", "2", "3", "4"]:
                    raise ValidationError(f"{path.name}: Q{qid} must contain exactly options 1..4")
            else:
                if "Ответ представьте в развернутой форме" not in qblock:
                    raise ValidationError(f"{path.name}: Q{qid} open question missing response guidance")

            qtype_by_id[qid] = qtype
            found_ids.append(qid)

        if mix != {"single": 2, "multiple": 2, "open": 1}:
            raise ValidationError(f"{path.name}: section mix must be 2/2/1, got {mix}")

    if found_ids != list(range(1, 31)):
        raise ValidationError(f"{path.name}: question IDs must be Q1..Q30")

    return qtype_by_id


def validate_template_v2(path: Path, variant: int, qtype_by_id: dict[int, str]) -> None:
    text = read_text(path)
    heading = f"# Вариант {variant}. Ответы по дисциплине «Математические основы ИИ»"
    if not text.startswith(heading):
        raise ValidationError(f"{path.name}: invalid H1 heading")

    required_fields = ["Данные студента:", "ФИО:", "Номер группы:", "Email:"]
    for field in required_fields:
        if field not in text:
            raise ValidationError(f"{path.name}: missing field '{field}'")

    if "Инструкция:" not in text:
        raise ValidationError(f"{path.name}: missing instruction block")

    qids = [int(x) for x in re.findall(r"^### Q(\d+) - .+$", text, flags=re.MULTILINE)]
    if qids != list(range(1, 31)):
        raise ValidationError(f"{path.name}: expected Q1..Q30 response blocks")

    for qid in range(1, 31):
        m = re.search(rf"^### Q{qid} - .+$", text, flags=re.MULTILINE)
        if not m:
            raise ValidationError(f"{path.name}: missing Q{qid} block")
        start = m.end()
        next_m = re.search(r"^### Q\d+ - .+$", text[start:], flags=re.MULTILINE)
        end = start + next_m.start() if next_m else len(text)
        block = text[start:end]

        qtype = parse_type_label(block)
        if qtype != qtype_by_id[qid]:
            raise ValidationError(
                f"{path.name}: Q{qid} type mismatch between test and template ({qtype} vs {qtype_by_id[qid]})"
            )

        if qtype == "open":
            if "Ответ:" not in block:
                raise ValidationError(f"{path.name}: Q{qid} open block missing answer label")
            tail = block.split("Ответ:", 1)[1]
            lines = tail.splitlines()
            blank_count = 0
            for line in lines:
                if line.strip() == "":
                    blank_count += 1
                else:
                    break
            if blank_count < 7:
                raise ValidationError(
                    f"{path.name}: Q{qid} open block must include at least 7 blank lines"
                )
        else:
            if "Ответ (номер(а) выбранных вариантов):" not in block:
                raise ValidationError(f"{path.name}: Q{qid} closed block missing answer slot header")


def validate_key_v2(path: Path, variant: int, qtype_by_id: dict[int, str]) -> None:
    text = read_text(path)
    heading = f"# Вариант {variant}. Ключи ответов (для преподавателя)"
    if not text.startswith(heading):
        raise ValidationError(f"{path.name}: invalid key H1 heading")

    qids = [int(x) for x in re.findall(r"^### Q(\d+) - .+$", text, flags=re.MULTILINE)]
    if qids != list(range(1, 31)):
        raise ValidationError(f"{path.name}: expected Q1..Q30 in key")

    for qid in range(1, 31):
        m = re.search(rf"^### Q{qid} - .+$", text, flags=re.MULTILINE)
        if not m:
            raise ValidationError(f"{path.name}: missing Q{qid}")
        start = m.end()
        next_m = re.search(r"^### Q\d+ - .+$", text[start:], flags=re.MULTILINE)
        end = start + next_m.start() if next_m else len(text)
        block = text[start:end]

        if "Тип:" not in block:
            raise ValidationError(f"{path.name}: Q{qid} missing type block")
        if "Формулировка вопроса:" not in block:
            raise ValidationError(f"{path.name}: Q{qid} missing wording block")

        qtype = parse_type_label(block)
        if qtype != qtype_by_id[qid]:
            raise ValidationError(
                f"{path.name}: Q{qid} type mismatch between test and key ({qtype} vs {qtype_by_id[qid]})"
            )

        if qtype == "single":
            if not re.search(r"Правильный ответ:\s*\d+\.\s+", block):
                raise ValidationError(f"{path.name}: Q{qid} single-answer block malformed")
        elif qtype == "multiple":
            if not re.search(r"Правильные ответы:\s*\d+(,\s*\d+)+", block):
                raise ValidationError(f"{path.name}: Q{qid} multiple-answer block malformed")
        else:
            if "Эталонный ответ:" not in block:
                raise ValidationError(f"{path.name}: Q{qid} open reference answer missing")
            if "Критерии оценивания:" not in block:
                raise ValidationError(f"{path.name}: Q{qid} open grading criteria missing")

        if not re.search(
            r"^Техническая трассируемость:\s+.+\s*->\s+.+\s*->\s+.+\s*->\s*Q\d+\s*$",
            block,
            flags=re.MULTILINE,
        ):
            raise ValidationError(f"{path.name}: Q{qid} technical traceability missing")

        if not re.search(r"^Источник материала:\s+.+$", block, flags=re.MULTILINE):
            raise ValidationError(f"{path.name}: Q{qid} source material line missing")


def detect_qtype_legacy(block: str) -> str:
    if "Выберите один верный вариант." in block:
        return "single"
    if "Выберите все верные варианты." in block:
        return "multiple"
    return "open"


def validate_test_file_legacy(path: Path, variant: int) -> dict[int, str]:
    text = read_text(path)
    heading = f"# Вариант {variant}. Тест по дисциплине «Математические основы ИИ»"
    if not text.startswith(heading):
        raise ValidationError(f"{path.name}: invalid H1 heading")

    q_separators = re.findall(r"^### Q\d+ - .+\n\n---\n\n", text, flags=re.MULTILINE)
    if len(q_separators) != 30:
        raise ValidationError(f"{path.name}: each question must have separator after H3")

    sections = get_sections(text)
    if len(sections) != 6:
        raise ValidationError(f"{path.name}: expected 6 sections, got {len(sections)}")

    qtype_by_id: dict[int, str] = {}
    found_ids: list[int] = []

    for _title, start, end in sections:
        block = text[start:end]
        questions = get_questions(block)
        if len(questions) != 5:
            raise ValidationError(f"{path.name}: each section must have 5 questions")

        mix = {"single": 0, "multiple": 0, "open": 0}
        for qid, _qtitle, qstart, qend in questions:
            qblock = block[qstart:qend]
            qtype = detect_qtype_legacy(qblock)
            mix[qtype] += 1

            if qtype in {"single", "multiple"}:
                options = re.findall(r"^- [A-D]\) ", qblock, flags=re.MULTILINE)
                if len(options) != 4:
                    raise ValidationError(f"{path.name}: Q{qid} must have exactly 4 options")

            qtype_by_id[qid] = qtype
            found_ids.append(qid)

        if mix != {"single": 2, "multiple": 2, "open": 1}:
            raise ValidationError(f"{path.name}: section mix must be 2/2/1, got {mix}")

    if found_ids != list(range(1, 31)):
        raise ValidationError(f"{path.name}: question IDs must be Q1..Q30")

    return qtype_by_id


def validate_template_legacy(path: Path, variant: int, qtype_by_id: dict[int, str]) -> None:
    text = read_text(path)
    heading = f"# Вариант {variant}. Ответы по дисциплине «Математические основы ИИ»"
    if not text.startswith(heading):
        raise ValidationError(f"{path.name}: invalid H1 heading")

    if not re.search(r"ФИО:\n\nНомер группы:\n\nEmail:\n", text):
        raise ValidationError(f"{path.name}: missing required student fields or blank lines")

    qids = [int(x) for x in re.findall(r"^### Q(\d+) - Ответ$", text, flags=re.MULTILINE)]
    if qids != list(range(1, 31)):
        raise ValidationError(f"{path.name}: expected Q1..Q30 response blocks")

    for qid in range(1, 31):
        pattern = re.compile(rf"^### Q{qid} - Ответ$", re.MULTILINE)
        m = pattern.search(text)
        if not m:
            raise ValidationError(f"{path.name}: missing Q{qid} block")
        start = m.end()
        next_m = re.search(r"^### Q\d+ - Ответ$", text[start:], flags=re.MULTILINE)
        end = start + next_m.start() if next_m else len(text)
        block = text[start:end]

        qtype = qtype_by_id[qid]
        if qtype == "open":
            if "Ответ:" not in block:
                raise ValidationError(f"{path.name}: Q{qid} open block missing answer label")
            tail = block.split("Ответ:", 1)[1]
            lines = tail.splitlines()
            blank_count = 0
            for line in lines:
                if line.strip() == "":
                    blank_count += 1
                else:
                    break
            if blank_count < 7:
                raise ValidationError(
                    f"{path.name}: Q{qid} open block must include at least 7 blank lines"
                )


def validate_key_legacy(path: Path, variant: int, qtype_by_id: dict[int, str]) -> None:
    text = read_text(path)
    heading = f"# Вариант {variant}. Ключ преподавателя по дисциплине «Математические основы ИИ»"
    if not text.startswith(heading):
        raise ValidationError(f"{path.name}: invalid key H1 heading")

    qids = [int(x) for x in re.findall(r"^### Q(\d+) - ", text, flags=re.MULTILINE)]
    if qids != list(range(1, 31)):
        raise ValidationError(f"{path.name}: expected Q1..Q30 in key")

    for qid in range(1, 31):
        pattern = re.compile(rf"^### Q{qid} - .+$", re.MULTILINE)
        m = pattern.search(text)
        if not m:
            raise ValidationError(f"{path.name}: missing Q{qid}")
        start = m.end()
        next_m = re.search(r"^### Q\d+ - .+$", text[start:], flags=re.MULTILINE)
        end = start + next_m.start() if next_m else len(text)
        block = text[start:end]

        qtype = qtype_by_id[qid]
        if qtype == "single" and "Правильный ответ:" not in block:
            raise ValidationError(f"{path.name}: Q{qid} single answer key missing")
        if qtype == "multiple" and "Правильные ответы:" not in block:
            raise ValidationError(f"{path.name}: Q{qid} multiple answer key missing")
        if qtype == "open":
            if "Эталонный ответ:" not in block:
                raise ValidationError(f"{path.name}: Q{qid} open reference answer missing")
            if "Критерии оценивания:" not in block:
                raise ValidationError(f"{path.name}: Q{qid} open grading criteria missing")

        if not re.search(r"Трассируемость:\s+\S+\s*->\s+\S+\s*->\s+\S+\s*->\s*Q\d+", block):
            raise ValidationError(f"{path.name}: Q{qid} traceability chain missing")


def validate_traceability_json(path: Path) -> None:
    if not path.exists():
        raise ValidationError(f"{path.name}: missing traceability json")

    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("traceability", [])
    if len(items) != 60:
        raise ValidationError("question_traceability.json: expected 60 entries")

    required = {"variant", "question_id", "topic_id", "material_id", "module_slug", "source_path"}
    for idx, item in enumerate(items, start=1):
        missing = required.difference(item.keys())
        if missing:
            raise ValidationError(
                f"question_traceability.json: item #{idx} missing keys: {sorted(missing)}"
            )


def strip_technical_lines(text: str) -> str:
    cleaned: list[str] = []
    for line in text.splitlines():
        if line.startswith("Техническая трассируемость:"):
            continue
        if line.startswith("Источник материала:"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def strip_code_fragments(text: str) -> str:
    return re.sub(r"`[^`]*`", " ", text)


def validate_language_contract(out_dir: Path) -> None:
    student_files = [
        out_dir / "test_variant_1.md",
        out_dir / "test_variant_2.md",
        out_dir / "answer_template_1.md",
        out_dir / "answer_template_2.md",
        out_dir / "README.md",
    ]
    key_files = [
        out_dir / "answer_key_1.md",
        out_dir / "answer_key_2.md",
    ]

    def check_one(path: Path, threshold: float, strip_tech: bool) -> None:
        text = read_text(path)
        if strip_tech:
            text = strip_technical_lines(text)
        text = strip_code_fragments(text)

        low_text = text.lower()
        for pattern in FORBIDDEN_ANGLO_PATTERNS:
            if re.search(pattern, low_text):
                raise ValidationError(
                    f"{path.name}: forbidden anglo phrase detected by pattern '{pattern}'"
                )

        latin_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_/.-]*", text)
        latin_filtered = []
        for token in latin_tokens:
            normalized = token.lower().strip(".,;:!?()[]{}\"'")
            if not normalized:
                continue
            if normalized in ALLOWED_LATIN_TOKENS:
                continue
            if normalized.startswith("q") and normalized[1:].isdigit():
                continue
            latin_filtered.append(normalized)

        all_words = re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", text)
        ratio = len(latin_filtered) / max(1, len(all_words))
        if ratio > threshold:
            raise ValidationError(
                f"{path.name}: latin token share too high ({ratio:.3f} > {threshold:.3f})"
            )

    for path in student_files:
        check_one(path, threshold=0.06, strip_tech=False)
    for path in key_files:
        check_one(path, threshold=0.10, strip_tech=True)


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    workspace_root = (project_root / args.workspace_root).resolve()
    out_dir = workspace_root / "05-tests-v1v2"

    validate_file_completeness(out_dir)

    if args.legacy_format:
        qtypes_v1 = validate_test_file_legacy(out_dir / "test_variant_1.md", 1)
        qtypes_v2 = validate_test_file_legacy(out_dir / "test_variant_2.md", 2)

        validate_template_legacy(out_dir / "answer_template_1.md", 1, qtypes_v1)
        validate_template_legacy(out_dir / "answer_template_2.md", 2, qtypes_v2)

        validate_key_legacy(out_dir / "answer_key_1.md", 1, qtypes_v1)
        validate_key_legacy(out_dir / "answer_key_2.md", 2, qtypes_v2)
    else:
        qtypes_v1 = validate_test_file_v2(out_dir / "test_variant_1.md", 1)
        qtypes_v2 = validate_test_file_v2(out_dir / "test_variant_2.md", 2)

        validate_template_v2(out_dir / "answer_template_1.md", 1, qtypes_v1)
        validate_template_v2(out_dir / "answer_template_2.md", 2, qtypes_v2)

        validate_key_v2(out_dir / "answer_key_1.md", 1, qtypes_v1)
        validate_key_v2(out_dir / "answer_key_2.md", 2, qtypes_v2)

        validate_language_contract(out_dir)

    validate_traceability_json(out_dir / "question_traceability.json")

    mode = "legacy" if args.legacy_format else "academic_ru_v2"
    log(f"all validation checks passed ({mode})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"[08_validate][ERROR] {exc}", flush=True)
        raise SystemExit(1)
