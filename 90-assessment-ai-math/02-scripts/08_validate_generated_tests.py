#!/usr/bin/env python3
"""Validate generated tests and answer artifacts against transfer-quality contracts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

QUESTION_RE = re.compile(r"^### Q(\d+) - (.+)$", re.MULTILINE)
SECTION_RE = re.compile(r"^## (.+)$", re.MULTILINE)
LATEX_RE = re.compile(r"\$[^$\n]+\$")

ALLOWED_LATIN_TOKENS = {
    "rnn",
    "lstm",
    "gru",
    "seq2seq",
    "ffn",
    "f1",
    "email",
    "latex",
    "utc",
    "markdown",
    "tests-v1v2",
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
    r"\balignment\b",
]

META_DIALOG_PATTERNS = [
    r"преподавател[ьяю]\s+важно",
    r"студент(у|а)?\s+должен\s+показать",
    r"студент(у|а)?\s+нужно\s+доказать",
    r"провер[ьяе]тся\s+преподавателем",
    r"для\s+проверки\s+преподавателем",
]

OBLIGATION_MARKERS = re.compile(r"\b(обязательно|необходимо|требуется|нужно)\b", re.IGNORECASE)
OPTIONAL_MARKERS = re.compile(r"\b(можно|дополнительно|при желании)\b", re.IGNORECASE)

RUSSIAN_STOPWORDS = {
    "и",
    "в",
    "во",
    "на",
    "по",
    "для",
    "при",
    "как",
    "что",
    "это",
    "или",
    "из",
    "к",
    "с",
    "не",
    "но",
    "же",
    "а",
    "о",
    "об",
    "от",
    "до",
    "над",
    "под",
    "без",
    "если",
    "ли",
    "где",
}

PROJECT_MARKER_PATTERNS = [
    r"themes/",
    r"\.ipynb\b",
    r"\bguides\b",
    r"\bsolutions\b",
    r"\bruns?_",
    r"\brun[_-]",
    r"\bmaterial_id\b",
    r"\bmodule_slug\b",
    r"\bsource_path\b",
]

KNOWLEDGE_SCOPE_REQUIRED = "internet_general"
BASIC_MIN_SHARE = 0.80


class ValidationError(Exception):
    """Raised when generated files violate required contracts."""


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


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_compare_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", " ", text.lower())).strip()


def contains_latex(text: str) -> bool:
    return LATEX_RE.search(text) is not None


def get_sections(text: str) -> list[tuple[str, int, int]]:
    matches = list(SECTION_RE.finditer(text))
    sections: list[tuple[str, int, int]] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((match.group(1).strip(), start, end))
    return sections


def get_questions(block: str) -> list[tuple[int, str, int, int]]:
    matches = list(QUESTION_RE.finditer(block))
    questions: list[tuple[int, str, int, int]] = []
    for i, match in enumerate(matches):
        qid = int(match.group(1))
        title = match.group(2).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        questions.append((qid, title, start, end))
    return questions


def parse_type_label(block: str) -> str:
    match = re.search(r"^Тип:\s*(.+)$", block, flags=re.MULTILINE)
    if not match:
        return "unknown"
    label = match.group(1).strip().lower()
    if "один вариант" in label:
        return "single"
    if "несколько вариантов" in label:
        return "multiple"
    if "открытый вопрос" in label:
        return "open"
    return "unknown"


def extract_context_task(block: str) -> tuple[str, str]:
    context_match = re.search(
        r"Контекст:\n(.+?)\n\nЗадание:\n",
        block,
        flags=re.DOTALL,
    )
    if not context_match:
        raise ValidationError("missing or malformed 'Контекст' block")

    task_match = re.search(
        r"Задание:\n(.+?)\n\n(?:Инструкция:|Подсказка \(ход рассуждения\):)",
        block,
        flags=re.DOTALL,
    )
    if not task_match:
        raise ValidationError("missing or malformed 'Задание' block")

    context = context_match.group(1).strip()
    task = task_match.group(1).strip()
    if not context:
        raise ValidationError("empty context")
    if not task:
        raise ValidationError("empty task")
    return context, task


def extract_closed_instruction(block: str, qtype: str) -> str:
    match = re.search(
        r"Инструкция:\n(.+?)\n\nВарианты ответа:\n",
        block,
        flags=re.DOTALL,
    )
    if not match:
        raise ValidationError("missing or malformed 'Инструкция' block for closed question")
    instruction = normalize_spaces(match.group(1))

    expected = {
        "single": "Выберите один вариант ответа.",
        "multiple": "Выберите все верные варианты ответа.",
    }
    if instruction != expected[qtype]:
        raise ValidationError(f"closed question has invalid instruction: '{instruction}'")
    return instruction


def extract_open_hint_steps(block: str) -> list[str]:
    match = re.search(
        r"Подсказка \(ход рассуждения\):\n((?:\d+\.\s+.+\n?)+)\nОтвет представьте",
        block,
        flags=re.DOTALL,
    )
    if not match:
        raise ValidationError("missing or malformed 'Подсказка (ход рассуждения)' block")

    step_lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    steps: list[str] = []
    for idx, line in enumerate(step_lines, start=1):
        expected_prefix = f"{idx}. "
        if not line.startswith(expected_prefix):
            raise ValidationError("hint steps must be sequentially numbered from 1")
        steps.append(line[len(expected_prefix) :].strip())

    if not (2 <= len(steps) <= 3):
        raise ValidationError("open question must contain 2-3 hint steps")
    if any(not step for step in steps):
        raise ValidationError("open question contains empty hint step")
    return steps


def validate_file_completeness(out_dir: Path) -> None:
    expected = [
        "test_variant_1.md",
        "test_variant_2.md",
        "answer_template_1.md",
        "answer_template_2.md",
        "answer_key_1.md",
        "answer_key_2.md",
        "README.md",
        "question_traceability.json",
    ]
    missing = [name for name in expected if not (out_dir / name).exists()]
    if missing:
        raise ValidationError(f"missing generated files: {', '.join(missing)}")


def validate_test_file_v2(path: Path, variant: int) -> dict[int, dict[str, Any]]:
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

    found_ids: list[int] = []
    meta: dict[int, dict[str, Any]] = {}

    for _title, start, end in sections:
        section_block = text[start:end]
        questions = get_questions(section_block)
        if len(questions) != 5:
            raise ValidationError(f"{path.name}: each section must have 5 questions")

        mix = {"single": 0, "multiple": 0, "open": 0}
        for qid, qtitle, qstart, qend in questions:
            qblock = section_block[qstart:qend]
            qtype = parse_type_label(qblock)
            if qtype == "unknown":
                raise ValidationError(f"{path.name}: Q{qid} has unknown type")
            mix[qtype] += 1

            try:
                context, task = extract_context_task(qblock)
            except ValidationError as exc:
                raise ValidationError(f"{path.name}: Q{qid} {exc}") from exc

            if qtype in {"single", "multiple"}:
                try:
                    _ = extract_closed_instruction(qblock, qtype)
                except ValidationError as exc:
                    raise ValidationError(f"{path.name}: Q{qid} {exc}") from exc

                option_matches = re.findall(r"^([1-4])\.\s+(.+)$", qblock, flags=re.MULTILINE)
                option_nums = [num for num, _ in option_matches]
                if option_nums != ["1", "2", "3", "4"]:
                    raise ValidationError(f"{path.name}: Q{qid} must contain exactly options 1..4")
                options = [text.strip() for _, text in option_matches]
                hint_steps: list[str] = []
            else:
                if "Ответ представьте в развернутой форме" not in qblock:
                    raise ValidationError(f"{path.name}: Q{qid} open question missing response guidance")
                try:
                    hint_steps = extract_open_hint_steps(qblock)
                except ValidationError as exc:
                    raise ValidationError(f"{path.name}: Q{qid} {exc}") from exc
                options = []

            meta[qid] = {
                "question_id": qid,
                "title": qtitle,
                "qtype": qtype,
                "context": context,
                "task": task,
                "options": options,
                "hint_steps": hint_steps,
                "block": qblock,
                "has_latex": contains_latex(qblock),
            }
            found_ids.append(qid)

        if mix != {"single": 2, "multiple": 2, "open": 1}:
            raise ValidationError(f"{path.name}: section mix must be 2/2/1, got {mix}")

    if found_ids != list(range(1, 31)):
        raise ValidationError(f"{path.name}: question IDs must be Q1..Q30")
    return meta


def validate_template_v2(path: Path, variant: int, test_meta: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    text = read_text(path)
    heading = f"# Вариант {variant}. Ответы по дисциплине «Математические основы ИИ»"
    if not text.startswith(heading):
        raise ValidationError(f"{path.name}: invalid H1 heading")

    required_fields = ["Данные студента:", "ФИО:", "Номер группы:", "Email:", "Инструкция:"]
    for field in required_fields:
        if field not in text:
            raise ValidationError(f"{path.name}: missing field '{field}'")

    matches = list(QUESTION_RE.finditer(text))
    qids = [int(m.group(1)) for m in matches]
    if qids != list(range(1, 31)):
        raise ValidationError(f"{path.name}: expected Q1..Q30 response blocks")

    meta: dict[int, dict[str, Any]] = {}
    for i, match in enumerate(matches):
        qid = int(match.group(1))
        title = match.group(2).strip()
        if title != test_meta[qid]["title"]:
            raise ValidationError(f"{path.name}: Q{qid} title mismatch with test file")

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        qtype = parse_type_label(block)
        if qtype != test_meta[qid]["qtype"]:
            raise ValidationError(f"{path.name}: Q{qid} type mismatch with test file")

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
                raise ValidationError(f"{path.name}: Q{qid} open block must include at least 7 blank lines")
        else:
            if "Ответ (номер(а) выбранных вариантов):" not in block:
                raise ValidationError(f"{path.name}: Q{qid} closed block missing answer slot header")

        meta[qid] = {"title": title, "qtype": qtype}

    return meta


def _parse_key_technical_block(block: str, qid: int, filename: str) -> dict[str, Any]:
    if "Технический блок:" not in block:
        raise ValidationError(f"{filename}: Q{qid} missing technical block")

    trace_match = re.search(
        r"^- Трассируемость:\s+(.+?)\s*->\s*(.+?)\s*->\s*(.+?)\s*->\s*Q(\d+)\s*$",
        block,
        flags=re.MULTILINE,
    )
    if not trace_match:
        raise ValidationError(f"{filename}: Q{qid} malformed technical traceability line")
    if int(trace_match.group(4)) != qid:
        raise ValidationError(f"{filename}: Q{qid} traceability line points to another question id")

    source_match = re.search(r"^- Источник материала:\s+.+$", block, flags=re.MULTILINE)
    if not source_match:
        raise ValidationError(f"{filename}: Q{qid} missing source material line")

    risk_match = re.search(r"^- Уровень риска:\s+(normal|high)\s*$", block, flags=re.MULTILINE)
    if not risk_match:
        raise ValidationError(f"{filename}: Q{qid} missing risk level line")

    latex_match = re.search(r"^- Требуется LaTeX:\s+(да|нет)\s*$", block, flags=re.MULTILINE)
    if not latex_match:
        raise ValidationError(f"{filename}: Q{qid} missing LaTeX flag line")

    style_match = re.search(r"^- Профиль стиля:\s+.+$", block, flags=re.MULTILINE)
    if not style_match:
        raise ValidationError(f"{filename}: Q{qid} missing style profile line")

    difficulty_match = re.search(r"^- Уровень сложности:\s+(basic|medium)\s*$", block, flags=re.MULTILINE)
    if not difficulty_match:
        raise ValidationError(f"{filename}: Q{qid} missing difficulty level line")

    scope_match = re.search(r"^- Область знаний:\s+(.+)$", block, flags=re.MULTILINE)
    if not scope_match:
        raise ValidationError(f"{filename}: Q{qid} missing knowledge scope line")

    return {
        "trace_topic_id": trace_match.group(1).strip(),
        "trace_material_id": trace_match.group(2).strip(),
        "trace_module_slug": trace_match.group(3).strip(),
        "risk_level": risk_match.group(1),
        "requires_latex": latex_match.group(1) == "да",
        "style_profile": style_match.group(0).split(":", 1)[1].strip(),
        "difficulty_level": difficulty_match.group(1),
        "knowledge_scope": scope_match.group(1).strip(),
    }


def validate_key_v2(path: Path, variant: int, test_meta: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    text = read_text(path)
    heading = f"# Вариант {variant}. Ключи ответов (для преподавателя)"
    if not text.startswith(heading):
        raise ValidationError(f"{path.name}: invalid key H1 heading")

    matches = list(QUESTION_RE.finditer(text))
    qids = [int(m.group(1)) for m in matches]
    if qids != list(range(1, 31)):
        raise ValidationError(f"{path.name}: expected Q1..Q30 in key")

    meta: dict[int, dict[str, Any]] = {}
    for i, match in enumerate(matches):
        qid = int(match.group(1))
        title = match.group(2).strip()
        if title != test_meta[qid]["title"]:
            raise ValidationError(f"{path.name}: Q{qid} title mismatch with test file")

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]

        qtype = parse_type_label(block)
        if qtype != test_meta[qid]["qtype"]:
            raise ValidationError(f"{path.name}: Q{qid} type mismatch with test file")

        try:
            context, task = extract_context_task(block)
        except ValidationError as exc:
            raise ValidationError(f"{path.name}: Q{qid} {exc}") from exc

        if normalize_spaces(context) != normalize_spaces(test_meta[qid]["context"]):
            raise ValidationError(f"{path.name}: Q{qid} context mismatch between test and key")
        if normalize_spaces(task) != normalize_spaces(test_meta[qid]["task"]):
            raise ValidationError(f"{path.name}: Q{qid} task mismatch between test and key")

        if qtype in {"single", "multiple"}:
            instruction_match = re.search(r"Инструкция:\n(.+?)\n\n", block, flags=re.DOTALL)
            if not instruction_match:
                raise ValidationError(f"{path.name}: Q{qid} missing or malformed 'Инструкция' block for closed question")
            instruction = normalize_spaces(instruction_match.group(1))
            expected_instruction = {
                "single": "Выберите один вариант ответа.",
                "multiple": "Выберите все верные варианты ответа.",
            }[qtype]
            if instruction != expected_instruction:
                raise ValidationError(f"{path.name}: Q{qid} closed question has invalid instruction")
        else:
            hint_match = re.search(
                r"Подсказка \(ход рассуждения\):\n((?:\d+\.\s+.+\n?)+)\nЭталонный ответ:",
                block,
                flags=re.DOTALL,
            )
            if not hint_match:
                raise ValidationError(f"{path.name}: Q{qid} missing or malformed 'Подсказка (ход рассуждения)' block")
            step_lines = [line.strip() for line in hint_match.group(1).splitlines() if line.strip()]
            if not (2 <= len(step_lines) <= 3):
                raise ValidationError(f"{path.name}: Q{qid} open question must contain 2-3 hint steps")
            for idx, line in enumerate(step_lines, start=1):
                if not line.startswith(f"{idx}. "):
                    raise ValidationError(f"{path.name}: Q{qid} hint steps must be numbered from 1")

        correct_indices: list[int] = []
        if qtype == "single":
            answer_match = re.search(r"Правильный ответ:\s*(\d+)\.\s+(.+)$", block, flags=re.MULTILINE)
            if not answer_match:
                raise ValidationError(f"{path.name}: Q{qid} malformed single-answer block")
            correct_indices = [int(answer_match.group(1))]
        elif qtype == "multiple":
            answers_match = re.search(r"Правильные ответы:\s*(\d+(?:,\s*\d+)+)$", block, flags=re.MULTILINE)
            if not answers_match:
                raise ValidationError(f"{path.name}: Q{qid} malformed multiple-answer block")
            correct_indices = [int(x.strip()) for x in answers_match.group(1).split(",")]
            listed = [int(x) for x in re.findall(r"^- (\d+)\.\s+.+$", block, flags=re.MULTILINE)]
            if listed != correct_indices:
                raise ValidationError(f"{path.name}: Q{qid} multiple answers must match list items")
        else:
            if "Эталонный ответ:" not in block:
                raise ValidationError(f"{path.name}: Q{qid} open reference answer missing")
            if "Критерии оценивания:" not in block:
                raise ValidationError(f"{path.name}: Q{qid} open grading criteria missing")

        tech = _parse_key_technical_block(block, qid, path.name)
        meta[qid] = {
            "title": title,
            "qtype": qtype,
            "block": block,
            "has_latex": contains_latex(block),
            "correct_indices": correct_indices,
            **tech,
        }
    return meta


def validate_traceability_json(path: Path, legacy_mode: bool) -> dict[tuple[int, int], dict[str, Any]]:
    if not path.exists():
        raise ValidationError(f"{path.name}: missing traceability json")

    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("traceability", [])
    if len(items) != 60:
        raise ValidationError("question_traceability.json: expected 60 entries")

    base_required = {"variant", "question_id", "topic_id", "material_id", "module_slug", "source_path"}
    extended_required = {
        "title",
        "question_type",
        "context",
        "task",
        "options",
        "correct",
        "hint_steps",
        "risk_level",
        "requires_latex",
        "allowed_abbrev",
        "style_profile",
        "difficulty_level",
        "knowledge_scope",
        "section_title",
        "reference",
        "criteria",
    }
    required = base_required if legacy_mode else (base_required | extended_required)

    result: dict[tuple[int, int], dict[str, Any]] = {}
    seen: set[tuple[int, int]] = set()
    for idx, item in enumerate(items, start=1):
        missing = required.difference(item.keys())
        if missing:
            raise ValidationError(f"question_traceability.json: item #{idx} missing keys: {sorted(missing)}")

        variant = int(item["variant"])
        if variant not in {1, 2}:
            raise ValidationError(f"question_traceability.json: item #{idx} has invalid variant")

        qid_match = re.fullmatch(r"Q(\d+)", str(item["question_id"]))
        if not qid_match:
            raise ValidationError(f"question_traceability.json: item #{idx} has invalid question_id")
        qid = int(qid_match.group(1))
        if not (1 <= qid <= 30):
            raise ValidationError(f"question_traceability.json: item #{idx} has out-of-range question_id")

        key = (variant, qid)
        if key in seen:
            raise ValidationError(f"question_traceability.json: duplicate mapping for {key}")
        seen.add(key)

        if not legacy_mode:
            qtype = str(item["question_type"])
            if qtype not in {"single", "multiple", "open"}:
                raise ValidationError(f"question_traceability.json: item #{idx} has invalid question_type")

            risk = str(item["risk_level"])
            if risk not in {"normal", "high"}:
                raise ValidationError(f"question_traceability.json: item #{idx} has invalid risk_level")
            if not isinstance(item["requires_latex"], bool):
                raise ValidationError(f"question_traceability.json: item #{idx} requires_latex must be boolean")
            if not isinstance(item["allowed_abbrev"], list):
                raise ValidationError(f"question_traceability.json: item #{idx} allowed_abbrev must be list")
            if not str(item["style_profile"]).strip():
                raise ValidationError(f"question_traceability.json: item #{idx} style_profile must be non-empty")
            if str(item["difficulty_level"]) not in {"basic", "medium"}:
                raise ValidationError(f"question_traceability.json: item #{idx} has invalid difficulty_level")
            if str(item["knowledge_scope"]) != KNOWLEDGE_SCOPE_REQUIRED:
                raise ValidationError(
                    f"question_traceability.json: item #{idx} must use knowledge_scope={KNOWLEDGE_SCOPE_REQUIRED}"
                )

        result[key] = item

    expected_keys = {(variant, qid) for variant in (1, 2) for qid in range(1, 31)}
    if set(result.keys()) != expected_keys:
        raise ValidationError("question_traceability.json: expected complete (variant, Q1..Q30) coverage")
    return result


def validate_sync_with_traceability(
    test_by_variant: dict[int, dict[int, dict[str, Any]]],
    key_by_variant: dict[int, dict[int, dict[str, Any]]],
    trace_map: dict[tuple[int, int], dict[str, Any]],
) -> None:
    for variant in (1, 2):
        for qid in range(1, 31):
            q_test = test_by_variant[variant][qid]
            q_key = key_by_variant[variant][qid]
            trace = trace_map[(variant, qid)]

            if str(trace["title"]).strip() != q_test["title"]:
                raise ValidationError(f"traceability mismatch: variant {variant} Q{qid} title differs from test")
            if str(trace["question_type"]).strip() != q_test["qtype"]:
                raise ValidationError(f"traceability mismatch: variant {variant} Q{qid} type differs from test")
            if normalize_spaces(str(trace["context"])) != normalize_spaces(q_test["context"]):
                raise ValidationError(f"traceability mismatch: variant {variant} Q{qid} context differs from test")
            if normalize_spaces(str(trace["task"])) != normalize_spaces(q_test["task"]):
                raise ValidationError(f"traceability mismatch: variant {variant} Q{qid} task differs from test")

            if q_test["qtype"] in {"single", "multiple"}:
                trace_options = [str(x).strip() for x in trace.get("options", [])]
                if trace_options != q_test["options"]:
                    raise ValidationError(f"traceability mismatch: variant {variant} Q{qid} options differ from test")

            if str(trace["risk_level"]) != q_key["risk_level"]:
                raise ValidationError(f"traceability mismatch: variant {variant} Q{qid} risk level differs from key")
            if bool(trace["requires_latex"]) != bool(q_key["requires_latex"]):
                raise ValidationError(f"traceability mismatch: variant {variant} Q{qid} LaTeX flag differs from key")
            if str(trace["style_profile"]).strip() != str(q_key["style_profile"]).strip():
                raise ValidationError(f"traceability mismatch: variant {variant} Q{qid} style profile differs from key")
            if str(trace["difficulty_level"]).strip() != str(q_key["difficulty_level"]).strip():
                raise ValidationError(f"traceability mismatch: variant {variant} Q{qid} difficulty differs from key")
            if str(trace["knowledge_scope"]).strip() != str(q_key["knowledge_scope"]).strip():
                raise ValidationError(f"traceability mismatch: variant {variant} Q{qid} knowledge scope differs from key")

            if q_test["qtype"] == "single":
                trace_indices = [int(i) + 1 for i in trace.get("correct", [])]
                if q_key["correct_indices"] != trace_indices:
                    raise ValidationError(f"traceability mismatch: variant {variant} Q{qid} single correct answer differs")
            elif q_test["qtype"] == "multiple":
                trace_indices = [int(i) + 1 for i in trace.get("correct", [])]
                if q_key["correct_indices"] != trace_indices:
                    raise ValidationError(
                        f"traceability mismatch: variant {variant} Q{qid} multiple correct answers differ"
                    )


def content_tokens(text: str) -> list[str]:
    raw = re.findall(r"[0-9A-Za-zА-Яа-яЁё]+", text.lower())
    return [w for w in raw if len(w) >= 4 and w not in RUSSIAN_STOPWORDS]


def has_semantic_duplication(context: str, option: str) -> bool:
    option_tokens = content_tokens(option)
    context_tokens = content_tokens(context)
    if len(option_tokens) < 5:
        return False

    context_joined = " ".join(context_tokens)
    option_joined = " ".join(option_tokens)
    if option_joined and option_joined in context_joined:
        return True

    for i in range(len(option_tokens) - 4):
        phrase = " ".join(option_tokens[i : i + 5])
        if phrase and phrase in context_joined:
            return True
    return False


def validate_anti_leakage(
    test_by_variant: dict[int, dict[int, dict[str, Any]]],
    trace_map: dict[tuple[int, int], dict[str, Any]],
) -> None:
    for variant in (1, 2):
        for qid in range(1, 31):
            q = test_by_variant[variant][qid]
            if q["qtype"] not in {"single", "multiple"}:
                continue

            context_norm = normalize_compare_text(q["context"])
            risk_level = str(trace_map[(variant, qid)]["risk_level"])
            for option in q["options"]:
                option_norm = normalize_compare_text(option)
                if option_norm and len(option_norm) >= 20 and option_norm in context_norm:
                    raise ValidationError(
                        f"anti-leakage violation: variant {variant} Q{qid} context fully contains option text"
                    )

                if risk_level == "high" and has_semantic_duplication(q["context"], option):
                    raise ValidationError(
                        f"anti-leakage violation: variant {variant} Q{qid} high-risk context is too close to option wording"
                    )


def validate_modality(
    test_by_variant: dict[int, dict[int, dict[str, Any]]],
    trace_map: dict[tuple[int, int], dict[str, Any]],
) -> None:
    for variant in (1, 2):
        for qid in range(1, 31):
            q = test_by_variant[variant][qid]
            if q["qtype"] not in {"single", "multiple"}:
                continue
            if not OBLIGATION_MARKERS.search(q["task"]):
                continue

            trace = trace_map[(variant, qid)]
            options = [str(x) for x in trace.get("options", [])]
            correct = [int(i) for i in trace.get("correct", [])]
            for idx in correct:
                if not (0 <= idx < len(options)):
                    raise ValidationError(f"modality check: variant {variant} Q{qid} has invalid correct option index")
                if OPTIONAL_MARKERS.search(options[idx]):
                    raise ValidationError(
                        f"modality conflict: variant {variant} Q{qid} has mandatory task but optional wording in correct answer"
                    )


def validate_latex_requirements(
    test_by_variant: dict[int, dict[int, dict[str, Any]]],
    key_by_variant: dict[int, dict[int, dict[str, Any]]],
    trace_map: dict[tuple[int, int], dict[str, Any]],
) -> None:
    for variant in (1, 2):
        for qid in range(1, 31):
            trace = trace_map[(variant, qid)]
            if not bool(trace.get("requires_latex")):
                continue
            if not test_by_variant[variant][qid]["has_latex"]:
                raise ValidationError(f"latex contract: variant {variant} Q{qid} missing $...$ marker in test")
            if not key_by_variant[variant][qid]["has_latex"]:
                raise ValidationError(f"latex contract: variant {variant} Q{qid} missing $...$ marker in key")


def strip_technical_lines(text: str) -> str:
    cleaned: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "Технический блок:":
            continue
        if stripped.startswith("- Трассируемость:"):
            continue
        if stripped.startswith("- Источник материала:"):
            continue
        if stripped.startswith("- Уровень риска:"):
            continue
        if stripped.startswith("- Требуется LaTeX:"):
            continue
        if stripped.startswith("- Профиль стиля:"):
            continue
        if stripped.startswith("- Уровень сложности:"):
            continue
        if stripped.startswith("- Область знаний:"):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def strip_code_fragments(text: str) -> str:
    return re.sub(r"`[^`]*`", " ", text)


def validate_language_contract(out_dir: Path, dynamic_allowed_tokens: set[str]) -> None:
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

    allowed_tokens = set(ALLOWED_LATIN_TOKENS)
    allowed_tokens.update(x.lower() for x in dynamic_allowed_tokens if x.strip())

    def check_one(path: Path, threshold: float, strip_tech: bool, student_mode: bool) -> None:
        text = read_text(path)
        if strip_tech:
            text = strip_technical_lines(text)
        text = strip_code_fragments(text)
        low = text.lower()

        for pattern in FORBIDDEN_ANGLO_PATTERNS:
            if re.search(pattern, low):
                raise ValidationError(f"{path.name}: forbidden anglo phrase detected by pattern '{pattern}'")

        for pattern in META_DIALOG_PATTERNS:
            if re.search(pattern, low):
                raise ValidationError(f"{path.name}: meta-dialog phrase detected by pattern '{pattern}'")

        if student_mode:
            if re.search(r"\b(single|multiple|open)\b", low):
                raise ValidationError(f"{path.name}: internal technical markers single|multiple|open are forbidden")
            if re.search(r"(^|\n)\s*(?:- )?[A-D][\)\.]\s+", text):
                raise ValidationError(f"{path.name}: A/B/C/D option markers are forbidden")

        latin_tokens = re.findall(r"[A-Za-z][A-Za-z0-9_/.-]*", text)
        latin_filtered: list[str] = []
        for token in latin_tokens:
            normalized = token.lower().strip(".,;:!?()[]{}\"'")
            if not normalized:
                continue
            if normalized in allowed_tokens:
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
        check_one(path, threshold=0.07, strip_tech=False, student_mode=True)
    for path in key_files:
        check_one(path, threshold=0.11, strip_tech=True, student_mode=False)


def validate_internet_orientation(out_dir: Path) -> None:
    student_files = [
        out_dir / "test_variant_1.md",
        out_dir / "test_variant_2.md",
        out_dir / "answer_template_1.md",
        out_dir / "answer_template_2.md",
        out_dir / "README.md",
    ]
    for path in student_files:
        text = strip_code_fragments(read_text(path))
        low = text.lower()
        for pattern in PROJECT_MARKER_PATTERNS:
            if re.search(pattern, low):
                raise ValidationError(
                    f"{path.name}: project-specific marker detected by pattern '{pattern}'"
                )


def validate_difficulty_profile(trace_map: dict[tuple[int, int], dict[str, Any]]) -> None:
    for variant in (1, 2):
        subset = [trace_map[(variant, qid)] for qid in range(1, 31)]
        basic = sum(1 for item in subset if str(item.get("difficulty_level")) == "basic")
        share = basic / max(1, len(subset))
        if share < BASIC_MIN_SHARE:
            raise ValidationError(
                f"difficulty profile: variant {variant} basic share too low ({share:.3f} < {BASIC_MIN_SHARE:.3f})"
            )
        if basic < 24:
            raise ValidationError(f"difficulty profile: variant {variant} must contain at least 24 basic questions")

        invalid_scope = [item["question_id"] for item in subset if str(item.get("knowledge_scope")) != KNOWLEDGE_SCOPE_REQUIRED]
        if invalid_scope:
            raise ValidationError(
                f"difficulty profile: variant {variant} has non-{KNOWLEDGE_SCOPE_REQUIRED} scope in {invalid_scope}"
            )


def collect_dynamic_allowed_tokens(trace_map: dict[tuple[int, int], dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for item in trace_map.values():
        for token in item.get("allowed_abbrev", []):
            normalized = str(token).strip()
            if normalized:
                tokens.add(normalized)
    return tokens


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
        match = pattern.search(text)
        if not match:
            raise ValidationError(f"{path.name}: missing Q{qid} block")
        start = match.end()
        next_match = re.search(r"^### Q\d+ - Ответ$", text[start:], flags=re.MULTILINE)
        end = start + next_match.start() if next_match else len(text)
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
                raise ValidationError(f"{path.name}: Q{qid} open block must include at least 7 blank lines")


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
        match = pattern.search(text)
        if not match:
            raise ValidationError(f"{path.name}: missing Q{qid}")
        start = match.end()
        next_match = re.search(r"^### Q\d+ - .+$", text[start:], flags=re.MULTILINE)
        end = start + next_match.start() if next_match else len(text)
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
        _ = validate_traceability_json(out_dir / "question_traceability.json", legacy_mode=True)
        mode = "legacy"
    else:
        test_v1 = validate_test_file_v2(out_dir / "test_variant_1.md", 1)
        test_v2 = validate_test_file_v2(out_dir / "test_variant_2.md", 2)
        _ = validate_template_v2(out_dir / "answer_template_1.md", 1, test_v1)
        _ = validate_template_v2(out_dir / "answer_template_2.md", 2, test_v2)
        key_v1 = validate_key_v2(out_dir / "answer_key_1.md", 1, test_v1)
        key_v2 = validate_key_v2(out_dir / "answer_key_2.md", 2, test_v2)

        test_by_variant = {1: test_v1, 2: test_v2}
        key_by_variant = {1: key_v1, 2: key_v2}
        trace_map = validate_traceability_json(out_dir / "question_traceability.json", legacy_mode=False)

        validate_sync_with_traceability(test_by_variant, key_by_variant, trace_map)
        validate_anti_leakage(test_by_variant, trace_map)
        validate_modality(test_by_variant, trace_map)
        validate_latex_requirements(test_by_variant, key_by_variant, trace_map)
        validate_difficulty_profile(trace_map)
        validate_internet_orientation(out_dir)
        dynamic_tokens = collect_dynamic_allowed_tokens(trace_map)
        validate_language_contract(out_dir, dynamic_tokens)
        mode = "transfer_academic_ru"

    log(f"all validation checks passed ({mode})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"[08_validate][ERROR] {exc}", flush=True)
        raise SystemExit(1)
