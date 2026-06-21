#!/usr/bin/env python3
"""Generate two assessment variants, answer templates, and teacher keys."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SEED = 20260621

QUESTION_BANK: dict[str, dict[str, list[dict[str, Any]]]] = {
    "foundations": {
        "single": [
            {
                "title": "Назначение padding в последовательностях",
                "prompt": "Какова основная цель добавления padding-токенов при пакетной обработке последовательностей?",
                "options": [
                    "Сделать длины последовательностей одинаковыми в батче",
                    "Увеличить словарь токенов",
                    "Удалить редкие токены",
                    "Снизить число признаков в эмбеддинге",
                ],
                "correct": [0],
            },
            {
                "title": "Форма тензора входа",
                "prompt": "Какая форма тензора типична для батча токенизированных последовательностей без учёта признаков?",
                "options": [
                    "(batch_size, seq_len)",
                    "(seq_len, vocab_size)",
                    "(batch_size, vocab_size)",
                    "(num_layers, batch_size)",
                ],
                "correct": [0],
            },
            {
                "title": "Роль masking",
                "prompt": "Что обеспечивает masking в задачах с padding?",
                "options": [
                    "Игнорирование служебных позиций при вычислениях",
                    "Принудительное увеличение длины последовательности",
                    "Случайную перестановку токенов",
                    "Удаление всех повторяющихся слов",
                ],
                "correct": [0],
            },
        ],
        "multiple": [
            {
                "title": "Базовые метрики качества",
                "prompt": "Какие метрики обычно относят к базовым для классификации последовательностей?",
                "options": [
                    "Accuracy",
                    "Precision/Recall",
                    "BLEU",
                    "F1-score",
                ],
                "correct": [0, 1, 3],
            },
            {
                "title": "Токенизация и устойчивость данных",
                "prompt": "Какие эффекты обычно дают унифицированные правила токенизации?",
                "options": [
                    "Повышают воспроизводимость подготовки данных",
                    "Снижают неоднозначность разбора текста",
                    "Гарантируют нулевую ошибку модели",
                    "Делают сравнение экспериментов более корректным",
                ],
                "correct": [0, 1, 3],
            },
            {
                "title": "Подготовка входных последовательностей",
                "prompt": "Что обычно учитывают при подготовке последовательностей фиксированной длины?",
                "options": [
                    "Стратегию усечения длинных примеров",
                    "Положение padding-токенов",
                    "Случайную замену всех токенов на одинаковые",
                    "Согласованность с mask-механизмом",
                ],
                "correct": [0, 1, 3],
            },
        ],
        "open": [
            {
                "title": "Нормализация входа",
                "prompt": "Объясните, почему единый протокол токенизации и padding важен для сравнимости результатов между моделями.",
                "reference": "Единый протокол исключает различия, вызванные предобработкой, и делает сравнение моделей по метрикам корректным.",
                "criteria": [
                    "Упомянуты токенизация и padding как источник систематических различий",
                    "Пояснено влияние на воспроизводимость и сопоставимость метрик",
                    "Ответ структурирован и логически завершён",
                ],
            },
            {
                "title": "Компромисс padding/truncation",
                "prompt": "Опишите компромисс между padding и truncation при подготовке корпусных данных для sequence-моделей.",
                "reference": "Padding повышает совместимость батчей, но добавляет пустые позиции; truncation снижает шум и стоимость, но может терять информацию.",
                "criteria": [
                    "Описаны плюсы и минусы padding",
                    "Описаны плюсы и риски truncation",
                    "Сделан вывод о выборе стратегии под задачу",
                ],
            },
        ],
    },
    "rnn": {
        "single": [
            {
                "title": "Преимущество LSTM",
                "prompt": "Какое свойство LSTM главным образом помогает удерживать долгосрочный контекст?",
                "options": [
                    "Наличие механизма ворот для управления памятью",
                    "Отсутствие рекуррентных связей",
                    "Прямая замена токенов на one-hot",
                    "Полная независимость от длины последовательности",
                ],
                "correct": [0],
            },
            {
                "title": "Смысл hidden state",
                "prompt": "Что отражает hidden state в рекуррентной сети?",
                "options": [
                    "Сжатое представление предыдущего контекста",
                    "Список всех токенов корпуса",
                    "Матрицу эмбеддингов словаря",
                    "Только текущий токен без истории",
                ],
                "correct": [0],
            },
            {
                "title": "Особенность GRU",
                "prompt": "Как обычно характеризуют GRU по сравнению с LSTM?",
                "options": [
                    "Более компактная архитектура с меньшим числом ворот",
                    "Полное отсутствие состояния",
                    "Невозможность работать с последовательностями",
                    "Обязательная двунаправленность",
                ],
                "correct": [0],
            },
        ],
        "multiple": [
            {
                "title": "Компоненты базового seq2seq",
                "prompt": "Какие элементы обычно входят в базовый seq2seq-подход?",
                "options": [
                    "Encoder",
                    "Decoder",
                    "Рекуррентная передача состояния",
                    "Обязательная свёрточная голова",
                ],
                "correct": [0, 1, 2],
            },
            {
                "title": "Трудности обучения RNN",
                "prompt": "Какие причины часто приводят к нестабильному обучению рекуррентных моделей?",
                "options": [
                    "Затухание/взрыв градиентов",
                    "Слишком длинные последовательности",
                    "Наличие любого optimizer без исключений",
                    "Неподходящий learning rate",
                ],
                "correct": [0, 1, 3],
            },
            {
                "title": "Диагностика RNN",
                "prompt": "Какие сигналы указывают на проблемы обобщения в RNN?",
                "options": [
                    "Рост разрыва между train и validation",
                    "Резкие колебания loss",
                    "Стабильное снижение обоих loss",
                    "Частая деградация на длинных примерах",
                ],
                "correct": [0, 1, 3],
            },
        ],
        "open": [
            {
                "title": "Выбор между GRU и LSTM",
                "prompt": "Сравните практические сценарии выбора GRU и LSTM в учебной задаче sequence learning.",
                "reference": "GRU часто выбирают как более лёгкую модель для ограниченных ресурсов, LSTM — когда критично более гибко контролировать память на длинном контексте.",
                "criteria": [
                    "Сопоставлены вычислительные и качественные аспекты",
                    "Есть привязка к типу данных/длине контекста",
                    "Дан аргументированный вывод",
                ],
            },
            {
                "title": "План диагностики RNN",
                "prompt": "Предложите короткий план диагностики, если RNN переобучается уже на ранних эпохах.",
                "reference": "Проверить разрыв train/val, скорректировать регуляризацию, раннюю остановку, размер модели и learning rate, затем повторно оценить стабильность метрик.",
                "criteria": [
                    "Есть шаги наблюдения и коррекции",
                    "Указаны минимум две управляемые гиперпараметрические меры",
                    "Показана связь действий и ожидаемого эффекта",
                ],
            },
        ],
    },
    "attention": {
        "single": [
            {
                "title": "Смысл attention-весов",
                "prompt": "Что показывают attention-веса в модели последовательностей?",
                "options": [
                    "Относительную важность элементов контекста для текущего шага",
                    "Только длину входной последовательности",
                    "Число эпох обучения",
                    "Порядок слоёв в оптимизаторе",
                ],
                "correct": [0],
            },
            {
                "title": "Alignment",
                "prompt": "Какую задачу решает alignment в механизме внимания?",
                "options": [
                    "Связывает позиции входа и выхода при генерации",
                    "Удаляет необходимость в данных валидации",
                    "Заменяет функцию потерь",
                    "Отменяет токенизацию",
                ],
                "correct": [0],
            },
            {
                "title": "Интерпретация attention-карт",
                "prompt": "Как корректно трактовать attention-карты в учебном анализе?",
                "options": [
                    "Как индикатор распределения фокуса модели",
                    "Как доказательство причинности без ограничений",
                    "Как замену всем метрикам качества",
                    "Как признак, что модель не требует тестирования",
                ],
                "correct": [0],
            },
        ],
        "multiple": [
            {
                "title": "Преимущества внимания",
                "prompt": "Какие эффекты обычно связывают с добавлением внимания в seq2seq?",
                "options": [
                    "Более явная работа с релевантным контекстом",
                    "Улучшение интерпретируемости поведения",
                    "Гарантия идеальной генерации",
                    "Снижение риска потери важной информации",
                ],
                "correct": [0, 1, 3],
            },
            {
                "title": "Ограничения attention-анализа",
                "prompt": "Какие ограничения важно учитывать при анализе attention-карт?",
                "options": [
                    "Вес не равен строгой причинности",
                    "Качество интерпретации зависит от контекста задачи",
                    "Карты всегда однозначны и не требуют проверки",
                    "Нужна связь с метриками и ошибками",
                ],
                "correct": [0, 1, 3],
            },
            {
                "title": "Сигналы полезного внимания",
                "prompt": "Какие наблюдения чаще всего говорят, что механизм внимания используется осмысленно?",
                "options": [
                    "Фокус смещается в зависимости от текущего шага декодера",
                    "Есть связь между attention и снижением ошибок",
                    "Веса всегда равномерны по всем позициям",
                    "Сложные примеры показывают осмысленные зоны фокуса",
                ],
                "correct": [0, 1, 3],
            },
        ],
        "open": [
            {
                "title": "Чтение attention-карт",
                "prompt": "Опишите короткий алгоритм интерпретации attention-карты на конкретном примере последовательности.",
                "reference": "Нужно сопоставить шаг выхода с входными позициями, выделить зоны максимального веса, сравнить с ожидаемыми связями и проверить согласованность с ошибками/метриками.",
                "criteria": [
                    "Есть поэтапная процедура анализа",
                    "Упомянуто сопоставление с ожидаемыми зависимостями",
                    "Добавлена проверка по качественным/количественным сигналам",
                ],
            },
            {
                "title": "Attention vs фиксированный контекст",
                "prompt": "Сравните подход с attention и подход с фиксированным контекст-вектором в seq2seq.",
                "reference": "Attention позволяет динамически перераспределять фокус по входу, фиксированный вектор ограничивает модель единым сжатием всей последовательности.",
                "criteria": [
                    "Сравнены механизмы представления контекста",
                    "Показано влияние на длинные последовательности",
                    "Сделан вывод о практическом выборе",
                ],
            },
        ],
    },
    "transformer_encoder": {
        "single": [
            {
                "title": "Роль positional encoding",
                "prompt": "Зачем в encoder добавляют positional encoding?",
                "options": [
                    "Чтобы модель учитывала порядок токенов",
                    "Чтобы исключить self-attention",
                    "Чтобы убрать нормализацию",
                    "Чтобы уменьшить размер словаря",
                ],
                "correct": [0],
            },
            {
                "title": "Смысл self-attention",
                "prompt": "Что позволяет self-attention в encoder-пайплайне?",
                "options": [
                    "Оценивать взаимосвязи между позициями входа",
                    "Только копировать вход без преобразования",
                    "Устранять необходимость в батчах",
                    "Гарантировать отсутствие ошибок на тесте",
                ],
                "correct": [0],
            },
            {
                "title": "Multi-head идея",
                "prompt": "Почему multi-head attention обычно полезен?",
                "options": [
                    "Разные головы могут учить разные типы зависимостей",
                    "Полностью отменяет потребность в данных",
                    "Заменяет функцию потерь",
                    "Убирает потребность в позиционной информации",
                ],
                "correct": [0],
            },
        ],
        "multiple": [
            {
                "title": "Состав encoder-блока",
                "prompt": "Какие элементы входят в типичный encoder-блок Transformer?",
                "options": [
                    "Multi-head self-attention",
                    "Feed-forward sublayer",
                    "Residual connections и нормализация",
                    "Только рекуррентный hidden state",
                ],
                "correct": [0, 1, 2],
            },
            {
                "title": "Masking в encoder",
                "prompt": "Когда в encoder особенно важен padding mask?",
                "options": [
                    "Когда в батче есть последовательности разной длины",
                    "Когда нужно исключить вклад padding-позиций",
                    "Когда модель обучается без токенизации",
                    "Когда важно не искажать attention-распределение",
                ],
                "correct": [0, 1, 3],
            },
            {
                "title": "Диагностика encoder",
                "prompt": "Какие сигналы полезны при диагностике encoder-модели?",
                "options": [
                    "Динамика train/validation метрик",
                    "Стабильность attention-распределений",
                    "Полный отказ от валидации",
                    "Кривые обучения по эпохам",
                ],
                "correct": [0, 1, 3],
            },
        ],
        "open": [
            {
                "title": "Пайплайн encoder",
                "prompt": "Кратко опишите encoder-пайплайн от входных токенов до итоговых представлений.",
                "reference": "Токены переводятся в эмбеддинги, добавляется позиционная информация, затем проходят через стек encoder-блоков self-attention + FFN + residual/normalization.",
                "criteria": [
                    "Перечислены ключевые этапы пайплайна",
                    "Корректно описана роль позиционной информации",
                    "Указана блочная структура encoder",
                ],
            },
            {
                "title": "Недостаток позиции",
                "prompt": "Что может произойти, если позиционная информация в encoder задана недостаточно выразительно?",
                "reference": "Модель хуже различает порядок токенов, что ведёт к ошибкам в задачах, где порядок критичен для смысла.",
                "criteria": [
                    "Объяснено влияние на учет порядка",
                    "Связь с потенциальной деградацией качества",
                    "Дан пример или тип задач, где риск высок",
                ],
            },
        ],
    },
    "autoregression": {
        "single": [
            {
                "title": "Назначение causal mask",
                "prompt": "Зачем в decoder-only моделях нужен causal mask?",
                "options": [
                    "Чтобы исключить просмотр будущих токенов",
                    "Чтобы убрать loss-функцию",
                    "Чтобы отключить обучение батчами",
                    "Чтобы модель не использовала контекст",
                ],
                "correct": [0],
            },
            {
                "title": "Цель next-token prediction",
                "prompt": "Какова базовая цель обучения autoregressive language model?",
                "options": [
                    "Предсказать следующий токен по предыдущему контексту",
                    "Предсказать длину словаря",
                    "Свернуть весь текст в один класс",
                    "Определить номер эпохи",
                ],
                "correct": [0],
            },
            {
                "title": "Генерация в decoder-only",
                "prompt": "Как обычно строится генерация в decoder-only модели на инференсе?",
                "options": [
                    "Последовательным добавлением новых токенов",
                    "Параллельной выдачей всех токенов без контекста",
                    "Случайной перестановкой обучающего корпуса",
                    "Только одним шагом независимо от длины",
                ],
                "correct": [0],
            },
        ],
        "multiple": [
            {
                "title": "Стабилизация autoregression",
                "prompt": "Какие практики часто помогают стабилизировать обучение autoregressive модели?",
                "options": [
                    "Контроль learning rate",
                    "Мониторинг validation loss",
                    "Полный отказ от регуляризации",
                    "Ограничение длины контекста под ресурсы",
                ],
                "correct": [0, 1, 3],
            },
            {
                "title": "Показатели качества генерации",
                "prompt": "Какие признаки обычно указывают на деградацию качества генерации?",
                "options": [
                    "Повторяемость фрагментов",
                    "Потеря связности контекста",
                    "Стабильное разнообразие и согласованность",
                    "Рост ошибок на валидационном наборе",
                ],
                "correct": [0, 1, 3],
            },
            {
                "title": "Декодирование и компромиссы",
                "prompt": "Какие факторы учитывают при выборе стратегии декодирования?",
                "options": [
                    "Баланс между разнообразием и устойчивостью",
                    "Риск повторов в длинной генерации",
                    "Наличие padding mask в encoder-only модели",
                    "Требования к управляемости вывода",
                ],
                "correct": [0, 1, 3],
            },
        ],
        "open": [
            {
                "title": "Диагностика повторов",
                "prompt": "Предложите план диагностики ситуации, когда модель в autoregressive режиме начинает повторяться.",
                "reference": "Проверить параметры декодирования, динамику loss, длину контекста, затем скорректировать стратегию сэмплирования и регуляризацию с повторной оценкой качества.",
                "criteria": [
                    "Указаны источники проблемы на уровне обучения и инференса",
                    "Есть конкретные шаги корректировки",
                    "План включает повторную проверку метрик/примеров",
                ],
            },
            {
                "title": "Train vs inference",
                "prompt": "Объясните ключевые различия между autoregressive обучением и autoregressive инференсом.",
                "reference": "На обучении модель видит корректный префикс и учится прогнозировать следующий токен, на инференсе она опирается на собственные предсказания, что может накапливать ошибки.",
                "criteria": [
                    "Отражены различия в доступном контексте",
                    "Пояснён риск накопления ошибок на инференсе",
                    "Ответ логично структурирован",
                ],
            },
        ],
    },
    "full_transformer": {
        "single": [
            {
                "title": "Связь encoder и decoder",
                "prompt": "Как decoder в full Transformer использует выход encoder?",
                "options": [
                    "Через механизм cross-attention к encoder-представлениям",
                    "Через полное игнорирование encoder",
                    "Через случайную замену входа",
                    "Через удаление позиционной информации",
                ],
                "correct": [0],
            },
            {
                "title": "Teacher forcing",
                "prompt": "Для чего на обучении encoder-decoder моделей применяют teacher forcing?",
                "options": [
                    "Для стабилизации обучения по целевой последовательности",
                    "Для отключения функции потерь",
                    "Для отказа от валидации",
                    "Для исключения decoder из модели",
                ],
                "correct": [0],
            },
            {
                "title": "Cross-attention сигнал",
                "prompt": "Что является ключевым источником информации для cross-attention в decoder?",
                "options": [
                    "Скрытые представления encoder",
                    "Только номер эпохи",
                    "Только длина словаря",
                    "Случайный шум без связи с входом",
                ],
                "correct": [0],
            },
        ],
        "multiple": [
            {
                "title": "Шаги инференса full Transformer",
                "prompt": "Какие шаги обычно входят в инференс encoder-decoder модели?",
                "options": [
                    "Кодирование входной последовательности",
                    "Пошаговое декодирование целевого выхода",
                    "Использование накопленного контекста decoder",
                    "Отказ от оценки качества результата",
                ],
                "correct": [0, 1, 2],
            },
            {
                "title": "Диагностика качества",
                "prompt": "Какие сигналы важны при диагностике encoder-decoder решения?",
                "options": [
                    "Разбор ошибок по типам примеров",
                    "Сравнение train/validation динамики",
                    "Игнорирование ошибок на редких классах",
                    "Анализ стабильности на длинных последовательностях",
                ],
                "correct": [0, 1, 3],
            },
            {
                "title": "Риски полного контура",
                "prompt": "Какие риски типичны для полного encoder-decoder контура?",
                "options": [
                    "Ошибки согласования между encoder и decoder",
                    "Деградация на длинных зависимостях",
                    "Абсолютная нечувствительность к качеству данных",
                    "Рост вычислительной стоимости обучения",
                ],
                "correct": [0, 1, 3],
            },
        ],
        "open": [
            {
                "title": "Оценивание полного контура",
                "prompt": "Предложите критерии оценивания качества full Transformer в задаче последовательного преобразования.",
                "reference": "Следует учитывать точность на валидации, устойчивость на сложных/длинных примерах, согласованность выхода с входным контекстом и повторяемость результатов.",
                "criteria": [
                    "Есть количественные и качественные критерии",
                    "Покрыты устойчивость и согласованность",
                    "Критерии применимы к учебному контексту",
                ],
            },
            {
                "title": "Train/inference разрыв",
                "prompt": "Объясните, как несоответствие между режимом обучения и инференса влияет на encoder-decoder модель.",
                "reference": "На обучении модель опирается на корректные целевые префиксы, на инференсе — на собственные предсказания, поэтому ошибки могут накапливаться и ухудшать дальние шаги.",
                "criteria": [
                    "Корректно описан источник разрыва",
                    "Показано влияние на качество последовательности",
                    "Дана логичная аргументация",
                ],
            },
        ],
    },
}


def log(message: str) -> None:
    print(f"[07_generate][INFO] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate module-based assessment variants.")
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--workspace-root", default="90-assessment-ai-math", help="Assessment workspace root.")
    parser.add_argument("--seed", type=int, default=SEED, help="Deterministic seed value.")
    return parser.parse_args()


def select_items(items: list[dict[str, Any]], count: int, variant: int) -> list[dict[str, Any]]:
    if not items:
        return []
    offset = (variant - 1) * count
    return [items[(offset + i) % len(items)] for i in range(count)]


def build_section_questions(topic_id: str, variant: int) -> list[dict[str, Any]]:
    bank = QUESTION_BANK.get(topic_id)
    if not bank:
        return []

    singles = select_items(bank["single"], 2, variant)
    multiples = select_items(bank["multiple"], 2, variant)
    opens = select_items(bank["open"], 1, variant)

    questions = []
    for item in singles:
        questions.append({**item, "qtype": "single"})
    for item in multiples:
        questions.append({**item, "qtype": "multiple"})
    for item in opens:
        questions.append({**item, "qtype": "open"})
    return questions


def option_letter(index: int) -> str:
    return chr(ord("A") + index)


def render_test_variant(variant: int, modules: list[dict[str, Any]], traceability: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        f"# Вариант {variant}. Тест по дисциплине «Математические основы ИИ»",
        "",
        "---",
        "",
    ]

    q_counter = 1
    for sec_idx, module in enumerate(modules):
        if sec_idx > 0:
            lines.extend(["", "---", ""])
        lines.append(f"## {module['module_title']}")
        lines.append("")

        questions = build_section_questions(module["topic_id"], variant)
        for local_idx, q in enumerate(questions):
            if local_idx > 0:
                lines.extend(["", "---", ""])
            lines.append(f"### Q{q_counter} - {q['title']}")
            lines.append("")
            lines.append("---")
            lines.append("")

            if q["qtype"] == "single":
                lines.append("Выберите один верный вариант.")
                lines.append("")
                lines.append(q["prompt"])
                lines.append("")
                for idx, option in enumerate(q["options"]):
                    lines.append(f"- {option_letter(idx)}) {option}")
            elif q["qtype"] == "multiple":
                lines.append("Выберите все верные варианты.")
                lines.append("")
                lines.append(q["prompt"])
                lines.append("")
                for idx, option in enumerate(q["options"]):
                    lines.append(f"- {option_letter(idx)}) {option}")
            else:
                lines.append("Дайте развёрнутый ответ (3-5 предложений).")
                lines.append("")
                lines.append(q["prompt"])

            material_pool = module.get("materials", [])
            if material_pool:
                material = material_pool[(local_idx + variant - 1) % len(material_pool)]
                material_id = material.get("material_id", "n/a")
                source_path = material.get("source_path", "n/a")
            else:
                material_id = "n/a"
                source_path = "n/a"

            traceability.append(
                {
                    "variant": variant,
                    "question_id": f"Q{q_counter}",
                    "question_type": q["qtype"],
                    "topic_id": module["topic_id"],
                    "module_slug": module["module_slug"],
                    "material_id": material_id,
                    "source_path": source_path,
                    "title": q["title"],
                    "correct": q.get("correct", []),
                    "reference": q.get("reference", ""),
                    "criteria": q.get("criteria", []),
                    "options": q.get("options", []),
                }
            )

            q_counter += 1

    return "\n".join(lines).rstrip() + "\n"


def render_answer_template(variant: int, modules: list[dict[str, Any]], traceability: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        f"# Вариант {variant}. Ответы по дисциплине «Математические основы ИИ»",
        "",
        "ФИО:",
        "",
        "Номер группы:",
        "",
        "Email:",
        "",
        "---",
        "",
    ]

    trace = [x for x in traceability if x["variant"] == variant]
    q_by_id = {x["question_id"]: x for x in trace}

    q_counter = 1
    for sec_idx, module in enumerate(modules):
        if sec_idx > 0:
            lines.extend(["", "---", ""])
        lines.append(f"## {module['module_title']}")
        lines.append("")

        for local_idx in range(5):
            if local_idx > 0:
                lines.extend(["", "---", ""])
            qid = f"Q{q_counter}"
            q = q_by_id[qid]
            lines.append(f"### {qid} - Ответ")
            lines.append("")
            if q["question_type"] == "open":
                lines.append("Ответ:")
                lines.extend(["", "", "", "", "", "", "", ""])
            else:
                lines.append("Ответ: ______________________________")
                lines.append("")
            q_counter += 1

    # Keep trailing blank lines so open questions at file end preserve 7+ answer lines.
    return "\n".join(lines) + "\n"


def render_answer_key(variant: int, modules: list[dict[str, Any]], traceability: list[dict[str, Any]]) -> str:
    lines: list[str] = [
        f"# Вариант {variant}. Ключ преподавателя по дисциплине «Математические основы ИИ»",
        "",
        "---",
        "",
    ]

    trace = [x for x in traceability if x["variant"] == variant]
    q_by_id = {x["question_id"]: x for x in trace}

    q_counter = 1
    for sec_idx, module in enumerate(modules):
        if sec_idx > 0:
            lines.extend(["", "---", ""])
        lines.append(f"## {module['module_title']}")
        lines.append("")

        for local_idx in range(5):
            if local_idx > 0:
                lines.extend(["", "---", ""])
            qid = f"Q{q_counter}"
            q = q_by_id[qid]
            lines.append(f"### {qid} - {q['title']}")
            lines.append("")

            if q["question_type"] == "single":
                idx = int(q["correct"][0])
                letter = option_letter(idx)
                option = q["options"][idx]
                lines.append(f"Правильный ответ: {letter}) {option}")
            elif q["question_type"] == "multiple":
                pairs = []
                for idx in q["correct"]:
                    i = int(idx)
                    pairs.append(f"{option_letter(i)}) {q['options'][i]}")
                lines.append("Правильные ответы: " + "; ".join(pairs))
            else:
                lines.append("Эталонный ответ: " + str(q["reference"]))
                lines.append("")
                lines.append("Критерии оценивания:")
                for criterion in q["criteria"]:
                    lines.append(f"- {criterion}")

            lines.append("")
            lines.append(
                "Трассируемость: "
                f"{q['topic_id']} -> {q['material_id']} -> {q['module_slug']} -> {qid}"
            )
            lines.append(f"Источник материала: {q['source_path']}")
            q_counter += 1

    return "\n".join(lines).rstrip() + "\n"


def render_readme(traceability: list[dict[str, Any]]) -> str:
    lines = [
        "# Assessment Test Package",
        "",
        "Содержимое каталога:",
        "- `test_variant_1.md`",
        "- `test_variant_2.md`",
        "- `answer_template_1.md`",
        "- `answer_template_2.md`",
        "- `answer_key_1.md`",
        "- `answer_key_2.md`",
        "",
        "Контракты:",
        "- 2 варианта теста",
        "- 6 разделов в варианте",
        "- 30 вопросов в варианте",
        "- микс 2 single + 2 multiple + 1 open на раздел",
        "",
    ]

    for variant in (1, 2):
        subset = [x for x in traceability if x["variant"] == variant]
        singles = sum(1 for x in subset if x["question_type"] == "single")
        multiples = sum(1 for x in subset if x["question_type"] == "multiple")
        opens = sum(1 for x in subset if x["question_type"] == "open")
        lines.append(
            f"- Вариант {variant}: total={len(subset)}, single={singles}, multiple={multiples}, open={opens}"
        )

    lines.extend([
        "",
        "Для полной трассируемости используйте файл `question_traceability.json`.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    _ = args.seed
    project_root = Path(args.project_root).resolve()
    workspace_root = (project_root / args.workspace_root).resolve()

    module_index_path = workspace_root / "03-topic-context" / "module_index.json"
    module_index = json.loads(module_index_path.read_text(encoding="utf-8"))
    modules = module_index.get("modules", [])

    out_dir = workspace_root / "05-tests-v1v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    traceability: list[dict[str, Any]] = []

    test_v1 = render_test_variant(1, modules, traceability)
    test_v2 = render_test_variant(2, modules, traceability)
    template_v1 = render_answer_template(1, modules, traceability)
    template_v2 = render_answer_template(2, modules, traceability)
    key_v1 = render_answer_key(1, modules, traceability)
    key_v2 = render_answer_key(2, modules, traceability)

    (out_dir / "test_variant_1.md").write_text(test_v1, encoding="utf-8")
    (out_dir / "test_variant_2.md").write_text(test_v2, encoding="utf-8")
    (out_dir / "answer_template_1.md").write_text(template_v1, encoding="utf-8")
    (out_dir / "answer_template_2.md").write_text(template_v2, encoding="utf-8")
    (out_dir / "answer_key_1.md").write_text(key_v1, encoding="utf-8")
    (out_dir / "answer_key_2.md").write_text(key_v2, encoding="utf-8")
    (out_dir / "README.md").write_text(render_readme(traceability), encoding="utf-8")
    (out_dir / "question_traceability.json").write_text(
        json.dumps({"traceability": traceability}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    log(f"generated questions: {len(traceability)}")
    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
