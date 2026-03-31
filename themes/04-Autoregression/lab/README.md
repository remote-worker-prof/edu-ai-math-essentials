# Лабораторный практикум по авторегрессионному моделированию

## Назначение
Практикум включает обязательный двухчастный маршрут:
1. `ЛР01`: декодерный трансформер (decoder-only Transformer) на детерминированной синтетике.
2. `ЛР02`: перенос на реальный корпус `Tiny Shakespeare`.

Цель блока — последовательно закрепить механику причинной маски (causal mask), а затем проверить её прикладную переносимость на реальных данных.

## Структура
- `01_decoder_only_causal_toy.ipynb` — стартовая тетрадь `ЛР01`.
- `02_decoder_only_tiny_shakespeare.ipynb` — стартовая тетрадь `ЛР02`.
- `solutions/01_decoder_only_causal_toy_solution.ipynb` — решение `ЛР01`.
- `solutions/02_decoder_only_tiny_shakespeare_solution.ipynb` — решение `ЛР02`.
- `guides/00_autoregression_prerequisites.md` — входной минимум.
- `guides/01_decoder_only_toy_walkthrough.md` — разбор `ЛР01`.
- `guides/02_autoregression_debugging_playbook.md` — диагностика `ЛР01`.
- `guides/03_tiny_shakespeare_walkthrough.md` — разбор `ЛР02`.
- `guides/04_tiny_shakespeare_debugging_playbook.md` — диагностика `ЛР02`.

## Карта курса
Общая линия курса:
1. Шаг 1 = `01-RNN / ЛР01`
2. Шаг 2 = `01-RNN / ЛР02`
3. Шаг 3 = `01-RNN / ЛР03`
4. Шаг 4 = `02-Attention / ЛР01`
5. Шаг 5 = `03-Transformer / ЛР01`
6. Шаг 6 = `03-Transformer / ЛР02`
7. Шаг 7 = `04-Autoregression / ЛР01`
8. Шаг 8 = `04-Autoregression / ЛР02`

## Канонический маршрут студента
1. Прочитать [guides/00_autoregression_prerequisites.md](./guides/00_autoregression_prerequisites.md).
2. Пройти [guides/01_decoder_only_toy_walkthrough.md](./guides/01_decoder_only_toy_walkthrough.md).
3. Выполнить [01_decoder_only_causal_toy.ipynb](./01_decoder_only_causal_toy.ipynb).
4. При затруднениях использовать [guides/02_autoregression_debugging_playbook.md](./guides/02_autoregression_debugging_playbook.md).
5. После второй попытки свериться с [solutions/01_decoder_only_causal_toy_solution.ipynb](./solutions/01_decoder_only_causal_toy_solution.ipynb).
6. Пройти [guides/03_tiny_shakespeare_walkthrough.md](./guides/03_tiny_shakespeare_walkthrough.md).
7. Выполнить [02_decoder_only_tiny_shakespeare.ipynb](./02_decoder_only_tiny_shakespeare.ipynb).
8. При затруднениях использовать [guides/04_tiny_shakespeare_debugging_playbook.md](./guides/04_tiny_shakespeare_debugging_playbook.md).
9. После второй попытки свериться с [solutions/02_decoder_only_tiny_shakespeare_solution.ipynb](./solutions/02_decoder_only_tiny_shakespeare_solution.ipynb).

## Профили выполнения `ЛР02`
Оба профиля детерминированы и используют фиксированное зерно случайности.

### `CPU-friendly` (обязательный)
- уменьшенный объём текста;
- укороченная длина контекста;
- умеренный размер модели;
- быстрый проверочный прогон на центральном процессоре.

### `GPU-friendly` (расширенный)
- больший объём текста;
- более длинный контекст;
- более ёмкая модель;
- расширенный прогон для лучшего качества генерации.

## Критерии завершения
### `ЛР01`
- тестовая токенная точность не ниже `0.97`;
- тестовая перплексия не выше `1.30`;
- корректное продолжение выполняется минимум в `18 из 20` фиксированных запусков;
- диагностика внимания подтверждает отсутствие доступа к будущим позициям.

### `ЛР02` (профиль `CPU-friendly`)
- тестовая перплексия лучше базового частотного ориентира;
- в детерминированной генерации осмысленный шаблон соблюдается минимум в `14 из 20` фиксированных запусков;
- диагностика внимания подтверждает отсутствие доступа в будущее.

### `ЛР02` (профиль `GPU-friendly`)
- тестовая перплексия лучше результата `CPU-friendly` профиля;
- в детерминированной генерации осмысленный шаблон соблюдается минимум в `16 из 20` фиксированных запусков;
- диагностика внимания подтверждает отсутствие доступа в будущее.

## Запуск
Команды выполняются из корня репозитория.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r themes/04-Autoregression/lab/requirements.txt
python3 -m ipykernel install --user --name autoregression-lab --display-name "Python (.venv) Autoregression Lab"
.venv/bin/jupyter notebook
```

## Дополнительные материалы
- [../theory/theory.md](../theory/theory.md) — каноническая теория темы.
- [../../00-Foundations/showcases/cards/06_shakespeare.md](../../00-Foundations/showcases/cards/06_shakespeare.md) — карточка корпуса `Shakespeare`.
- [../../03-Transformer/theory/theory.md](../../03-Transformer/theory/theory.md) — предыдущий теоретический шаг.
