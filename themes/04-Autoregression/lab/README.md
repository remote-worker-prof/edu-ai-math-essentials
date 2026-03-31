# Лабораторный практикум по авторегрессионному моделированию

## Назначение
Практикум содержит одну последовательную лабораторную работу:
1. `декодерный трансформер` (decoder-only Transformer) на детерминированном синтетическом корпусе для предсказания следующего токена (next-token prediction).

Цель блока — показать переход от классификации последовательности к пошаговой генерации при помощи причинной маски (causal mask).

## Структура
- `01_decoder_only_causal_toy.ipynb` — стартовая учебная тетрадь (notebook) с `TODO`.
- `solutions/01_decoder_only_causal_toy_solution.ipynb` — полное решение.
- `guides/00_autoregression_prerequisites.md` — входной минимум перед запуском.
- `guides/01_decoder_only_toy_walkthrough.md` — пошаговый маршрут выполнения.
- `guides/02_autoregression_debugging_playbook.md` — диагностический порядок при затруднениях.

## Карта курса
Общая линия курса:
1. Шаг 1 = `01-RNN / ЛР01`
2. Шаг 2 = `01-RNN / ЛР02`
3. Шаг 3 = `01-RNN / ЛР03`
4. Шаг 4 = `02-Attention / ЛР01`
5. Шаг 5 = `03-Transformer / ЛР01`
6. Шаг 6 = `03-Transformer / ЛР02`
7. Шаг 7 = `04-Autoregression / ЛР01`

## Детерминированный маршрут студента
1. Прочитать [guides/00_autoregression_prerequisites.md](./guides/00_autoregression_prerequisites.md).
2. Прочитать [guides/01_decoder_only_toy_walkthrough.md](./guides/01_decoder_only_toy_walkthrough.md).
3. Выполнить `TODO` в [01_decoder_only_causal_toy.ipynb](./01_decoder_only_causal_toy.ipynb) строго по порядку.
4. При затруднениях открыть [guides/02_autoregression_debugging_playbook.md](./guides/02_autoregression_debugging_playbook.md).
5. После второй попытки сравнить решение с [solutions/01_decoder_only_causal_toy_solution.ipynb](./solutions/01_decoder_only_causal_toy_solution.ipynb).

## Контрольные критерии завершения
Работа считается завершённой, если одновременно выполнено:
- тестовая токенная точность не ниже `0.97`;
- тестовая перплексия не выше `1.30`;
- в детерминированной генерации корректный шаблон выполняется минимум в `18 из 20` запусков;
- в диагностике внимания отсутствуют обращения к будущим позициям.

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

### Варианты среды выполнения
Учебная тетрадь поддерживает те же режимы, что и предыдущие блоки:
- локальный автоматический режим: `auto`;
- локальный режим центрального процессора: `local-cpu`;
- локальный режим графического процессора: `local-gpu`;
- облачные режимы `Google Colab` и `Kaggle`.

Подробные инструкции:
- [../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md](../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md)
- [../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md](../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md)

## Дополнительные материалы
- [../theory/theory.md](../theory/theory.md) — каноническая теория темы.
- [../../03-Transformer/theory/theory.md](../../03-Transformer/theory/theory.md) — предыдущий теоретический шаг.

## Частые вопросы
### Почему здесь только одна лабораторная?
Потому что блок вводный и должен оставаться простым: сначала фиксируется причинная маска и генерация следующего токена, затем уже расширяется масштаб данных.

### Почему выбран синтетический корпус?
Чтобы обеспечить воспроизводимость, прозрачную диагностику и учебно-методическую последовательность без лишнего вычислительного шума.

### Что дальше после этого блока?
Следующий естественный шаг — увеличение сложности генерации и переход к более крупным языковым моделям на той же авторегрессионной идее.
