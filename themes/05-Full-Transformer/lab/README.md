# Лабораторный практикум по теме `05-Full-Transformer`

> Статус: **срочный анонс-пакет** для вводного занятия. Полный учебный пакет `ЛР05` будет опубликован отдельным обновлением.

## Назначение

`ЛР05` — финальная лабораторная по построению и обучению полноценного трансформера «кодировщик-декодировщик» (encoder-decoder Transformer) для предсказания токенов текста на корпусе `WikiText-2 (raw-v1)`.

## Что уже есть сейчас

- стартовая тетрадь-скелет с обязательными `TODO`;
- каркас тетради-решения;
- дорожная карта для студентов и преподавателя;
- зафиксированный методический контракт полного пакета.

## Структура

- `01_full_transformer_wikitext2.ipynb` — стартовая тетрадь-скелет `ЛР05`.
- `solutions/01_full_transformer_wikitext2_solution.ipynb` — тетрадь-решение (скелет) `ЛР05`.
- `guides/00_lab5_roadmap.md` — что объяснять студентам сейчас и что будет в полном пакете.
- `../theory/theory.md` — вводная теория темы.
- `requirements.txt` — базовые зависимости.

## Канонический вводный маршрут студента (сейчас)

1. Открыть [../theory/theory.md](../theory/theory.md).
2. Прочитать [guides/00_lab5_roadmap.md](./guides/00_lab5_roadmap.md).
3. Открыть [01_full_transformer_wikitext2.ipynb](./01_full_transformer_wikitext2.ipynb).
4. Просмотреть `TODO 1..6`, понять будущий поток работы (workflow) и критерии.

## Контракт будущей полной ЛР05

Обязательный pipeline:

`контракт данных -> интуиция -> формализация -> ручной пример -> TODO -> мини-проверки (mini-checks) -> диагностика`.

Обязательные проверки:
- фиксированное зерно случайности (seed);
- фиксированные индексные разбиения;
- отсутствие доступа к будущему в декодере;
- сравнение с базовым ориентиром (baseline) по перплексии (perplexity);
- детерминированные контрольные продолжения (continuation probes).

## Запуск

Команды из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r themes/05-Full-Transformer/lab/requirements.txt
python3 -m ipykernel install --user --name full-transformer-lab --display-name "Python (.venv) Full Transformer Lab"
.venv/bin/jupyter notebook
```

## Связь с предыдущей темой

Предыдущий шаг курса:
- [../../04-Autoregression/lab/README.md](../../04-Autoregression/lab/README.md)
