# Лабораторный практикум по Attention

## Назначение
Практикум содержит одно задание на синтетических данных:
1. `GRU` encoder-decoder с `Luong attention` для `seq2seq`.

Цель практикума — показать, как `attention` снимает fixed-context bottleneck обычного `seq2seq`, научить читать тензоры `query/key/value`-логики и интерпретировать `attention_scores` через тепловую карту.

## Структура
- `01_gru_seq2seq_attention_reverse_toy.ipynb` — задание с `TODO`.
- `solutions/01_gru_seq2seq_attention_reverse_toy_solution.ipynb` — полное решение.

Основной ноутбук и решение синхронизированы 1:1: одинаковая структура разделов и одинаковые контрольные точки.

## Запуск
Команды выполняются из корня репозитория.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r themes/02-Attention/lab/requirements.txt
python -m ipykernel install --user --name attention-lab --display-name "Python (.venv) Attention Lab"
jupyter notebook
```

## Порядок прохождения
1. `01_gru_seq2seq_attention_reverse_toy.ipynb`

## Как читать формы тензоров
Базовый шаблон:
- `N` — число объектов в наборе;
- `T_in` — длина входной последовательности encoder;
- `T_out` — длина выходной последовательности decoder;
- `H` — размер скрытого пространства;
- `V` — размер словаря.

Ключевой принцип:
- `encoder_outputs` хранит представления всех входных позиций, а не только одно финальное состояние;
- `attention_scores` имеет форму `(batch, T_out, T_in)` и показывает, куда decoder смотрит на каждом шаге.

Проверка на каждом шаге:
1. До обучения: сравнить фактические формы `encoder_input`, `decoder_input`, `decoder_target`.
2. После сборки модели: проверить форму выхода `model.output_shape`.
3. После `predict`: проверить формы `context`, `attention_scores`, `probs`.

## Методический маршрут
Рекомендуемый цикл работы в ноутбуке:
1. `Контракт данных` и `Таблица форм тензоров`.
2. `Контракт модели` и `Мини-теория`.
3. `Генерация данных` -> `Мини-проверка данных`.
4. `Модель` -> `Мини-проверка модели` -> `Трассировка одного примера`.
5. `Как идет обучение внутри эпохи` -> `Обучение` -> `Мини-проверка обучения`.
6. `Оценка и диагностика`.
7. `Карта внимания одного примера` -> `Что ожидать на практике`.

## Схема валидации
В работе используется тот же подход, что и в блоке `RNN`:
- сначала формируется отдельный `test` через `train_test_split`;
- затем обучение выполняется на `train` с `validation_split` внутри `model.fit`.

Это разделяет:
- настройку обучения (`train`/`validation`),
- финальную независимую оценку (`test`).

## API-конвенция
Лабораторная полностью использует `Functional API`, потому что attention требует:
- два входа модели;
- промежуточные тензоры `encoder_outputs`, `decoder_outputs`, `context`;
- отдельный доступ к `attention_scores` для анализа.

## Связь с теорией
- Раздел 1 в `theory.md` -> зачем plain `seq2seq` нужен attention.
- Раздел 2 -> интуиция `query`, `key`, `value`.
- Раздел 3 -> формулы Luong attention и формы тензоров.
- Раздел 4 -> отличие `cross-attention` от `self-attention`.
- Раздел 5 -> почему следующий шаг после этой ЛР — облегченный `Transformer`.

## Дополнительные материалы
- TensorFlow tutorial: NMT with attention — <https://www.tensorflow.org/tutorials/text/nmt_with_attention>
- Keras layer API: `Attention` — <https://keras.io/api/layers/attention_layers/attention/>
- TensorFlow tutorial: Transformer — <https://www.tensorflow.org/text/tutorials/transformer>

## Типичные проблемы
### Attention-модель
- Симптом: `attention_scores` имеет неожиданную форму.
  Причина: перепутаны местами `query` и `value`.
  Исправление: вызывать `Attention([decoder_outputs, encoder_outputs])`.
- Симптом: attention не может смотреть по всем входным позициям.
  Причина: encoder возвращает только финальное состояние.
  Исправление: включить `return_sequences=True` у `enc_gru`.

### Метрики и интерпретация
- Симптом: `token_accuracy` растет, а `exact_match` заметно отстает.
  Причина: ошибки остаются хотя бы в одном шаге многих последовательностей.
  Исправление: проверить сборку модели, число эпох и корректность teacher forcing.
- Симптом: heatmap выглядит шумной и плохо читается.
  Причина: PAD-позиции не скрыты при визуализации.
  Исправление: показывать только значимые строки и столбцы.

### Сравнение с plain seq2seq
- Симптом: разница с baseline почти не видна.
  Причина: слишком короткие последовательности или слишком легкая задача.
  Исправление: использовать `ENC_LEN=10` и переменную длину `4..10`, как в ноутбуке.
