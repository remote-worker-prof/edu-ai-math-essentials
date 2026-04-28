# Лабораторный практикум по RNN

## Назначение
Практикум содержит три последовательных задания на синтетических данных:
1. `SimpleRNN` в формате `many-to-one`.
2. `LSTM` в формате `many-to-many`.
3. `GRU` encoder-decoder в формате `seq2seq`.

Цель практикума — научиться уверенно читать формы тензоров, правильно связывать входы/выходы модели и диагностировать обучение по кривым качества.

## Структура
- `01_simple_rnn_many_to_one_toy.ipynb` — задание с `TODO`.
- `02_lstm_many_to_many_toy.ipynb` — задание с `TODO`.
- `03_gru_seq2seq_reverse_toy.ipynb` — задание с `TODO`.
- `solutions/01_simple_rnn_many_to_one_toy_solution.ipynb` — полное решение.
- `solutions/02_lstm_many_to_many_toy_solution.ipynb` — полное решение.
- `solutions/03_gru_seq2seq_reverse_toy_solution.ipynb` — полное решение.

Основные ноутбуки и решения синхронизированы 1:1: одинаковая структура разделов и одинаковые контрольные точки.

## Карта курса
Общая учебная линия сейчас устроена так:
1. Шаг 1 = `01-RNN / ЛР01` — `SimpleRNN many-to-one`
2. Шаг 2 = `01-RNN / ЛР02` — `LSTM many-to-many`
3. Шаг 3 = `01-RNN / ЛР03` — `GRU seq2seq`
4. Шаг 4 = `02-Attention / ЛР01` — `GRU seq2seq + attention`

Важно: `01-RNN / ЛР03` остаётся внутри блока `RNN`. Мы не создаём отдельную тему только ради `seq2seq`, и `02-Attention / ЛР01` не является “новой ЛР03”.

## Beginner Route
Если студенты приходят почти с нуля, рекомендуемый маршрут такой:
1. Пройти общий foundations-вход: [../../00-Foundations/README.md](../../00-Foundations/README.md).
2. Прочитать shared-guide по формам и метрикам: [../../00-Foundations/guides/01_sequence_shapes_and_metrics.md](../../00-Foundations/guides/01_sequence_shapes_and_metrics.md).
3. Прочитать [guides/00_prerequisites_and_notation.md](./guides/00_prerequisites_and_notation.md).
4. Пройти [guides/01_simple_rnn_many_to_one_beginner.md](./guides/01_simple_rnn_many_to_one_beginner.md), затем Шаг 1 = `01-RNN / ЛР01`.
5. Пройти [guides/02_lstm_many_to_many_beginner.md](./guides/02_lstm_many_to_many_beginner.md), затем Шаг 2 = `01-RNN / ЛР02`.
6. До `seq2seq` повторить shared-guide по токенам и decoder shift: [../../00-Foundations/guides/02_tokens_padding_and_decoder_shift.md](../../00-Foundations/guides/02_tokens_padding_and_decoder_shift.md).
7. Прочитать [guides/03_gru_seq2seq_tokens_beginner.md](./guides/03_gru_seq2seq_tokens_beginner.md), затем Шаг 3 = `01-RNN / ЛР03`.
8. Только после этого переходить к Шагу 4 = `02-Attention / ЛР01`.

## Self-study Route
Если блок `RNN` нужно пройти полностью без преподавателя, используйте такой маршрут.

### `01-RNN / ЛР01` `SimpleRNN many-to-one` (Шаг 1)
1. [../../00-Foundations/README.md](../../00-Foundations/README.md)
2. [../../00-Foundations/guides/01_sequence_shapes_and_metrics.md](../../00-Foundations/guides/01_sequence_shapes_and_metrics.md)
3. [guides/00_prerequisites_and_notation.md](./guides/00_prerequisites_and_notation.md)
4. [guides/01_simple_rnn_many_to_one_beginner.md](./guides/01_simple_rnn_many_to_one_beginner.md)
5. [guides/01_simple_rnn_walkthrough.md](./guides/01_simple_rnn_walkthrough.md)
6. [01_simple_rnn_many_to_one_toy.ipynb](./01_simple_rnn_many_to_one_toy.ipynb)
7. [solutions/01_simple_rnn_many_to_one_toy_solution.ipynb](./solutions/01_simple_rnn_many_to_one_toy_solution.ipynb)
8. [../../00-Foundations/guides/04_self_study_debugging_playbook.md](../../00-Foundations/guides/04_self_study_debugging_playbook.md)

### `01-RNN / ЛР02` `LSTM many-to-many` (Шаг 2)
1. [../../00-Foundations/README.md](../../00-Foundations/README.md)
2. [../../00-Foundations/guides/01_sequence_shapes_and_metrics.md](../../00-Foundations/guides/01_sequence_shapes_and_metrics.md)
3. [guides/00_prerequisites_and_notation.md](./guides/00_prerequisites_and_notation.md)
4. [guides/02_lstm_many_to_many_beginner.md](./guides/02_lstm_many_to_many_beginner.md)
5. [guides/02_lstm_walkthrough.md](./guides/02_lstm_walkthrough.md)
6. [02_lstm_many_to_many_toy.ipynb](./02_lstm_many_to_many_toy.ipynb)
7. [solutions/02_lstm_many_to_many_toy_solution.ipynb](./solutions/02_lstm_many_to_many_toy_solution.ipynb)
8. [../../00-Foundations/guides/04_self_study_debugging_playbook.md](../../00-Foundations/guides/04_self_study_debugging_playbook.md)

### `01-RNN / ЛР03` `GRU seq2seq` (Шаг 3)
1. [../../00-Foundations/README.md](../../00-Foundations/README.md)
2. [../../00-Foundations/guides/02_tokens_padding_and_decoder_shift.md](../../00-Foundations/guides/02_tokens_padding_and_decoder_shift.md)
3. [guides/00_prerequisites_and_notation.md](./guides/00_prerequisites_and_notation.md)
4. [guides/03_gru_seq2seq_tokens_beginner.md](./guides/03_gru_seq2seq_tokens_beginner.md)
5. [guides/03_gru_seq2seq_walkthrough.md](./guides/03_gru_seq2seq_walkthrough.md)
6. [03_gru_seq2seq_reverse_toy.ipynb](./03_gru_seq2seq_reverse_toy.ipynb)
7. [solutions/03_gru_seq2seq_reverse_toy_solution.ipynb](./solutions/03_gru_seq2seq_reverse_toy_solution.ipynb)
8. [../../00-Foundations/guides/04_self_study_debugging_playbook.md](../../00-Foundations/guides/04_self_study_debugging_playbook.md)

## Companion Guides
- [../../00-Foundations/README.md](../../00-Foundations/README.md) — общий вход в sequence-курс: shared-guides, warm-up notebooks и optional real-data showcases.
- [../../00-Foundations/guides/01_sequence_shapes_and_metrics.md](../../00-Foundations/guides/01_sequence_shapes_and_metrics.md) — единый нулевой слой по формам `(batch, time, features)`, `loss`, `accuracy`, `return_sequences`, `token_accuracy`, `exact_match`.
- [../../00-Foundations/guides/02_tokens_padding_and_decoder_shift.md](../../00-Foundations/guides/02_tokens_padding_and_decoder_shift.md) — общий вход в `seq2seq`: словарь, `PAD/SOS/EOS`, masking и сдвиг decoder.
- [../../00-Foundations/guides/04_self_study_debugging_playbook.md](../../00-Foundations/guides/04_self_study_debugging_playbook.md) — единый self-study маршрут диагностики для sequence-ноутбуков.
- [guides/00_prerequisites_and_notation.md](./guides/00_prerequisites_and_notation.md) — базовые понятия, формы тензоров, `loss`, `accuracy`, `train/validation/test`.
- [guides/01_simple_rnn_many_to_one_beginner.md](./guides/01_simple_rnn_many_to_one_beginner.md) — интуиция `many-to-one`, `h_T`, `sigmoid`, `binary_crossentropy`.
- [guides/01_simple_rnn_walkthrough.md](./guides/01_simple_rnn_walkthrough.md) — пошаговый разбор всех `TODO` ЛР1 с expected outputs и типовыми ошибками.
- [guides/02_lstm_many_to_many_beginner.md](./guides/02_lstm_many_to_many_beginner.md) — cumulative sum, `return_sequences=True`, `token_accuracy` vs `sequence_accuracy`.
- [guides/02_lstm_walkthrough.md](./guides/02_lstm_walkthrough.md) — пошаговый разбор всех `TODO` ЛР2, включая `sequence_accuracy`.
- [guides/03_gru_seq2seq_tokens_beginner.md](./guides/03_gru_seq2seq_tokens_beginner.md) — словарь токенов, `PAD/SOS/EOS`, `Embedding`, `teacher forcing`, `exact_match`.
- [guides/03_gru_seq2seq_walkthrough.md](./guides/03_gru_seq2seq_walkthrough.md) — пошаговый разбор всех `TODO` `01-RNN / ЛР03`, включая decoder shift и `exact_match`.

## Запуск
Команды выполняются из корня репозитория.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r themes/01-RNN/lab/requirements.txt
python3 -m ipykernel install --user --name rnn-lab --display-name "Python (.venv) RNN Lab"
.venv/bin/jupyter notebook
```

### Runtime Варианты
TensorFlow notebook'и этого блока теперь можно запускать:
- локально через `auto`, `local-cpu`, `local-gpu`;
- в `Google Colab` через `colab-cpu` или `colab-gpu`;
- в `Kaggle` через `kaggle-cpu` или `kaggle-gpu`.

Подробный guide с объяснением всех режимов:
[../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md](../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md)

Если нужен именно локальный GPU и вы не уверены в версиях `TensorFlow` / `CUDA`, дополнительно откройте:
[../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md](../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md)

## Порядок прохождения
1. Шаг 1 = `01-RNN / ЛР01` -> `01_simple_rnn_many_to_one_toy.ipynb`
2. Шаг 2 = `01-RNN / ЛР02` -> `02_lstm_many_to_many_toy.ipynb`
3. Шаг 3 = `01-RNN / ЛР03` -> `03_gru_seq2seq_reverse_toy.ipynb`
4. Следующий блок после этого README: Шаг 4 = `02-Attention / ЛР01`

## Что читать перед каждой ЛР
| ЛР | Что открыть до ноутбука |
|---|---|
| `01-RNN / ЛР01` `SimpleRNN many-to-one` | `../../00-Foundations/README.md` + `../../00-Foundations/guides/01_sequence_shapes_and_metrics.md` + `00_prerequisites_and_notation.md` + `01_simple_rnn_many_to_one_beginner.md` + `01_simple_rnn_walkthrough.md` |
| `01-RNN / ЛР02` `LSTM many-to-many` | `../../00-Foundations/README.md` + `../../00-Foundations/guides/01_sequence_shapes_and_metrics.md` + `00_prerequisites_and_notation.md` + `02_lstm_many_to_many_beginner.md` + `02_lstm_walkthrough.md` |
| `01-RNN / ЛР03` `GRU seq2seq` | `../../00-Foundations/README.md` + `../../00-Foundations/guides/02_tokens_padding_and_decoder_shift.md` + `00_prerequisites_and_notation.md` + `03_gru_seq2seq_tokens_beginner.md` + `03_gru_seq2seq_walkthrough.md` |

## Shape Bridge по ключевым TODO
Этот блок нужен как быстрый `Shape Bridge` перед кодом: что именно должно получиться на выходе каждого ключевого шага.

| ЛР | Ключевой TODO-переход | Ожидаемая форма |
|---|---|---|
| `ЛР01` | вход в модель `SimpleRNN` | `(batch, time, features)` |
| `ЛР01` | выход binary head | `(batch, 1)` |
| `ЛР02` | выход `LSTM(return_sequences=True)` | `(batch, time, units)` |
| `ЛР02` | целевой тензор для token-level проверки | `(batch, time, 1)` |
| `ЛР03` | `encoder_input` / `decoder_input` | `(N, T_in)` / `(N, T_out)` |
| `ЛР03` | `decoder_target` | `(N, T_out, 1)` |

Минимальная практика перед обучением: рядом с каждым TODO проверять rank и длины осей через `assert`, чтобы ошибка формы ловилась до `model.fit`.

## Как читать формы тензоров
Базовый шаблон:
- `N` — число объектов в наборе;
- `T` — длина временной оси;
- `F` — число признаков на шаге;
- `V` — размер словаря (для `seq2seq`).

Ключевой принцип: в рекуррентных задачах порядок осей должен быть `(batch, time, features)`.

Проверка на каждом шаге:
1. До обучения: сравнить фактические формы `X`, `y` с таблицей в ноутбуке.
2. После сборки модели: проверить форму выхода `model.output_shape`.
3. После `predict`: проверить форму `probs/preds` и сопоставить с целевым тензором.

## Методический маршрут
Рекомендуемый цикл работы в каждом ноутбуке:
1. `Что нужно знать до старта` и `Интуиция задачи без формул`.
2. `Контракт данных`, `Таблица форм тензоров`, `Шпаргалка по обозначениям и формам`.
3. `Контракт модели`, `Мини-теория`, `Ручной разбор одного примера`.
4. `Генерация данных` -> `Мини-проверка данных`.
5. `Модель` -> `Мини-проверка модели` -> `Трассировка одного примера`.
6. `Как идет обучение внутри эпохи` -> `Обучение` -> `Мини-проверка обучения`.
7. `Оценка и диагностика` -> `Как читать графики и метрики`.
8. `Если застряли: порядок диагностики` -> `Чек-лист перед сдачей` -> `Вопросы для самопроверки`.

## Схема валидации
Во всех трех работах используется единый подход:
- сначала формируется отдельный `test` через `train_test_split`;
- затем обучение выполняется на `train` с `validation_split` внутри `model.fit`.

Это разделяет:
- настройку обучения (`train`/`validation`),
- финальную независимую оценку (`test`).

## API-конвенция
- ЛР1 и ЛР2: только `Sequential` + `add(...)`.
- ЛР3: гибрид — encoder/decoder блоки через `Sequential.add(...)`, объединение двух входов через функциональный граф (`Functional API`).

## Связь с теорией
- Раздел «Форматы последовательных задач» в `theory.md` -> понимание `many-to-one`, `many-to-many`, `seq2seq`.
- Раздел «Канонический блок обозначений и формул» -> проверка математических обозначений перед реализацией.
- Раздел «Как это выглядит в лабораторных» -> прямое соответствие формул и тензоров из кода.
- Раздел «Внутренняя логика обучения по типам ячеек» -> интерпретация кривых и поведения обучения.

Beginner-guides не заменяют `theory.md`: теория остаётся каноническим источником формул, а guides дают мягкий вход, ручные примеры и словарь.

## Дополнительные материалы
- [../theory/theory.md](../theory/theory.md) — основная теория по `SimpleRNN`, `LSTM`, `GRU`, `seq2seq`.
- [../../00-Foundations/README.md](../../00-Foundations/README.md) — общий prerequisites-layer, warm-up examples и optional showcases до или между лабораторными.
- `00-initial/01-RNN/Конспект - рекуррентные нейронные сети для начинающих.pdf` — вводный конспект по теме.
- После Шага 3 = `01-RNN / ЛР03` логичный следующий шаг — Шаг 4 = `02-Attention / ЛР01`.

## FAQ
### Что делать, если я путаюсь в формах?
Сначала открыть `00_prerequisites_and_notation.md`, затем проговорить каждую ось словами: сколько объектов, сколько шагов, сколько признаков.

### Что делать, если я не понимаю `axis`?
Вернуться к форме тензора и буквально подписать оси руками: `0 = batch`, `1 = time`, `2 = features`. Для ЛР1 и ЛР2 суммирование и накопление идут по временной оси `1`.

### Почему в ЛР2 есть `token_accuracy`, если токенизации там нет?
В ЛР2 `token_accuracy` означает “доля правильных ответов по шагам времени”. Это не NLP tokenization. Полный разбор есть в `02_lstm_many_to_many_beginner.md`.

### Когда появляется настоящая работа с токенами?
Только в ЛР3: там уже есть словарь, целочисленные токены, `PAD`, `SOS`, `EOS`, `Embedding`, маска и `exact_match`.

### Что делать, если notebook запускается не по порядку?
Лучший способ восстановиться — `Restart & Run All`. Подробный маршрут есть в `../../00-Foundations/guides/04_self_study_debugging_playbook.md`.

### Как понять, что форма `y` неправильная?
Сравнить её с таблицей форм в notebook и с ближайшей мини-проверкой. Для ЛР1 ожидается `(N,)`, для ЛР2 — `(N, T, 1)`.

### Когда лучше читать теорию?
Не обязательно целиком до старта. Удобнее проходить лабораторную и точечно возвращаться в `theory.md`, когда нужно разобраться в формуле или термине.

### Как пользоваться solution notebook, чтобы не превратить работу в copy-paste?
Сначала сделать свою попытку, затем открыть walkthrough для текущего шага, и только после этого сверять solution notebook по одному блоку, а не целиком.

## Типичные проблемы
### ЛР1 (`SimpleRNN`, `many-to-one`)
- Симптом: `accuracy` около `0.5`.
  Причина: неверное правило построения меток.
  Исправление: проверить `sum > 0` и форму `y`.
- Симптом: ошибка входной формы.
  Причина: перепутан порядок осей.
  Исправление: вход должен иметь вид `(batch, time, features)`.

### ЛР2 (`LSTM`, `many-to-many`)
- Симптом: выход имеет форму `(batch, units)`.
  Причина: не включен `return_sequences=True`.
  Исправление: проверить конфигурацию `LSTM`.
- Симптом: `token_accuracy` высокая, `sequence_accuracy` низкая.
  Причина: локальные ошибки присутствуют хотя бы на одном шаге многих последовательностей.
  Исправление: проверить разметку и улучшить обучение.
- Симптом: студент думает, что `token_accuracy` связана с текстовой токенизацией.
  Причина: термин `token` используется в более общем смысле как “один элемент последовательности”.
  Исправление: открыть `02_lstm_many_to_many_beginner.md` и развести понятия “шаг времени” и “tokenization”.

### ЛР3 (`GRU`, `seq2seq`)
- Симптом: низкий `exact_match` при приемлемом `token_accuracy`.
  Причина: нарушен сдвиг `decoder_input`/`decoder_target`.
  Исправление: проверить схемы `[SOS] + rev` и `rev + [EOS]`.
- Симптом: нестабильная оценка последовательности.
  Причина: PAD-токены учитываются в `exact_match`.
  Исправление: считать метрику только по маске `target != PAD`.
- Симптом: студент не понимает, зачем нужны `PAD`, `SOS`, `EOS`.
  Причина: пропущен входной слой про словарь токенов и decoder shift.
  Исправление: до ноутбука прочитать `03_gru_seq2seq_tokens_beginner.md`.
