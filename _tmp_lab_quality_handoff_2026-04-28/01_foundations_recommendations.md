# Рекомендации для `themes/00-Foundations`

`00-Foundations` уже выполняет роль общего входа: формы последовательностей,
токены, `PAD/SOS/EOS`, masking, teacher forcing, attention heatmap,
self-study debugging и runtime guides. Следующая доработка должна сделать этот
слой еще более каноническим и машинно проверяемым.

## 1. Входной маршрут

Стоит закрепить один короткий "нулевой маршрут" для совсем нового студента:

1. Что такое последовательность и почему у нее есть ось времени.
2. Что означает форма `(batch, time, features)`.
3. Как читать `X`, `y`, `tokens`, `targets`, `mask`.
4. Чем отличаются many-to-one, many-to-many, seq2seq, decoder-only и
   encoder-decoder постановки.
5. Какой notebook открыть первым и какие проверки должны сработать.

Сейчас маршруты уже есть, но агенту стоит добавить более явный переход:

- "если путаешь оси, не иди дальше";
- "если не понимаешь `PAD/SOS/EOS`, не открывай seq2seq";
- "если не понимаешь causal mask, не открывай autoregression";
- "если не понимаешь `encoder_input`, `decoder_input`, `decoder_target`,
  не открывай full transformer".

## 2. Warm-up examples

Для `examples/` важно сохранить малый CPU-масштаб. В каждом warm-up notebook-е
рекомендуется добавить или усилить:

- явный data-contract в начале;
- таблицу ожидаемых форм;
- один ручной пример до модели;
- мини-проверки через `assert`;
- понятный финальный блок "что теперь должно стать ясным";
- русские Google-style docstrings, если есть helper-функции:
  `Аргументы:`, `Возвращает:`, `Исключения:`.

Матрицы, toy tokens и masks оформить многострочно. Это особенно важно для
новичка: визуальное расположение строк должно совпадать с учебной идеей
батча или последовательности.

## 3. Единый glossary

Сейчас словарь распределен по guides. Стоит добавить или усилить общий
glossary, который связывает термины из всех пяти ЛР:

- `batch`;
- `time`;
- `features`;
- `embedding_dim`;
- `hidden_state`;
- `logits`;
- `probabilities`;
- `teacher forcing`;
- `decoder shift`;
- `padding mask`;
- `causal mask`;
- `cross-attention`;
- `attention_scores`;
- `perplexity`;
- `token_accuracy`;
- `exact_match`;
- `generation gate`;
- `baseline`.

Для каждого термина желательно дать:

- короткое объяснение "на пальцах";
- форму тензора, если термин тензорный;
- где впервые используется;
- типичную ошибку;
- какой `assert` или diagnostic помогает поймать ошибку.

## 4. CPU/GPU/cloud expectations

Runtime guides уже сильные, но их можно связать с лабораторными контрактами:

- явно разделить `CPU-friendly`, `GPU-friendly`, `Colab`, `Kaggle`;
- указать, какие notebook-и обязаны проходить на CPU;
- какие notebook-и только структурно проверяются без тяжелого обучения;
- какие acceptance-прогоны требуют GPU;
- какие ошибки должны быть ранними и понятными, если GPU недоступен.

Особенно важно закрепить правило:

```text
Если выбран GPU-профиль, скрытого fallback на CPU быть не должно.
```

Для heavy notebook-ов лучше иметь preflight-блок, который падает до загрузки
данных и обучения, если окружение не соответствует профилю.

## 5. Self-study checks

В `04_self_study_debugging_playbook.md` стоит добавить универсальную таблицу:

| Симптом | Вероятная причина | Что проверить | Где исправлять |
|---|---|---|---|
| loss не падает | target сдвинут неверно | `decoder_input` и `decoder_target` | data-contract |
| accuracy высокая, генерация плохая | метрика не соответствует задаче | `exact_match`, generation gate | evaluation |
| attention смотрит на `PAD` | mask не применена | `padding_mask`, heatmap | model call |
| decoder видит будущее | causal mask неверна | верхний треугольник scores | mask builder |

Эта таблица должна стать общей диагностической рамкой для всех тем.

## 6. Машинные проверки для Foundations

Для `00-Foundations` стоит добавить contract-check, который проверяет:

- все referenced guides и notebooks из README существуют;
- все warm-up notebooks читаются как JSON;
- в code cells нет `Args:`, `Returns:`, `Raises:`;
- helper docstrings используют `Аргументы:`, `Возвращает:`,
  `Исключения:`;
- toy matrices/tokens/masks объявлены многострочно;
- runtime guides содержат `local-cpu`, `local-gpu`, `colab-cpu`,
  `colab-gpu`, `kaggle-cpu`, `kaggle-gpu`;
- glossary покрывает термины, которые используются в README тем `01` ... `05`.

## 7. Что не делать в Foundations

- Не превращать `00-Foundations` в шестую лабораторную с тяжелыми заданиями.
- Не переносить туда все объяснения из тематических ЛР.
- Не добавлять обязательное тяжелое обучение в warm-up.
- Не менять датасетные постановки тематических блоков через foundations-правку.

Главная роль `00-Foundations` — снять входной страх перед формами, масками,
runtime и диагностикой, чтобы тематические ЛР читались спокойнее.

