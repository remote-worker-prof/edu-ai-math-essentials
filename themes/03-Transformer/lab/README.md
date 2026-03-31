# Лабораторный практикум по Transformer

## Назначение
Практикум содержит две последовательные лабораторные:
1. `Transformer encoder` на synthetic order-sensitive classification.
2. `Transformer encoder` на реальном `IMDB` для binary sentiment classification.

Цель блока — мягко перевести студентов от `cross-attention` в `seq2seq` к `self-attention`, positional embedding и encoder-only Transformer.

## Структура
- `01_transformer_encoder_order_toy.ipynb` — starter notebook с `TODO`.
- `02_transformer_encoder_imdb.ipynb` — starter notebook с `TODO`.
- `solutions/01_transformer_encoder_order_toy_solution.ipynb` — полное решение `ЛР01`.
- `solutions/02_transformer_encoder_imdb_solution.ipynb` — полное решение `ЛР02`.

Solutions остаются в репозитории, но считаются материалом для сравнения только после своей попытки и debugging-step.

## Карта курса
Общая линия курса:
1. Шаг 1 = `01-RNN / ЛР01`
2. Шаг 2 = `01-RNN / ЛР02`
3. Шаг 3 = `01-RNN / ЛР03`
4. Шаг 4 = `02-Attention / ЛР01`
5. Шаг 5 = `03-Transformer / ЛР01`
6. Шаг 6 = `03-Transformer / ЛР02`

Важно: этот блок специально ограничен **encoder-only Transformer** в формате `toy -> real data`. Полный encoder-decoder Transformer и causal mask остаются на следующий шаг в [../../04-Autoregression/lab/README.md](../../04-Autoregression/lab/README.md).

## Опционально, Если Нужно Повторить Прошлую Тему
Открывайте этот блок только если после `02-Attention` плавают термины или чтение heatmap:
- [../../00-Foundations/README.md](../../00-Foundations/README.md)
- [../../00-Foundations/guides/03_attention_heatmaps.md](../../00-Foundations/guides/03_attention_heatmaps.md)
- [../../02-Attention/theory/theory.md](../../02-Attention/theory/theory.md)

Если эта база уже устойчива, переходите сразу к фиксированному треку ниже.

## Фиксированный Трек Студента
Это один канонический маршрут для self-study и для работы с преподавателем.

1. Прочитать [guides/00_transformer_prerequisites.md](./guides/00_transformer_prerequisites.md).
2. Прочитать [guides/01_self_attention_and_positional_encoding_beginner.md](./guides/01_self_attention_and_positional_encoding_beginner.md).
3. Пройти [guides/02_transformer_encoder_toy_walkthrough.md](./guides/02_transformer_encoder_toy_walkthrough.md).
4. Сделать свою первую попытку в [01_transformer_encoder_order_toy.ipynb](./01_transformer_encoder_order_toy.ipynb).
5. Если застряли, открыть [guides/04_transformer_debugging_playbook.md](./guides/04_transformer_debugging_playbook.md) и сделать вторую попытку в `ЛР01`.
6. Только после второй попытки сравнить результат с [solutions/01_transformer_encoder_order_toy_solution.ipynb](./solutions/01_transformer_encoder_order_toy_solution.ipynb).
7. Пройти [guides/03_transformer_encoder_imdb_walkthrough.md](./guides/03_transformer_encoder_imdb_walkthrough.md).
8. Сделать свою первую попытку в [02_transformer_encoder_imdb.ipynb](./02_transformer_encoder_imdb.ipynb), используя ready reuse-block из `ЛР01`.
9. Если застряли, снова открыть [guides/04_transformer_debugging_playbook.md](./guides/04_transformer_debugging_playbook.md) и сделать вторую попытку в `ЛР02`.
10. Только после второй попытки сравнить результат с [solutions/02_transformer_encoder_imdb_solution.ipynb](./solutions/02_transformer_encoder_imdb_solution.ipynb).

## Ключевые Материалы
- [guides/00_transformer_prerequisites.md](./guides/00_transformer_prerequisites.md) — что повторить после `02-Attention`.
- [guides/01_self_attention_and_positional_encoding_beginner.md](./guides/01_self_attention_and_positional_encoding_beginner.md) — мягкий вход в `self-attention`, positional embedding и encoder block.
- [guides/02_transformer_encoder_toy_walkthrough.md](./guides/02_transformer_encoder_toy_walkthrough.md) — пошаговый разбор `ЛР01`.
- [guides/03_transformer_encoder_imdb_walkthrough.md](./guides/03_transformer_encoder_imdb_walkthrough.md) — пошаговый разбор `ЛР02` как transfer/reuse-работы.
- [guides/04_transformer_debugging_playbook.md](./guides/04_transformer_debugging_playbook.md) — типичные поломки в masking, shapes и attention scores.

## Запуск
Команды выполняются из корня репозитория.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r themes/03-Transformer/lab/requirements.txt
python3 -m ipykernel install --user --name transformer-lab --display-name "Python (.venv) Transformer Lab"
.venv/bin/jupyter notebook
```

### Runtime Варианты
TensorFlow notebook'и этого блока теперь можно запускать:
- локально через `auto`, `local-cpu`, `local-gpu`;
- в `Google Colab` через `colab-cpu` или `colab-gpu`;
- в `Kaggle` через `kaggle-cpu` или `kaggle-gpu`.

Полный runtime-guide для понятного выбора между локальным и облачным запуском:
[../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md](../../00-Foundations/guides/05_local_tensorflow_gpu_notebooks.md)

Если вы хотите локальный GPU и не уверены в версиях `TensorFlow` / `CUDA`, дополнительно откройте:
[../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md](../../00-Foundations/guides/06_tensorflow_cuda_version_selection.md)

## Критерии Завершения
### `03-Transformer / ЛР01`
Работа считается законченной, если одновременно выполнено следующее:
- модель различает два ручных примера с перестановкой `7` и `3`;
- итоговая метрика держится на уровне `test_acc >= 0.95`;
- attention heatmap строится по непустой части последовательности и не включает padded хвост.

### `03-Transformer / ЛР02`
Работа считается законченной, если одновременно выполнено следующее:
- итоговая метрика держится на уровне `test_acc >= 0.75` на текущем subset;
- показан хотя бы один декодированный review;
- attention heatmap строится по первым содержательным токенам review, а не по padded хвосту.

## Как Читать Формы Тензоров
Базовый шаблон:
- `N` — число объектов;
- `T` — длина последовательности;
- `E` — размер embedding / model dimension;
- `H` — число attention heads;
- `V` — размер словаря.

Ключевые тензоры:
- `tokens`: `(N, T)`
- `embeddings`: `(N, T, E)`
- `attention_scores`: `(N, H, T, T)`
- `y_pred`: `(N, 1)`

Проверка на каждом шаге:
1. До обучения: проверить формы `X_train`, `y_train`.
2. После embedding: проверить `(batch, time, embed_dim)`.
3. После trace-model: проверить форму `attention_scores`.
4. После pooling/classifier head: убедиться, что модель возвращает одну вероятность на объект.

## API-Конвенция
Обе лабораторные используют один и тот же шаблон:
- `Functional API`;
- custom layers `TokenAndPositionEmbedding` и `TransformerEncoderBlock`;
- отдельный tracing-path для `attention_scores`;
- выход классификатора `y_pred -> (N, 1)`;
- `sigmoid` в финальном слое;
- `binary_crossentropy` как функция потерь.

## Связь С Теорией
- [../theory/theory.md](../theory/theory.md) — каноническая теория блока `Transformer`.
- Раздел 1 в `theory.md` -> переход от `cross-attention` к `self-attention`.
- Раздел 2 -> scaled dot-product attention.
- Раздел 3 -> positional embedding.
- Раздел 4 и 5 -> `multi-head attention`, residual, `LayerNormalization`, feed-forward block.
- Раздел 6 -> padding mask.

Guides не заменяют теорию, а помогают пройти её в понятном для студента порядке.

## Дополнительные Материалы
- [../../02-Attention/lab/README.md](../../02-Attention/lab/README.md) — предыдущий шаг курса.
- [../../00-Foundations/showcases/cards/01_imdb.md](../../00-Foundations/showcases/cards/01_imdb.md) — dataset card для `IMDB`.

## FAQ
### Почему здесь нет полного encoder-decoder Transformer?
Потому что первая версия темы делает мягкий переход от attention к Transformer. Сначала осваиваются `self-attention`, positional embedding и encoder block, а decoder и causal mask остаются на следующий блок.

### Почему первая лабораторная synthetic?
Потому что toy-задача изолированно показывает роль порядка. Если один и тот же набор токенов может давать разные метки, студент сразу видит, зачем нужен positional embedding.

### Что именно переносится из `ЛР01` в `ЛР02`?
В `ЛР02` переиспользуется уже понятный transformer reuse-block: `TokenAndPositionEmbedding`, `masked_average` и `TransformerEncoderBlock`. Новая работа начинается не с повторной реализации слоёв, а с переноса их на real text classification.

### Что дальше после этого блока?
Следующий естественный шаг:
- [../../04-Autoregression/lab/README.md](../../04-Autoregression/lab/README.md) — `decoder-only Transformer` + `causal mask` в детерминированной генеративной лабораторной;
- затем полный `encoder-decoder Transformer`.

## Типичные Проблемы
### Навигация
- Симптом: непонятно, когда открывать solution.  
  Причина: смешан обязательный и дополнительный трек.  
  Исправление: держаться `Фиксированного трека студента` и открывать solution только после debugging-step и второй попытки.

### Encoder block
- Симптом: `attention_scores` имеет странную форму.  
  Причина: перепутано ожидание формы у `MultiHeadAttention`.  
  Исправление: помнить про `(batch, heads, T, T)`.
- Симптом: модель обучается, но attention шумный.  
  Причина: `PAD`-токены участвуют во внимании.  
  Исправление: передавать padding mask в attention и учитывать её при pooling.

### Позиции и порядок
- Симптом: модель плохо различает последовательности с одинаковым набором токенов.  
  Причина: positional embedding не добавлен или сломан.  
  Исправление: проверить `TokenAndPositionEmbedding`.

### Реальные тексты
- Симптом: `IMDB` работает слишком медленно.  
  Причина: слишком большая длина последовательности или слишком большой subset.  
  Исправление: держать `maxlen=200` и умеренный train subset.
- Симптом: attention-карта review плохо читается.  
  Причина: показывается слишком длинный текст или padded хвост.  
  Исправление: обрезать визуализацию до первых содержательных токенов.
