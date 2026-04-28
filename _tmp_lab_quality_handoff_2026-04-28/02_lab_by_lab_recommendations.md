# Рекомендации по пяти тематическим ЛР

Под "5 ЛР" здесь понимаются пять тематических блоков:

- `themes/01-RNN`;
- `themes/02-Attention`;
- `themes/03-Transformer`;
- `themes/04-Autoregression`;
- `themes/05-Full-Transformer`.

Внутри каждого блока нужно учитывать все starter notebooks и solution
notebooks. Главная цель следующей доработки — не поменять постановки, а
сделать учебный контракт одинаково ясным и машинно закрепленным.

## 01-RNN

Текущий блок уже хорошо выстроен как последовательность:

- `SimpleRNN many-to-one`;
- `LSTM many-to-many`;
- `GRU seq2seq`.

Что стоит усилить:

- Перед каждым `TODO` явно связывать форму тензора с действием студента:
  почему здесь `(batch, time, features)`, где появляется последняя временная
  позиция, где нужен `return_sequences=True`, где начинается decoder shift.
- Добавить короткие "shape bridges" перед ключевыми code cells:
  "из `(N, T, F)` получаем `(N, 1)`", "из `(N, T)` получаем `(N, T, V)`".
- В student notebooks оставить guided skeleton, но усилить `assert`-подсказки:
  expected rank, expected batch size, expected time length, expected output
  channels.
- В solution notebooks сделать код максимально читаемым:
  русские docstrings, промежуточные имена, пустые строки между этапами,
  многострочные toy arrays.
- В README или финале notebook-а добавить единый report checklist:
  постановка, формы, метрики, sanity-checks, графики обучения, пример
  предсказания, диагностика ошибки.

Что закрепить проверкой:

- все пары starter/solution существуют;
- starter содержит `TODO`, solution не содержит нерешенных `TODO`;
- expected shapes явно встречаются рядом с ключевыми проверками;
- нет английских docstring-секций `Args:`, `Returns:`, `Raises:`;
- runtime-маркеры не расходятся между starter и solution.

## 02-Attention

Сильная сторона блока — он продолжает `seq2seq` и вводит attention как
снятие fixed-context bottleneck. Следующий шаг — сделать интерпретацию
`query/key/value` еще более "зеркальной" и визуальной.

Что стоит усилить:

- Добавить beginner-мостик:
  `query` = "что decoder сейчас ищет",
  `key` = "по каким признакам encoder предлагает позиции",
  `value` = "какую информацию encoder отдаёт, если позиция выбрана".
- Пояснить attention scores как таблицу сопоставления:
  строки = шаги decoder, столбцы = позиции encoder.
- Перед heatmap явно проговаривать, какие оси у картинки и где должен быть
  `PAD`.
- Развести три объекта:
  raw scores, normalized attention weights, context vector.
- Показать, как mask меняет смысл heatmap: `PAD` не должен получать внимание.

Что закрепить проверкой:

- notebook-и содержат markers `query`, `key`, `value`, `attention_scores`,
  `context`, `padding_mask`;
- heatmap block существует и подписывает оси;
- shape contract для `attention_scores` фиксирует `(batch, T_out, T_in)` или
  локально принятую эквивалентную форму;
- mask contract запрещает внимание к `PAD`;
- starter/solution имеют одинаковые разделы диагностики.

## 03-Transformer

В этом блоке уже есть два notebook-а: toy encoder и IMDB encoder. По текущему
срезу именно здесь заметны английские docstrings `Args:`/`Returns:`, поэтому
русификация и выравнивание стиля должны быть первым практическим шагом.

Что стоит усилить:

- Перевести docstrings всех helper-функций на русские Google-style секции:
  `Аргументы:`, `Возвращает:`, `Исключения:`.
- Добавить beginner-объяснение encoder как "перечитывания всех токенов с
  учетом всех остальных токенов", а self-attention как таблицы связей
  `token -> token`.
- Пояснить positional encoding отдельно:
  без позиции набор токенов похож на мешок, а с позицией модель видит порядок.
- Развести `padding_mask` и attention visualization:
  mask участвует в вычислении, heatmap помогает проверить смысл.
- Для IMDB добавить короткий мостик от toy sequence к real text:
  что меняется в данных, а что остается тем же в формах.

Что закрепить проверкой:

- отсутствуют `Args:`, `Returns:`, `Raises:` в code cells;
- присутствуют `Аргументы:`, `Возвращает:` для helper-функций;
- starter/solution consistency для обоих notebook-ов;
- есть markers `padding_mask`, `attention_scores`, `positional`, `encoder`;
- многострочно оформлены списки toy tokens, shape tuples и демонстрационные
  матрицы, если они объявляются явно.

## 04-Autoregression

Блок сложный и важный: causal mask, Tiny Shakespeare, CPU/GPU контуры,
perplexity, baseline, generation gates. Здесь особенно нужен строгий runtime
и leakage contract.

Что стоит усилить:

- Явно развести `ЛР02 CPU` и `ЛР02 GPU` как два разных образовательных
  сценария, а не просто две скорости одной работы.
- В каждом notebook-е перед моделью дать "causal mask на пальцах":
  токен может смотреть в прошлое и в себя, но не в будущие позиции.
- Добавить маленький ручной пример mask-матрицы, где каждая строка матрицы
  написана на отдельной строке.
- Объяснить leakage check:
  если decoder видит будущий target, метрики становятся нечестными.
- Разделить `perplexity`, `baseline`, `generation gate`:
  perplexity измеряет среднюю уверенность модели, baseline задает нижнюю
  планку, generation gate проверяет поведение на фиксированных подсказках.
- Для GPU notebook-а сделать ранний preflight главным учебным контрактом:
  если GPU-профиль выбран, GPU должен быть реально доступен.

Что закрепить проверкой:

- markers `causal_mask`, `leakage`, `perplexity`, `baseline`,
  `generation gate`, `gpu_preflight`;
- CPU notebook не требует GPU;
- GPU notebook не делает скрытый fallback на CPU;
- heavy training не запускается в обычном CI;
- shape/mask diagnostics есть и в starter, и в solution;
- русские docstrings и многострочные tensor/mask examples.

## 05-Full-Transformer

Финальный блок должен быть особенно строгим, потому что он объединяет encoder,
decoder, causal mask, cross-attention и полноценный data-contract. Сейчас в
README уже есть важные определения `encoder_input`, `decoder_input`,
`decoder_target`; их стоит сделать центральной учебной осью всех материалов.

Что стоит усилить:

- В начале notebook-а дать полный data-contract:
  `encoder_input`, `target`, `decoder_input`, `decoder_target`.
- Показать ручной пример окна:
  какие tokens уходят в encoder, какие tokens становятся target, почему
  `decoder_input = [SOS] + target[:-1]`.
- Развести маски:
  padding mask отвечает за пустые позиции, causal mask запрещает будущее,
  cross-attention mask связывает decoder с допустимыми encoder-позициями.
- Объяснить encoder-decoder Transformer как две сцены:
  encoder строит память по входу, decoder по одному шагу читает свою историю
  и обращается к памяти encoder.
- Для solution на `WikiText-2` явно пояснить, что dataset отличается от
  starter-а, но учебный контракт тензоров должен оставаться тем же.
- В финальном отчете требовать не только метрики, но и объяснение:
  почему `decoder_input` и `decoder_target` сдвинуты, где применялась causal
  mask, что подтверждает отсутствие leakage.

Что закрепить проверкой:

- markers `encoder_input`, `decoder_input`, `decoder_target`, `target`;
- markers `padding_mask`, `causal_mask`, `cross_attention`;
- starter/solution alignment несмотря на разные датасеты;
- GPU profile падает ранней диагностикой, если GPU недоступен;
- generation gates и baseline checks присутствуют;
- report checklist содержит data-contract, masks, metrics, leakage check,
  examples and limitations.

## Общий порядок внедрения

1. Сначала добавить contract-check скрипты без изменения содержимого notebook-ов
   и зафиксировать текущую картину.
2. Затем русифицировать docstrings и comments в одной теме за раз.
3. Потом привести arrays/tensors/masks к многострочному виду.
4. После этого усилить beginner-мостики и report checklists.
5. В конце подключить проверки в локальный маршрут и CI.

Такой порядок снижает риск случайно изменить учебную постановку или сломать
starter/solution alignment.

