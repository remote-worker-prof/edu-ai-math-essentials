# Вариант 2. Ключ преподавателя по дисциплине «Математические основы ИИ»

---

## Раздел 1. Foundations

### Q1 - Роль masking

Правильный ответ: A) Игнорирование служебных позиций при вычислениях

Трассируемость: foundations -> mat_0003 -> 01-foundations -> Q1
Источник материала: themes/00-Foundations/examples/02_minimal_keras_sequence_classifier.ipynb

---

### Q2 - Назначение padding в последовательностях

Правильный ответ: A) Сделать длины последовательностей одинаковыми в батче

Трассируемость: foundations -> mat_0004 -> 01-foundations -> Q2
Источник материала: themes/00-Foundations/examples/03_tokenization_padding_masking.ipynb

---

### Q3 - Подготовка входных последовательностей

Правильные ответы: A) Стратегию усечения длинных примеров; B) Положение padding-токенов; D) Согласованность с mask-механизмом

Трассируемость: foundations -> mat_0005 -> 01-foundations -> Q3
Источник материала: themes/00-Foundations/examples/04_attention_heatmap_toy.ipynb

---

### Q4 - Базовые метрики качества

Правильные ответы: A) Accuracy; B) Precision/Recall; D) F1-score

Трассируемость: foundations -> mat_0006 -> 01-foundations -> Q4
Источник материала: themes/00-Foundations/showcases/01_imdb_many_to_one_showcase.ipynb

---

### Q5 - Компромисс padding/truncation

Эталонный ответ: Padding повышает совместимость батчей, но добавляет пустые позиции; truncation снижает шум и стоимость, но может терять информацию.

Критерии оценивания:
- Описаны плюсы и минусы padding
- Описаны плюсы и риски truncation
- Сделан вывод о выборе стратегии под задачу

Трассируемость: foundations -> mat_0007 -> 01-foundations -> Q5
Источник материала: themes/00-Foundations/showcases/02_jena_climate_lstm_gru_showcase.ipynb

---

## Раздел 2. RNN

### Q6 - Особенность GRU

Правильный ответ: A) Более компактная архитектура с меньшим числом ворот

Трассируемость: rnn -> mat_0018 -> 02-rnn -> Q6
Источник материала: themes/01-RNN/lab/02_lstm_many_to_many_toy.ipynb

---

### Q7 - Преимущество LSTM

Правильный ответ: A) Наличие механизма ворот для управления памятью

Трассируемость: rnn -> mat_0019 -> 02-rnn -> Q7
Источник материала: themes/01-RNN/lab/03_gru_seq2seq_reverse_toy.ipynb

---

### Q8 - Диагностика RNN

Правильные ответы: A) Рост разрыва между train и validation; B) Резкие колебания loss; D) Частая деградация на длинных примерах

Трассируемость: rnn -> mat_0020 -> 02-rnn -> Q8
Источник материала: themes/01-RNN/lab/guides/00_prerequisites_and_notation.md

---

### Q9 - Компоненты базового seq2seq

Правильные ответы: A) Encoder; B) Decoder; C) Рекуррентная передача состояния

Трассируемость: rnn -> mat_0021 -> 02-rnn -> Q9
Источник материала: themes/01-RNN/lab/guides/00_self_study_debugging_playbook.md

---

### Q10 - План диагностики RNN

Эталонный ответ: Проверить разрыв train/val, скорректировать регуляризацию, раннюю остановку, размер модели и learning rate, затем повторно оценить стабильность метрик.

Критерии оценивания:
- Есть шаги наблюдения и коррекции
- Указаны минимум две управляемые гиперпараметрические меры
- Показана связь действий и ожидаемого эффекта

Трассируемость: rnn -> mat_0022 -> 02-rnn -> Q10
Источник материала: themes/01-RNN/lab/guides/01_simple_rnn_many_to_one_beginner.md

---

## Раздел 3. Attention

### Q11 - Интерпретация attention-карт

Правильный ответ: A) Как индикатор распределения фокуса модели

Трассируемость: attention -> mat_0034 -> 03-attention -> Q11
Источник материала: themes/02-Attention/lab/guides/00_attention_prerequisites.md

---

### Q12 - Смысл attention-весов

Правильный ответ: A) Относительную важность элементов контекста для текущего шага

Трассируемость: attention -> mat_0035 -> 03-attention -> Q12
Источник материала: themes/02-Attention/lab/guides/01_gru_seq2seq_attention_beginner.md

---

### Q13 - Сигналы полезного внимания

Правильные ответы: A) Фокус смещается в зависимости от текущего шага декодера; B) Есть связь между attention и снижением ошибок; D) Сложные примеры показывают осмысленные зоны фокуса

Трассируемость: attention -> mat_0036 -> 03-attention -> Q13
Источник материала: themes/02-Attention/lab/guides/02_attention_walkthrough.md

---

### Q14 - Преимущества внимания

Правильные ответы: A) Более явная работа с релевантным контекстом; B) Улучшение интерпретируемости поведения; D) Снижение риска потери важной информации

Трассируемость: attention -> mat_0037 -> 03-attention -> Q14
Источник материала: themes/02-Attention/lab/guides/03_attention_debugging_playbook.md

---

### Q15 - Attention vs фиксированный контекст

Эталонный ответ: Attention позволяет динамически перераспределять фокус по входу, фиксированный вектор ограничивает модель единым сжатием всей последовательности.

Критерии оценивания:
- Сравнены механизмы представления контекста
- Показано влияние на длинные последовательности
- Сделан вывод о практическом выборе

Трассируемость: attention -> mat_0038 -> 03-attention -> Q15
Источник материала: themes/02-Attention/lab/solutions/01_gru_seq2seq_attention_reverse_toy_solution.ipynb

---

## Раздел 4. Transformer Encoder

### Q16 - Multi-head идея

Правильный ответ: A) Разные головы могут учить разные типы зависимостей

Трассируемость: transformer_encoder -> mat_0041 -> 04-transformer-encoder -> Q16
Источник материала: themes/03-Transformer/lab/02_transformer_encoder_imdb.ipynb

---

### Q17 - Роль positional encoding

Правильный ответ: A) Чтобы модель учитывала порядок токенов

Трассируемость: transformer_encoder -> mat_0042 -> 04-transformer-encoder -> Q17
Источник материала: themes/03-Transformer/lab/guides/00_transformer_prerequisites.md

---

### Q18 - Диагностика encoder

Правильные ответы: A) Динамика train/validation метрик; B) Стабильность attention-распределений; D) Кривые обучения по эпохам

Трассируемость: transformer_encoder -> mat_0043 -> 04-transformer-encoder -> Q18
Источник материала: themes/03-Transformer/lab/guides/01_self_attention_and_positional_encoding_beginner.md

---

### Q19 - Состав encoder-блока

Правильные ответы: A) Multi-head self-attention; B) Feed-forward sublayer; C) Residual connections и нормализация

Трассируемость: transformer_encoder -> mat_0044 -> 04-transformer-encoder -> Q19
Источник материала: themes/03-Transformer/lab/guides/02_transformer_encoder_toy_walkthrough.md

---

### Q20 - Недостаток позиции

Эталонный ответ: Модель хуже различает порядок токенов, что ведёт к ошибкам в задачах, где порядок критичен для смысла.

Критерии оценивания:
- Объяснено влияние на учет порядка
- Связь с потенциальной деградацией качества
- Дан пример или тип задач, где риск высок

Трассируемость: transformer_encoder -> mat_0045 -> 04-transformer-encoder -> Q20
Источник материала: themes/03-Transformer/lab/guides/03_transformer_encoder_imdb_walkthrough.md

---

## Раздел 5. Autoregression

### Q21 - Генерация в decoder-only

Правильный ответ: A) Последовательным добавлением новых токенов

Трассируемость: autoregression -> mat_0051 -> 05-autoregression -> Q21
Источник материала: themes/04-Autoregression/lab/02_decoder_only_tiny_shakespeare.ipynb

---

### Q22 - Назначение causal mask

Правильный ответ: A) Чтобы исключить просмотр будущих токенов

Трассируемость: autoregression -> mat_0052 -> 05-autoregression -> Q22
Источник материала: themes/04-Autoregression/lab/02_decoder_only_tiny_shakespeare_gpu.ipynb

---

### Q23 - Декодирование и компромиссы

Правильные ответы: A) Баланс между разнообразием и устойчивостью; B) Риск повторов в длинной генерации; D) Требования к управляемости вывода

Трассируемость: autoregression -> mat_0053 -> 05-autoregression -> Q23
Источник материала: themes/04-Autoregression/lab/guides/00_autoregression_prerequisites.md

---

### Q24 - Стабилизация autoregression

Правильные ответы: A) Контроль learning rate; B) Мониторинг validation loss; D) Ограничение длины контекста под ресурсы

Трассируемость: autoregression -> mat_0054 -> 05-autoregression -> Q24
Источник материала: themes/04-Autoregression/lab/guides/01_decoder_only_toy_walkthrough.md

---

### Q25 - Train vs inference

Эталонный ответ: На обучении модель видит корректный префикс и учится прогнозировать следующий токен, на инференсе она опирается на собственные предсказания, что может накапливать ошибки.

Критерии оценивания:
- Отражены различия в доступном контексте
- Пояснён риск накопления ошибок на инференсе
- Ответ логично структурирован

Трассируемость: autoregression -> mat_0055 -> 05-autoregression -> Q25
Источник материала: themes/04-Autoregression/lab/guides/02_autoregression_debugging_playbook.md

---

## Раздел 6. Full Transformer

### Q26 - Cross-attention сигнал

Правильный ответ: A) Скрытые представления encoder

Трассируемость: full_transformer -> mat_0066 -> 06-full-transformer -> Q26
Источник материала: themes/05-Full-Transformer/lab/guides/00_full_transformer_prerequisites.md

---

### Q27 - Связь encoder и decoder

Правильный ответ: A) Через механизм cross-attention к encoder-представлениям

Трассируемость: full_transformer -> mat_0067 -> 06-full-transformer -> Q27
Источник материала: themes/05-Full-Transformer/lab/guides/01_full_transformer_walkthrough.md

---

### Q28 - Риски полного контура

Правильные ответы: A) Ошибки согласования между encoder и decoder; B) Деградация на длинных зависимостях; D) Рост вычислительной стоимости обучения

Трассируемость: full_transformer -> mat_0068 -> 06-full-transformer -> Q28
Источник материала: themes/05-Full-Transformer/lab/guides/02_full_transformer_debugging_playbook.md

---

### Q29 - Шаги инференса full Transformer

Правильные ответы: A) Кодирование входной последовательности; B) Пошаговое декодирование целевого выхода; C) Использование накопленного контекста decoder

Трассируемость: full_transformer -> mat_0069 -> 06-full-transformer -> Q29
Источник материала: themes/05-Full-Transformer/lab/solutions/01_full_transformer_wikitext2_solution.ipynb

---

### Q30 - Train/inference разрыв

Эталонный ответ: На обучении модель опирается на корректные целевые префиксы, на инференсе — на собственные предсказания, поэтому ошибки могут накапливаться и ухудшать дальние шаги.

Критерии оценивания:
- Корректно описан источник разрыва
- Показано влияние на качество последовательности
- Дана логичная аргументация

Трассируемость: full_transformer -> mat_0070 -> 06-full-transformer -> Q30
Источник материала: themes/05-Full-Transformer/theory/theory.md
