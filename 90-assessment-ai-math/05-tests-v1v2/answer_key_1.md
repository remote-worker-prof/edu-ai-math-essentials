# Вариант 1. Ключ преподавателя по дисциплине «Математические основы ИИ»

---

## Раздел 1. Foundations

### Q1 - Назначение padding в последовательностях

Правильный ответ: A) Сделать длины последовательностей одинаковыми в батче

Трассируемость: foundations -> mat_0002 -> 01-foundations -> Q1
Источник материала: themes/00-Foundations/examples/01_numpy_sequence_basics.ipynb

---

### Q2 - Форма тензора входа

Правильный ответ: A) (batch_size, seq_len)

Трассируемость: foundations -> mat_0003 -> 01-foundations -> Q2
Источник материала: themes/00-Foundations/examples/02_minimal_keras_sequence_classifier.ipynb

---

### Q3 - Базовые метрики качества

Правильные ответы: A) Accuracy; B) Precision/Recall; D) F1-score

Трассируемость: foundations -> mat_0004 -> 01-foundations -> Q3
Источник материала: themes/00-Foundations/examples/03_tokenization_padding_masking.ipynb

---

### Q4 - Токенизация и устойчивость данных

Правильные ответы: A) Повышают воспроизводимость подготовки данных; B) Снижают неоднозначность разбора текста; D) Делают сравнение экспериментов более корректным

Трассируемость: foundations -> mat_0005 -> 01-foundations -> Q4
Источник материала: themes/00-Foundations/examples/04_attention_heatmap_toy.ipynb

---

### Q5 - Нормализация входа

Эталонный ответ: Единый протокол исключает различия, вызванные предобработкой, и делает сравнение моделей по метрикам корректным.

Критерии оценивания:
- Упомянуты токенизация и padding как источник систематических различий
- Пояснено влияние на воспроизводимость и сопоставимость метрик
- Ответ структурирован и логически завершён

Трассируемость: foundations -> mat_0006 -> 01-foundations -> Q5
Источник материала: themes/00-Foundations/showcases/01_imdb_many_to_one_showcase.ipynb

---

## Раздел 2. RNN

### Q6 - Преимущество LSTM

Правильный ответ: A) Наличие механизма ворот для управления памятью

Трассируемость: rnn -> mat_0017 -> 02-rnn -> Q6
Источник материала: themes/01-RNN/lab/01_simple_rnn_many_to_one_toy.ipynb

---

### Q7 - Смысл hidden state

Правильный ответ: A) Сжатое представление предыдущего контекста

Трассируемость: rnn -> mat_0018 -> 02-rnn -> Q7
Источник материала: themes/01-RNN/lab/02_lstm_many_to_many_toy.ipynb

---

### Q8 - Компоненты базового seq2seq

Правильные ответы: A) Encoder; B) Decoder; C) Рекуррентная передача состояния

Трассируемость: rnn -> mat_0019 -> 02-rnn -> Q8
Источник материала: themes/01-RNN/lab/03_gru_seq2seq_reverse_toy.ipynb

---

### Q9 - Трудности обучения RNN

Правильные ответы: A) Затухание/взрыв градиентов; B) Слишком длинные последовательности; D) Неподходящий learning rate

Трассируемость: rnn -> mat_0020 -> 02-rnn -> Q9
Источник материала: themes/01-RNN/lab/guides/00_prerequisites_and_notation.md

---

### Q10 - Выбор между GRU и LSTM

Эталонный ответ: GRU часто выбирают как более лёгкую модель для ограниченных ресурсов, LSTM — когда критично более гибко контролировать память на длинном контексте.

Критерии оценивания:
- Сопоставлены вычислительные и качественные аспекты
- Есть привязка к типу данных/длине контекста
- Дан аргументированный вывод

Трассируемость: rnn -> mat_0021 -> 02-rnn -> Q10
Источник материала: themes/01-RNN/lab/guides/00_self_study_debugging_playbook.md

---

## Раздел 3. Attention

### Q11 - Смысл attention-весов

Правильный ответ: A) Относительную важность элементов контекста для текущего шага

Трассируемость: attention -> mat_0033 -> 03-attention -> Q11
Источник материала: themes/02-Attention/lab/01_gru_seq2seq_attention_reverse_toy.ipynb

---

### Q12 - Alignment

Правильный ответ: A) Связывает позиции входа и выхода при генерации

Трассируемость: attention -> mat_0034 -> 03-attention -> Q12
Источник материала: themes/02-Attention/lab/guides/00_attention_prerequisites.md

---

### Q13 - Преимущества внимания

Правильные ответы: A) Более явная работа с релевантным контекстом; B) Улучшение интерпретируемости поведения; D) Снижение риска потери важной информации

Трассируемость: attention -> mat_0035 -> 03-attention -> Q13
Источник материала: themes/02-Attention/lab/guides/01_gru_seq2seq_attention_beginner.md

---

### Q14 - Ограничения attention-анализа

Правильные ответы: A) Вес не равен строгой причинности; B) Качество интерпретации зависит от контекста задачи; D) Нужна связь с метриками и ошибками

Трассируемость: attention -> mat_0036 -> 03-attention -> Q14
Источник материала: themes/02-Attention/lab/guides/02_attention_walkthrough.md

---

### Q15 - Чтение attention-карт

Эталонный ответ: Нужно сопоставить шаг выхода с входными позициями, выделить зоны максимального веса, сравнить с ожидаемыми связями и проверить согласованность с ошибками/метриками.

Критерии оценивания:
- Есть поэтапная процедура анализа
- Упомянуто сопоставление с ожидаемыми зависимостями
- Добавлена проверка по качественным/количественным сигналам

Трассируемость: attention -> mat_0037 -> 03-attention -> Q15
Источник материала: themes/02-Attention/lab/guides/03_attention_debugging_playbook.md

---

## Раздел 4. Transformer Encoder

### Q16 - Роль positional encoding

Правильный ответ: A) Чтобы модель учитывала порядок токенов

Трассируемость: transformer_encoder -> mat_0040 -> 04-transformer-encoder -> Q16
Источник материала: themes/03-Transformer/lab/01_transformer_encoder_order_toy.ipynb

---

### Q17 - Смысл self-attention

Правильный ответ: A) Оценивать взаимосвязи между позициями входа

Трассируемость: transformer_encoder -> mat_0041 -> 04-transformer-encoder -> Q17
Источник материала: themes/03-Transformer/lab/02_transformer_encoder_imdb.ipynb

---

### Q18 - Состав encoder-блока

Правильные ответы: A) Multi-head self-attention; B) Feed-forward sublayer; C) Residual connections и нормализация

Трассируемость: transformer_encoder -> mat_0042 -> 04-transformer-encoder -> Q18
Источник материала: themes/03-Transformer/lab/guides/00_transformer_prerequisites.md

---

### Q19 - Masking в encoder

Правильные ответы: A) Когда в батче есть последовательности разной длины; B) Когда нужно исключить вклад padding-позиций; D) Когда важно не искажать attention-распределение

Трассируемость: transformer_encoder -> mat_0043 -> 04-transformer-encoder -> Q19
Источник материала: themes/03-Transformer/lab/guides/01_self_attention_and_positional_encoding_beginner.md

---

### Q20 - Пайплайн encoder

Эталонный ответ: Токены переводятся в эмбеддинги, добавляется позиционная информация, затем проходят через стек encoder-блоков self-attention + FFN + residual/normalization.

Критерии оценивания:
- Перечислены ключевые этапы пайплайна
- Корректно описана роль позиционной информации
- Указана блочная структура encoder

Трассируемость: transformer_encoder -> mat_0044 -> 04-transformer-encoder -> Q20
Источник материала: themes/03-Transformer/lab/guides/02_transformer_encoder_toy_walkthrough.md

---

## Раздел 5. Autoregression

### Q21 - Назначение causal mask

Правильный ответ: A) Чтобы исключить просмотр будущих токенов

Трассируемость: autoregression -> mat_0050 -> 05-autoregression -> Q21
Источник материала: themes/04-Autoregression/lab/01_decoder_only_causal_toy.ipynb

---

### Q22 - Цель next-token prediction

Правильный ответ: A) Предсказать следующий токен по предыдущему контексту

Трассируемость: autoregression -> mat_0051 -> 05-autoregression -> Q22
Источник материала: themes/04-Autoregression/lab/02_decoder_only_tiny_shakespeare.ipynb

---

### Q23 - Стабилизация autoregression

Правильные ответы: A) Контроль learning rate; B) Мониторинг validation loss; D) Ограничение длины контекста под ресурсы

Трассируемость: autoregression -> mat_0052 -> 05-autoregression -> Q23
Источник материала: themes/04-Autoregression/lab/02_decoder_only_tiny_shakespeare_gpu.ipynb

---

### Q24 - Показатели качества генерации

Правильные ответы: A) Повторяемость фрагментов; B) Потеря связности контекста; D) Рост ошибок на валидационном наборе

Трассируемость: autoregression -> mat_0053 -> 05-autoregression -> Q24
Источник материала: themes/04-Autoregression/lab/guides/00_autoregression_prerequisites.md

---

### Q25 - Диагностика повторов

Эталонный ответ: Проверить параметры декодирования, динамику loss, длину контекста, затем скорректировать стратегию сэмплирования и регуляризацию с повторной оценкой качества.

Критерии оценивания:
- Указаны источники проблемы на уровне обучения и инференса
- Есть конкретные шаги корректировки
- План включает повторную проверку метрик/примеров

Трассируемость: autoregression -> mat_0054 -> 05-autoregression -> Q25
Источник материала: themes/04-Autoregression/lab/guides/01_decoder_only_toy_walkthrough.md

---

## Раздел 6. Full Transformer

### Q26 - Связь encoder и decoder

Правильный ответ: A) Через механизм cross-attention к encoder-представлениям

Трассируемость: full_transformer -> mat_0065 -> 06-full-transformer -> Q26
Источник материала: themes/05-Full-Transformer/lab/01_full_transformer_tiny_shakespeare.ipynb

---

### Q27 - Teacher forcing

Правильный ответ: A) Для стабилизации обучения по целевой последовательности

Трассируемость: full_transformer -> mat_0066 -> 06-full-transformer -> Q27
Источник материала: themes/05-Full-Transformer/lab/guides/00_full_transformer_prerequisites.md

---

### Q28 - Шаги инференса full Transformer

Правильные ответы: A) Кодирование входной последовательности; B) Пошаговое декодирование целевого выхода; C) Использование накопленного контекста decoder

Трассируемость: full_transformer -> mat_0067 -> 06-full-transformer -> Q28
Источник материала: themes/05-Full-Transformer/lab/guides/01_full_transformer_walkthrough.md

---

### Q29 - Диагностика качества

Правильные ответы: A) Разбор ошибок по типам примеров; B) Сравнение train/validation динамики; D) Анализ стабильности на длинных последовательностях

Трассируемость: full_transformer -> mat_0068 -> 06-full-transformer -> Q29
Источник материала: themes/05-Full-Transformer/lab/guides/02_full_transformer_debugging_playbook.md

---

### Q30 - Оценивание полного контура

Эталонный ответ: Следует учитывать точность на валидации, устойчивость на сложных/длинных примерах, согласованность выхода с входным контекстом и повторяемость результатов.

Критерии оценивания:
- Есть количественные и качественные критерии
- Покрыты устойчивость и согласованность
- Критерии применимы к учебному контексту

Трассируемость: full_transformer -> mat_0069 -> 06-full-transformer -> Q30
Источник материала: themes/05-Full-Transformer/lab/solutions/01_full_transformer_wikitext2_solution.ipynb
