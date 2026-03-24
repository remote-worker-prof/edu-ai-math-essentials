# Dataset Card: English-Spanish Pairs (`spa-eng`)

## Тип задачи
`seq2seq` translation with `cross-attention`.

## Подходящие архитектуры
- encoder-decoder `GRU`
- encoder-decoder `LSTM`
- `seq2seq + attention`

## Загрузка
Через архив англо-испанских пар из корпуса `manythings/Anki`, как в официальном TensorFlow tutorial.

## Ожидаемые формы
- `encoder_input -> (N, T_in)`
- `decoder_input -> (N, T_out)`
- `decoder_target -> (N, T_out)`
- `attention_scores -> (N, T_out, T_in)`

## CPU-friendly subset
- `1000..3000` пар предложений;
- короткие фразы;
- словарный лимит для fast-mode;
- `2..4` эпохи на компактной модели.

## Главный учебный вывод
Attention нужен не только в synthetic reverse-задаче: на реальных парных последовательностях он даёт интерпретируемую карту того, как decoder соотносит выход с входом.

## Следующий шаг
После понимания этой карты внимания логичный следующий мостик — облегчённый `Transformer`.
