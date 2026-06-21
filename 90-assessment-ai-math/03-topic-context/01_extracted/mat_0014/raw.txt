# Dataset Card: Shakespeare

## Тип задачи
Character-level next-token prediction and text generation.

## Подходящие архитектуры
- `GRU`
- `LSTM`
- stacked recurrent models

## Загрузка
```python
path_to_file = tf.keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt",
)
```

## Ожидаемые формы
- после векторизации и нарезки:
  - `input -> (N, T)`
  - `target -> (N, T)`
- logits:
  - `(N, T, V)`

## CPU-friendly subset
- укороченный текстовый корпус;
- `sequence_length=80..100`;
- `embedding_dim=64`;
- `GRU(128)` и `1..2` эпохи для демонстрации формы и sampling loop.

## Главный учебный вывод
Здесь `seq2seq` упрощается до задачи предсказания следующего символа, а recurrent state становится особенно наглядным в режиме генерации.

## Следующий шаг
Подходит как мостик от классификации к генеративным sequence-задачам.
