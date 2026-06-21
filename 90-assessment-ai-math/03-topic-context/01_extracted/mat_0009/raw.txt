# Dataset Card: IMDB

## Тип задачи
`many-to-one` binary text classification.

## Подходящие архитектуры
- `Embedding + SimpleRNN`
- `Embedding + LSTM`
- `Embedding + GRU`
- bidirectional RNN как следующий шаг

## Загрузка
```python
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=8000)
```

## Ожидаемые формы
- после загрузки: список последовательностей разной длины;
- после padding: `X -> (N, T)`;
- метки: `y -> (N,)`.

## CPU-friendly subset
- `num_words=8000`
- `maxlen=120..160`
- `train_subset=2000..4000`
- `test_subset=1000..2000`
- `1..2` эпохи на маленьких моделях

## Главный учебный вывод
Один и тот же `many-to-one` формат переносится с synthetic token sequences на реальные тексты почти без изменения общей логики: токены, embedding, одна итоговая метка на всю последовательность.

## Следующий шаг
Перейти к `Reuters`, если нужна multiclass-классификация, или к `spa-eng`, если нужен уже `seq2seq`.
