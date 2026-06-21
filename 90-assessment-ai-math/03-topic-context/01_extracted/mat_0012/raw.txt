# Dataset Card: Reuters

## Тип задачи
`many-to-one` multiclass topic classification.

## Подходящие архитектуры
- `Embedding + LSTM`
- `Embedding + GRU`
- простые bag-of-words baselines для сравнения

## Загрузка
```python
from tensorflow import keras
(x_train, y_train), (x_test, y_test) = keras.datasets.reuters.load_data(num_words=10000)
```

## Ожидаемые формы
- после padding: `X -> (N, T)`
- классы: `y -> (N,)`
- число классов: `46`

## CPU-friendly subset
- `train_subset=3000..5000`
- `test_subset=1000..2000`
- `maxlen=120..160`
- `1..2` эпохи на маленькой модели

## Главный учебный вывод
По сравнению с `IMDB` меняется не логика последовательного кодирования, а тип цели: вместо бинарной метки появляется multiclass head с `softmax`.

## Следующий шаг
Использовать после `IMDB`, если нужна более содержательная классификация новостных тем.
