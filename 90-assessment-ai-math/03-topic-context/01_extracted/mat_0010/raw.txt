# Dataset Card: Jena Climate

## Тип задачи
Multivariate time-series forecasting.

## Подходящие архитектуры
- `LSTM`
- `GRU`
- stacked recurrent models

## Загрузка
```python
zip_path = tf.keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
    fname="jena_climate_2009_2016.csv.zip",
    extract=True,
)
```

## Ожидаемые формы
- сырые признаки после hourly sub-sampling: таблица `(time, features)`;
- после нарезки окон: `X -> (N, T, F)`;
- для single-step forecast: `y -> (N,)`.

## CPU-friendly subset
- первые `2500..5000` часов после subsampling;
- окно `24` или `48` шагов;
- `1..2` эпохи;
- `LSTM(32)` и `GRU(32)` без лишней глубины.

## Главный учебный вывод
RNN-подходы одинаково естественно работают не только на текстах, но и на числовых мультифичевых временных рядах, где особенно важно читать форму `(batch, time, features)`.

## Следующий шаг
Сравнить этот сценарий с `UCI HAR`, если хочется перейти от forecasting к time-series classification.
