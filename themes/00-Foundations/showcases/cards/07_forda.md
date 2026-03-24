# Dataset Card: FordA

## Тип задачи
Univariate time-series classification.

## Подходящие архитектуры
- `GRU`
- `LSTM`
- 1D CNN
- lightweight Transformer encoder как future bridge

## Загрузка
Через TSV-файлы из UCR archive, как в Keras example.

## Ожидаемые формы
- сырые series: `(N, 500)`
- после добавления channel axis: `X -> (N, 500, 1)`
- метки: `y -> (N,)`

## CPU-friendly subset
- полный train/test уже сравнительно компактный;
- для fast-mode можно брать `1500..2500` train examples;
- `1..3` эпохи на небольшой модели.

## Главный учебный вывод
Даже простой одномерный sensor-like ряд естественно ложится в shape `(batch, time, features)` и позволяет сравнивать recurrent и non-recurrent sequence-модели на одном и том же интерфейсе.

## Следующий шаг
Использовать как мостик к более широкому разговору о sequence-моделях за пределами чистых RNN.
