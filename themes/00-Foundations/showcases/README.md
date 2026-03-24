# Real-Data Showcases for Sequence Models

## Назначение
Этот каталог нужен для второго круга обучения:
- сначала понять synthetic labs;
- потом увидеть те же идеи на реальных датасетах;
- после этого уже обсуждать масштабирование и более современные архитектуры.

Showcase notebook'и специально сделаны в fast-mode:
- маленькие подвыборки;
- компактные модели;
- CPU-friendly режим;
- фокус на одном учебном выводе, а не на максимальном качестве.

## Первая Волна: Runnable Demos
1. [01_imdb_many_to_one_showcase.ipynb](./01_imdb_many_to_one_showcase.ipynb) — `many-to-one` sentiment classification на `IMDB`, сравнение `SimpleRNN`, `LSTM`, `GRU`.
2. [02_jena_climate_lstm_gru_showcase.ipynb](./02_jena_climate_lstm_gru_showcase.ipynb) — multivariate forecasting на `Jena Climate`, сравнение `LSTM` и `GRU`.
3. [03_spa_eng_seq2seq_attention_showcase.ipynb](./03_spa_eng_seq2seq_attention_showcase.ipynb) — маленький `seq2seq + attention` на англо-испанских парах и heatmap для одного примера.

## Вторая Волна: Dataset Cards
- [cards/01_imdb.md](./cards/01_imdb.md)
- [cards/02_jena_climate.md](./cards/02_jena_climate.md)
- [cards/03_spa_eng.md](./cards/03_spa_eng.md)
- [cards/04_reuters.md](./cards/04_reuters.md)
- [cards/05_uci_har.md](./cards/05_uci_har.md)
- [cards/06_shakespeare.md](./cards/06_shakespeare.md)
- [cards/07_forda.md](./cards/07_forda.md)

## Как использовать showcase-трек
1. Не идти сюда до того, как понятны формы и synthetic labs.
2. Для каждого notebook сначала читать dataset card.
3. Не пытаться “выжать максимум метрики” на fast-mode подвыборке.
4. Смотреть на то, как меняются данные, формы и диагностика при переходе от toy-примеров к real data.

## Единый Шаблон Чтения
Для каждого датасета фиксируйте:
- тип задачи;
- базовую форму входа и цели;
- какие архитектуры здесь естественны;
- какой CPU-friendly subset достаточен для демонстрации;
- какой один главный методический вывод вы хотите показать студенту.
