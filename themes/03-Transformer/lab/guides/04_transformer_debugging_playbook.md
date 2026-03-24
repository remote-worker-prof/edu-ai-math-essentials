# Debugging Playbook: `03-Transformer`

## Зачем нужен этот файл
В Transformer-ноутбуках студенты часто застревают не на общей идее, а на одной из трёх конкретных зон:
- формы attention;
- positional embedding;
- padding mask.

Поэтому здесь полезнее не общий совет “проверь код”, а короткий порядок диагностики.

## Самый надёжный режим работы
1. Запускать notebook сверху вниз.
2. После правок в середине использовать `Restart & Run All`.
3. Сначала проверять формы, потом метрики.
4. Сначала проверять mask, потом heatmap.

## Shape-first маршрут
Минимум, который стоит распечатать:

```python
print(X_batch.shape)
print(mask.shape)
print(embedded_batch.shape)
print(attention_scores.shape)
```

Для encoder-only Transformer ожидается:
- `tokens -> (batch, T)`
- `mask -> (batch, T)`
- `embeddings -> (batch, T, E)`
- `attention_scores -> (batch, heads, T, T)`

## Где искать ошибку
### 1. Данные
Признаки:
- метрики около случайного уровня;
- ручной пример противоречит label;
- `PAD` не равен нулю или padded хвост выглядит как обычные токены.

Что делать:
- вывести пару примеров;
- вручную проверить rule label;
- проверить длины до и после padding.

### 2. Positional embedding
Признаки:
- модель плохо различает перестановки;
- attention выглядит разумно, но порядок всё равно теряется;
- output shape после embedding не совпадает с `(batch, T, E)`.

Что делать:
- проверить, что позиционный индекс создаётся по длине `T`;
- проверить, что token embedding и position embedding имеют одинаковый `embed_dim`;
- проверить, что они складываются, а не конкатенируются.

### 3. Attention mask
Признаки:
- attention смотрит в padded хвост;
- pooled output зависит от числа `PAD`;
- heatmap шумит на последних позициях.

Что делать:
- проверить `mask = (tokens != 0)`;
- проверить форму mask перед `MultiHeadAttention`;
- проверить, что та же mask участвует в pooling.

### 4. Classifier head
Признаки:
- `ValueError` на `fit`;
- shape logits не совпадает с `y`;
- loss не соответствует финальному слою.

Что делать:
- для binary classification держать один sigmoid-logit или эквивалентный binary head;
- сверить форму `y`;
- отдельно проверить pooled vector shape.

## Минимальный маршрут восстановления
1. `Restart & Run All`.
2. Проверить shapes.
3. Проверить один ручной пример.
4. Проверить mask и padded хвост.
5. Проверить `attention_scores`.
6. Только потом сверяться с solution notebook.

## Как пользоваться solution notebook
Правильный порядок:
1. Своя попытка.
2. Walkthrough для текущего блока.
3. Повторная попытка.
4. Только после этого solution notebook.

Неправильный порядок:
- открыть полное решение;
- скопировать custom layer;
- получить зелёные ячейки без понимания, зачем нужен positional embedding или mask.

## Что должно остаться
- привычка проверять не только `accuracy`, но и `attention_scores`;
- понимание, что большинство transformer-ошибок сидит в shapes и masking;
- спокойный порядок диагностики: данные -> embedding -> mask -> attention -> pooling -> metrics.
