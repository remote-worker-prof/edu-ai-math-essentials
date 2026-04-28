# 03-Transformer Theory: Self-Attention, Position, Masks Для Новичка

## Кому читать
- Тем, кто освоил attention и хочет понять, что меняется в transformer encoder.
- Тем, кто путает self-attention и cross-attention.
- Тем, кому нужен мягкий вход в positional encoding и mask semantics.

## Интуиция на пальцах
Transformer encoder заменяет рекуррентный проход на слой внимания:
каждый токен смотрит на другие токены в той же последовательности.

Две ключевые идеи:
1. `self-attention` связывает позиции напрямую.
2. `positional` информация возвращает модели чувство порядка.

## Контракт данных
- `tokens.shape = (batch, time)`
- `embeddings.shape = (batch, time, d_model)`
- `padding_mask.shape = (batch, time)`
- `attention_scores.shape = (batch, heads, time, time)` или `(batch, time, time)`

Смысл маски:
- `padding_mask` выключает пустые позиции `PAD`,
- в encoder нет causal-запрета на будущее (он нужен в decoder-темах позже).

## Формализация (минимум формул)
Линейные проекции:

`Q = XW_Q, K = XW_K, V = XW_V`

Scaled dot-product:

`Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + M) V`

Encoder-блок:

`H1 = LayerNorm(X + MHA(X))`

`H2 = LayerNorm(H1 + FFN(H1))`

Это ядро обеих лабораторных темы.

## Ручной мини-пример
Пусть токены:

```text
tokens =
[
  [8, 4, 9, 0],
  [3, 1, 7, 5],
]

padding_mask =
[
  [1, 1, 1, 0],
  [1, 1, 1, 1],
]
```

Для первого объекта последняя позиция `PAD`, поэтому в attention она не должна получать значимые веса как ключ.

## Где это в TODO
- TODO по positional embedding: добавить/проверить позиционную информацию.
- TODO по attention: проверить форму `attention_scores`.
- TODO по маске: убедиться, что `padding_mask` применяется корректно.
- TODO по диагностике: визуально проверить карту внимания и интерпретацию осей.

## Типичные ошибки
- Считать, что self-attention уже автоматически знает порядок без positional признаков.
- Пропускать маскирование `PAD` и получать шумные веса внимания.
- Путать `encoder-only` с full-transformer.
- Сравнивать только accuracy без проверки attention-диагностики.

## Мини-чеклист
- Я понимаю разницу между self-attention и cross-attention.
- Я могу объяснить, зачем нужен делитель `sqrt(d_k)`.
- Я знаю, где в ноутбуке проверяется `padding_mask`.
- Я готов к decoder-only шагу:
  [../../04-Autoregression/theory/theory.md](../../04-Autoregression/theory/theory.md).
