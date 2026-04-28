# 05-Full-Transformer Theory: Строгий Encoder-Decoder Контракт Для Финальной ЛР

## Кому читать
- Тем, кто дошёл до финальной темы и хочет собрать целостную картину.
- Тем, кто путает `encoder_input`, `decoder_input`, `decoder_target`.
- Тем, кому нужна строгая логика `padding/causal/cross` масок в одном месте.

## Интуиция на пальцах
В full-transformer decoder предсказывает цель, опираясь на два источника:
1. прошлые токены самой цели (decoder self-attention),
2. память encoder по входной последовательности (cross-attention).

Поэтому data-contract здесь строже, чем в предыдущих темах.

## Контракт данных
- `encoder_input.shape = (batch, src_len)`
- `decoder_input.shape = (batch, tgt_len)` (обычно `SOS + target[:-1]`)
- `decoder_target.shape = (batch, tgt_len)` или `(batch, tgt_len, 1)`

Контракт масок:
- `padding_mask_encoder`: скрывает `PAD` на стороне source.
- `causal_mask_decoder`: запрещает доступ decoder к будущим позициям target.
- `cross_attention_mask`: разрешает decoder смотреть только на валидные source-позиции.

## Формализация (минимум формул)
Условное цепное правило:

`P(y_1...y_T | x) = Π_t P(y_t | y_<t, x)`

Attention в decoder self-attention:

`softmax(Q_dec K_dec^T / sqrt(d_k) + causal_mask)`

Attention в cross-attention:

`softmax(Q_dec K_enc^T / sqrt(d_k) + cross_mask)`

Loss и perplexity:

`L = token_cross_entropy`

`PPL = exp(L)`

## Ручной мини-пример
Пусть:

```text
encoder_input = [12, 7, 5, 0, 0]
decoder_target = [31, 9, 14, 2]
decoder_input  = [SOS, 31, 9, 14]
```

Тогда:
- decoder шаг `t=2` может смотреть только на `SOS,31,9` (causal),
- и на валидные source-токены `12,7,5` (cross + padding).

## Где это в TODO
- TODO по сборке датасета: правильно сформировать три входа/цель.
- TODO по маскам: проверить causal/padding/cross контракты.
- TODO по обучению: baseline/perplexity/generation gates.
- TODO по диагностике: убедиться, что внимания в будущее нет.

## Типичные ошибки
- Сдвигают `decoder_target` неверно и теряют teacher-forcing логику.
- Используют только одну маску там, где нужны несколько.
- Проверяют только `test_perplexity`, игнорируя generation gates.
- Не валидируют форму входов до `model.fit`.

## Мини-чеклист
- Я могу устно объяснить роль каждой из трёх масок.
- Я знаю, почему `decoder_input` и `decoder_target` не совпадают.
- Я проверяю и `perplexity`, и generation-quality gates.
- Я умею связать финальную ЛР с предыдущими шагами курса:
  [../../00-Foundations/theory/theory.md](../../00-Foundations/theory/theory.md).
