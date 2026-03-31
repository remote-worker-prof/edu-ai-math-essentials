# Transformer encoder: следующий шаг после attention

Место в линии курса:
- предыдущий шаг: `02-Attention / ЛР01`;
- текущие лабораторные этой темы: `03-Transformer / ЛР01`, `03-Transformer / ЛР02`;
- общий порядок курса: Шаг 5 и Шаг 6.

## 1. Что меняется после `cross-attention`

В `02-Attention / ЛР01` decoder смотрел на выходы encoder:

$$
\text{context}_t = \mathrm{Attention}(q_t^{dec}, K^{enc}, V^{enc})
$$

Это был пример `cross-attention`:
- запросы приходят из одной последовательности;
- память для просмотра приходит из другой.

Следующий логический шаг — разрешить последовательности смотреть на самих себя.

Так появляется `self-attention`:
- каждый токен строит свой `query`, `key`, `value`;
- токены оценивают полезность друг друга;
- вместо рекуррентного прохода модель сразу связывает далекие позиции через attention.

## 2. Scaled dot-product attention

Пусть матрицы признаков последовательности:

$$
X \in \mathbb{R}^{T \times d_{model}}
$$

Из них строятся:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

Далее считаются score'ы совместимости:

$$
S = \frac{QK^\top}{\sqrt{d_k}}
$$

После этого по каждой строке применяется `softmax`:

$$
A = \mathrm{softmax}(S)
$$

И формируется новый набор представлений:

$$
\mathrm{Attention}(Q, K, V) = AV
$$

Практический смысл:
- каждая строка `A` показывает, на какие позиции входа смотрит текущий токен;
- деление на $\sqrt{d_k}$ стабилизирует шкалу score'ов;
- выход имеет ту же временную длину `T`, но уже содержит контекст от других позиций.

## 3. Почему Transformer не может жить без позиции

`Self-attention` сам по себе не знает порядок токенов.

Если переставить токены местами, attention видит тот же набор векторов, но не понимает:
- кто был раньше;
- кто был позже;
- где начало и конец.

Поэтому к token embeddings добавляют позиционную информацию.

Есть два популярных варианта:
- `sinusoidal positional encoding`;
- обучаемый `positional embedding`.

В этой теме используется **learned positional embedding**, потому что он проще:
- легче читать в notebook;
- хорошо поддерживается Keras;
- достаточно для первого знакомства с Transformer encoder.

## 4. Multi-head attention

Один attention-head — это один способ смотреть на последовательность.

`Multi-head attention` запускает несколько голов параллельно:

$$
\mathrm{head}_i = \mathrm{Attention}(Q_i, K_i, V_i)
$$

Затем головы объединяются:

$$
\mathrm{MHA}(X) = \mathrm{Concat}(\mathrm{head}_1, \dots, \mathrm{head}_h)W_O
$$

Интуиция:
- одна голова может сильнее реагировать на локальные связи;
- другая — на дальние зависимости;
- третья — на важные служебные токены.

В учебных notebook'ах этого курса достаточно:
- `num_heads = 2`;
- маленькое `embed_dim`;
- один encoder block.

## 5. Transformer encoder block

Минимальный encoder block состоит из двух частей:

1. `MultiHeadAttention`
2. feed-forward sublayer

Обе части окружены residual connections и нормализацией:

$$
H_1 = \mathrm{LayerNorm}(X + \mathrm{MHA}(X))
$$

$$
H_2 = \mathrm{LayerNorm}(H_1 + \mathrm{FFN}(H_1))
$$

Где:

$$
\mathrm{FFN}(h) = W_2 \sigma(W_1 h + b_1) + b_2
$$

Практический смысл блока:
- attention перемешивает информацию между позициями;
- FFN дообрабатывает каждый временной шаг независимо;
- residual connections помогают обучению;
- `LayerNormalization` стабилизирует значения.

## 6. Padding mask

Если последовательности дополняются `PAD`-токенами, attention не должен считать их настоящими словами.

Иначе возникают проблемы:
- модель тратит внимание на пустые позиции;
- интерпретация heatmap становится шумной;
- pooled-представление ухудшается.

Поэтому в encoder block передаётся `padding mask`.

В Keras `MultiHeadAttention` ждёт булеву маску формы:

$$
(batch, T_{query}, T_{key})
$$

или форму, которую можно broadcast'ить до неё.

Для self-attention на padded последовательностях достаточно логики:
- `mask = (tokens != PAD)`;
- затем эта маска используется внутри attention;
- и та же маска учитывается при pooling.

## 7. Почему в этой теме только encoder-only Transformer

Полный Transformer обычно включает:
- encoder;
- decoder;
- masked self-attention в decoder;
- cross-attention decoder к encoder.

Но для первого блока это слишком плотный скачок.

Поэтому здесь мы ограничиваемся **encoder-only** версией:
- понятнее переход от attention к self-attention;
- проще увидеть роль positional embedding;
- проще держать CPU-friendly ноутбуки;
- можно сразу применить блок к классификации.

Логика курса становится такой:
- `seq2seq + attention` показал, как смотреть на другие позиции;
- `Transformer encoder` показывает, как сделать attention базовым механизмом обработки последовательности.

## 8. Как это выглядит в лабораторных

### `03-Transformer / ЛР01`
Synthetic toy-задача на порядок токенов.

Её смысл:
- один и тот же набор токенов может давать разные ответы;
- значит, модели мало знать только “какие токены были”;
- ей нужно знать **где** они были.

Эта лабораторная специально подчёркивает пользу positional embedding.

### `03-Transformer / ЛР02`
`IMDB` sentiment classification.

Здесь тот же encoder block переносится на реальные тексты:
- вход становится длиннее;
- метка остаётся одна на всю последовательность;
- self-attention уже работает не на synthetic tokens, а на реальных review-последовательностях.

## 9. Что должно остаться после темы

- `self-attention` — это способ токенов смотреть друг на друга внутри одной последовательности;
- positional embedding нужен, чтобы Transformer не терял порядок;
- `multi-head attention` — это несколько параллельных способов читать контекст;
- encoder block = attention + feed-forward + residual + normalization;
- следующий логический шаг после этой темы — блок [../../04-Autoregression/lab/README.md](../../04-Autoregression/lab/README.md) с decoder-only/causal Transformer, а затем полный encoder-decoder Transformer.

## 10. Навигация по курсу

### Что было до этого
`02-Attention / ЛР01` — `cross-attention` как мостик от `seq2seq` к Transformer.

### Что сейчас
- `03-Transformer / ЛР01` — order-sensitive toy classification.
- `03-Transformer / ЛР02` — `IMDB` classification на Transformer encoder.

### Что дальше
После этого блока можно отдельно изучать:
- [../../04-Autoregression/lab/README.md](../../04-Autoregression/lab/README.md) — decoder-only Transformer и causal mask;
- полный encoder-decoder Transformer;
- более крупные текстовые модели и языковое моделирование.
