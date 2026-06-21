# Полный трансформер «кодировщик-декодировщик» (encoder-decoder): финальная лабораторная `ЛР05`

Место в линии курса:
- предыдущий шаг: `04-Autoregression / ЛР01 + ЛР02 (CPU/GPU)`;
- текущий шаг: `05-Full-Transformer / ЛР05`;
- цель: собрать и обучить полный трансформер для предсказания токенов продолжения текста.

## 1. Что меняется после темы 04

В теме `04` мы работали с декодерным режимом (decoder-only):

$$
P(y_{1:T}) = \prod_{t=1}^{T} P(y_t \mid y_{<t}).
$$

В `ЛР05` появляется входная последовательность `x` для кодировщика, и декодер прогнозирует цель с учётом `x`:

$$
P(y_{1:T} \mid x) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, x).
$$

Поэтому декодер использует два типа внимания:
1. самовнимание (self-attention) по уже известной части цели;
2. перекрёстное внимание (cross-attention) к выходам кодировщика.

## 2. Контракт данных

Для каждого окна текста фиксируем:
- `encoder_input = ids[i : i + SRC_LEN]`;
- `target = ids[i + SRC_LEN : i + SRC_LEN + TGT_LEN]`;
- `decoder_input = [SOS] + target[:-1]`;
- `decoder_target = target`.

Смысл: на шаге `t` декодер получает корректный предыдущий токен (teacher forcing) и предсказывает текущий.

## 3. Вывод функции потерь

### 3.1 Цепное правило вероятностей

$$
P(y_{1:T} \mid x) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, x).
$$

Логарифм:

$$
\log P(y_{1:T} \mid x) = \sum_{t=1}^{T} \log P(y_t \mid y_{<t}, x).
$$

### 3.2 Отрицательное лог-правдоподобие

Минимизируем отрицательное среднее по валидным позициям:

$$
\mathcal{L}_{\mathrm{NLL}} = -\frac{1}{T_{\mathrm{valid}}}
\sum_{t=1}^{T} m_t\,\log P(y_t \mid y_{<t}, x),
\quad
T_{\mathrm{valid}} = \sum_{t=1}^{T} m_t.
$$

При one-hot цели это перекрёстная энтропия (cross-entropy) на токенах.

### 3.3 Перплексия

$$
\mathrm{PPL} = e^{\mathcal{L}_{\mathrm{NLL}}}.
$$

Интерпретация: средний «эффективный объём неопределённости» модели на одном токене.

## 4. Причинная маска и отсутствие утечки в будущее

Пусть `S` — оценки внимания декодера до `softmax`.
Для запрещённых позиций `j > i` применяем маску:

$$
S'_{i,j} = -\infty, \quad j > i.
$$

После `softmax`:

$$
\alpha_{i,j} = 0, \quad j > i.
$$

Значит, выход на позиции `i` зависит только от позиций `\le i` и от выходов кодировщика в cross-attention.

## 5. Масштабирование оценок внимания

В скалярном произведении внимания используется делитель:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V.
$$

Почему нужен `1/\sqrt{d_k}`: при росте размерности `d_k` дисперсия `QK^\top` растёт, `softmax` становится слишком «острым», а градиенты деградируют. Делитель нормирует масштаб логитов.

## 6. Датасеты в starter и solution

В `ЛР05` специально используются разные корпуса:
- `starter`: `Tiny Shakespeare`;
- `solution`: `WikiText-2` (текстовые `train/valid/test` из GitHub-источника).

Это сделано методически:
1. `starter` остаётся контролируемым и прозрачным для самостоятельной работы;
2. `solution` показывает перенос той же архитектуры на более сложную текстовую статистику.

## 7. Профили выполнения

Оба ноутбука поддерживают единый выбор профиля:
- `CPU-friendly`: целевой бюджет `40-60` минут;
- `GPU-friendly`: целевой бюджет `30-45` минут.

Если выбран `GPU-friendly`, но графический процессор недоступен, тетрадь должна завершиться понятной ошибкой конфигурации без скрытого CPU-fallback.

## 8. Критерии завершения

### Starter (`Tiny Shakespeare`)
1. `test_perplexity < baseline_perplexity`;
2. `success_count >= 18` из `20`;
3. `mean_match_ratio >= 0.70`, где `mean_match_ratio` — доля точных (`argmax`) посимвольных совпадений по фиксированным `probes`;
4. диагностика внимания подтверждает отсутствие доступа к будущим позициям.

### Solution (`WikiText-2`)
1. `test_perplexity < baseline_perplexity`;
2. `success_count >= 16` из `20`;
3. `mean_match_ratio >= 0.60`, где `mean_match_ratio` считается как `top-k hit ratio` (эталонный токен попадает в `k` лучших предсказаний на каждом шаге);
4. диагностика внимания подтверждает отсутствие доступа к будущим позициям.

## 9. Навигация

- Практика: [../lab/README.md](../lab/README.md)
- Входной минимум: [../lab/guides/00_full_transformer_prerequisites.md](../lab/guides/00_full_transformer_prerequisites.md)
- Пошаговый разбор: [../lab/guides/01_full_transformer_walkthrough.md](../lab/guides/01_full_transformer_walkthrough.md)
- Диагностика: [../lab/guides/02_full_transformer_debugging_playbook.md](../lab/guides/02_full_transformer_debugging_playbook.md)

## 10. Расшифровка каждой закорючки: полный encoder-decoder контракт

Этот раздел добавлен как мост от формул к финальной лабораторной. Он сохраняет весь вывод выше и разбирает, что означает каждый tensor в `ЛР05`.

### 10.1 Что это значит словами

Full-transformer отвечает на вопрос: как decoder предсказывает продолжение, если у него есть отдельная source-память encoder.

Если совсем с нуля:
- encoder читает исходный фрагмент;
- decoder получает предыдущие target-токены;
- decoder предсказывает текущий target-токен;
- cross-attention позволяет decoder смотреть на encoder-память.

Если профессионально:
- модель оценивает условное распределение $P(y_{1:T}\mid x)$;
- decoder self-attention ограничивается causal mask;
- cross-attention ограничивается padding mask источника.

### 10.2 Что означает каждый символ

| Символ | Смысл | Shape-contract |
|---|---|---|
| $x$ | source-последовательность encoder | `(batch, src_len)` |
| $y_t$ | target-токен на шаге `t` | scalar token id |
| $y_{<t}$ | target-префикс до шага `t` | длина `t-1` |
| $m_t$ | валидность target-позиции | `0/1` |
| $Q_{dec}$ | decoder queries | `(batch, tgt_len, d_k)` |
| $K_{enc},V_{enc}$ | encoder memory | `(batch, src_len, d_k/d_v)` |
| $M$ | mask в attention logits | broadcast к attention score shape |
| $\mathcal{L}_{\mathrm{NLL}}$ | отрицательное log-likelihood | scalar |

### 10.3 Data-contract без двусмысленности

```text
encoder_input:
  ids[i : i + SRC_LEN]
  shape = (batch, SRC_LEN)

target:
  ids[i + SRC_LEN : i + SRC_LEN + TGT_LEN]
  shape = (batch, TGT_LEN)

decoder_input:
  [SOS] + target[:-1]
  shape = (batch, TGT_LEN)

decoder_target:
  target
  shape = (batch, TGT_LEN)
```

Правило: `decoder_input` — что decoder уже видит; `decoder_target` — что он обязан предсказать.

### 10.4 Мини-числовой пример teacher forcing

Если:

```text
target = [31, 9, 14, 2]
SOS = 1
```

то:

```text
decoder_input  = [1, 31, 9, 14]
decoder_target = [31, 9, 14, 2]
```

На позиции `0` decoder видит `SOS` и должен предсказать `31`.
На позиции `1` decoder видит `31` и должен предсказать `9`.

### 10.5 Разбор масок

```text
padding mask:
  выключает PAD в encoder/decoder токенах

causal mask:
  запрещает decoder self-attention смотреть на future target positions

cross-attention mask:
  запрещает decoder смотреть на PAD-позиции encoder_input
```

Если перепутать causal и cross mask, модель либо увидит будущее, либо потеряет доступ к source-памяти.

### 10.6 Где это в TODO

- TODO data contract: собрать `encoder_input`, `decoder_input`, `decoder_target`.
- TODO mask contract: проверить padding, causal и cross masks отдельно.
- TODO training: сравнить `test_perplexity` с `baseline_perplexity`.
- TODO generation: проверить `success_count` и `mean_match_ratio`.
- TODO diagnostics: подтвердить отсутствие future leakage.

### 10.7 Типичная ошибка и способ поймать её

Ошибка: сделать `decoder_input == decoder_target`.

Антидот:
1. Вывести один пример до обучения.
2. Проверить, что `decoder_input[0] == SOS`.
3. Проверить, что `decoder_input[1:] == decoder_target[:-1]`.
4. Только после этого запускать обучение.
