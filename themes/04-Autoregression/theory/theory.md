# 04-Autoregression Theory: Причинная Цепочка От Causal Mask До Perplexity

## Кому читать
- Тем, кто впервые переходит от классификации к генерации токенов.
- Тем, кто хочет понять `leakage` и зачем нужен строгий causal mask.
- Тем, кто работает с CPU/GPU-профилями и боится скрытого fallback.

## Интуиция на пальцах
В авторегрессии модель предсказывает следующий токен только по прошлому контексту.
Если модель видит будущее, задача становится нечестной, а метрики теряют смысл.

Поэтому в теме всегда держим причинную цепочку:
`causal mask -> no leakage -> честная cross-entropy -> честная perplexity -> корректные generation gates`.

## Контракт данных
- `token_windows.shape = (batch, context)`
- `targets.shape = (batch, context)`
- `padding_mask.shape = (batch, context)`
- `causal_mask.shape = (context, context)` и это нижнетреугольная матрица

Runtime-контракт:
- `CPU` и `GPU` маршруты разделены явно;
- при выбранном GPU-профиле скрытый CPU fallback запрещён;
- `gpu_preflight` обязателен до длинного запуска.

## Формализация (минимум формул)
Цепное правило:

`P(x_1...x_T) = Π_t P(x_t | x_<t)`

Оптимизируем среднюю token-level cross-entropy:

`L = -mean_t log P(x_t | x_<t)`

Перплексия:

`PPL = exp(L)`

Если `causal_mask` корректен и leakage нет, `PPL` и generation-gates можно интерпретировать как честную оценку качества модели.

## Ручной мини-пример
Для `context=4` causal mask:

```text
[
  [1, 0, 0, 0],
  [1, 1, 0, 0],
  [1, 1, 1, 0],
  [1, 1, 1, 1],
]
```

Строка `i` показывает, на какие позиции можно смотреть при предсказании токена в позиции `i`.

## Где это в TODO
- TODO по маске: построить и проверить causal mask.
- TODO по leakage-check: убедиться, что внимание в будущее блокировано.
- TODO по метрикам: baseline/perplexity/generation gates.
- TODO по GPU: пройти `gpu_preflight` и соблюдать runtime budget.

## Типичные ошибки
- Путают `padding_mask` и `causal_mask`.
- Оценивают качество только по train-loss.
- Считают любое GPU-ускорение допустимым, даже со скрытым CPU fallback.
- Интерпретируют generation без baseline/perplexity-контекста.

## Мини-чеклист
- Я могу объяснить, почему leakage ломает смысл метрик.
- Я умею проверить нижнетреугольность causal mask.
- Я понимаю разницу между CPU- и GPU-контрактами в этой теме.
- Я готов к full-transformer:
  [../../05-Full-Transformer/theory/theory.md](../../05-Full-Transformer/theory/theory.md).
