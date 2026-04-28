# edu-ai-math-essentials

Учебный репозиторий по математическим основам ИИ с последовательным маршрутом:
`00-Foundations -> 01-RNN -> 02-Attention -> 03-Transformer -> 04-Autoregression -> 05-Full-Transformer`.

## Локальный quality-маршрут

```bash
python3 scripts/validate_notebooks.py
python3 scripts/check_notebook_contracts.py
python3 scripts/check_lab_quality_contracts.py
python3 scripts/check_runtime_gpu_contracts.py
git diff --check
git status --short --branch
```

Проверки выше не запускают тяжелое обучение и валидируют только структуру,
контракты и учебную согласованность материалов.
