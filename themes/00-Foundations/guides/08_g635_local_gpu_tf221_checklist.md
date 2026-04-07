# G635 Local GPU Checklist (TensorFlow 2.21) — для клона с G615

Цель: быстро поднять локальный GPU-контур на новом ноутбуке (класс `RTX 50xx`) и подтвердить запуск хотя бы одного показательного `solution`-ноутбука из `ЛР 04`.

## 0) Быстрый смысл
- Базовый курс держит `tensorflow>=2.16,<2.20` для совместимости.
- Для `RTX 50xx` в теме `04` зафиксирован расширенный recovery: `tensorflow==2.21.0` + `nvidia-cuda-nvcc-cu12>=12.8`.
- Критично: в GPU-ноутбуках `ЛР 04` есть runtime-предзагрузка CUDA-библиотек из `.venv` до импорта TensorFlow.

## Быстрый one-command путь
Если хотите без ручной пошаговой возни, используйте helper-скрипт:

```bash
themes/04-Autoregression/lab/scripts/setup_local_gpu_tf221.sh
```

Он:
- ставит базовые зависимости `ЛР04` без tensorflow-downgrade;
- фиксирует `tensorflow[and-cuda]==2.21.0` и `nvidia-cuda-nvcc-cu12>=12.8`;
- регистрирует Jupyter kernel `students-ai-gpu-tf221`.

## 1) Preflight железа
- [ ] Проверить, что драйвер и GPU видны:

```bash
nvidia-smi -L
nvidia-smi
```

Ожидается видимый `GPU 0`.

## 2) Подготовить окружение проекта
Из корня репозитория:

- [ ] (если `.venv` ещё нет)
```bash
python3 -m venv .venv
```

- [ ] Базовые зависимости курса (без tensorflow-пина из `requirements.txt`, чтобы не тянуть downgrade)
```bash
.venv/bin/python -m pip install --upgrade pip
grep -Ev '^[[:space:]]*tensorflow([[:space:]]*[<>=!~].*)?$' \
  themes/04-Autoregression/lab/requirements.txt > /tmp/lr04-no-tf.req
.venv/bin/python -m pip install -r /tmp/lr04-no-tf.req
```

## 3) Переключить TensorFlow на локальный GPU-стек для RTX 50xx
- [ ] Применить расширенный recovery-стек:

```bash
.venv/bin/python -m pip install --upgrade 'tensorflow[and-cuda]==2.21.0' 'nvidia-cuda-nvcc-cu12>=12.8'
```

- [ ] Проверить версию TensorFlow:

```bash
.venv/bin/python - <<'PY'
import tensorflow as tf
print(tf.__version__)
print(tf.test.is_built_with_cuda())
PY
```

Ожидается `2.21.0` и `True` для `is_built_with_cuda`.

## 4) Зарегистрировать Jupyter kernel
- [ ]
```bash
.venv/bin/python -m ipykernel install --user --name students-ai-gpu-tf221 --display-name 'Python (.venv) GPU TF2.21'
```

## 5) Показательный запуск solution-ноутбука из ЛР 04
- [ ] Запустить неинтерактивный прогон GPU solution:

```bash
GPU_TRAINING_BUDGET_MINUTES=5 \
  themes/04-Autoregression/lab/scripts/execute_gpu_solution.sh \
  themes/04-Autoregression/lab/solutions/runs_g635_demo
```

## 6) Критерий успеха
- [ ] Проверить итоговый JSON:

`themes/04-Autoregression/lab/solutions/runs_g635_demo/02_decoder_only_tiny_shakespeare_gpu_solution.summary.json`

Минимальный PASS:
- `overall_pass = true`
- `generation_pass = true`
- `baseline_pass = true`
- `success_count >= 19`

## 7) Что делать, если снова `INVALID_PTX` / `INVALID_HANDLE`
- [ ] Не переписывать модель сразу.
- [ ] Проверить, что запущен именно GPU-ноутбук `ЛР 04`, где есть `gpu_preflight()` и предзагрузка CUDA-библиотек.
- [ ] Перезапустить kernel (`Restart & Run All`).
- [ ] Если нестабильно — временно переключиться на `local-cpu`/`colab-gpu`/`kaggle-gpu` для прохождения учебного маршрута.

---

## Факт проверки (этот хост, 2026-04-06)
Показательный прогон выполнен:
- summary: `themes/04-Autoregression/lab/solutions/runs_g635_demo/02_decoder_only_tiny_shakespeare_gpu_solution.summary.json`
- результат: `overall_pass=true`, `success_count=19/20`, `test_perplexity=8.5983`, `stop_reason=time_budget_reached`, `timed_budget_minutes=5.0`.
