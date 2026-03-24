# TensorFlow Runtime Modes for Local, Colab, and Kaggle

Этот guide объясняет, как запускать TensorFlow notebook'и курса:
- локально;
- локально с принудительным CPU;
- локально с GPU;
- в `Google Colab`;
- в `Kaggle Notebooks`.

Главная идея теперь одна и та же во всех TensorFlow notebook'ах:
- в начале notebook есть ячейка `Выбор runtime`;
- студент меняет только `RUNTIME_MODE`;
- notebook сам настраивает TensorFlow и, при необходимости, облачный bootstrap.

Этот guide отвечает на вопрос **"где и как запускать notebook"**.

Если вам нужен именно **локальный GPU** и вы не уверены в версиях `TensorFlow` / `CUDA`,
дополнительно откройте:

`themes/00-Foundations/guides/06_tensorflow_cuda_version_selection.md`

## Что такое `RUNTIME_MODE`
Во всех TensorFlow notebook'ах есть одинаковый набор режимов:

| `RUNTIME_MODE` | Когда использовать | Что делает notebook |
|---|---|---|
| `auto` | лучший вариант по умолчанию | пытается использовать GPU, если TensorFlow его реально видит; иначе остаётся на CPU |
| `local-cpu` | локальный запуск без GPU | принудительно скрывает GPU и работает только на CPU |
| `local-gpu` | локальный запуск с обязательным GPU | требует видимый GPU и останавливается с понятной ошибкой, если GPU нет |
| `colab-cpu` | запуск в `Google Colab` на CPU | настраивает CPU-only режим в Colab |
| `colab-gpu` | запуск в `Google Colab` с GPU | требует включённый GPU accelerator |
| `kaggle-cpu` | запуск в `Kaggle Notebooks` на CPU | настраивает CPU-only режим в Kaggle |
| `kaggle-gpu` | запуск в `Kaggle Notebooks` с GPU | требует включённый GPU accelerator |

Если вы меняете `RUNTIME_MODE`, дальше нужен `Restart & Run All`.

## Что выбирать на практике
### Если вы запускаете notebook локально и не хотите разбираться с GPU
Оставляйте:

```python
RUNTIME_MODE = "auto"
```

или явно ставьте:

```python
RUNTIME_MODE = "local-cpu"
```

### Если у вас локальная `Linux + NVIDIA` машина и вы хотите ускорение
Ставьте:

```python
RUNTIME_MODE = "local-gpu"
```

Но сначала подготовьте локальную среду.

Официальный локальный GPU-path курса:
- `Linux + NVIDIA`;
- или `Windows + WSL2 + Ubuntu`, если вы работаете на Windows;
- native Windows GPU-путь для современных версий TensorFlow в курсе не считается основным.

Из корня репозитория:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r themes/00-Foundations/requirements.txt
python3 -m pip install --upgrade 'tensorflow[and-cuda]>=2.16,<2.20'
.venv/bin/python -m ipykernel install --user --name students-ai-gpu --display-name "Python (.venv) GPU"
.venv/bin/jupyter notebook
```

Потом в notebook:

```python
RUNTIME_MODE = "local-gpu"
```

Если `local-gpu` запускается, но дальше на очень новой видеокарте появляются ошибки вида
`CUDA_ERROR_INVALID_PTX` или `CUDA_ERROR_INVALID_HANDLE`, это обычно уже не проблема
самого notebook, а совместимости текущего TensorFlow GPU-стека с конкретной картой.

В таком случае нормальный учебный fallback такой:
- переключиться на `local-cpu`;
- или унести запуск в `colab-gpu`;
- или унести запуск в `kaggle-gpu`.

Если хотите увидеть **реальный пример** такого сценария на современной `NVIDIA` laptop GPU,
откройте:

`themes/00-Foundations/guides/07_tensorflow_blackwell_local_gpu_case_study.md`

Если хотите понять, **как именно думать про версии** и когда нужно смотреть official
compatibility tables, переходите в guide `06`.

### Если своего GPU нет или его жалко для длинных запусков
Используйте `Colab` или `Kaggle`.

Для `Colab`:

```python
RUNTIME_MODE = "colab-cpu"
```

или

```python
RUNTIME_MODE = "colab-gpu"
```

Для `Kaggle`:

```python
RUNTIME_MODE = "kaggle-cpu"
```

или

```python
RUNTIME_MODE = "kaggle-gpu"
```

## Как работают Colab и Kaggle
Cloud-сценарий разделяется на два варианта.

### Вариант 1. Репозиторий уже открыт в облаке
Например:
- вы уже клонировали курс в `Colab`;
- или открыли notebook внутри проекта в `Kaggle`.

Тогда notebook просто:
- находит корень репозитория;
- при необходимости ставит только недостающие course-пакеты;
- не переустанавливает TensorFlow и Jupyter-слой поверх облачной среды.

### Вариант 2. В облако загружен только сам notebook
Тогда notebook может попытаться сам скачать курс, но для этого в первой ячейке нужен публичный HTTPS URL репозитория:

```python
COURSE_REPO_HTTPS_URL = "https://github.com/<org>/<repo>.git"
```

Пока там стоит placeholder, cloud auto-bootstrap не сработает. Это сделано специально, чтобы студент видел понятную причину, а не непонятный `git clone` failure.

## Почему cloud-режим не ставит TensorFlow заново
В `Colab` и `Kaggle` TensorFlow и Jupyter-окружение обычно уже управляются платформой.

Поэтому notebook:
- читает `NOTEBOOK_REQUIREMENTS`;
- ставит только course-зависимости;
- пропускает `tensorflow`, `jupyter`, `ipykernel`, `nbconvert`, `nbformat`.

Это уменьшает риск сломать облачный runtime переустановкой тяжелого стека.

## Как проверить, что выбранный режим действительно сработал
После первой runtime-ячейки notebook печатает короткий summary:
- какой режим был запрошен;
- какая среда определена;
- где корень репозитория;
- видит ли TensorFlow GPU;
- какой вычислительный режим выбран в итоге.

Типичные ожидания:
- `local-cpu` -> `visible GPUs: []`, `compute device: CPU`
- `local-gpu` -> хотя бы один GPU виден, `compute device: GPU`
- `auto` -> GPU используется только если TensorFlow его реально видит

## Быстрые команды для локальной диагностики
Проверка хоста:

```bash
nvidia-smi
```

Проверка окружения проекта:

```bash
.venv/bin/python - <<'PY'
import tensorflow as tf
print("built_with_cuda =", tf.test.is_built_with_cuda())
print("gpus =", tf.config.list_physical_devices("GPU"))
PY
```

## Если `*-gpu` выбран, а GPU нет
Это штатный сценарий ошибки.

Notebook должен остановиться и сказать понятными словами:
- переключиться на `*-cpu`;
- или включить GPU accelerator в `Colab/Kaggle`;
- или закончить локальную настройку GPU.

Тихого fallback из `*-gpu` в CPU специально нет, чтобы студент не думал, что он обучается на GPU, когда это не так.

## Если GPU есть, но notebook всё равно падает
Самые частые причины:
- локальный TensorFlow GPU-стек ещё не дружит с очень новой картой;
- accelerator не включён в `Colab` или `Kaggle`;
- TensorFlow видит GPU, но конкретные kernels падают уже во время реальной работы.

Практический порядок действий:
1. Для первого запуска вернуться на `auto` или `local-cpu`.
2. Если нужен именно GPU, попробовать `colab-gpu` или `kaggle-gpu`.
3. Только потом возвращаться к локальной GPU-отладке.

## Что делать, если нужно просто безопасно запуститься
Если не хочется думать о среде, выбирайте:

```python
RUNTIME_MODE = "auto"
```

Это самый спокойный режим для первого запуска.
