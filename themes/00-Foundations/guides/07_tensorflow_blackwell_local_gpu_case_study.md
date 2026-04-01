# Local TensorFlow GPU on a New NVIDIA Laptop GPU: Conservative Case Study

Этот guide показывает **реальный локальный кейс**, где:
- `nvidia-smi` работает;
- TensorFlow в `.venv` уже собран с `CUDA`;
- видеокарта видна хотя бы частично;
- но полноценная notebook-like нагрузка всё равно падает.

Это не guide про "как победить всё любой ценой". Это guide про то, **как аккуратно
думать**, когда локальный GPU-path на очень новой карте ведёт себя нестабильно.

## Актуализация: что реально сработало в текущей `.venv` (31 марта 2026)
Ниже зафиксирован именно рабочий путь для этой машины (`Ubuntu 24.04` + `RTX 5070 Ti`),
который позволил запустить `GPU`-тетради темы `04` без `CPU`-fallback.

### Что пришлось сделать
1. Оставить системный драйвер NVIDIA как есть (`nvidia-smi` должен работать).
2. В активной `.venv` установить `TensorFlow` с `CUDA`-рантаймом:
```bash
python3 -m pip install --upgrade 'tensorflow[and-cuda]==2.21.0'
python3 -m pip install --upgrade 'nvidia-cuda-nvcc-cu12>=12.8'
```
3. Перед импортом `TensorFlow` предзагрузить CUDA-библиотеки из `.venv`
   (это встроено в `GPU`-ноутбуки темы `04` в runtime-ячейке):
   - обновление `LD_LIBRARY_PATH` путями из `.venv/lib/python*/site-packages/nvidia/*/lib`;
   - `ctypes.CDLL(..., RTLD_GLOBAL)` для ключевых библиотек (`cudart`, `cublas`,
     `cudnn`, `cufft`, `curand`, `cusolver`, `cusparse`, `nccl`, `nvjitlink`).
4. Запустить `gpu_preflight()`:
   - `runtime_mode == local-gpu`;
   - `tf.config.list_physical_devices('GPU')` непустой;
   - проходят `matmul` на `/GPU:0` и короткий `train_on_batch`.

### Что это означает методически
- Проблема была не в математике лабораторных и не в архитектуре модели.
- Критичен именно корректный Python-рантайм для `CUDA` внутри `.venv`.
- После прохождения `gpu_preflight()` повторная переустановка тяжёлого стека не нужна.

## Исходные Условия Кейса
Хост:
- `Ubuntu 24.04.4 LTS`
- `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- `driver 580.126.09`
- `nvidia-smi` работает

Текущее course-окружение:
- `Python 3.12.3`
- `.venv`
- `tensorflow==2.19.1`
- `tf.test.is_built_with_cuda() == True`

Что показывал `tf.sysconfig.get_build_info()` в этом окружении:
- `cuda_version = 12.5.1`
- `cudnn_version = 9`
- `cuda_compute_capabilities = sm_60, sm_70, sm_80, sm_89, compute_90`

Это уже хороший сигнал для мышления:
- TensorFlow собран с `CUDA`;
- но в колесе не видно нативной поддержки `sm_120`;
- значит для `compute capability 12.0` TensorFlow может уходить в `PTX JIT`.

## Что Именно Проверяли
Консервативная проверка была такой:

1. Не трогать host-driver и system `CUDA`.
2. Сначала проверить уже существующую `.venv`.
3. Потом сделать ровно один low-risk эксперимент в новой `.venv` с более новым stable wheel.
4. Если это не помогает, остановить локальную GPU-эскалацию и зафиксировать safe fallback.

Проверки в baseline:
- `nvidia-smi`
- версия `Python`
- `pip list` по `tensorflow` и `nvidia-*`
- `tf.sysconfig.get_build_info()`
- `tf.config.list_physical_devices("GPU")`
- простой `tf.matmul` на `/GPU:0`
- маленький synthetic `Embedding -> LSTM -> Dense` `train_on_batch`

Важно:
- одного `tf.config.list_physical_devices("GPU")` недостаточно;
- нужна хотя бы одна **notebook-like** проверка с реальными kernels.

## Что Получилось В Существующей `.venv`
Baseline дал такую картину:

- `nvidia-smi` работал;
- TensorFlow видел `GPU:0`;
- serial `tf.matmul` на GPU проходил;
- но маленький `Keras`/`RNN` шаг падал с ошибками:
  - `CUDA_ERROR_INVALID_PTX`
  - затем `CUDA_ERROR_INVALID_HANDLE`
  - в проблемном узле типа `Cast`

Практический смысл:
- это уже не сценарий "`GPU` вообще не настроен";
- это и не доказательство, что notebook сломан;
- это сценарий "`GPU` виден, но реальные kernels падают".

Именно такой сценарий курс считает **compatibility problem**, а не обычной student-ошибкой.

## Что Дал Изолированный Апдейт До `2.21.0`
В отдельной `.venv-tf221-gpucheck` был выполнен low-risk эксперимент:

```bash
python3 -m venv .venv-tf221-gpucheck
source .venv-tf221-gpucheck/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install 'tensorflow[and-cuda]==2.21.0'
```

После этого произошло не "исправление", а **другая проблема**:
- TensorFlow уже не регистрировал GPU;
- `tf.config.list_physical_devices("GPU")` возвращал пустой список;
- в логах было `Cannot dlopen some GPU libraries`.

Дальше был применён documented venv-level symlink fix из official TensorFlow pip-guide:
- симлинки на `nvidia/*/lib/*.so*` в каталог `tensorflow`;
- симлинк на `ptxas` в `$VIRTUAL_ENV/bin/ptxas`.

Но даже после этого в новой `.venv`:
- GPU всё ещё не регистрировался;
- значит isolated upgrade **не дал usable local GPU path**.

Это важно:
- более новый wheel не обязан автоматически чинить новый GPU;
- иногда вы не переходите из "GPU виден, но kernels падают" в "всё хорошо";
- иногда вы переходите в другой класс проблемы: "`GPU` вообще не виден в новом venv".

## Как Правильно Классифицировать Этот Кейс
Для этого кейса верная классификация такая:

- это **не** ошибка лабораторной;
- это **не** признак того, что студент неправильно пишет `RNN`;
- это **не** повод вслепую переставлять `CUDA` и `cuDNN`;
- это **не** доказательство, что любой более новый TensorFlow wheel сразу спасёт ситуацию.

Это вот что:
- host-driver в целом жив;
- TensorFlow GPU-path частично работает;
- но локальный стек для этой новой карты пока нестабилен для реальной notebook-like нагрузки.

## Важное различие: сбой IDE и сбой вычислительного контура
Иногда во время длинного запуска падает интерфейс IDE (например, `code: 139` в VS Code).
Это отдельный класс проблемы и он не равен ошибке обучения модели.

Как различать:
- сбой IDE: окно закрывается, но процесс `nbconvert` может продолжать работу в терминале;
- сбой вычислительного контура TensorFlow: в выполненной тетради фиксируются ошибки
  `CUDA`/`XLA`/`Keras` и итоговые проверки не проходят.

Где смотреть фактический результат обучения:
1. `summary.json` (главный итоговый источник);
2. `*.executed.ipynb` (источник для восстановления summary, если лог неполный);
3. только затем консольный лог запуска.

## Что Из Этого Должен Сделать Студент
Если вы просто хотите пройти лабораторную:
- используйте `RUNTIME_MODE = "auto"` для первого безопасного запуска;
- если локальный GPU уже однажды повёл себя нестабильно, переключайтесь на `RUNTIME_MODE = "local-cpu"`.

Если вам нужен именно GPU для учебной работы:
- переходите в `colab-gpu`;
- или переходите в `kaggle-gpu`.

Если вы уже упёрлись в `INVALID_PTX` / `INVALID_HANDLE`:
- не чините notebook вслепую;
- не переписывайте архитектуру модели в надежде, что ошибка исчезнет;
- сначала зафиксируйте рабочий запуск на `CPU` или cloud `GPU`.

## Что Было Подтверждено Как Safe Fallback
На этом же проекте был отдельно подтверждён `local-cpu` сценарий:
- runtime-ячейка из `themes/01-RNN/lab/01_simple_rnn_many_to_one_toy.ipynb`
  корректно переключалась в `local-cpu`;
- `compute device` становился `CPU`;
- `tf.config.get_visible_devices("GPU")` возвращал пустой список;
- маленький `SimpleRNN` training step на `CPU` проходил.

Это и есть правильный учебный итог:
- не обязательно добиваться локального GPU любой ценой;
- важно сохранить **стабильный** путь прохождения курса.

## Полезная Практическая Нюансировка
Если вы проверяете `local-cpu`, не путайте два вопроса:

- `tf.config.list_physical_devices("GPU")`
- `tf.config.get_visible_devices("GPU")`

Первый вопрос говорит:
- "какие GPU вообще физически есть на машине"

Второй вопрос говорит:
- "какие GPU реально оставлены видимыми для текущего runtime"

Для `local-cpu` важнее именно второй вопрос.

## Короткая Версия Для Студента
Можно запомнить так:

- "Здесь сломана не сама лабораторная и не базовая установка драйвера."
- "TensorFlow уже видит или частично видит GPU, но wheel ещё не очень дружит с новой архитектурой карты."
- "Если реальные kernels падают, это compatibility issue, а не доказательство, что студент неправильно сделал lab."
- "Для курса нормальный fallback: `auto`, `local-cpu`, `colab-gpu`, `kaggle-gpu`."

## Главная Мысль Этого Guide
На очень новой `NVIDIA` карте полезно уметь различать три состояния:

1. `nvidia-smi` не работает -> host/setup проблема.
2. TensorFlow не видит GPU -> setup-проблема внутри Python env.
3. TensorFlow видит GPU, но реальные kernels падают -> compatibility-проблема.

Этот кейс полезен именно тем, что показывает:
- третий сценарий реален;
- а попытка "просто обновить wheel" может привести и ко второму сценарию тоже;
- поэтому для курса нужен не героизм, а спокойный, безопасный fallback.
