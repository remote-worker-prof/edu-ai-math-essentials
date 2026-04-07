# Foundations for Sequence Labs

## Назначение
`00-Foundations` — это общий нулевой блок перед `01-RNN` и `02-Attention`.

Его задача:
- выровнять словарь по формам тензоров и метрикам;
- спокойно ввести токены, `PAD/SOS/EOS`, masking и teacher forcing;
- показать, как читать attention heatmap до большой лабораторной;
- дать короткие warm-up notebooks, которые можно пройти на CPU за несколько минут.

Этот блок не заменяет тематические guides внутри `01-RNN` и `02-Attention`. Он снимает общие входные трудности до того, как студент откроет основную лабораторную.

## Структура
- [guides/01_sequence_shapes_and_metrics.md](./guides/01_sequence_shapes_and_metrics.md) — формы `(batch, time, features)`, `loss` vs `accuracy`, `return_sequences`, `token_accuracy` vs `exact_match`.
- [guides/02_tokens_padding_and_decoder_shift.md](./guides/02_tokens_padding_and_decoder_shift.md) — словарь, `PAD/SOS/EOS`, masking, teacher forcing, сдвиг decoder.
- [guides/03_attention_heatmaps.md](./guides/03_attention_heatmaps.md) — `query/key/value`, score-matrix, `context`, чтение heatmap.
- [guides/04_self_study_debugging_playbook.md](./guides/04_self_study_debugging_playbook.md) — единый маршрут диагностики для self-study режима.
- [guides/05_local_tensorflow_gpu_notebooks.md](./guides/05_local_tensorflow_gpu_notebooks.md) — единый runtime-guide для `local / Colab / Kaggle` и режимов `CPU / GPU`.
- [guides/06_tensorflow_cuda_version_selection.md](./guides/06_tensorflow_cuda_version_selection.md) — как думать про версии `TensorFlow` и `CUDA`, если нужен локальный GPU.
- [guides/07_tensorflow_blackwell_local_gpu_case_study.md](./guides/07_tensorflow_blackwell_local_gpu_case_study.md) — консервативный кейс для очень новой `NVIDIA` laptop GPU.
- [guides/08_g635_local_gpu_tf221_checklist.md](./guides/08_g635_local_gpu_tf221_checklist.md) — практический чеклист миграции `G615 -> G635` с `TensorFlow 2.21`.
- `examples/` — короткие полностью решённые warm-up notebooks.
- `showcases/` — необязательные real-data demos и dataset cards для второго круга.
- [requirements.txt](./requirements.txt) — зависимости для foundations, warm-up и showcase notebook'ов.

## Маршруты
### Совсем с нуля
1. [guides/01_sequence_shapes_and_metrics.md](./guides/01_sequence_shapes_and_metrics.md)
2. [examples/01_numpy_sequence_basics.ipynb](./examples/01_numpy_sequence_basics.ipynb)
3. [examples/02_minimal_keras_sequence_classifier.ipynb](./examples/02_minimal_keras_sequence_classifier.ipynb)
4. [guides/04_self_study_debugging_playbook.md](./guides/04_self_study_debugging_playbook.md)
5. После этого переходить в [../01-RNN/lab/README.md](../01-RNN/lab/README.md)

### Понимаю формы и метрики, но хочу быстро выровнять sequence-термины
1. [guides/02_tokens_padding_and_decoder_shift.md](./guides/02_tokens_padding_and_decoder_shift.md)
2. [examples/03_tokenization_padding_masking.ipynb](./examples/03_tokenization_padding_masking.ipynb)
3. [guides/04_self_study_debugging_playbook.md](./guides/04_self_study_debugging_playbook.md)
4. Дальше идти в `01-RNN / ЛР03` или `02-Attention / ЛР01`

### Готов к seq2seq / attention и хочу только короткий refresher
1. [guides/03_attention_heatmaps.md](./guides/03_attention_heatmaps.md)
2. [examples/04_attention_heatmap_toy.ipynb](./examples/04_attention_heatmap_toy.ipynb)
3. [showcases/README.md](./showcases/README.md) — если нужен мостик от synthetic labs к реальным данным
4. Затем открыть [../02-Attention/lab/README.md](../02-Attention/lab/README.md)

## Warm-up Examples
- [examples/01_numpy_sequence_basics.ipynb](./examples/01_numpy_sequence_basics.ipynb) — ручной разбор `shape`, `axis`, `sum`, `cumsum` и построения меток.
- [examples/02_minimal_keras_sequence_classifier.ipynb](./examples/02_minimal_keras_sequence_classifier.ipynb) — минимальный `Embedding + SimpleRNN/GRU` на synthetic data.
- [examples/03_tokenization_padding_masking.ipynb](./examples/03_tokenization_padding_masking.ipynb) — словарь, `PAD/SOS/EOS`, decoder shift и mask без тяжёлого обучения.
- [examples/04_attention_heatmap_toy.ipynb](./examples/04_attention_heatmap_toy.ipynb) — ручная score-matrix, softmax, `context` и маленький Keras `Attention`.

## Real-Data Showcases
Материалы в `showcases/` не обязательны перед первой попыткой лабораторной.

Их правильный момент:
- после того, как понятны toy-задачи;
- когда хочется увидеть, как те же идеи переносятся на реальные данные;
- когда нужен мостик от учебной reverse-задачи к прикладному sequence modeling.

Рекомендуемый порядок:
1. [showcases/01_imdb_many_to_one_showcase.ipynb](./showcases/01_imdb_many_to_one_showcase.ipynb)
2. [showcases/02_jena_climate_lstm_gru_showcase.ipynb](./showcases/02_jena_climate_lstm_gru_showcase.ipynb)
3. [showcases/03_spa_eng_seq2seq_attention_showcase.ipynb](./showcases/03_spa_eng_seq2seq_attention_showcase.ipynb)
4. Dataset cards в [showcases/cards/](./showcases/cards/)

## Запуск
Команды выполняются из корня репозитория.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r themes/00-Foundations/requirements.txt
python3 -m ipykernel install --user --name foundations-lab --display-name "Python (.venv) Foundations"
.venv/bin/jupyter notebook
```

### Runtime Варианты Для TensorFlow Notebook'ов
Базовый student-flow теперь один и тот же во всех TensorFlow notebook'ах:
- в первой runtime-ячейке выбрать `RUNTIME_MODE`;
- для первого запуска безопасно оставлять `auto`;
- если нужен только CPU, выбрать `local-cpu`;
- если нужен локальный GPU, выбрать `local-gpu`;
- если хочется сэкономить локальные ресурсы, выбрать `colab-*` или `kaggle-*`.

Полный guide:
[guides/05_local_tensorflow_gpu_notebooks.md](./guides/05_local_tensorflow_gpu_notebooks.md)

Если IDE падает при долгом запуске (`code: 139` в VS Code), используйте блок
`Если VS Code падает с code: 139` в этом же guide `05`.

Если нужен именно локальный GPU и вы не уверены в версиях `TensorFlow` / `CUDA`, используйте:
[guides/06_tensorflow_cuda_version_selection.md](./guides/06_tensorflow_cuda_version_selection.md)

Короткая локальная версия:
```bash
source .venv/bin/activate
python3 -m pip install --upgrade 'tensorflow[and-cuda]>=2.16,<2.20'
.venv/bin/python -m ipykernel install --user --name students-ai-gpu --display-name "Python (.venv) GPU"
.venv/bin/jupyter notebook
```

Короткая облачная версия:
- `Google Colab` -> `RUNTIME_MODE = "colab-cpu"` или `RUNTIME_MODE = "colab-gpu"`
- `Kaggle` -> `RUNTIME_MODE = "kaggle-cpu"` или `RUNTIME_MODE = "kaggle-gpu"`
- если в облаке загружен только notebook, замените `COURSE_REPO_HTTPS_URL` на публичный HTTPS URL курса

### Ограничения бесплатных облачных аккаунтов
Краткая актуализация по `Colab` и `Kaggle` вынесена в:
[guides/05_local_tensorflow_gpu_notebooks.md](./guides/05_local_tensorflow_gpu_notebooks.md)

Что важно помнить:
- лимиты облака динамические и могут меняться;
- перед длинным запуском проверяйте текущие квоты на официальных страницах платформ.

Короткий маршрут для локального GPU:
1. Сначала открыть `05` и выбрать runtime-сценарий.
2. Если нужен просто рабочий запуск, этого достаточно.
3. Если нужен именно локальный GPU и есть вопросы по версиям, открыть `06`.
4. Для пошаговой миграции на новый ноутбук использовать:
   [guides/08_g635_local_gpu_tf221_checklist.md](./guides/08_g635_local_gpu_tf221_checklist.md).

## Связь С Основными Темами
- После этого блока основной теоретический трек начинается в [../01-RNN/theory/theory.md](../01-RNN/theory/theory.md).
- Практический трек RNN начинается в [../01-RNN/lab/README.md](../01-RNN/lab/README.md).
- Трек attention продолжает тот же курс в [../02-Attention/lab/README.md](../02-Attention/lab/README.md).

## Что Должно Остаться После Foundations
- уверенное чтение форм `(batch, time, features)`;
- понимание, чем `loss` отличается от `accuracy`;
- интуиция, зачем нужны `PAD/SOS/EOS`, masking и teacher forcing;
- привычка смотреть на attention как на score-matrix, а не как на “магический блок”;
- готовность открывать лабораторные ноутбуки без резкого когнитивного скачка.
