# Avito candidate generation

Воспроизводимый пайплайн генерации до 50 объявлений для каждого поискового
запроса. Данные должны находиться в папке `dataset/`.

## Установка

```bash
uv sync
```

## B0: глобальный word TF-IDF

Короткая проверка на нескольких запросах:

```bash
uv run avito-experiment run \
  --config configs/b0_tfidf.toml \
  --query-limit 5 \
  --output-dir /tmp/avito-smoke
```

Полный запуск для всех benchmark-запросов:

```bash
uv run avito-experiment run --config configs/b0_tfidf.toml
```

Локальная proxy-валидация:

```bash
uv run avito-experiment evaluate --config configs/b0_tfidf.toml
```

Результаты сохраняются в `runs/b0_tfidf/`:

- `answer.csv` — файл для отправки;
- `predictions.parquet` — кандидаты со score и rank;
- `config.toml` — копия конфигурации;
- `run.json` — параметры и время запуска.

Валидация строится из `train.parquet`: запросы делятся группами по
нормализованному тексту, а положительные объявления ограничиваются текущим
корпусом `benchmark_items.parquet`. Результат сохраняется отдельно в
`runs/b0_tfidf_validation/` и не смешивается с файлом для отправки.

## B1: TF-IDF с параметрами и географией

B1 сохраняет тот же retrieval-пайплайн, но дополняет текст параметрами
объявления и запроса. Сначала выбираются до 40 кандидатов из той же локации,
затем список дополняется глобальными кандидатами до 50.

```bash
uv run avito-experiment evaluate --config configs/b1_tfidf_geo.toml
uv run avito-experiment run --config configs/b1_tfidf_geo.toml
```

Сравнить все выполненные validation-запуски:

```bash
uv run avito-experiment compare
```

Таблица также сохраняется в `runs/validation_summary.csv`.

## B2: word + char TF-IDF

B2 добавляет к B1 символьный TF-IDF по заголовкам. Он устойчивее к опечаткам,
словоформам и частичным совпадениям. Word- и char-кандидаты объединяются через
Reciprocal Rank Fusion, после чего сохраняется прежняя схема из 40 локальных и
10 глобальных кандидатов.

```bash
uv run avito-experiment evaluate --config configs/b2_tfidf_hybrid.toml
uv run avito-experiment run --config configs/b2_tfidf_hybrid.toml
```

## B3: CatBoost learning-to-rank

B3 расширяет retrieval-пул до 200 кандидатов, а затем выбирает лучшие 50 с
помощью `CatBoostRanker`. Модель обучается на query-группах из train и hard
negatives из текущего retrieval. Признаки включают word/char score и rank,
RRF, совпадение географии и категории, а также свойства объявления.

```bash
uv run avito-experiment evaluate --config configs/b3_catboost_ranker.toml
uv run avito-experiment run --config configs/b3_catboost_ranker.toml
```

Для запуска на CUDA-машине можно заменить `task_type = "CPU"` на `"GPU"` в
конфигурации эксперимента.

## Ablation: вклад параметров и географии

Две промежуточные конфигурации меняют относительно B0 только один компонент:

```bash
uv run avito-experiment evaluate --config configs/a1_tfidf_params.toml
uv run avito-experiment evaluate --config configs/a2_tfidf_geo.toml
```

- `A1` добавляет текстовые параметры без географического ограничения;
- `A2` добавляет географию, но оставляет только исходный текст заголовка.

## Результаты

На фиксированной proxy-validation из 2 452 поисковых контекстов:

| Эксперимент | Recall@10 | Recall@20 | Recall@50 |
|---|---:|---:|---:|
| B0: word TF-IDF по заголовку | 0.0890 | 0.1305 | 0.2127 |
| A1: только текстовые параметры | 0.0928 | 0.1371 | 0.2072 |
| A2: только география | 0.3810 | 0.4590 | 0.5293 |
| B1: TF-IDF с параметрами и географией | 0.4419 | 0.5425 | 0.6457 |
| B2: word + char TF-IDF с географией | 0.4941 | 0.6031 | **0.7155** |
| B3: CatBoost reranker над pool-200 | **0.5338** | **0.6408** | **0.7501** |

Это сравнительная локальная оценка, а не прогноз leaderboard: только 6.29%
положительных пар validation-fold относятся к объявлениям, присутствующим в
текущем benchmark-корпусе.

География — основной источник прироста: отдельно она повышает Recall@50 на
0.3166 относительно B0. Параметры без географии не дают прироста на Recall@50
(-0.0055), но вместе с географией добавляют ещё 0.1164. Значит, параметры полезны
прежде всего внутри более релевантного географического пула кандидатов.

Символьный TF-IDF добавляет ещё 0.0698 Recall@50 относительно B1. Улучшение
получено без нейросетевых эмбеддингов: за счёт устойчивого текстового поиска,
географического ограничения и объединения независимых ранжирований.

CatBoost-reranker добавляет ещё 0.0346 Recall@50 относительно B2. Расширение
пула само по себе даёт 0.7267, а обучаемый отбор повышает результат до 0.7501.
Таким образом, прирост связан и с большим покрытием кандидатов, и с
learning-to-rank поверх hard negatives.
