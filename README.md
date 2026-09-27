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

## Первый результат

На фиксированной proxy-validation из 2 452 поисковых контекстов:

| Эксперимент | Recall@10 | Recall@20 | Recall@50 |
|---|---:|---:|---:|
| B0: word TF-IDF по заголовку | 0.0890 | 0.1305 | 0.2127 |

Это сравнительная локальная оценка, а не прогноз leaderboard: только 6.29%
положительных пар validation-fold относятся к объявлениям, присутствующим в
текущем benchmark-корпусе.
