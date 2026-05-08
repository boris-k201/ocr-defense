## OCR Defense (рендер + обфускации + оценка OCR)

### Установка

Используется локальный `./venv`.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Веб‑интерфейс (FastAPI)

Запуск:

```bash
uvicorn webapp.app:app --host 127.0.0.1 --port 8000
```

Страницы:
- `http://127.0.0.1:8000/` — рендер + настройки защиты текста
- `http://127.0.0.1:8000/testing` — запуск тестирования и просмотр результатов

### `ocr-defense.py def` — рендер текста в PNG (с обфускациями или без)

#### Базовый рендер

```bash
echo "Привет мир" | python ocr-defense.py def -i - -o out.png
```

#### Рендер с защитными возмущениями

```bash
echo "Привет мир" | python ocr-defense.py def -i - -o out_obf.png --attack watermark --random-seed 1 -v
```

Доступные режимы `--attack`: `none`, `semantic`, `diacritics`, `image_patch`, `watermark`, `distortions`, `adv_docvqa`, `all`.

`--attack` выбирает, какие подсистемы применять. Значения параметров каждой подсистемы берутся из `config.json` → `attack`.

### `ocr-defense.py eval` — тестирование OCR-движков и сохранение метрик

Пример запуска (текст из стандартного потока ввода, результаты в JSON):

```bash
echo "Привет мир" | python ocr-defense.py eval -i - -o ocr_results.json --attack all
```

Выбор движков:

```bash
echo "Привет мир" | python ocr-defense.py eval -i - --engines tesseract,trocr,donut,easyocr -o ocr_results.json
```

#### Dataset-mode (анализ набора текстов одним запуском)

Если входной файл содержит несколько записей, можно включить `--dataset-mode`.
Тогда `--input` делится по `--separator`, каждая часть оценивается отдельно, и в `--output` пишется единый JSON с полем `items`.

```bash
python ocr-defense.py eval \
  -i dataset.txt \
  -o ocr_results_dataset.json \
  --attack all \
  --dataset-mode \
  --separator "\n---\n" \
  --print-samples 3
```

Полезные флаги dataset-mode:
- `--dataset-mode` — включить пакетную оценку;
- `--separator` — разделитель записей;
- `--keep-empty` — не отбрасывать пустые записи;
- `--print-samples N` — вывести в stdout первые `N` обработанных записей.

`paddleocr` требует установленного `paddlepaddle`. На очень новых версиях Python колёса могут отсутствовать (рекомендуется Python 3.10–3.12).

### Единая конфигурация (`config.json`)

`ocr-defense.py def` и `ocr-defense.py eval` читают параметры из одного JSON-файла (путь задаётся `--config`).

Структура:

- `render` — параметры рендера изображения;
- `attack.semantic` — semantic/GА-параметры;
- `attack.diacritics` — параметры диакритик;
- `attack.image_patch` — параметры патчей;
- `attack.watermark` — параметры watermark;
- `attack.distortions` — параметры искажений;
- `attack.adv_docvqa` — параметры adv_docvqa.

Пример:

```json
{
  "render": {
    "image_width": 900,
    "image_height": 220,
    "margin": 10,
    "font_size": 28,
    "dpi": 96,
    "text_color": "#111111",
    "background_color": "#ffffff",
    "font_path": "fonts/PT_Sans/PTSans-Regular.ttf"
  },
  "attack": {
    "semantic": {
      "language": "auto",
      "max_changed_words": 3
    },
    "diacritics": {
      "budget_per_word": 5,
      "diacritics_probability": 0.6
    },
    "image_patch": {
      "max_patches_per_line": 1,
      "effects": ["bbox", "pixel", "text"]
    },
    "watermark": {
      "text_lines": ["CONFIDENTIAL"],
      "color": "#606060",
      "alpha": 80
    },
    "distortions": {
      "enable_skew": true,
      "enable_rotate": true,
      "enable_warp": true,
      "enable_strikethrough": true
    },
    "adv_docvqa": {
      "model_name": "donut",
      "checkpoint": "naver-clova-ix/donut-base-finetuned-docvqa",
      "local_files_only": true
    }
  }
}
```

Пример файла конфигурации со всеми полями доступен в `config.json` в корне проекта.

`font_path` в `render` опционален: если задан, используется для поддерживаемых символов, а для остальных — системный шрифт (fallback).

### Web Testing: single + dataset mode

На странице `http://127.0.0.1:8000/testing` теперь доступны оба режима:
- обычный (`dataset_mode` выключен) — оценка одного текста;
- пакетный (`dataset_mode` включен) — split текста по `separator` и оценка всех частей.

На страницах `Render` и `Testing` также можно загрузить исходный текст из файла (кнопка "Загрузить текст из файла").

### Построение графиков метрик по датасету

Для результата `ocr-defense.py eval --dataset-mode` добавлен скрипт `plot-dataset-metrics.py`.
Он строит отдельные PNG-графики для каждого OCR-движка:
- CER: `original` vs `attacked`
- WER: `original` vs `attacked`

Пример:

```bash
python plot-dataset-metrics.py -i ocr_results_dataset.json -o plots
```

### Docker

Сборка:

```bash
docker build -t ocr-defense .
```

Запуск:

```bash
docker run --rm -p 8000:8000 ocr-defense
```

После запуска откройте `http://127.0.0.1:8000/`.

