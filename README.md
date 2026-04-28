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
echo "Привет мир" | python ocr-defense.py def -i - -o out_obf.png --attack diacritics --random-seed 1 -v
```

Доступные режимы `--attack`: `none`, `semantic`, `diacritics`, `image_patch`, `all`.

### `ocr-defense.py eval` — тестирование OCR-движков и сохранение метрик

Пример запуска (текст из стандартного потока ввода, результаты в JSON):

```bash
echo "Привет мир" | python ocr-defense.py eval -i - -o ocr_results.json --attack all
```

Выбор движков:

```bash
echo "Привет мир" | python ocr-defense.py eval -i - --engines tesseract,trocr,easyocr -o ocr_results.json
```

`paddleocr` требует установленного `paddlepaddle`. На очень новых версиях Python колёса могут отсутствовать (рекомендуется Python 3.10–3.12).

### Конфигурация рендера (`config.json`)

`ocr-defense.py def` и `ocr-defense.py eval` читают параметры из JSON (путь задаётся `--config`).

Пример:

```json
{
  "image_width": 900,
  "image_height": 220,
  "font_size": 28,
  "dpi": 96,
  "text_color": "#111111",
  "background_color": "#ffffff",
  "font_path": "fonts/PT_Sans/PTSans-Regular.ttf"
}
```

`font_path` опционален: если задан, используется для поддерживаемых символов, а для остальных — системный шрифт (fallback).

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

