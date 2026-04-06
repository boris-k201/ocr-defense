## OCR Defense (рендер + обфускации + оценка OCR)

### Установка

Используется локальный `./venv`.

```bash
./venv/bin/pip install -r requirements.txt
```

### `main.py` — рендер текста в PNG (с обфускациями или без)

#### Базовый рендер

```bash
echo "Привет мир" | ./venv/bin/python main.py -i - -o out.png
```

#### Рендер с защитными возмущениями

```bash
echo "Привет мир" | ./venv/bin/python main.py -i - -o out_obf.png --attack diacritics --random-seed 1 -v
```

Доступные режимы `--attack`: `none`, `semantic`, `diacritics`, `image_patch`, `all`.

#### Конфигурация рендера (`config.json`)

`main.py` и `evaluate.py` читают параметры из JSON (путь задаётся `--config`).

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

### `evaluate.py` — тестирование OCR-движков и сохранение метрик

Пример запуска (текст из stdin, результаты в JSON):

```bash
echo "Привет мир" | ./venv/bin/python evaluate.py -i - -o ocr_results.json --attack all
```

Выбор движков:

```bash
echo "Привет мир" | ./venv/bin/python evaluate.py -i - --engines tesseract,trocr,easyocr -o ocr_results.json
```

`paddleocr` требует установленного `paddlepaddle`. На очень новых версиях Python колёса могут отсутствовать (рекомендуется Python 3.10–3.12).

### Веб‑интерфейс (FastAPI)

Запуск:

```bash
./venv/bin/uvicorn webapp.app:app --host 127.0.0.1 --port 8000
```

Страницы:
- `http://127.0.0.1:8000/` — рендер + advanced настройки атак
- `http://127.0.0.1:8000/testing` — запуск тестирования и просмотр результатов

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

