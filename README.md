# 3D Model Soft

Перетворює фото одного предмета на модель GLB через Hunyuan3D-2mini. Gradio показує модель і дозволяє завантажити її. Для UV-текстури можна увімкнути Hunyuan Paint.

## Запуск

Потрібні Docker Compose та NVIDIA GPU з доступом до CUDA з контейнерів. У Windows використовуйте Docker Desktop з Linux-контейнерами і WSL 2.

```powershell
docker compose up --build -d
```

Відкрийте [інтерфейс](http://127.0.0.1:7861). Перший запуск генерації може завантажувати ваги. Щоб змінити порт або початковий стан перемикача Paint, задайте `WEB_PORT` або `HUNYUAN_PAINT_DEFAULT=0` у `.env`.

```powershell
docker compose logs -f web generator
docker compose down
```

## Структура

- `services/web/` містить інтерфейс, завантаження фото й видачу GLB.
- `services/generator/` містить внутрішній HTTP-сервіс, генерацію форми та Hunyuan Paint. Він має доступ до GPU й обробляє один запит за раз.
- `data/` містить локальні ваги, кеші та тестові фото. Compose монтує лише `hunyuan_cache`, `hunyuan_paint_weights` та `u2net`.
- `experiments/texture/` містить окремий прототип текстурування готової моделі.
- `docs/` містить інструкції та попередні порівняння.

Браузер підключається до `web`. Той передає фото й параметри у `generator` через внутрішню мережу Compose та отримує GLB. Назовні опубліковано лише порт інтерфейсу.

## Тести без GPU

```powershell
uv run --extra ui python -m unittest discover -s tests -v
```

## Режими генерації

- **Fast (use Math)** запускає Photo2Mesh 2.4 AI-Assisted. CLIP визначає категорію, Wonder3D генерує шість RGB-ракурсів і шість карт нормалей, а Photo2Mesh будує Visual Hull, уточнює поверхню за нормалями та запікає текстуру в GLB. Wonder3D потребує NVIDIA GPU. Ваги завантажуються під час першого запиту й кешуються в `data/photo2mesh_models`.
- **Quality (use 3D model)** використовує Hunyuan3D та опційний Hunyuan Paint у GPU-контейнері `generator`.
