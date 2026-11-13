# 3D Model Soft

Застосунок перетворює одне фото на GLB через Hunyuan3D-2mini.
Gradio показує модель і дозволяє завантажити файл. Колір вершин
проєктується з фото; опція mirror-back переносить його також на задню сторону.
Це наближене забарвлення, а не повний текстурний атлас.

## Запуск у Docker

Потрібні Docker з підтримкою NVIDIA GPU та NVIDIA Container Toolkit.

```powershell
docker compose up --build
```

Відкрийте http://127.0.0.1:7860. Перший запуск завантажує ваги;
кеш зберігається у data/. Зупинка: Ctrl+C, потім `docker compose down`.
Docker створює окремі середовища UI та моделі з різними залежностями.

## Локальний UI

```powershell
uv sync --locked --extra ui
$env:HUNYUAN_PYTHON = "C:\path\to\hunyuan\python.exe"
uv run --extra ui python scripts/run_photo_to_3d.py
```

HUNYUAN_PYTHON має вказувати на вже налаштоване середовище Hunyuan
із CUDA, torch, hy3dgen, rembg та залежностями з Dockerfile.
Extra ui встановлює лише інтерфейс. Для відтворюваного повного встановлення
використовуйте Dockerfile. GRADIO_SERVER_PORT змінює порт (типово 7860).

## Код

- scripts/run_photo_to_3d.py — UI, запуск процесу моделі, кешування GLB.
- scripts/generate_hunyuan.py — видалення фону, Hunyuan3D і експорт.
- scripts/photo_vertex_colors.py — перенесення кольорів із фото.
- src/3d_model_soft/ui/gradio_compat.py — сумісність Model3D з Gradio 6.31.0.

Старий сценарій Zero123++ / space carving видалено.
project_plan.md і docs/reconstruction_comparison.md залишені як історія.
Локальні дані й ваги в data/ не входять у репозиторій.

## Перевірка без GPU

```powershell
uv run --extra ui python -m unittest discover -s tests -v
```

Ці тести не завантажують модель і не підтверджують якість GPU-генерації.

## Експеримент: локальна AI-текстура готової моделі

Потрібне окреме середовище Hunyuan з `xatlas==0.0.11` (встановлення:
`uv pip install --python data/hunyuan_runtime/Scripts/python.exe xatlas==0.0.11`).
BLIP-base описує фото на CPU, Stable Diffusion 1.5 img2img отримує опис
і фрагмент поверхні; CPU offload економить VRAM. Потім xatlas будує UV,
видимі ділянки отримують колір фото, приховані — згенерований матеріал.
GPU-генерація форми повторно не запускається.

```powershell
$env:HF_HOME = "$PWD/data/hunyuan_cache"
$env:U2NET_HOME = "$PWD/data/u2net"
$env:NUMBA_DISABLE_JIT = "1"
& .\data\hunyuan_runtime\Scripts\python.exe scripts/local_texture.py `
  data/test_photos/01_simple_apple.jpg "$HOME/Downloads/model.glb" data/my_texture
```

Вихідна папка має бути новою. Вона містить `description.json`, фрагмент
фото `material_reference.png`, згенерований `material.png`, UV `texture.png`
і готовий `model.glb`. Оригінальний GLB не змінюється.
Параметр `--material "red apple skin with tiny yellow speckles"` дозволяє
уточнити англійський опис; `--seed` задає варіант генерації.

Це прототип для предметів з переважно однорідним матеріалом. BLIP дає
короткий опис, а не достовірну специфікацію матеріалу. Приховані деталі
вигадані; написи, кілька різних матеріалів і складні форми не відновлюються.
Проєкція фото передбачає фронт моделі вздовж +Z; можливі шви та повторення
малюнка. Перший запуск завантажує відкриті ваги, далі обчислення локальні.

Перевірка UV без завантаження моделей:
`data/hunyuan_runtime/Scripts/python.exe -m unittest discover -s tests -p test_texture_bake.py -v`

Багаторакурсний Paint: [локальне налаштування](docs/local_paint.md).

[Текстури до та після: зображення й пояснення](docs/texture_comparison.md).
