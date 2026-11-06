# Ракурси й камери для реконструкції

`Zero123ViewGenerator.generate_views(photo, seed)` отримує одну RGB-сітку
Zero123++ v1.2 розміром **640 × 960**. Генератор читає плитки зліва направо,
зверху вниз і видаляє тло з кожної окремо. Плитки залишаються **320 × 320**
у вихідних координатах. Це зберігає відповідність силуету проєкції камери.

Результат кожного виклику має власну папку `<output_root>/<id>/views/`.
У ній `view_00.png` … `view_05.png` у форматі RGBA та `cameras.json`.
Папка `views` публікується після запису всіх файлів. Повторний виклик
створює новий набір і залишає попередній доступним для перегляду.

## Формат камер

`cameras.json` відображає назву кожного PNG на параметри камери. Нижче
показано перший запис. Розмір кадру читається з відповідного PNG.

```json
{
  "view_00.png": {"azimuth": 30.0, "elevation": 20.0, "distance": 4.0, "fov": 30.0}
}
```

| PNG | Рядок, стовпець | Азимут | Підняття |
| --- | --- | --- | --- |
| `view_00.png` | 1, 1 | 30° | 20° |
| `view_01.png` | 1, 2 | 90° | −10° |
| `view_02.png` | 2, 1 | 150° | 20° |
| `view_03.png` | 2, 2 | 210° | −10° |
| `view_04.png` | 3, 1 | 270° | 20° |
| `view_05.png` | 3, 2 | 330° | −10° |

Камери дивляться на початок координат. Вісь +Y спрямована вгору, азимут
0° відповідає +Z, 90° — +X. Відлік азимуту прив'язаний до вхідного виду.
Вертикальне поле огляду v1.2 становить 30°. `distance=4.0` задає вибрану
нормалізовану відстань у координатах проєкту, яку можна змінити в
конструкторі. Модель не надає каліброваної відстані. Геометрична точність
також залежить від узгодженості згенерованих ракурсів і масок.
[Параметри Zero123++](https://github.com/SUDO-AI-3D/zero123plus#camera-parameters).
[Порядок нарізки в демонстрації авторів](https://github.com/SUDO-AI-3D/zero123plus/blob/7d0315c31be6eb906b34cf07d91310f8e12e9b95/gradio_app.py#L100-L115).

`ViewsRepository.load(views_dir, alpha_threshold=128)` повертає кортеж
`View` у порядку назв файлів. Силует — `alpha >= alpha_threshold`, а колір
береться з RGB-каналів. У PNG зберігаються всі значення альфа-каналу.
Поріг можна задати цілим числом від 1 до 255. Репозиторій перевіряє
відповідність PNG і камер, відхиляє пошкоджені записи та назви, що
посилаються за межі папки. Той самий формат підтримують синтетичні набори.

## Запуск із Zero123++

Для роботи з PNG потрібен Pillow. Для конкретної реалізації видалення тла
встановіть опційний CPU пакет `background-removal`. Він використовує
реалізацію `RembgRemover` з [завдання 2](https://github.com/Lenuar23/3d-model_generator/tree/dcab4186809b6b2ef0330936f2c56875fae45600).

```powershell
uv sync --locked --extra background-removal
```

Модель запускається окремим процесом. Її NumPy 1.26.4 та PyTorch не
встановлюються в середовище проєкту, яке використовує NumPy 2.5.4 або новіший.
Збережіть код запуску з перевіреного [коміту завдання 4](https://github.com/Lenuar23/3d-model_generator/tree/09b9c5408c457028b15faec0ed67c9cc67c33c2e)
в окремій папці. Якщо середовище моделі ще не створено, виконайте команди
нижче. Потрібні Python 3.12 і CUDA GPU.

```powershell
git fetch origin feat/zero123-notebook
git worktree add --detach data/zero123_task4_source 09b9c5408c457028b15faec0ed67c9cc67c33c2e
uv venv --python 3.12 data/zero123_runtime
uv pip install --python data/zero123_runtime/Scripts/python.exe -r data/zero123_task4_source/notebooks/zero123_requirements.in
$env:HF_HOME = (Join-Path (Get-Location) 'data/zero123_cache')
$env:HF_HUB_DOWNLOAD_TIMEOUT = '120'
```

Перша генерація завантажує приблизно 5,58 ГБ ваг Zero123++ v1.2. Ваги
поширюються за CC BY-NC 4.0. `u2netp` також завантажує свої ваги при
першому використанні, після чого сесія видалення тла повторно використовується
для всіх шести кадрів. [Умови авторів Zero123++](https://github.com/SUDO-AI-3D/zero123plus#license).

У Python основного середовища (`.venv`) запустіть цей приклад, замінивши
`photo.jpg` шляхом до власного фото.

```python
from importlib import import_module

ai = import_module("3d_model_soft.infrastructure.ai.zero123_generator")
rembg = import_module("3d_model_soft.infrastructure.ai.rembg_remover")
storage = import_module("3d_model_soft.infrastructure.storage.views_repository")

backend = ai.Zero123SubprocessBackend(
    "data/zero123_runtime/Scripts/python.exe",
    "data/zero123_task4_source/scripts/run_zero123.py",
    vae_tiling=True,
)
generator = ai.Zero123ViewGenerator(backend, rembg.RembgRemover("u2netp"), "data/generated_views")
views_dir = generator.generate_views("photo.jpg", seed=42)
views = storage.ViewsRepository().load(views_dir)
```

Для вже збереженої сітки v1.2 можна передати власний бекенд. Це функція
`(photo: Path, seed: int) -> np.ndarray`, яка повертає RGB `uint8` форми
`(960, 640, 3)`. Вона дозволяє повторити сегментацію й запис камер без
запуску GPU. Викликаючий код має забезпечити, що сітка справді отримана
від v1.2 і ракурси залишилися в початковому порядку.

## Перевірка

Десять збережених сіток із завдання 4 пройшли нарізку та реальне
видалення тла через `RembgRemover("u2netp")` на CPU. Записано й прочитано
**60 із 60** RGBA PNG розміром 320 × 320. Усі маски при порозі 128 містили
пікселі переднього плану. Камери з кожної папки пройшли перевірку порядку,
кутів і розмірів.

Окремо перевірено повний запуск для фото `05_complex_rubber_duck.jpg`
із seed 42 і 28 кроками. Zero123++ виконався в ізольованому середовищі
на NVIDIA GeForce RTX 4050 Laptop GPU. Репозиторій прочитав усі шість
нових PNG і камер. Кількості пікселів переднього плану за порядком ракурсів
становили 26 141, 30 109, 27 542, 25 070, 30 734 і 28 123. Збіг із
попередньою сіткою для того самого фото перевірено за цими кількостями.

Тест інтеграції читає синтетичний набір Студента 1 через `ViewsRepository`,
порівнює камери й силуети та передає отримані `View` до `carve`. Результати
для реальних фото підтверджують роботу формату й нарізки. Якість силуетів
і точність 3D-геометрії потребують оцінки на наступних етапах.
