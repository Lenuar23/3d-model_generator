# 3D модель з фотографії: структура і ту-ду

## Пайплайн

```
фото → видалення фону → ракурси (Zero123++) → підтвердити / перегенерувати
     → маски → space carving → marching cubes → mesh → текстура → GLB → viewer
```

Претреновані моделі беремо лише для окремих кроків (фон, ракурси). Реконструкцію в 3D пишемо самі.

---

## Архітектура (шари)

```
photo3d/
├── pyproject.toml
├── README.md
├── src/photo3d/
│   ├── domain/                  # чисті сутності й інтерфейси, без залежностей від бібліотек ШІ
│   │   ├── models.py            # Camera, View, ViewSet, VoxelGrid, Mesh
│   │   └── interfaces.py        # BackgroundRemover, ViewGenerator, MeshExporter (Protocol)
│   │
│   ├── services/                # наша логіка (алгоритми)
│   │   ├── camera_math.py       # проєкція 3D → піксель
│   │   ├── carving.py           # space carving
│   │   ├── meshing.py           # marching cubes, чистка
│   │   └── texturing.py         # колір вершин з ракурсів
│   │
│   ├── infrastructure/          # все зовнішнє: моделі, файли, формати
│   │   ├── ai/
│   │   │   ├── rembg_remover.py       # реалізує BackgroundRemover
│   │   │   └── zero123_generator.py   # реалізує ViewGenerator
│   │   ├── storage/
│   │   │   └── views_repository.py    # читання/запис views/ і cameras.json
│   │   └── export/
│   │       └── glb_exporter.py        # реалізує MeshExporter
│   │
│   ├── app/                     # сценарії використання, склеювання шарів
│   │   ├── generate_views.py    # фото → ракурси
│   │   ├── reconstruct.py       # ракурси → model.glb
│   │   └── container.py         # збирає реалізації (dependency injection)
│   │
│   └── ui/                      # лише презентація
│       ├── gradio_app.py
│       └── viewer/index.html    # three.js
│
├── notebooks/                   # Colab для Zero123++
├── data/                        # у .gitignore
│   ├── test_photos/
│   └── synthetic/
└── tests/
```

**Правило залежностей (головне, що тримає архітектуру чистою):**

```
ui → app → services → domain
            ↑
infrastructure (реалізує інтерфейси з domain)
```

- `domain` не імпортує нічого з інших шарів.
- `services` знають лише про `domain`. Вони не знають про `rembg`, Gradio чи GLB.
- `infrastructure` імплементує інтерфейси з `domain/interfaces.py`.
- `app` бере сервіси та інфраструктуру й складає з них сценарії.
- `ui` викликає тільки `app`.

Це не перевантаження: шарів п'ять, кожен з 1–4 файлів. Додавати ще шари (наприклад, окремий `api/` чи `core/`) не потрібно.

**Контракт між нами:**
- Студент 2 віддає `views/` з PNG (RGBA, маска = альфа-канал) і `cameras.json` (`azimuth`, `elevation`, `distance`, `fov` для кожного ракурсу).
- Студент 1 віддає функцію `reconstruct(views_dir) -> model.glb`.

---

## Правила гілок

- Гілка створюється від свіжого `main`: `git checkout main && git pull && git checkout -b <гілка>`.
- Одна задача = одна гілка = один Pull Request.
- Назви: `feat/...` (нова функція), `chore/...` (допоміжне), `docs/...` (документи).
- Мерджимо в `main` через PR, інший студент дивиться код.

---

## Студент 1 (Мікс)

- [ ] **Каркас шарів і інтерфейси**
  Гілка: `feat/project-skeleton`
  Що в ній: папки `domain/services/infrastructure/app/ui`, `models.py`, `interfaces.py`, заглушки функцій, `.gitignore`, README.

- [ ] **Модель камери**
  Гілка: `feat/camera-model`
  Що в ній: `services/camera_math.py`, `project(points_3d, camera) -> pixels`, тести проєкції.

- [ ] **Синтетичні дані**
  Гілка: `feat/synthetic-data`
  Що в ній: скрипт, який рендерить силуети готового 3D-об'єкта з тих самих кутів, що й Zero123++; це ground truth для тестів.

- [ ] **Space carving**
  Гілка: `feat/space-carving`
  Що в ній: `services/carving.py`, voxel grid, відсікання вокселів за масками, тест на синтетиці.

- [ ] **Побудова mesh**
  Гілка: `feat/meshing`
  Що в ній: `services/meshing.py`, marching cubes, згладжування, видалення дрібних артефактів.

- [ ] **Текстурування**
  Гілка: `feat/texturing`
  Що в ній: `services/texturing.py`, перевірка видимості вершин, колір з ракурсів.

- [ ] **Стійкість до шуму**
  Гілка: `feat/carving-robustness`
  Що в ній: правило «воксель лишається, якщо він у силуеті у K із N ракурсів», підбір K на реальних ракурсах.

- [ ] **Експорт GLB**
  Гілка: `feat/glb-export`
  Що в ній: `infrastructure/export/glb_exporter.py`, перевірка у viewer.

- [ ] **Сценарій reconstruct і контейнер**
  Гілка: `feat/reconstruct-usecase`
  Що в ній: `app/reconstruct.py`, `app/container.py`, `reconstruct(views_dir) -> model.glb`.

- [ ] **Технічна частина звіту**
  Гілка: `docs/technical-report`
  Що в ній: опис алгоритму, формули, обмеження, результати на синтетиці.

- [ ] **Ревʼю PR напарника та інтеграція**
  Гілка: `chore/integration`
  Що в ній: виправлення на стику частин, фінальна перевірка всього пайплайну.

---

## Студент 2 (Мішікс)

- [ ] **Colab і оточення**
  Гілка: `chore/colab-setup`
  Що в ній: ноутбук у `notebooks/`, який клонує репо, ставить залежності й перевіряє GPU.

- [ ] **Видалення фону**
  Гілка: `feat/bg-removal`
  Що в ній: `infrastructure/ai/rembg_remover.py`, клас реалізує `BackgroundRemover`, повертає RGBA.

- [ ] **Тестові фото**
  Гілка: `chore/test-photos`
  Що в ній: 10 фото предметів (проста форма, складна, блискуча, прозора) у `data/test_photos/`; лише список/опис у репо, якщо файли важкі.

- [ ] **Запуск Zero123++**
  Гілка: `feat/zero123-notebook`
  Що в ній: ноутбук, який бере фото й видає сітку ракурсів, перевірено на всіх тестових фото.

- [ ] **Нарізка ракурсів і cameras.json**
  Гілка: `feat/view-generator`
  Що в ній: `infrastructure/ai/zero123_generator.py` і `storage/views_repository.py`; сітка ріжеться на PNG, кожен проходить `remove_bg`, пишеться `cameras.json` за контрактом; `generate_views(photo, seed) -> views_dir`.

- [ ] **Сценарій generate_views**
  Гілка: `feat/generate-views-usecase`
  Що в ній: `app/generate_views.py`, що викликає генератор і повертає шлях до ракурсів.

- [ ] **UI на Gradio**
  Гілка: `feat/gradio-ui`
  Що в ній: `ui/gradio_app.py`, завантаження фото, показ ракурсів, кнопки «Підтвердити» (викликає `reconstruct`) і «Перегенерувати» (новий seed).

- [ ] **Viewer на three.js**
  Гілка: `feat/threejs-viewer`
  Що в ній: `ui/viewer/index.html`, `GLTFLoader`, `OrbitControls`, автообертання (turntable).

- [ ] **Таблиця тестів**
  Гілка: `docs/test-results`
  Що в ній: прогін 10 об'єктів, таблиця «працює / ламається і чому», скріншоти.

- [ ] **Презентація**
  Гілка: `docs/presentation`
  Що в ній: слайди зі скріншотами та демо.

---

## Ризики

1. Невузгоджені ракурси від Zero123++ можуть «з'їдати» частини об'єкта при space carving. Тест на синтетиці показує, де помилка: в алгоритмі чи в ракурсах.
2. Верх об'єкта менш точний через фіксовані кути. Описати у звіті як обмеження.
3. Блискучі та прозорі предмети погано сегментуються. Описати у звіті як відомий збій.
4. Формат `cameras.json` змінюємо лише разом.
