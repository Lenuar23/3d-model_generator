# Локальний багаторакурсний Hunyuan Paint

Paint працює з готовим mesh і фото. Він рендерить карти нормалей та позицій,
генерує шість узгоджених ракурсів і переносить їх на UV-текстуру.
У вебзастосунку виберіть «Текстурувати з Hunyuan Paint». У Docker цей
параметр увімкнений за замовчуванням. Команда нижче запускає Paint
окремо для готової моделі.

## Налаштоване середовище Windows

- Python: `data/hunyuan_runtime/Scripts/python.exe`.
- PyTorch 2.5.1 + CUDA 12.1; Diffusers 0.32.2; Transformers 4.48.3.
- Hunyuan3D-2: `data/hunyuan_source`, revision `f8db63096c8282cb27354314d896feba5ba6ff8a`.
- CUDA 12.1.1: `data/cuda_toolkit` (nvcc, cudart, cccl з NVIDIA redist,
  контрольні SHA256 перевірені).
- Microsoft Build Tools 2022 + MSVC 14.38, Windows SDK.
- Додаткові пакети: xatlas 0.0.11, pygltflib, ninja, wheel, fast-simplification 0.2.0.
- Ваги: `data/hunyuan_paint_weights`, репозиторій `tencent/Hunyuan3D-2`,
  revision `9cd649ba6913f7a852e3286bad86bfa9a2d83dcf`.

Для локального збирання растеризатора використано вузький заголовок
`c10/cuda/CUDAStream.h` замість `ATen/cuda/CUDAContext.h` та відповідний
`c10::cuda::getCurrentCUDAStream()`. Це уникає зайвих залежностей заголовків
cuBLAS/cuSPARSE; логіка растеризації не змінюється.
Команда збірки з налаштованим середовищем: `cmd /c data\build_rasterizer.cmd`.

## Запуск

```powershell
& .\data\hunyuan_runtime\Scripts\python.exe scripts/paint_multiview.py `
  data/test_photos/01_simple_apple.jpg "$HOME/Downloads/model.glb" data/paint_trial `
  --view-size 512 --max-faces 60000
```

Папка результату має бути новою. Вихід: шість `view_XX.png`, `model.glb`
і `report.json` з параметрами та піковою виділеною CUDA-пам'яттю.
Використовується 512 — штатний розмір Paint. Проба 256 дала сильні кольорові
артефакти, тому цей режим прибрано. Режим CPU offload повільніший і потребує вільної RAM.
Форма спрощується лише у вихідній копії; оригінальний GLB не перезаписується.
Перш ніж оцінювати якість, перевірте ракурси: прихована сторона є генерацією,
а не відновленням невідомих деталей із фото.

Генератор шуму multiview налаштовано на CPU. Використовується model CPU
offload: sequential offload несумісний із прямим читанням learned embeddings
у поточній реалізації Paint (помилка meta tensor).

Перевірено на RTX 4050 Laptop 6 GB: шість ракурсів 512, UV 1024,
60 000 трикутників. Пікова пам’ять, виділена PyTorch: 4 393 466 880 байт
(це не повне використання VRAM усіма процесами). Результат: data/paint_trial_512/.
