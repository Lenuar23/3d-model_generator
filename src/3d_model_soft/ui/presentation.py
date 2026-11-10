"""Application text and styling shared by the Gradio callbacks."""

TEXT = {
    "Згенеруйте шість ракурсів предмета з одного фото.": "Create six views of an object from one photo.",
    "Фото предмета": "Object photo",
    "Згенерувати ракурси": "Generate views",
    "Перегенерувати": "Regenerate",
    "Підтвердити": "Build 3D model",
    "Стан": "Status",
    "Завантажити GLB": "Download GLB",
    "Перегляд 3D-моделі": "3D model preview",
    "Ракурси": "Views",
    "Ракурс": "View",
    "Завантажте фото предмета.": "Upload an object photo.",
    "Фото готове. Згенеруйте ракурси.": "Photo ready. Generate views.",
    "Готуємо ракурси…": "Preparing views…",
    "Реконструюємо модель…": "Building the 3D model…",
    "Модель готова. Перегляньте її у 3D або завантажте GLB.": "Model ready. Explore it in 3D or download the GLB.",
    "Реконструкція поки недоступна.": "Reconstruction is currently unavailable.",
    "Показано 6 готових демо-ракурсів.": "Showing six saved demo views.",
    "Zero123++ згенерував 6 ракурсів.": "Zero123++ generated six views.",
    "Згенеровано 6 ракурсів.": "Generated six views.",
    "Не вдалося згенерувати ракурси. Перевірте фото або журнал запуску.": "View generation failed. Check the photo or application log.",
    "Не вдалося реконструювати модель. Перевірте журнал запуску або спробуйте інші ракурси.": "Reconstruction failed. Check the application log or try other views.",
    "Жива генерація Zero123++ недоступна. Запустіть демо у Windows із налаштованим середовищем Zero123++.": "Live Zero123++ generation is unavailable. Run the demo on Windows with Zero123++ configured.",
    "Обертайте модель мишею, наближайте колесом.": "Drag to rotate. Scroll to zoom.",
}


def translate(text, language):
    if language == "en":
        for source, target in sorted(TEXT.items(), key=lambda pair: -len(pair[0])):
            text = text.replace(source, target)
    return text


CSS = """
.gradio-container { max-width: 1380px !important; margin: auto; padding: 32px !important; }
#studio-header { padding: 16px 0 24px; }
#studio-header h1 { font-size: 34px; letter-spacing: -1px; }
#studio-header p { opacity: .75; font-size: 16px; }
.studio-card { border: 1px solid var(--border-color-primary); border-radius: 20px;
 padding: 20px; background: var(--block-background-fill); }
.gradio-container button { border-radius: 12px !important; }
#model-preview { border-radius: 16px; overflow: hidden; }
@media (max-width: 700px) { .gradio-container { padding: 12px !important; }
 .studio-card { padding: 12px; } }
"""
