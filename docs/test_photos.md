# Тестові фото предметів

Набір від 4 листопада 2026 року: 10 справжніх фотографій із Wikimedia Commons.
Три предмети мають просту форму, три — складну, два — блискучу поверхню,
два — прозорі. Це вхідні фото для майбутніх перевірок видалення фону,
генерації ракурсів і реконструкції; результати цих перевірок ще не отримані.

Усі 10 файлів завантажено до `data/test_photos/`, декодовано та візуально
переглянуто: головний предмет розпізнається, люди відсутні. Фото з дрібним
предметом серед безладу, обрізаною качкою та водяним знаком відхилено.
Тіні, відображення й складне тло частини фінальних кадрів залишені як
контрольовані складнощі. У чайника внизу праворуч є невеликий сторонній
фрагмент; у прозорої пляшки крізь скло видно траву. Це явно позначені
складні приклади, а не ground truth геометрії чи калібровані ракурси.

Бінарні фото ігноруються Git відповідно до наявного правила `data/`.
У репозиторії зберігаються цей опис і
[маніфест](test_photos_manifest.json): точна URL завантаження, сторінка
джерела, автор, ліцензія, розмір, SHA-256 і очікувана складність.
Маніфест описує саме завантажені байти: це або вихідне фото, або мініатюра
Wikimedia. Локальних обрізань, ретушування та видалення фону немає.
Режим JPEG також записано в маніфесті: ложка має режим `L` (градації сірого).
Перед передаванням фото адаптеру видалення фону відкривайте його через
Pillow і явно виконуйте `np.array(image.convert("RGB"))`; вихідні файли
набору при цьому залишаються без змін.

Відновлення набору зі стандартною бібліотекою Python, без нових залежностей:

```powershell
python scripts/download_test_photos.py
python scripts/download_test_photos.py --check
```

Перша команда завантажує відсутні файли та перевіряє байти за маніфестом.
Повторний запуск перевіряє наявні файли без повторного завантаження.
Змінені користувачем файли не перезаписуються: команда зупиняється з помилкою.
Для окремої копії використовуйте `--output-dir <папка>`.
`--check` працює без мережі. Якщо Commons тимчасово відповідає HTTP 429,
повторіть запуск пізніше; якщо вміст джерела змінився, SHA-256 не пройде
перевірку — оновлювати маніфест слід лише після повторної перевірки фото.

Категорії позначають головну складність: матеріал і геометрія можуть
перетинатися, наприклад, дужка замка також блищить. Очікувані проблеми
нижче є гіпотезами для майбутніх тестів, а не висновками про роботу пайплайну.

## Перелік і очікувані складнощі

| File | Category | Size | Subject / expected challenge |
| --- | --- | --- | --- |
| `01_simple_apple.jpg` | simple | 960 x 870 | Червоне яблуко: Опукла форма; перевірити заглиблення біля плодоніжки та контактну тінь. |
| `02_simple_tennis_ball.jpg` | simple | 960 x 955 | Тенісний м’яч: Майже сфера; ворс, шов і логотип можуть створити помилкові деталі. |
| `03_simple_soap.jpg` | simple | 960 x 711 | Алеппське мило: Прямокутна форма з округлими ребрами; зберегти кути й рельєф штампа. |
| `04_complex_pine_cone.jpg` | complex | 612 x 726 | Соснова шишка: Тонкі лусочки, глибокі проміжки та самооклюзія; ризик злиття деталей. |
| `05_complex_rubber_duck.jpg` | complex | 960 x 1280 | Гумова качка: Дзьоб і виступи; контроль асиметрії, тіней та відділення від стіни й підлоги. |
| `06_complex_padlock.jpg` | complex | 960 x 703 | Навісний замок: Отвір дужки й тонка перемичка; світлий метал на білому тлі може зникнути з маски. |
| `07_shiny_spoon.jpg` | shiny | 960 x 1280 | Сталева супова ложка: Дзеркальні відблиски, тонка ручка й увігнута чаша; силует не описує увігнутість. |
| `08_shiny_silver_teapot.jpg` | shiny | 960 x 594 | Срібний чайник: Відображення оточення, носик, ручка й ніжки; унизу кадру є невеликий сторонній предмет. |
| `09_transparent_bottle.jpg` | transparent | 960 x 720 | Порожня прозора пляшка: Крізь скло видно траву; заломлення й слабкий контур ускладнюють сегментацію. |
| `10_transparent_goblet.jpg` | transparent | 960 x 1285 | Порожній скляний келих: Прозора чаша й тонка ніжка на градієнтному тлі; ризик розривів маски. |

## Авторство й умови використання

Для кожної фотографії наведено окреме джерело й обрану ліцензію.
CC BY вимагає атрибуції; CC BY-SA також застосовує ShareAlike до похідних
матеріалів. Зберігайте зазначення авторства, посилання на джерело та
ліцензію й опис змін при публікації скріншотів або похідних зображень.
Наявність фото в цьому наборі не змінює його ліцензію.
Для багатоліцензійних джерел обрано Creative Commons варіант із таблиці.

| File | Source and author | License |
| --- | --- | --- |
| `01_simple_apple.jpg` | [File:Red Apple.jpg](https://commons.wikimedia.org/wiki/File:Red_Apple.jpg) - Abhijit Tembhekar from Mumbai, India | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0) |
| `02_simple_tennis_ball.jpg` | [File:Tennis ball 01.jpg](https://commons.wikimedia.org/wiki/File:Tennis_ball_01.jpg) - Fcb981 at English Wikipedia | [CC BY 2.5](https://creativecommons.org/licenses/by/2.5) |
| `03_simple_soap.jpg` | [File:Savon d'Alep Laurier 12% - 3.jpg](https://commons.wikimedia.org/wiki/File:Savon_d%27Alep_Laurier_12%25_-_3.jpg) - Lionel Allorge | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) |
| `04_complex_pine_cone.jpg` | [File:Pinus nigra cone.jpg](https://commons.wikimedia.org/wiki/File:Pinus_nigra_cone.jpg) - Menchi | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| `05_complex_rubber_duck.jpg` | [File:1 Yellow rubber duck.jpg](https://commons.wikimedia.org/wiki/File:1_Yellow_rubber_duck.jpg) - Pieter de derde | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `06_complex_padlock.jpg` | [File:Padlock infront of isolated white background 01.jpg](https://commons.wikimedia.org/wiki/File:Padlock_infront_of_isolated_white_background_01.jpg) - Jonatan Svensson Glad | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `07_shiny_spoon.jpg` | [File:Stainless Steel Soup Spoon.jpg](https://commons.wikimedia.org/wiki/File:Stainless_Steel_Soup_Spoon.jpg) - Paolomarco | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `08_shiny_silver_teapot.jpg` | [File:Silver teapot by Thomas Holland, 1806-7, Dayton Art Institute.JPG](https://commons.wikimedia.org/wiki/File:Silver_teapot_by_Thomas_Holland,_1806-7,_Dayton_Art_Institute.JPG) - Wmpearl | [CC0](https://creativecommons.org/publicdomain/zero/1.0/deed.en) |
| `09_transparent_bottle.jpg` | [File:Clear empty glass bottle.jpg](https://commons.wikimedia.org/wiki/File:Clear_empty_glass_bottle.jpg) - Santeri Viinamäki | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) |
| `10_transparent_goblet.jpg` | [File:White Wine Glass LACMA M.2004.176.5.jpg](https://commons.wikimedia.org/wiki/File:White_Wine_Glass_LACMA_M.2004.176.5.jpg) - Koloman Moser (Austria, 1868-1918), E. Bakalowits & Söhne (Austria, Vienna, active 1845-circa 1914) | [Public domain](https://creativecommons.org/publicdomain/mark/1.0/) |

Для келиха LACMA автори в джерелі — творці скляного предмета;
постачальник фотографії — Los Angeles County Museum of Art.
Статус public domain підтверджено на окремій сторінці файлу.
Дата перевірки джерел і ліцензій: **2026-11-04**.
