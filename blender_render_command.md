# Команда Рендера (type2)

## Рекомендуемая команда

```bash
blender -b -P render_script_type2.py -- \
  --object_path_pkl ./example_material/example_object_path.pkl \
  --parent_dir ./example_material \
  --light_mode uniform \
  --camera_pose z-circular-elevated
```

## Разбор параметров

- `blender`
  - Запускает Blender из терминала.

- `-b`
  - Режим без интерфейса (background).

- `-P render_script_type2.py`
  - Запускает Python-скрипт рендера внутри Blender.

- `--`
  - Разделитель между аргументами Blender и аргументами скрипта.

- `--object_path_pkl ./example_material/example_object_path.pkl`
  - Путь к `.pkl`, где лежит список путей к 3D-моделям.
  - Это входной батч для рендера.

- `--parent_dir ./example_material`
  - Корневая папка для результатов.
  - Скрипт пишет файлы в `./example_material/rendered_imgs/<part_id>/...`.

- `--light_mode uniform`
  - Режим освещения.
  - Доступные значения:
    - `uniform` (рекомендуется): два равномерных источника света с противоположных сторон.
    - `random`: случайные источники света.
    - `camera`: свет из позиции камеры.
    - `basic`: простое освещение через композитор (в Blender 5 может автоматически откатиться, если нужные ноды недоступны).

- `--camera_pose z-circular-elevated`
  - Режим траектории камеры.
  - Доступные значения:
    - `random`: случайные направления камеры.
    - `z-circular`: круговой обход вокруг оси Z.
    - `z-circular-elevated`: круговой обход вокруг Z с наклоном по высоте.

## Другие доступные аргументы

- `--num_images <int>`
  - Количество ракурсов на одну модель.
  - По умолчанию: `20`.

- `--camera_dist_min <float>`
  - Минимальная дистанция камеры до объекта.
  - По умолчанию: `2.0`.

- `--camera_dist_max <float>`
  - Максимальная дистанция камеры до объекта.
  - По умолчанию: `2.0`.

- `--fast_mode`
  - Флаг быстрого режима (булевый переключатель).

- `--extract_material`
  - Включает режим извлечения материалов.
  - В текущем скрипте по умолчанию выключен (включается только флагом).

- `--delete_material`
  - Удаляет материалы перед рендером.

- `--uniform_light_direction <x y z>`
  - Направление света для `uniform`.
  - По умолчанию: `0.09387503 -0.63953443 -0.7630093`.

- `--basic_ambient <float>`
  - Компонента ambient для режима `basic`.
  - По умолчанию: `0.3`.

- `--basic_diffuse <float>`
  - Компонента diffuse для режима `basic`.
  - По умолчанию: `0.7`.

## Быстрые примеры

Рендер только 4 ракурсов:

```bash
blender -b -P render_script_type2.py -- \
  --object_path_pkl ./example_material/example_object_path.pkl \
  --parent_dir ./example_material \
  --light_mode uniform \
  --camera_pose z-circular-elevated \
  --num_images 4
```

Случайный свет и случайная камера:

```bash
blender -b -P render_script_type2.py -- \
  --object_path_pkl ./example_material/example_object_path.pkl \
  --parent_dir ./example_material \
  --light_mode random \
  --camera_pose random
```
