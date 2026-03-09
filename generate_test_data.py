"""
Генерация синтетических тестовых данных для CV-системы детекции занятости парковки.

Создаёт:
  - Парковку на 100 машиномест (10 секций × 10 мест)
  - 4 камеры с перекрывающимися зонами обзора
  - GeoJSON-разметку парковочных мест (вид сверху)
  - JSON-калибровку каждой камеры (реперные точки + рабочая зона)
  - Несколько тестовых сценариев с разной занятостью

Использование:
  python generate_test_data.py
"""

import cv2
import numpy as np
import json
import os
import random
import math


def cv2_imwrite_unicode(filepath, img, params=None):
    """cv2.imwrite wrapper that handles Unicode/Cyrillic paths on Windows."""
    ext = os.path.splitext(filepath)[1]
    if params:
        result, encoded = cv2.imencode(ext, img, params)
    else:
        result, encoded = cv2.imencode(ext, img)
    if result:
        encoded.tofile(filepath)
    return result


# ─── Конфигурация парковки ────────────────────────────────────────────────
PARK_IDX = 1
NUM_SECTIONS = 10       # секций (пар рядов)
SPOTS_PER_ROW = 10      # мест в ряду
SPOT_W = 55             # ширина места (пиксели BEV)
SPOT_H = 100            # глубина места
GAP_X = 8               # зазор между местами по X
GAP_Y = 6               # зазор между рядами в секции
ROAD_WIDTH = 70          # ширина проезда между секциями
MARGIN = 50             # отступы от края

# Размер итогового изображения "камеры"
CAM_W, CAM_H = 1280, 720

# Размер bird-eye view (план парковки) — вычисляется автоматически
BEV_W = MARGIN * 2 + SPOTS_PER_ROW * (SPOT_W + GAP_X) - GAP_X
BEV_H = MARGIN * 2 + NUM_SECTIONS * (2 * SPOT_H + GAP_Y) + (NUM_SECTIONS - 1) * ROAD_WIDTH

# ─── Конфигурация камер (4 камеры) ────────────────────────────────────────
# Каждая камера "смотрит" на свою четверть парковки с перекрытием ~20%
CAMERA_CONFIGS = {
    1: {
        "name": "Камера 1 (северо-запад)",
        "bev_region": [0, 0, BEV_W * 0.6, BEV_H * 0.6],  # область BEV
        "cam_corners": np.array([
            [180, 100],   # верх-лево (дальний край)
            [1100, 100],  # верх-право
            [1280, 680],  # низ-право (ближний край)
            [0, 680],     # низ-лево
        ], dtype=np.float32),
    },
    2: {
        "name": "Камера 2 (северо-восток)",
        "bev_region": [BEV_W * 0.4, 0, BEV_W, BEV_H * 0.6],
        "cam_corners": np.array([
            [180, 100],
            [1100, 100],
            [1280, 680],
            [0, 680],
        ], dtype=np.float32),
    },
    3: {
        "name": "Камера 3 (юго-запад)",
        "bev_region": [0, BEV_H * 0.4, BEV_W * 0.6, BEV_H],
        "cam_corners": np.array([
            [180, 100],
            [1100, 100],
            [1280, 680],
            [0, 680],
        ], dtype=np.float32),
    },
    4: {
        "name": "Камера 4 (юго-восток)",
        "bev_region": [BEV_W * 0.4, BEV_H * 0.4, BEV_W, BEV_H],
        "cam_corners": np.array([
            [180, 100],
            [1100, 100],
            [1280, 680],
            [0, 680],
        ], dtype=np.float32),
    },
}


def generate_parking_spots():
    """Генерирует координаты 100 парковочных мест (вид сверху).

    Планировка:
      - 10 секций, каждая содержит 2 ряда мест (лицом друг к другу)
      - Между секциями — проезды
      - В каждом ряду 5 мест (итого 10 мест на секцию)

    Returns:
        list: [{id, polygon, center}, ...]
    """
    spots = []
    spot_id = 1

    for section in range(NUM_SECTIONS):
        section_y = MARGIN + section * (2 * SPOT_H + GAP_Y + ROAD_WIDTH)

        for row_in_section in range(2):
            y_base = section_y + row_in_section * (SPOT_H + GAP_Y)

            for col in range(SPOTS_PER_ROW):
                x_base = MARGIN + col * (SPOT_W + GAP_X)

                # Небольшое случайное отклонение для реализма
                jitter_x = random.uniform(-1, 1)
                jitter_y = random.uniform(-1, 1)

                polygon = [
                    [x_base + jitter_x,          y_base + jitter_y],
                    [x_base + SPOT_W + jitter_x, y_base + jitter_y],
                    [x_base + SPOT_W + jitter_x, y_base + SPOT_H + jitter_y],
                    [x_base + jitter_x,          y_base + SPOT_H + jitter_y],
                    [x_base + jitter_x,          y_base + jitter_y],  # замыкаем
                ]
                spots.append({
                    "id": spot_id,
                    "polygon": polygon,
                    "center": [x_base + SPOT_W // 2, y_base + SPOT_H // 2],
                })
                spot_id += 1

    return spots


def create_geojson(spots, filepath):
    """Сохраняет разметку в формате GeoJSON."""
    features = []
    for s in spots:
        features.append({
            "type": "Feature",
            "properties": {
                "id": s["id"],
                "type": "parking_spot"
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [s["polygon"]]
            }
        })
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)
    print(f"  ✓ GeoJSON: {filepath}  ({len(spots)} мест)")


def get_camera_bev_corners(cam_cfg):
    """Возвращает 4 угла региона BEV для камеры."""
    r = cam_cfg["bev_region"]
    return np.array([
        [r[0], r[1]],
        [r[2], r[1]],
        [r[2], r[3]],
        [r[0], r[3]],
    ], dtype=np.float32)


def create_calibration(camera_idx, cam_cfg, filepath):
    """Сохраняет калибровку камеры (4 реперные точки + рабочая зона).

    Args:
        camera_idx: номер камеры
        cam_cfg: конфигурация камеры из CAMERA_CONFIGS
        filepath: путь для сохранения JSON
    """
    bev_corners = get_camera_bev_corners(cam_cfg)
    cam_corners = cam_cfg["cam_corners"]

    calib = {
        "camera_idx": camera_idx,
        "park_idx": PARK_IDX,
        "camera_name": cam_cfg["name"],
        "reference_points": [],
        "work_zone": []
    }

    # Реперные точки (4 пары: мировые координаты ↔ координаты камеры)
    for bev_pt, cam_pt in zip(bev_corners.tolist(), cam_corners.tolist()):
        calib["reference_points"].append({
            "world": {"x": bev_pt[0], "y": bev_pt[1]},
            "image": {"x": cam_pt[0], "y": cam_pt[1]},
        })

    # Рабочая зона — полигон внутри которого камера надёжно распознаёт.
    # Немного сужаем от краёв (исключаем проблемные дальние/крайние зоны)
    r = cam_cfg["bev_region"]
    shrink_x = (r[2] - r[0]) * 0.05
    shrink_y = (r[3] - r[1]) * 0.08
    calib["work_zone"] = [
        {"x": r[0] + shrink_x, "y": r[1] + shrink_y * 2},  # дальний край — сильнее обрезаем
        {"x": r[2] - shrink_x, "y": r[1] + shrink_y * 2},
        {"x": r[2] - shrink_x, "y": r[3] - shrink_y},
        {"x": r[0] + shrink_x, "y": r[3] - shrink_y},
    ]

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Калибровка камеры {camera_idx}: {filepath}")


def draw_car_topdown(img, center, spot_w, spot_h, color=None, angle=0):
    """Рисует стилизованный автомобиль (вид сверху) на парковочном месте.

    Args:
        img: изображение для рисования
        center: центр парковочного места (x, y)
        spot_w: ширина места
        spot_h: высота места
        color: цвет машины (BGR) или None для случайного
        angle: угол поворота в градусах (небольшой, для реализма)
    """
    cx, cy = int(center[0]), int(center[1])
    if color is None:
        colors = [
            (40, 40, 40),       # чёрный
            (180, 180, 180),    # серебристый
            (200, 30, 30),      # красный
            (30, 30, 200),      # синий
            (220, 220, 220),    # белый
            (30, 80, 30),       # тёмно-зелёный
            (60, 60, 100),      # тёмно-синий
            (100, 100, 120),    # серый
            (50, 120, 180),     # голубой
            (40, 30, 120),      # бордовый
        ]
        color = random.choice(colors)

    car_w = int(spot_w * 0.72)
    car_h = int(spot_h * 0.82)

    # Небольшое случайное смещение — машина не всегда идеально по центру
    offset_x = random.randint(-3, 3)
    offset_y = random.randint(-4, 4)
    cx += offset_x
    cy += offset_y

    # Корпус
    x1 = cx - car_w // 2
    y1 = cy - car_h // 2
    x2 = cx + car_w // 2
    y2 = cy + car_h // 2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

    # Тень (лёгкая)
    shadow_color = tuple(max(c - 60, 0) for c in color)
    cv2.rectangle(img, (x1 + 2, y2), (x2 + 2, y2 + 3), shadow_color, -1)

    # Крыша (чуть меньше, светлее)
    roof_color = tuple(min(c + 35, 255) for c in color)
    rx1 = cx - car_w // 3
    ry1 = cy - car_h // 5
    rx2 = cx + car_w // 3
    ry2 = cy + car_h // 8
    cv2.rectangle(img, (rx1, ry1), (rx2, ry2), roof_color, -1)

    # Лобовое стекло
    glass_color = (160, 190, 210)
    gx1 = cx - car_w // 3
    gy1 = cy - car_h // 2 + 5
    gx2 = cx + car_w // 3
    gy2 = cy - car_h // 5
    cv2.rectangle(img, (gx1, gy1), (gx2, gy2), glass_color, -1)

    # Заднее стекло
    glass_color2 = (150, 180, 200)
    gx1 = cx - car_w // 3
    gy1 = cy + car_h // 8
    gx2 = cx + car_w // 3
    gy2 = cy + car_h // 8 + 8
    cv2.rectangle(img, (gx1, gy1), (gx2, gy2), glass_color2, -1)


def render_asphalt(width, height):
    """Рендерит текстуру асфальта."""
    # Базовый серый
    img = np.full((height, width, 3), (85, 85, 85), dtype=np.uint8)

    # Шум для текстуры
    noise = np.random.normal(0, 4, (height, width, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Небольшие пятна (имитация неровностей асфальта)
    for _ in range(width * height // 2000):
        cx = random.randint(0, width - 1)
        cy = random.randint(0, height - 1)
        r = random.randint(3, 12)
        shade = random.randint(-15, 15)
        cv2.circle(img, (cx, cy), r, (85 + shade, 85 + shade, 85 + shade), -1)

    return img


def render_bev_image(spots, occupied_ids):
    """Рендерит bird-eye-view изображение парковки (100 мест).

    Args:
        spots: список парковочных мест
        occupied_ids: set с ID занятых мест

    Returns:
        numpy array: BEV-изображение
    """
    img = render_asphalt(BEV_W, BEV_H)

    # Разметка проездов (жёлтые осевые линии)
    for section in range(NUM_SECTIONS - 1):
        road_center_y = MARGIN + (section + 1) * (2 * SPOT_H + GAP_Y) + section * ROAD_WIDTH + ROAD_WIDTH // 2
        # Пунктирная линия
        for x in range(0, BEV_W, 30):
            cv2.line(img, (x, road_center_y), (min(x + 15, BEV_W), road_center_y),
                     (0, 200, 220), 2, cv2.LINE_AA)

    # Стрелки направления на проездах
    for section in range(NUM_SECTIONS - 1):
        road_center_y = MARGIN + (section + 1) * (2 * SPOT_H + GAP_Y) + section * ROAD_WIDTH + ROAD_WIDTH // 2
        for x in range(MARGIN + 100, BEV_W - MARGIN, 250):
            # Стрелка вправо
            pts = np.array([
                [x, road_center_y + 12],
                [x + 20, road_center_y + 12],
                [x + 20, road_center_y + 18],
                [x + 35, road_center_y],
                [x + 20, road_center_y - 18],
                [x + 20, road_center_y - 12],
                [x, road_center_y - 12],
            ], dtype=np.int32)
            cv2.fillPoly(img, [pts], (200, 200, 200))

    # Парковочные линии и машины
    for s in spots:
        pts = np.array(s["polygon"][:-1], dtype=np.int32)

        # Белая разметка линий
        cv2.polylines(img, [pts], True, (220, 220, 220), 1, cv2.LINE_AA)

        # Номер места или машина
        cx, cy = int(s["center"][0]), int(s["center"][1])
        if s["id"] not in occupied_ids:
            # Пустое место — номер
            label = str(s["id"])
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            tx = cx - text_size[0] // 2
            ty = cy + text_size[1] // 2
            cv2.putText(img, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1, cv2.LINE_AA)
        else:
            draw_car_topdown(img, (cx, cy), SPOT_W, SPOT_H)

    return img


def get_camera_bev_region_image(bev_img, cam_cfg):
    """Вырезает из BEV-изображения область видимости камеры."""
    r = cam_cfg["bev_region"]
    x1, y1 = int(r[0]), int(r[1])
    x2, y2 = int(r[2]), int(r[3])

    # Ограничиваем координаты
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(bev_img.shape[1], x2)
    y2 = min(bev_img.shape[0], y2)

    return bev_img[y1:y2, x1:x2].copy()


def apply_perspective(bev_region, cam_cfg):
    """Применяет перспективное преобразование к вырезанной области BEV.

    Args:
        bev_region: вырезанная из BEV область камеры
        cam_cfg: конфигурация камеры

    Returns:
        numpy array: изображение камеры (перспектива)
    """
    h, w = bev_region.shape[:2]
    src_corners = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h],
    ], dtype=np.float32)

    cam_corners = cam_cfg["cam_corners"]
    H = cv2.getPerspectiveTransform(src_corners, cam_corners)
    cam_img = cv2.warpPerspective(
        bev_region, H, (CAM_W, CAM_H),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(35, 35, 35)
    )

    # Шум камеры
    noise = np.random.normal(0, 6, cam_img.shape).astype(np.int16)
    cam_img = np.clip(cam_img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Лёгкое размытие
    cam_img = cv2.GaussianBlur(cam_img, (3, 3), 0.4)

    return cam_img


def generate_test_scenario(spots, test_idx, occupancy_rate=0.5):
    """Генерирует один тестовый сценарий с заданной занятостью.

    Args:
        spots: все парковочные места
        test_idx: номер теста
        occupancy_rate: доля занятых мест (0.0 - 1.0)

    Returns:
        set: множество ID занятых мест
    """
    num_occupied = max(1, int(len(spots) * occupancy_rate))
    occupied_ids = set(random.sample([s["id"] for s in spots], num_occupied))
    return occupied_ids


def main():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)

    print("=" * 60)
    print("  СТАМП — Генерация тестовых данных для CV парковки")
    print(f"  Парковка: {NUM_SECTIONS * SPOTS_PER_ROW * 2} мест")
    print(f"  Камеры: {len(CAMERA_CONFIGS)}")
    print(f"  BEV размер: {BEV_W}x{BEV_H}")
    print("=" * 60)

    # 1. Генерация парковочных мест
    spots = generate_parking_spots()
    num_spots = len(spots)
    print(f"\n🅿️  Создано {num_spots} парковочных мест "
          f"({NUM_SECTIONS} секций × 2 ряда × {SPOTS_PER_ROW})")

    # 2. GeoJSON-разметка
    geojson_path = os.path.join(data_dir, f"park_{PARK_IDX}.geojson")
    create_geojson(spots, geojson_path)

    # 3. Калибровка для каждой камеры
    print(f"\n📷 Калибровка камер ({len(CAMERA_CONFIGS)} шт.):")
    for cam_idx, cam_cfg in CAMERA_CONFIGS.items():
        calib_path = os.path.join(data_dir, f"calibrate_{cam_idx}.json")
        create_calibration(cam_idx, cam_cfg, calib_path)

    # 4. Тестовые сценарии
    scenarios = [
        (1, 0.55, "55% занятость (обычная загрузка)"),
        (2, 0.15, "15% занятость (раннее утро)"),
        (3, 0.90, "90% занятость (час пик)"),
    ]

    # Ground truth для проверки
    ground_truth = {}

    for test_idx, rate, desc in scenarios:
        print(f"\n📸 Тест {test_idx}: {desc}")
        occupied_ids = generate_test_scenario(spots, test_idx, rate)
        occupied_count = len(occupied_ids)
        print(f"   Занято мест: {occupied_count}/{num_spots}"
              f" ({occupied_count/num_spots*100:.0f}%)")

        # Рендерим полный BEV
        bev_img = render_bev_image(spots, occupied_ids)

        # Сохраняем полный BEV (для отладки)
        bev_path = os.path.join(data_dir, f"bev_{PARK_IDX}_{test_idx}.png")
        cv2_imwrite_unicode(bev_path, bev_img)
        print(f"   ✓ BEV полный: {bev_path}")

        # Генерируем изображение для каждой камеры
        for cam_idx, cam_cfg in CAMERA_CONFIGS.items():
            # Вырезаем область камеры из BEV
            bev_region = get_camera_bev_region_image(bev_img, cam_cfg)

            # Перспективное преобразование
            cam_img = apply_perspective(bev_region, cam_cfg)

            # Сохраняем
            cam_path = os.path.join(data_dir,
                                     f"test_{cam_idx}_{PARK_IDX}_{test_idx}.jpg")
            cv2_imwrite_unicode(cam_path, cam_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"   ✓ Камера {cam_idx}: {cam_path}")

        # Ground truth
        gt = {}
        for s in spots:
            gt[str(s["id"])] = {"occupied": s["id"] in occupied_ids}
        ground_truth[str(test_idx)] = {
            "description": desc,
            "occupied_count": occupied_count,
            "total_spots": num_spots,
            "occupancy_rate": round(occupied_count / num_spots * 100, 1),
            "occupied_ids": sorted(list(occupied_ids)),
            "spots": gt,
        }

    # Сохраняем ground truth
    gt_path = os.path.join(data_dir, f"ground_truth_{PARK_IDX}.json")
    with open(gt_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Ground truth: {gt_path}")

    print("\n" + "=" * 60)
    print("  Генерация завершена!")
    print(f"  Данные: {data_dir}")
    print(f"  Мест: {num_spots}")
    print(f"  Камер: {len(CAMERA_CONFIGS)}")
    print(f"  Тестов: {len(scenarios)}")
    print("=" * 60)


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    main()
