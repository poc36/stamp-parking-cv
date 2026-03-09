"""
Визуализация результатов детекции занятости парковочных мест.

Рисует:
  - Полигоны мест на изображении камеры (цветовая градация по occupancy_pct)
  - Информационную панель с метриками
  - BEV-вид с разметкой
  - Мульти-камерный BEV с зонами камер
"""

import cv2
import json
import os
import numpy as np


# ─── Цвета ────────────────────────────────────────────────────────────────
COLOR_FREE = (0, 200, 0)         # зелёный — свободно
COLOR_OCCUPIED = (0, 0, 220)     # красный — занято
COLOR_UNCERTAIN = (0, 200, 220)  # жёлтый — неуверенно
COLOR_NO_COVERAGE = (120, 120, 120)  # серый — нет покрытия
COLOR_TEXT = (255, 255, 255)     # белый
COLOR_BG = (30, 30, 30)         # фон панели

# Цвета зон камер (полупрозрачные)
CAMERA_COLORS = {
    1: (255, 100, 100),   # красноватый
    2: (100, 255, 100),   # зеленоватый
    3: (100, 100, 255),   # синеватый
    4: (255, 255, 100),   # жёлтый
}


def occupancy_color(occupancy_pct: int) -> tuple:
    """Возвращает цвет BGR по проценту занятости (0–100).

    Градиент: зелёный (0%) → жёлтый (50%) → красный (100%)
    """
    pct = max(0, min(100, occupancy_pct))

    if pct < 50:
        # Зелёный → Жёлтый
        ratio = pct / 50
        r = int(0 + ratio * 0)
        g = int(200 - ratio * 0)
        b = int(0 + ratio * 200)
    else:
        # Жёлтый → Красный
        ratio = (pct - 50) / 50
        r = int(0)
        g = int(200 - ratio * 200)
        b = int(200 + ratio * 20)

    return (r, g, b)


def draw_results_on_camera(camera_image: np.ndarray,
                           markup,
                           calibration,
                           results: dict) -> np.ndarray:
    """
    Рисует результаты детекции на изображении камеры.

    Args:
        camera_image: исходное изображение
        markup: ParkingMarkup
        calibration: CameraCalibration
        results: dict {spot_id: {detected: bool, occupancy_pct: int, ...}}

    Returns:
        Изображение с наложенной визуализацией
    """
    vis = camera_image.copy()

    for spot in markup.spots:
        spot_id = spot["id"]
        cam_pts = calibration.transform_polygon_to_image(spot["polygon"])
        cam_pts_int = cam_pts.astype(np.int32)

        # Определяем цвет
        if spot_id in results:
            r = results[spot_id]
            in_zone = r.get("in_work_zone", True)
            if not in_zone:
                color = COLOR_NO_COVERAGE
                alpha = 0.15
            elif r["detected"]:
                occ = r.get("occupancy_pct", 100)
                color = occupancy_color(occ)
                alpha = 0.35
            else:
                color = COLOR_FREE
                alpha = 0.25
        else:
            color = COLOR_NO_COVERAGE
            alpha = 0.1

        # Полупрозрачная заливка
        overlay = vis.copy()
        cv2.fillPoly(overlay, [cam_pts_int], color)
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

        # Контур
        cv2.polylines(vis, [cam_pts_int], True, color, 1, cv2.LINE_AA)

        # ID места (только если в зоне)
        if spot_id in results and results[spot_id].get("in_work_zone", True):
            center = np.mean(cam_pts, axis=0).astype(int)
            label = f"{spot_id}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)[0]
            text_x = center[0] - text_size[0] // 2
            text_y = center[1] + text_size[1] // 2

            # Фон текста
            cv2.rectangle(vis,
                          (text_x - 1, text_y - text_size[1] - 1),
                          (text_x + text_size[0] + 1, text_y + 1),
                          COLOR_BG, -1)
            cv2.putText(vis, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLOR_TEXT, 1, cv2.LINE_AA)

    return vis


def draw_results_on_bev(bev_image: np.ndarray,
                        markup,
                        results: dict) -> np.ndarray:
    """BEV-визуализация — вид сверху с разметкой."""
    vis = bev_image.copy()

    for spot in markup.spots:
        spot_id = spot["id"]
        pts = np.array(spot["polygon"], dtype=np.int32)

        if spot_id in results:
            r = results[spot_id]
            in_zone = r.get("in_work_zone", True)
            if not in_zone:
                color = COLOR_NO_COVERAGE
            elif r["detected"]:
                occ = r.get("occupancy_pct", 100)
                color = occupancy_color(occ)
            else:
                color = COLOR_FREE
        else:
            color = COLOR_NO_COVERAGE

        # Полупрозрачная заливка
        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
        cv2.polylines(vis, [pts], True, color, 1, cv2.LINE_AA)

        # ID и статус
        cx, cy = int(spot["center"][0]), int(spot["center"][1])
        if spot_id in results:
            r = results[spot_id]
            if r.get("in_work_zone", True):
                status = "X" if r["detected"] else "O"
                label = f"{spot_id}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]
                tx = cx - text_size[0] // 2
                ty = cy + text_size[1] // 2
                cv2.putText(vis, label, (tx, ty),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLOR_TEXT, 1, cv2.LINE_AA)

    return vis


def create_info_panel(results: dict, width: int = 400, height: int = 720) -> np.ndarray:
    """Создаёт информационную панель с метриками."""
    panel = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    y = 30
    cv2.putText(panel, "PARKING OCCUPANCY", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)
    y += 10
    cv2.line(panel, (20, y), (width - 20, y), (80, 80, 80), 1)

    # Подсчёт с учётом рабочих зон
    in_zone = {k: v for k, v in results.items()
               if v.get("in_work_zone", True)}
    total = len(in_zone)
    occupied = sum(1 for r in in_zone.values() if r["detected"])
    free = total - occupied

    y += 35
    cv2.putText(panel, f"Total spots: {len(results)}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    y += 25
    cv2.putText(panel, f"In work zone: {total}", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

    y += 30
    cv2.circle(panel, (30, y - 5), 8, COLOR_FREE, -1)
    cv2.putText(panel, f"Free: {free}", (50, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_FREE, 1, cv2.LINE_AA)

    y += 28
    cv2.circle(panel, (30, y - 5), 8, COLOR_OCCUPIED, -1)
    cv2.putText(panel, f"Occupied: {occupied}", (50, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_OCCUPIED, 1, cv2.LINE_AA)

    y += 28
    occupancy_pct = (occupied / total * 100) if total > 0 else 0
    cv2.putText(panel, f"Occupancy: {occupancy_pct:.0f}%", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

    # Прогресс-бар
    y += 20
    bar_w = width - 60
    cv2.rectangle(panel, (20, y), (20 + bar_w, y + 18), (60, 60, 60), -1)
    fill_w = int(bar_w * occupancy_pct / 100)
    bar_color = COLOR_FREE if occupancy_pct < 50 else COLOR_UNCERTAIN if occupancy_pct < 80 else COLOR_OCCUPIED
    cv2.rectangle(panel, (20, y), (20 + fill_w, y + 18), bar_color, -1)
    cv2.rectangle(panel, (20, y), (20 + bar_w, y + 18), (100, 100, 100), 1)

    # Легенда цветов
    y += 40
    cv2.putText(panel, "LEGEND", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    y += 5
    cv2.line(panel, (20, y), (width - 20, y), (60, 60, 60), 1)

    y += 20
    cv2.circle(panel, (30, y - 3), 6, COLOR_FREE, -1)
    cv2.putText(panel, "Free (0-30%)", (45, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    y += 18
    cv2.circle(panel, (30, y - 3), 6, COLOR_UNCERTAIN, -1)
    cv2.putText(panel, "Uncertain (30-70%)", (45, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    y += 18
    cv2.circle(panel, (30, y - 3), 6, COLOR_OCCUPIED, -1)
    cv2.putText(panel, "Occupied (70-100%)", (45, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    y += 18
    cv2.circle(panel, (30, y - 3), 6, COLOR_NO_COVERAGE, -1)
    cv2.putText(panel, "No coverage", (45, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    # Детали по местам
    y += 30
    cv2.putText(panel, "SPOT DETAILS", (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    y += 5
    cv2.line(panel, (20, y), (width - 20, y), (60, 60, 60), 1)

    y += 18
    for spot_id in sorted(in_zone.keys()):
        if y > height - 15:
            cv2.putText(panel, f"... +{len(in_zone) - len([s for s in sorted(in_zone.keys()) if s <= spot_id]) + 1} more",
                        (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        (120, 120, 120), 1, cv2.LINE_AA)
            break
        r = in_zone[spot_id]
        status = "OCC" if r["detected"] else "FRE"
        color = COLOR_OCCUPIED if r["detected"] else COLOR_FREE
        occ = r.get("occupancy_pct", 0)
        method = r.get("method", "?")

        cv2.circle(panel, (28, y - 2), 4, color, -1)
        text = f"#{spot_id:3d} {status} {occ:3d}% [{method[:6]}]"
        cv2.putText(panel, text, (40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)
        y += 14

    return panel


def compose_visualization(camera_vis: np.ndarray,
                          bev_vis: np.ndarray,
                          info_panel: np.ndarray) -> np.ndarray:
    """Собирает финальное изображение: камера + BEV + инфо-панель."""
    cam_h, cam_w = camera_vis.shape[:2]
    panel_h, panel_w = info_panel.shape[:2]

    # Масштабируем BEV до высоты камеры
    bev_h, bev_w = bev_vis.shape[:2]
    scale = cam_h / bev_h
    new_bev_w = max(1, int(bev_w * scale))
    bev_resized = cv2.resize(bev_vis, (new_bev_w, cam_h))

    # Выравниваем высоту панели
    if panel_h != cam_h:
        info_panel = cv2.resize(info_panel, (panel_w, cam_h))

    # Горизонтальная склейка
    result = np.hstack([camera_vis, bev_resized, info_panel])

    return result


def draw_multi_camera_bev(bev_full: np.ndarray,
                          markup,
                          aggregated_results: dict,
                          data_dir: str = None,
                          camera_indices: list = None) -> np.ndarray:
    """Рисует мульти-камерный BEV с зонами камер и агрегированными результатами.

    Args:
        bev_full: полное BEV-изображение парковки
        markup: ParkingMarkup
        aggregated_results: агрегированные результаты
        data_dir: директория с данными (для чтения зон камер)
        camera_indices: индексы камер

    Returns:
        numpy array: визуализация
    """
    h, w = bev_full.shape[:2]

    # Масштабируем BEV до приемлемого размера для визуализации
    max_dim = 1200
    scale = min(max_dim / w, max_dim / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        vis = cv2.resize(bev_full, (new_w, new_h))
    else:
        vis = bev_full.copy()
        new_w, new_h = w, h
        scale = 1.0

    # Рисуем зоны камер
    if data_dir and camera_indices:
        for cam_idx in camera_indices:
            calib_path = os.path.join(data_dir, f"calibrate_{cam_idx}.json")
            if os.path.exists(calib_path):
                with open(calib_path, 'r') as f:
                    calib_data = json.load(f)

                if "work_zone" in calib_data and calib_data["work_zone"]:
                    wz_pts = [(int(p["x"] * scale), int(p["y"] * scale))
                              for p in calib_data["work_zone"]]
                    wz_np = np.array(wz_pts, dtype=np.int32)

                    cam_color = CAMERA_COLORS.get(cam_idx, (200, 200, 200))
                    overlay = vis.copy()
                    cv2.polylines(overlay, [wz_np], True, cam_color, 2, cv2.LINE_AA)
                    cv2.addWeighted(overlay, 0.8, vis, 0.2, 0, vis)

                    # Метка камеры
                    label_pt = wz_pts[0]
                    cv2.putText(vis, f"Cam {cam_idx}", (label_pt[0] + 5, label_pt[1] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, cam_color, 1, cv2.LINE_AA)

    # Рисуем результаты по местам
    for spot in markup.spots:
        spot_id = spot["id"]
        pts = np.array([(int(p[0] * scale), int(p[1] * scale))
                         for p in spot["polygon"]], dtype=np.int32)

        if spot_id in aggregated_results:
            r = aggregated_results[spot_id]
            if r.get("num_cameras", 0) == 0:
                color = COLOR_NO_COVERAGE
            elif r["detected"]:
                occ = r.get("occupancy_pct", 100)
                color = occupancy_color(occ)
            else:
                color = COLOR_FREE
        else:
            color = COLOR_NO_COVERAGE

        overlay = vis.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)
        cv2.polylines(vis, [pts], True, color, 1, cv2.LINE_AA)

    # Заголовок
    cv2.rectangle(vis, (0, 0), (new_w, 35), COLOR_BG, -1)
    cv2.putText(vis, "MULTI-CAMERA AGGREGATED VIEW", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_TEXT, 2, cv2.LINE_AA)

    # Статистика
    total = len(aggregated_results)
    occupied = sum(1 for r in aggregated_results.values() if r["detected"])
    free = total - occupied
    rate = occupied / total * 100 if total else 0

    stats_y = new_h - 8
    cv2.rectangle(vis, (0, stats_y - 25), (new_w, new_h), COLOR_BG, -1)
    stats_text = f"Total: {total} | Occupied: {occupied} | Free: {free} | Rate: {rate:.0f}%"
    cv2.putText(vis, stats_text, (10, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

    return vis
