"""
Модуль детекции занятости парковочных мест.

Три режима:
  1. YOLO — детекция автомобилей с проверкой пересечения с полигонами мест
  2. Feature-based — анализ текстурных/цветовых признаков внутри ROI (fallback)
  3. Hybrid — комбинация YOLO + feature-based

Поддержка:
  - Рабочая зона камеры (пропуск мест за пределами)
  - occupancy_pct (0–100%) — параметр занятости с учётом достоверности
  - MultiCameraAggregator — агрегация результатов от нескольких камер
"""

import cv2
import numpy as np
import json
from shapely.geometry import Polygon, box
from pathlib import Path


class ParkingMarkup:
    """Загрузка и работа с GeoJSON-разметкой парковочных мест."""

    def __init__(self, geojson_path: str):
        with open(geojson_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.spots = []
        for feature in data["features"]:
            coords = feature["geometry"]["coordinates"][0]
            spot_id = feature["properties"]["id"]
            # GeoJSON полигон (последняя точка = первая, убираем повтор)
            polygon = [(c[0], c[1]) for c in coords[:-1]]
            center = np.mean(polygon, axis=0)
            self.spots.append({
                "id": spot_id,
                "polygon": polygon,
                "center": center.tolist(),
                "shapely": Polygon(polygon),
            })

    def get_spot_by_id(self, spot_id: int):
        """Возвращает парковочное место по ID."""
        for s in self.spots:
            if s["id"] == spot_id:
                return s
        return None

    def get_all_ids(self):
        """Возвращает список всех ID парковочных мест."""
        return [s["id"] for s in self.spots]


class YOLODetector:
    """Детекция автомобилей с помощью YOLOv8."""

    # COCO классы: car=2, motorcycle=3, bus=5, truck=7
    VEHICLE_CLASSES = {2, 3, 5, 7}

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.25):
        from ultralytics import YOLO
        self.model = YOLO(model_name)
        self.confidence = confidence
        print(f"  ✓ YOLO модель загружена: {model_name}")

    def detect_vehicles(self, image: np.ndarray) -> list:
        """Детектирует транспортные средства на изображении.

        Returns:
            list of dict: [{bbox: [x1,y1,x2,y2], confidence: float, class_id: int}, ...]
        """
        results = self.model(image, conf=self.confidence, verbose=False)
        detections = []

        for r in results:
            for i, (bbox, conf, cls) in enumerate(
                zip(r.boxes.xyxy.cpu().numpy(),
                    r.boxes.conf.cpu().numpy(),
                    r.boxes.cls.cpu().numpy())
            ):
                class_id = int(cls)
                if class_id in self.VEHICLE_CLASSES:
                    detections.append({
                        "bbox": bbox.tolist(),
                        "confidence": float(conf),
                        "class_id": class_id,
                    })

        return detections


class FeatureDetector:
    """Feature-based детектор занятости (анализ текстур/цветов в ROI).

    Использует комбинацию edge density и color variance для определения
    занятости парковочного места. Места с машинами имеют значительно
    больше граней и цветовой дисперсии по сравнению с пустым асфальтом.
    """

    def __init__(self, edge_threshold: float = 4.0, var_threshold: float = 2000.0):
        self.edge_threshold = edge_threshold
        self.var_threshold = var_threshold

    def analyze_roi(self, image: np.ndarray, mask: np.ndarray) -> dict:
        """Анализирует ROI парковочного места.

        Args:
            image: bird-eye view изображение
            mask: маска полигона (binary, same size as image)

        Returns:
            dict с метриками
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Применяем маску
        masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
        masked_color = cv2.bitwise_and(image, image, mask=mask)

        # Считаем только пиксели внутри маски
        pixels = gray[mask > 0]
        if len(pixels) == 0:
            return {"edge_density": 0, "color_variance": 0, "mean_intensity": 0}

        # Edge density (Canny)
        edges = cv2.Canny(masked_gray, 50, 150)
        edge_pixels = edges[mask > 0]
        edge_density = np.sum(edge_pixels > 0) / len(pixels) * 100

        # Цветовая дисперсия
        color_pixels = masked_color[mask > 0]
        color_variance = np.var(color_pixels)

        # Средняя яркость
        mean_intensity = np.mean(pixels)

        return {
            "edge_density": float(edge_density),
            "color_variance": float(color_variance),
            "mean_intensity": float(mean_intensity),
        }


class OccupancyDetector:
    """Основной детектор занятости парковочных мест.

    Объединяет YOLO-детекцию и feature-based анализ.
    Поддерживает рабочие зоны камер и выдаёт occupancy_pct (0–100%).
    """

    def __init__(self, mode: str = "yolo", yolo_model: str = "yolov8n.pt"):
        """
        Args:
            mode: "yolo", "feature", или "hybrid"
            yolo_model: имя модели для YOLO
        """
        self.mode = mode
        self.yolo = None
        self.feature = FeatureDetector()

        if mode in ("yolo", "hybrid"):
            self.yolo = YOLODetector(yolo_model)

    def detect_on_camera_image(self, camera_image: np.ndarray,
                                markup: 'ParkingMarkup',
                                calibration,
                                respect_work_zone: bool = True) -> dict:
        """
        Определяет занятость мест на изображении камеры.

        Подход:
        1. YOLO детектирует машины на изображении камеры
        2. Полигоны парковочных мест преобразуются в координаты камеры
        3. Проверяется пересечение bboxes машин с полигонами мест

        Args:
            camera_image: изображение с камеры (numpy array BGR)
            markup: ParkingMarkup — разметка парковочных мест (BEV координаты)
            calibration: CameraCalibration — калибровка камеры
            respect_work_zone: учитывать ли рабочую зону камеры

        Returns:
            dict: {spot_id: {detected, confidence, occupancy_pct, method, in_work_zone}}
        """
        results = {}

        # Фильтрация по рабочей зоне
        active_spots = []
        for spot in markup.spots:
            in_wz = True
            if respect_work_zone and hasattr(calibration, 'is_spot_in_work_zone'):
                in_wz = calibration.is_spot_in_work_zone(spot["polygon"])

            if in_wz:
                active_spots.append(spot)
            else:
                # Место за пределами рабочей зоны — не анализируем
                results[spot["id"]] = {
                    "detected": False,
                    "confidence": 0.0,
                    "occupancy_pct": 0,
                    "method": "out_of_zone",
                    "in_work_zone": False,
                }

        if self.mode in ("yolo", "hybrid"):
            yolo_results = self._detect_yolo(camera_image, active_spots,
                                              markup, calibration)
            results.update(yolo_results)

        if self.mode in ("feature", "hybrid"):
            # Feature-based на bird-eye view
            bev_image = calibration.image_to_bev(camera_image)
            feature_results = self._detect_features(bev_image, active_spots, markup)

            if self.mode == "hybrid":
                # Объединяем: если YOLO не нашёл, но feature нашёл → feature
                for spot_id, feat_data in feature_results.items():
                    if spot_id not in results:
                        results[spot_id] = feat_data
                    else:
                        if results[spot_id].get("method") == "out_of_zone":
                            continue
                        if not results[spot_id]["detected"] and feat_data["detected"]:
                            results[spot_id] = feat_data
                            results[spot_id]["method"] = "hybrid_feature"
                        elif results[spot_id]["detected"] and feat_data["detected"]:
                            # Оба нашли — усредняем confidence
                            avg_conf = (results[spot_id]["confidence"] +
                                       feat_data["confidence"]) / 2
                            results[spot_id]["confidence"] = avg_conf
                            results[spot_id]["occupancy_pct"] = int(avg_conf * 100)
                            results[spot_id]["method"] = "hybrid_both"
            else:
                results.update(feature_results)

        return results

    def _detect_yolo(self, camera_image, active_spots, markup, calibration) -> dict:
        """YOLO-based детекция с occupancy_pct."""
        detections = self.yolo.detect_vehicles(camera_image)
        results = {}

        # Преобразуем полигоны активных мест в координаты камеры
        spot_polys_in_cam = {}
        for spot in active_spots:
            cam_pts = calibration.transform_polygon_to_image(spot["polygon"])
            spot_poly = Polygon(cam_pts.tolist())
            spot_polys_in_cam[spot["id"]] = spot_poly

        # Создаём shapely boxes из YOLO детекций
        det_boxes = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            det_boxes.append({
                "polygon": box(x1, y1, x2, y2),
                "confidence": d["confidence"],
            })

        # Проверяем пересечения
        for spot in active_spots:
            spot_id = spot["id"]
            spot_poly = spot_polys_in_cam[spot_id]
            occupied = False
            max_iou = 0.0
            best_conf = 0.0

            for db in det_boxes:
                if not spot_poly.is_valid or not db["polygon"].is_valid:
                    continue
                intersection = spot_poly.intersection(db["polygon"]).area
                union = spot_poly.union(db["polygon"]).area
                iou = intersection / union if union > 0 else 0

                # Какая часть bbox попадает в spot
                overlap_ratio = intersection / spot_poly.area if spot_poly.area > 0 else 0

                if overlap_ratio > 0.15 or iou > 0.1:
                    occupied = True
                    if db["confidence"] > best_conf:
                        best_conf = db["confidence"]
                        max_iou = max(iou, overlap_ratio)

            # occupancy_pct: confidence × IoU → 0–100%
            occ_pct = int(min(best_conf * max_iou * 150, 100)) if occupied else 0
            # Если уверенно занято — минимум 60%
            if occupied and occ_pct < 60:
                occ_pct = max(occ_pct, int(best_conf * 100))

            results[spot_id] = {
                "detected": occupied,
                "confidence": float(best_conf) if occupied else 0.0,
                "iou": float(max_iou),
                "occupancy_pct": occ_pct,
                "method": "yolo",
                "in_work_zone": True,
            }

        return results

    def _detect_features(self, bev_image, active_spots, markup) -> dict:
        """Feature-based детекция на BEV-изображении с occupancy_pct."""
        results = {}
        h, w = bev_image.shape[:2]

        for spot in active_spots:
            # Создаём маску для полигона
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(spot["polygon"], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)

            # Анализируем ROI
            metrics = self.feature.analyze_roi(bev_image, mask)

            # Комбинированная оценка
            edge_score = metrics["edge_density"] / self.feature.edge_threshold
            var_score = metrics["color_variance"] / self.feature.var_threshold

            # Взвешенная комбинация
            combined_score = (edge_score * 0.4 + var_score * 0.6)
            occupied = combined_score > 1.0

            confidence = min(combined_score, 1.0)
            occ_pct = int(confidence * 100) if occupied else int(combined_score * 50)

            results[spot["id"]] = {
                "detected": occupied,
                "confidence": float(confidence),
                "occupancy_pct": occ_pct,
                "metrics": metrics,
                "method": "feature",
                "in_work_zone": True,
            }

        return results


class MultiCameraAggregator:
    """Агрегация результатов детекции от нескольких камер.

    Для каждого парковочного места объединяет оценки от N камер,
    формируя итоговый параметр занятости.

    Стратегии агрегации:
      - "weighted_avg": взвешенное среднее по confidence
      - "max_confidence": выбор камеры с максимальным confidence
      - "majority_vote": голосование большинством
    """

    def __init__(self, strategy: str = "weighted_avg"):
        """
        Args:
            strategy: стратегия агрегации
        """
        self.strategy = strategy

    def aggregate(self, camera_results: dict, markup: 'ParkingMarkup') -> dict:
        """Агрегирует результаты от нескольких камер.

        Args:
            camera_results: {camera_idx: {spot_id: {detected, confidence, ...}}}
            markup: ParkingMarkup для получения списка всех мест

        Returns:
            dict: {spot_id: {detected, confidence, occupancy_pct, sources, ...}}
        """
        all_spot_ids = markup.get_all_ids()
        aggregated = {}

        for spot_id in all_spot_ids:
            # Собираем данные по этому месту от всех камер
            cam_data = []
            for cam_idx, results in camera_results.items():
                if spot_id in results:
                    r = results[spot_id]
                    # Пропускаем камеры, для которых место вне рабочей зоны
                    if not r.get("in_work_zone", True):
                        continue
                    cam_data.append({
                        "camera_idx": cam_idx,
                        "detected": r["detected"],
                        "confidence": r.get("confidence", 0),
                        "occupancy_pct": r.get("occupancy_pct", 0),
                        "method": r.get("method", "unknown"),
                    })

            if not cam_data:
                # Ни одна камера не видит это место
                aggregated[spot_id] = {
                    "detected": False,
                    "confidence": 0.0,
                    "occupancy_pct": 0,
                    "num_cameras": 0,
                    "sources": [],
                    "method": "no_coverage",
                }
                continue

            aggregated[spot_id] = self._aggregate_spot(spot_id, cam_data)

        return aggregated

    def _aggregate_spot(self, spot_id: int, cam_data: list) -> dict:
        """Агрегирует данные по одному месту от нескольких камер.

        Args:
            spot_id: ID парковочного места
            cam_data: список данных от камер

        Returns:
            dict: агрегированный результат
        """
        num_cameras = len(cam_data)
        sources = [{"camera": d["camera_idx"], "detected": d["detected"],
                     "confidence": d["confidence"]}
                    for d in cam_data]

        if self.strategy == "weighted_avg":
            return self._weighted_average(cam_data, num_cameras, sources)
        elif self.strategy == "max_confidence":
            return self._max_confidence(cam_data, num_cameras, sources)
        elif self.strategy == "majority_vote":
            return self._majority_vote(cam_data, num_cameras, sources)
        else:
            return self._weighted_average(cam_data, num_cameras, sources)

    def _weighted_average(self, cam_data, num_cameras, sources) -> dict:
        """Взвешенное среднее по confidence."""
        total_weight = 0
        weighted_occ = 0

        for d in cam_data:
            weight = d["confidence"] if d["confidence"] > 0 else 0.1
            if d["detected"]:
                weighted_occ += weight * d["occupancy_pct"]
            total_weight += weight

        avg_occ = weighted_occ / total_weight if total_weight > 0 else 0

        # Итоговое решение
        votes_occupied = sum(1 for d in cam_data if d["detected"])
        avg_confidence = sum(d["confidence"] for d in cam_data) / num_cameras

        # Если хотя бы половина камер считает занятым — занято
        detected = votes_occupied > num_cameras / 2
        # Или если одна камера уверена (>80%)
        if any(d["confidence"] > 0.8 and d["detected"] for d in cam_data):
            detected = True

        occ_pct = int(avg_occ) if detected else int(avg_occ * 0.5)

        return {
            "detected": detected,
            "confidence": round(avg_confidence, 3),
            "occupancy_pct": min(occ_pct, 100),
            "num_cameras": num_cameras,
            "sources": sources,
            "method": "aggregated_weighted_avg",
        }

    def _max_confidence(self, cam_data, num_cameras, sources) -> dict:
        """Выбор камеры с максимальным confidence."""
        best = max(cam_data, key=lambda d: d["confidence"])

        return {
            "detected": best["detected"],
            "confidence": best["confidence"],
            "occupancy_pct": best["occupancy_pct"],
            "num_cameras": num_cameras,
            "sources": sources,
            "method": "aggregated_max_confidence",
            "best_camera": best["camera_idx"],
        }

    def _majority_vote(self, cam_data, num_cameras, sources) -> dict:
        """Голосование большинством."""
        votes_occupied = sum(1 for d in cam_data if d["detected"])
        detected = votes_occupied > num_cameras / 2

        avg_confidence = sum(d["confidence"] for d in cam_data) / num_cameras
        avg_occ = sum(d["occupancy_pct"] for d in cam_data) / num_cameras

        return {
            "detected": detected,
            "confidence": round(avg_confidence, 3),
            "occupancy_pct": int(avg_occ),
            "num_cameras": num_cameras,
            "votes_for": votes_occupied,
            "votes_against": num_cameras - votes_occupied,
            "sources": sources,
            "method": "aggregated_majority_vote",
        }
