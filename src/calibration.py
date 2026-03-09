"""
Модуль калибровки камеры.

Вычисляет матрицу гомографии из реперных точек и выполняет
перспективное преобразование изображения камеры → вид сверху (bird's-eye view).

Поддержка:
  - Калибровка по 4+ реперным точкам
  - Рабочая зона камеры (work_zone) — полигон надёжной детекции
  - Преобразование точек и полигонов между системами координат
"""

import cv2
import json
import numpy as np
from shapely.geometry import Polygon, Point


class CameraCalibration:
    """Калибровка камеры по реперным точкам.

    Attributes:
        camera_idx: номер камеры
        park_idx: номер парковочного объекта
        camera_name: имя камеры
        H_img_to_bev: матрица гомографии image → BEV
        H_bev_to_img: матрица гомографии BEV → image
        work_zone: Shapely Polygon рабочей зоны (мировые координаты)
        work_zone_points: список точек рабочей зоны
    """

    def __init__(self, calibration_path: str):
        with open(calibration_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.camera_idx = data.get("camera_idx", 1)
        self.park_idx = data.get("park_idx", 1)
        self.camera_name = data.get("camera_name", f"Camera {self.camera_idx}")

        # Извлекаем пары точек
        world_pts = []
        image_pts = []
        for rp in data["reference_points"]:
            world_pts.append([rp["world"]["x"], rp["world"]["y"]])
            image_pts.append([rp["image"]["x"], rp["image"]["y"]])

        self.world_pts = np.array(world_pts, dtype=np.float32)
        self.image_pts = np.array(image_pts, dtype=np.float32)

        # Вычисляем матрицу гомографии: image → world (bird's-eye)
        self.H_img_to_bev, _ = cv2.findHomography(self.image_pts, self.world_pts)
        # Обратная матрица: world → image
        self.H_bev_to_img, _ = cv2.findHomography(self.world_pts, self.image_pts)

        # Рабочая зона камеры
        self.work_zone = None
        self.work_zone_points = []
        if "work_zone" in data and data["work_zone"]:
            wz_pts = [(p["x"], p["y"]) for p in data["work_zone"]]
            self.work_zone_points = wz_pts
            self.work_zone = Polygon(wz_pts)

    def image_to_bev(self, image: np.ndarray, bev_size: tuple = None) -> np.ndarray:
        """Преобразует изображение камеры → bird's-eye view."""
        if bev_size is None:
            max_x = int(np.max(self.world_pts[:, 0])) + 50
            max_y = int(np.max(self.world_pts[:, 1])) + 50
            bev_size = (max_x, max_y)

        bev = cv2.warpPerspective(image, self.H_img_to_bev, bev_size,
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0))
        return bev

    def bev_to_image(self, bev: np.ndarray, img_size: tuple = None) -> np.ndarray:
        """Преобразует bird's-eye → изображение камеры."""
        if img_size is None:
            max_x = int(np.max(self.image_pts[:, 0])) + 50
            max_y = int(np.max(self.image_pts[:, 1])) + 50
            img_size = (max_x, max_y)

        img = cv2.warpPerspective(bev, self.H_bev_to_img, img_size,
                                   flags=cv2.INTER_LINEAR)
        return img

    def transform_point_to_bev(self, img_point: np.ndarray) -> np.ndarray:
        """Преобразует точку из координат камеры → мировые координаты."""
        pt = np.array([[[img_point[0], img_point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H_img_to_bev)
        return transformed[0][0]

    def transform_point_to_image(self, bev_point: np.ndarray) -> np.ndarray:
        """Преобразует мировую точку → координаты камеры."""
        pt = np.array([[[bev_point[0], bev_point[1]]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.H_bev_to_img)
        return transformed[0][0]

    def transform_polygon_to_image(self, polygon_bev: list) -> np.ndarray:
        """Преобразует полигон из мировых координат → координаты камеры."""
        pts = np.array([[p] for p in polygon_bev], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pts, self.H_bev_to_img)
        return transformed.reshape(-1, 2)

    def is_in_work_zone(self, point) -> bool:
        """Проверяет, находится ли точка в рабочей зоне камеры.

        Args:
            point: (x, y) — мировые координаты

        Returns:
            bool: True если точка в рабочей зоне или зона не задана
        """
        if self.work_zone is None:
            return True  # нет зоны — считаем всё рабочим
        return self.work_zone.contains(Point(point[0], point[1]))

    def is_spot_in_work_zone(self, spot_polygon) -> bool:
        """Проверяет, находится ли парковочное место в рабочей зоне.

        Место считается в рабочей зоне, если его центр попадает в зону.

        Args:
            spot_polygon: list of (x, y) — полигон парковочного места

        Returns:
            bool: True если место в рабочей зоне или зона не задана
        """
        if self.work_zone is None:
            return True
        center = np.mean(spot_polygon, axis=0)
        return self.is_in_work_zone(center)

    def get_work_zone_mask(self, size: tuple) -> np.ndarray:
        """Возвращает бинарную маску рабочей зоны (мировые координаты).

        Args:
            size: (width, height) маски

        Returns:
            numpy array: бинарная маска (255 = рабочая зона, 0 = вне)
        """
        mask = np.zeros((size[1], size[0]), dtype=np.uint8)
        if self.work_zone_points:
            pts = np.array(self.work_zone_points, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else:
            mask[:] = 255  # нет зоны — всё рабочее
        return mask
