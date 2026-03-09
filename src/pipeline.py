"""
Основной pipeline детекции занятости парковочных мест.

Поддерживает:
  - Одиночную камеру: python src/pipeline.py --park_idx 1 --test_idx 1 --camera_idx 1
  - Мульти-камеру:    python src/pipeline.py --park_idx 1 --test_idx 1 --multi-camera
  - Режимы детекции:  yolo, feature, hybrid
  - Сравнение с GT:   --compare-gt
"""

import argparse
import json
import os
import sys
import time
import glob

# Добавляем путь к проекту
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import numpy as np

from calibration import CameraCalibration
from detection import ParkingMarkup, OccupancyDetector, MultiCameraAggregator
from utils import cv2_imread_unicode, cv2_imwrite_unicode
from visualization import (
    draw_results_on_camera,
    draw_results_on_bev,
    create_info_panel,
    compose_visualization,
    draw_multi_camera_bev,
)


def find_cameras(data_dir: str) -> list:
    """Находит все калибровочные файлы камер в директории данных.

    Returns:
        list of int: отсортированные индексы камер
    """
    pattern = os.path.join(data_dir, "calibrate_*.json")
    files = glob.glob(pattern)
    indices = []
    for f in files:
        basename = os.path.basename(f)
        # calibrate_<idx>.json
        try:
            idx = int(basename.replace("calibrate_", "").replace(".json", ""))
            indices.append(idx)
        except ValueError:
            continue
    return sorted(indices)


def run_single_camera_pipeline(park_idx: int, test_idx: int,
                                camera_idx: int = 1,
                                mode: str = "hybrid",
                                visualize: bool = True,
                                data_dir: str = None,
                                results_dir: str = None) -> dict:
    """
    Запускает pipeline детекции для одной камеры.

    Args:
        park_idx: номер парковочного объекта
        test_idx: номер теста
        camera_idx: номер камеры
        mode: режим детекции ("yolo", "feature", "hybrid")
        visualize: создавать ли визуализацию
        data_dir: путь к данным
        results_dir: путь для результатов

    Returns:
        dict: результат в формате ТЗ
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_dir is None:
        data_dir = os.path.join(project_root, "data")
    if results_dir is None:
        results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 60)
    print(f"  СТАМП CV Pipeline — Камера {camera_idx}")
    print(f"  Park {park_idx}, Test {test_idx}, Режим: {mode.upper()}")
    print("=" * 60)

    # ─── 1. Загрузка данных ───────────────────────────────────────────
    print("\n[1/5] Загрузка данных...")

    # Разметка
    geojson_path = os.path.join(data_dir, f"park_{park_idx}.geojson")
    if not os.path.exists(geojson_path):
        print(f"  ✗ Не найден файл разметки: {geojson_path}")
        return None
    markup = ParkingMarkup(geojson_path)
    print(f"  ✓ Разметка: {len(markup.spots)} мест")

    # Калибровка
    calib_path = os.path.join(data_dir, f"calibrate_{camera_idx}.json")
    if not os.path.exists(calib_path):
        print(f"  ✗ Не найден файл калибровки: {calib_path}")
        return None
    calibration = CameraCalibration(calib_path)
    print(f"  ✓ Калибровка: {calibration.camera_name}")
    if calibration.work_zone:
        print(f"  ✓ Рабочая зона: задана ({len(calibration.work_zone_points)} точек)")

    # Тестовое изображение
    img_path = os.path.join(data_dir, f"test_{camera_idx}_{park_idx}_{test_idx}.jpg")
    if not os.path.exists(img_path):
        img_path = os.path.join(data_dir, f"test_{camera_idx}_{park_idx}_{test_idx}.png")
    if not os.path.exists(img_path):
        print(f"  ✗ Не найдено тестовое изображение: {img_path}")
        return None
    camera_image = cv2_imread_unicode(img_path)
    if camera_image is None:
        print(f"  ✗ Не удалось прочитать изображение: {img_path}")
        return None
    print(f"  ✓ Изображение: {camera_image.shape[1]}×{camera_image.shape[0]}")

    # ─── 2. Калибровка — восстановление перспективы ────────────────────
    print("\n[2/5] Восстановление перспективы...")
    t0 = time.time()
    bev_image = calibration.image_to_bev(camera_image)
    t_calib = time.time() - t0
    print(f"  ✓ Bird's-eye view: {bev_image.shape[1]}×{bev_image.shape[0]} ({t_calib*1000:.0f}ms)")

    # ─── 3. Детекция занятости ────────────────────────────────────────
    print(f"\n[3/5] Детекция занятости (режим: {mode})...")
    t0 = time.time()
    detector = OccupancyDetector(mode=mode)
    results = detector.detect_on_camera_image(camera_image, markup, calibration)
    t_det = time.time() - t0

    in_zone = sum(1 for r in results.values() if r.get("in_work_zone", True))
    occupied_count = sum(1 for r in results.values()
                         if r["detected"] and r.get("in_work_zone", True))
    free_count = in_zone - occupied_count
    print(f"  ✓ В рабочей зоне: {in_zone} мест")
    print(f"  ✓ Занято: {occupied_count}")
    print(f"  ✓ Свободно: {free_count}")
    print(f"  ✓ Время детекции: {t_det*1000:.0f}ms")

    # ─── 4. Формирование результата ───────────────────────────────────
    print("\n[4/5] Формирование результата...")

    # Результат в формате ТЗ
    result_json = {
        "params": {
            "park_idx": park_idx,
            "calibrate_idx": calibration.camera_idx,
        },
        "result": {}
    }
    for spot_id, data in results.items():
        result_json["result"][str(spot_id)] = {
            "detected": data["detected"]
        }

    result_path = os.path.join(results_dir,
                                f"result_{park_idx}_{test_idx}_cam{camera_idx}.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Результат: {result_path}")

    # Расширенный результат с метриками
    detailed_result = {
        "params": {
            "park_idx": park_idx,
            "test_idx": test_idx,
            "camera_idx": camera_idx,
            "camera_name": calibration.camera_name,
            "mode": mode,
            "processing_time_ms": {
                "calibration": round(t_calib * 1000),
                "detection": round(t_det * 1000),
            }
        },
        "summary": {
            "total_spots": len(results),
            "in_work_zone": in_zone,
            "occupied": occupied_count,
            "free": free_count,
            "occupancy_rate": round(occupied_count / in_zone * 100, 1) if in_zone else 0,
        },
        "spots": {}
    }
    for spot_id, data in results.items():
        spot_data = {k: v for k, v in data.items() if k != "metrics"}
        detailed_result["spots"][str(spot_id)] = spot_data

    detailed_path = os.path.join(results_dir,
                                  f"detailed_{park_idx}_{test_idx}_cam{camera_idx}.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_result, f, indent=2, ensure_ascii=False, default=str)
    print(f"  ✓ Подробный результат: {detailed_path}")

    # ─── 5. Визуализация ──────────────────────────────────────────────
    if visualize:
        print("\n[5/5] Визуализация...")

        cam_vis = draw_results_on_camera(camera_image, markup, calibration, results)
        bev_vis = draw_results_on_bev(bev_image, markup, results)
        info = create_info_panel(results, width=380, height=camera_image.shape[0])
        final = compose_visualization(cam_vis, bev_vis, info)

        vis_path = os.path.join(results_dir,
                                 f"vis_{park_idx}_{test_idx}_cam{camera_idx}.png")
        cv2_imwrite_unicode(vis_path, final)
        print(f"  ✓ Визуализация: {vis_path}")

        bev_vis_path = os.path.join(results_dir,
                                     f"bev_{park_idx}_{test_idx}_cam{camera_idx}.png")
        cv2_imwrite_unicode(bev_vis_path, bev_vis)

        cam_vis_path = os.path.join(results_dir,
                                     f"cam_{park_idx}_{test_idx}_cam{camera_idx}.png")
        cv2_imwrite_unicode(cam_vis_path, cam_vis)
    else:
        print("\n[5/5] Визуализация пропущена (--no-visualize)")

    print("\n" + "=" * 60)
    print(f"  ✓ Pipeline (камера {camera_idx}) завершён!")
    rate = detailed_result['summary']['occupancy_rate']
    print(f"  Занятость: {occupied_count}/{in_zone} ({rate}%)")
    print("=" * 60)

    return results


def run_multi_camera_pipeline(park_idx: int, test_idx: int,
                               mode: str = "hybrid",
                               strategy: str = "weighted_avg",
                               visualize: bool = True,
                               data_dir: str = None,
                               results_dir: str = None) -> dict:
    """
    Запускает pipeline для всех камер и агрегирует результаты.

    Args:
        park_idx: номер парковочного объекта
        test_idx: номер теста
        mode: режим детекции
        strategy: стратегия агрегации ("weighted_avg", "max_confidence", "majority_vote")
        visualize: создавать ли визуализацию
        data_dir: путь к данным
        results_dir: путь для результатов

    Returns:
        dict: агрегированные результаты
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if data_dir is None:
        data_dir = os.path.join(project_root, "data")
    if results_dir is None:
        results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Находим все камеры
    camera_indices = find_cameras(data_dir)
    if not camera_indices:
        print("✗ Не найдено ни одной камеры!")
        return None

    print("\n" + "█" * 60)
    print(f"  СТАМП CV — МУЛЬТИ-КАМЕРНЫЙ PIPELINE")
    print(f"  Park {park_idx}, Test {test_idx}")
    print(f"  Камеры: {camera_indices}")
    print(f"  Режим: {mode.upper()}, Стратегия: {strategy}")
    print("█" * 60)

    # Загрузка разметки
    geojson_path = os.path.join(data_dir, f"park_{park_idx}.geojson")
    markup = ParkingMarkup(geojson_path)
    print(f"\n📋 Разметка: {len(markup.spots)} мест")

    # Обработка каждой камеры
    camera_results = {}
    total_time = 0

    for cam_idx in camera_indices:
        print(f"\n{'─' * 50}")
        results = run_single_camera_pipeline(
            park_idx=park_idx,
            test_idx=test_idx,
            camera_idx=cam_idx,
            mode=mode,
            visualize=visualize,
            data_dir=data_dir,
            results_dir=results_dir,
        )
        if results:
            camera_results[cam_idx] = results

    # Агрегация
    print(f"\n{'█' * 60}")
    print(f"  АГРЕГАЦИЯ ({strategy})")
    print(f"{'█' * 60}")

    aggregator = MultiCameraAggregator(strategy=strategy)
    aggregated = aggregator.aggregate(camera_results, markup)

    # Статистика
    total_spots = len(aggregated)
    occupied = sum(1 for r in aggregated.values() if r["detected"])
    free = total_spots - occupied
    covered = sum(1 for r in aggregated.values() if r["num_cameras"] > 0)
    multi_view = sum(1 for r in aggregated.values() if r["num_cameras"] > 1)

    print(f"\n📊 Результаты агрегации:")
    print(f"   Всего мест: {total_spots}")
    print(f"   Покрыто камерами: {covered}")
    print(f"   Перекрытие (>1 камеры): {multi_view}")
    print(f"   Занято: {occupied}")
    print(f"   Свободно: {free}")
    occ_rate = occupied / total_spots * 100 if total_spots else 0
    print(f"   Занятость: {occ_rate:.1f}%")

    # Сохранение агрегированного результата (формат ТЗ)
    result_json = {
        "params": {
            "park_idx": park_idx,
            "calibrate_idx": "multi",
            "cameras": camera_indices,
            "strategy": strategy,
        },
        "result": {}
    }
    for spot_id, data in aggregated.items():
        result_json["result"][str(spot_id)] = {
            "detected": data["detected"]
        }

    result_path = os.path.join(results_dir, f"result_{park_idx}_{test_idx}.json")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Результат (ТЗ формат): {result_path}")

    # Подробный агрегированный результат
    detailed_agg = {
        "params": {
            "park_idx": park_idx,
            "test_idx": test_idx,
            "mode": mode,
            "strategy": strategy,
            "cameras": camera_indices,
        },
        "summary": {
            "total_spots": total_spots,
            "covered_by_cameras": covered,
            "multi_camera_coverage": multi_view,
            "occupied": occupied,
            "free": free,
            "occupancy_rate": round(occ_rate, 1),
        },
        "spots": {}
    }
    for spot_id, data in aggregated.items():
        detailed_agg["spots"][str(spot_id)] = data

    detailed_path = os.path.join(results_dir,
                                  f"detailed_{park_idx}_{test_idx}_aggregated.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_agg, f, indent=2, ensure_ascii=False, default=str)
    print(f"✓ Подробный результат: {detailed_path}")

    # Визуализация мульти-камеры
    if visualize:
        print("\n📷 Общая визуализация мульти-камеры...")
        bev_path = os.path.join(data_dir, f"bev_{park_idx}_{test_idx}.png")
        if os.path.exists(bev_path):
            bev_full = cv2_imread_unicode(bev_path)
            if bev_full is not None:
                multi_bev = draw_multi_camera_bev(bev_full, markup, aggregated,
                                                   data_dir, camera_indices)
                multi_vis_path = os.path.join(results_dir,
                                               f"vis_{park_idx}_{test_idx}_multi.png")
                cv2_imwrite_unicode(multi_vis_path, multi_bev)
                print(f"✓ Мульти-камерная визуализация: {multi_vis_path}")

    print("\n" + "█" * 60)
    print(f"  ✓ Мульти-камерный pipeline завершён!")
    print(f"  Занятость: {occupied}/{total_spots} ({occ_rate:.1f}%)")
    print("█" * 60)

    return aggregated


def compare_with_ground_truth(results: dict, ground_truth_path: str, test_idx: int):
    """Сравнивает результаты с ground truth."""
    if not os.path.exists(ground_truth_path):
        print("  ⚠ Ground truth не найден, пропускаем сравнение")
        return

    with open(ground_truth_path, 'r') as f:
        gt = json.load(f)

    if str(test_idx) not in gt:
        print(f"  ⚠ Ground truth для теста {test_idx} не найден")
        return

    gt_data = gt[str(test_idx)]
    gt_spots = gt_data["spots"]

    correct = 0
    total = 0
    tp = 0  # true positive (правильно определено как занятое)
    fp = 0  # false positive (ошибочно определено как занятое)
    tn = 0  # true negative (правильно определено как свободное)
    fn = 0  # false negative (пропуск занятого)
    errors = []

    for spot_id_str, gt_spot in gt_spots.items():
        spot_id = int(spot_id_str)
        gt_occupied = gt_spot["occupied"]

        # Ищем предсказание
        pred_occupied = False
        if isinstance(results, dict):
            if spot_id in results:
                pred_occupied = results[spot_id].get("detected", False)
            elif spot_id_str in results:
                pred_occupied = results[spot_id_str].get("detected", False)

        total += 1
        if gt_occupied == pred_occupied:
            correct += 1
            if gt_occupied:
                tp += 1
            else:
                tn += 1
        else:
            if pred_occupied:
                fp += 1
            else:
                fn += 1
            errors.append({
                "spot_id": spot_id,
                "expected": gt_occupied,
                "predicted": pred_occupied,
            })

    accuracy = correct / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n{'═' * 50}")
    print(f"📊 СРАВНЕНИЕ С GROUND TRUTH (тест {test_idx})")
    print(f"{'═' * 50}")
    print(f"   GT описание: {gt_data.get('description', '?')}")
    print(f"   GT занятых:  {gt_data.get('occupied_count', '?')}/{gt_data.get('total_spots', '?')}")
    print(f"{'─' * 50}")
    print(f"   Accuracy:  {correct}/{total} ({accuracy:.1f}%)")
    print(f"   Precision: {precision:.1f}%")
    print(f"   Recall:    {recall:.1f}%")
    print(f"   F1-Score:  {f1:.1f}%")
    print(f"{'─' * 50}")
    print(f"   TP: {tp}  FP: {fp}")
    print(f"   TN: {tn}  FN: {fn}")

    if errors:
        print(f"\n   Ошибки ({len(errors)}):")
        for e in errors[:20]:  # показываем только первые 20
            exp = "OCCUPIED" if e["expected"] else "FREE"
            pred = "OCCUPIED" if e["predicted"] else "FREE"
            print(f"     Место #{e['spot_id']}: ожидалось {exp}, получено {pred}")
        if len(errors) > 20:
            print(f"     ... и ещё {len(errors) - 20} ошибок")

    return {
        "accuracy": round(accuracy, 1),
        "precision": round(precision, 1),
        "recall": round(recall, 1),
        "f1": round(f1, 1),
        "errors": len(errors),
    }


def main():
    parser = argparse.ArgumentParser(
        description="СТАМП CV Pipeline — Детекция занятости парковки"
    )
    parser.add_argument("--park_idx", type=int, default=1,
                        help="Номер парковочного объекта")
    parser.add_argument("--test_idx", type=int, default=1,
                        help="Номер теста")
    parser.add_argument("--camera_idx", type=int, default=1,
                        help="Номер камеры (для одиночного режима)")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["yolo", "feature", "hybrid"],
                        help="Режим детекции")
    parser.add_argument("--multi-camera", action="store_true",
                        help="Мульти-камерный режим (все камеры + агрегация)")
    parser.add_argument("--strategy", type=str, default="weighted_avg",
                        choices=["weighted_avg", "max_confidence", "majority_vote"],
                        help="Стратегия агрегации мульти-камеры")
    parser.add_argument("--no-visualize", action="store_true",
                        help="Не создавать визуализацию")
    parser.add_argument("--compare-gt", action="store_true",
                        help="Сравнить с ground truth")
    args = parser.parse_args()

    if args.multi_camera:
        results = run_multi_camera_pipeline(
            park_idx=args.park_idx,
            test_idx=args.test_idx,
            mode=args.mode,
            strategy=args.strategy,
            visualize=not args.no_visualize,
        )
    else:
        results = run_single_camera_pipeline(
            park_idx=args.park_idx,
            test_idx=args.test_idx,
            camera_idx=args.camera_idx,
            mode=args.mode,
            visualize=not args.no_visualize,
        )

    if results and args.compare_gt:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gt_path = os.path.join(project_root, "data",
                                f"ground_truth_{args.park_idx}.json")
        compare_with_ground_truth(results, gt_path, args.test_idx)


if __name__ == "__main__":
    main()
