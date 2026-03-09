"""
СТАМП CV — Веб-сервер для визуализации результатов детекции.

Запуск:
  python gui_server.py
  Открыть: http://localhost:8080

API:
  GET /api/spots       — GeoJSON разметка
  GET /api/results     — результаты детекции (агрегированные или по камере)
  GET /api/cameras     — список камер
  GET /api/ground-truth — ground truth
  GET /api/run         — запуск pipeline (feature mode)
  GET /data/<file>     — статические файлы данных
  GET /results/<file>  — файлы результатов
"""

import http.server
import json
import os
import sys
import urllib.parse
import mimetypes

# Путь к проекту
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
WEB_DIR = os.path.join(PROJECT_ROOT, "web")

# Добавляем src в путь
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

PORT = 8080


class ParkingAPIHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler для API парковки и статических файлов."""

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # API endpoints
        if path == "/api/spots":
            self.handle_spots(query)
        elif path == "/api/results":
            self.handle_results(query)
        elif path == "/api/cameras":
            self.handle_cameras()
        elif path == "/api/ground-truth":
            self.handle_ground_truth(query)
        elif path == "/api/run":
            self.handle_run_pipeline(query)
        elif path.startswith("/data/"):
            self.serve_file(DATA_DIR, path[6:])
        elif path.startswith("/results/"):
            self.serve_file(RESULTS_DIR, path[9:])
        else:
            # Статические файлы из web/
            self.serve_web_file(path)

    def handle_spots(self, query):
        """Возвращает GeoJSON разметку."""
        park_idx = query.get("park_idx", ["1"])[0]
        filepath = os.path.join(DATA_DIR, f"park_{park_idx}.geojson")
        self.serve_json_file(filepath)

    def handle_results(self, query):
        """Возвращает результаты детекции."""
        park_idx = query.get("park_idx", ["1"])[0]
        test_idx = query.get("test_idx", ["1"])[0]
        camera = query.get("camera", ["multi"])[0]

        if camera == "multi":
            filepath = os.path.join(RESULTS_DIR,
                                     f"detailed_{park_idx}_{test_idx}_aggregated.json")
        else:
            filepath = os.path.join(RESULTS_DIR,
                                     f"detailed_{park_idx}_{test_idx}_cam{camera}.json")

        if os.path.exists(filepath):
            self.serve_json_file(filepath)
        else:
            self.send_json({"error": f"Результаты не найдены: {filepath}",
                           "hint": "Запустите pipeline: /api/run?test_idx=1"}, 404)

    def handle_cameras(self):
        """Возвращает список камер с калибровкой."""
        cameras = []
        import glob
        for f in sorted(glob.glob(os.path.join(DATA_DIR, "calibrate_*.json"))):
            with open(f, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                cameras.append({
                    "camera_idx": data["camera_idx"],
                    "camera_name": data.get("camera_name", f"Camera {data['camera_idx']}"),
                    "has_work_zone": bool(data.get("work_zone")),
                    "work_zone": data.get("work_zone", []),
                })
        self.send_json({"cameras": cameras, "count": len(cameras)})

    def handle_ground_truth(self, query):
        """Возвращает ground truth."""
        park_idx = query.get("park_idx", ["1"])[0]
        filepath = os.path.join(DATA_DIR, f"ground_truth_{park_idx}.json")
        self.serve_json_file(filepath)

    def handle_run_pipeline(self, query):
        """Запускает pipeline и возвращает результаты."""
        park_idx = int(query.get("park_idx", ["1"])[0])
        test_idx = int(query.get("test_idx", ["1"])[0])
        mode = query.get("mode", ["feature"])[0]
        multi = query.get("multi", ["true"])[0] == "true"

        try:
            from pipeline import run_single_camera_pipeline, run_multi_camera_pipeline

            if multi:
                results = run_multi_camera_pipeline(
                    park_idx=park_idx,
                    test_idx=test_idx,
                    mode=mode,
                    visualize=True,
                    data_dir=DATA_DIR,
                    results_dir=RESULTS_DIR,
                )
            else:
                camera_idx = int(query.get("camera_idx", ["1"])[0])
                results = run_single_camera_pipeline(
                    park_idx=park_idx,
                    test_idx=test_idx,
                    camera_idx=camera_idx,
                    mode=mode,
                    visualize=True,
                    data_dir=DATA_DIR,
                    results_dir=RESULTS_DIR,
                )

            if results:
                # Serialize results
                serializable = {}
                for k, v in results.items():
                    serializable[str(k)] = {
                        key: val for key, val in v.items()
                        if not isinstance(val, (bytes, memoryview))
                    }
                self.send_json({"status": "ok", "spots": len(serializable)})
            else:
                self.send_json({"status": "error", "message": "Pipeline failed"}, 500)
        except Exception as e:
            self.send_json({"status": "error", "message": str(e)}, 500)

    def serve_web_file(self, path):
        """Раздаёт статические файлы из web/."""
        if path == "/" or path == "":
            path = "/index.html"

        filepath = os.path.join(WEB_DIR, path.lstrip("/"))
        if os.path.exists(filepath) and os.path.isfile(filepath):
            content_type, _ = mimetypes.guess_type(filepath)
            if content_type is None:
                content_type = "application/octet-stream"

            with open(filepath, 'rb') as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(content))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, f"File not found: {path}")

    def serve_file(self, base_dir, filename):
        """Раздаёт файлы из указанной директории."""
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath) and os.path.isfile(filepath):
            content_type, _ = mimetypes.guess_type(filepath)
            if content_type is None:
                content_type = "application/octet-stream"

            with open(filepath, 'rb') as f:
                content = f.read()

            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", len(content))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, f"File not found: {filename}")

    def serve_json_file(self, filepath):
        """Раздаёт JSON файл."""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.send_json(data)
        else:
            self.send_json({"error": "File not found"}, 404)

    def send_json(self, data, status=200):
        """Отправляет JSON ответ."""
        content = json.dumps(data, ensure_ascii=False, default=str).encode('utf-8')
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(content))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        """Краткий лог."""
        sys.stderr.write(f"[{self.log_date_time_string()}] {args[0]}\n")


def main():
    print("=" * 60)
    print("  СТАМП CV — Web GUI Server")
    print(f"  http://localhost:{PORT}")
    print("=" * 60)
    print(f"\n  Data:    {DATA_DIR}")
    print(f"  Results: {RESULTS_DIR}")
    print(f"  Web:     {WEB_DIR}")
    print(f"\n  Ctrl+C to stop\n")

    server = http.server.HTTPServer(("", PORT), ParkingAPIHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
