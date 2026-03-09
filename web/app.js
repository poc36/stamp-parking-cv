/**
 * СТАМП CV — Интерактивная карта парковки
 *
 * Canvas-рендеринг парковочных мест с цветовой кодировкой,
 * зонами камер, зумом и интерактивным взаимодействием.
 */

// ─── State ────────────────────────────────────────────────
let spots = [];           // GeoJSON features
let results = {};         // detection results
let cameras = [];         // camera configs
let showCameraZones = true;
let showSpotIds = true;

// Canvas state
let canvas, ctx;
let offsetX = 0, offsetY = 0;
let scale = 1;
let isDragging = false;
let dragStartX, dragStartY;
let hoveredSpot = null;

// ─── Initialization ──────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
    canvas = document.getElementById('parkingCanvas');
    ctx = canvas.getContext('2d');

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Canvas mouse events
    canvas.addEventListener('mousedown', onMouseDown);
    canvas.addEventListener('mousemove', onMouseMove);
    canvas.addEventListener('mouseup', onMouseUp);
    canvas.addEventListener('mouseleave', onMouseLeave);
    canvas.addEventListener('wheel', onWheel);

    // Load data
    await loadSpots();
    await loadCameras();
    await loadResults();
});

function resizeCanvas() {
    const container = document.getElementById('canvasContainer');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    render();
}

// ─── Data Loading ─────────────────────────────────────────
async function loadSpots() {
    try {
        const resp = await fetch('/api/spots?park_idx=1');
        const data = await resp.json();
        spots = data.features || [];
        document.getElementById('totalSpots').textContent = spots.length;
        fitToView();
        render();
    } catch (e) {
        console.error('Failed to load spots:', e);
    }
}

async function loadCameras() {
    try {
        const resp = await fetch('/api/cameras');
        const data = await resp.json();
        cameras = data.cameras || [];
        document.getElementById('cameraCount').textContent = cameras.length;

        // Populate camera select
        const sel = document.getElementById('cameraSelect');
        sel.innerHTML = '<option value="multi">Все камеры (агрегация)</option>';
        cameras.forEach(cam => {
            sel.innerHTML += `<option value="${cam.camera_idx}">
                Камера ${cam.camera_idx} — ${cam.camera_name}</option>`;
        });
    } catch (e) {
        console.error('Failed to load cameras:', e);
    }
}

async function loadResults() {
    const testIdx = document.getElementById('testSelect').value;
    const camVal = document.getElementById('cameraSelect').value;

    try {
        const url = `/api/results?park_idx=1&test_idx=${testIdx}&camera=${camVal}`;
        const resp = await fetch(url);
        const data = await resp.json();

        if (data.error) {
            console.warn('No results:', data.error);
            results = {};
            updateStats({});
            updateSpotsList({});
            render();
            return;
        }

        results = data.spots || {};
        updateStats(data.summary || {});
        updateSpotsList(results);
        render();
    } catch (e) {
        console.error('Failed to load results:', e);
    }
}

async function runPipeline() {
    const btn = document.getElementById('btnRunPipeline');
    btn.classList.add('loading');
    btn.innerHTML = '<span class="loading-spinner"></span> Анализ...';

    const testIdx = document.getElementById('testSelect').value;
    const mode = document.getElementById('modeSelect').value;
    const camVal = document.getElementById('cameraSelect').value;
    const multi = camVal === 'multi';

    try {
        let url = `/api/run?park_idx=1&test_idx=${testIdx}&mode=${mode}&multi=${multi}`;
        if (!multi) url += `&camera_idx=${camVal}`;

        const resp = await fetch(url);
        const data = await resp.json();

        if (data.status === 'ok') {
            await loadResults();
        } else {
            alert('Ошибка: ' + (data.message || 'Unknown'));
        }
    } catch (e) {
        alert('Ошибка запуска: ' + e.message);
    } finally {
        btn.classList.remove('loading');
        btn.innerHTML = '▶ Запустить анализ';
    }
}

// ─── Stats Update ─────────────────────────────────────────
function updateStats(summary) {
    const total = summary.total_spots || spots.length || 0;
    const occupied = summary.occupied || 0;
    const free = summary.free || (total - occupied);
    const rate = summary.occupancy_rate || (total > 0 ? (occupied / total * 100).toFixed(1) : 0);

    document.getElementById('statTotal').textContent = total;
    document.getElementById('statFree').textContent = free;
    document.getElementById('statOccupied').textContent = occupied;
    document.getElementById('statRate').textContent = rate + '%';

    const bar = document.getElementById('occupancyBar');
    bar.style.width = rate + '%';
}

// ─── Spots List ───────────────────────────────────────────
function updateSpotsList(spotsData) {
    const container = document.getElementById('spotsList');
    if (!spotsData || Object.keys(spotsData).length === 0) {
        container.innerHTML = '<div class="spots-placeholder">Нет данных. Запустите анализ.</div>';
        return;
    }

    let html = '';
    const keys = Object.keys(spotsData).sort((a, b) => parseInt(a) - parseInt(b));

    keys.forEach(spotId => {
        const r = spotsData[spotId];
        const detected = r.detected;
        const inZone = r.in_work_zone !== false;
        const pct = r.occupancy_pct || 0;
        const method = (r.method || '?').substring(0, 8);

        let dotColor, statusText, pctColor;
        if (!inZone) {
            dotColor = '#555';
            statusText = 'Вне зоны';
            pctColor = '#555';
        } else if (detected) {
            dotColor = pct > 70 ? '#ff1744' : pct > 30 ? '#ffc107' : '#00c853';
            statusText = 'Занято';
            pctColor = dotColor;
        } else {
            dotColor = '#00c853';
            statusText = 'Свободно';
            pctColor = '#00c853';
        }

        html += `<div class="spot-item" data-id="${spotId}" data-detected="${detected}"
                      data-inzone="${inZone}" onclick="highlightSpot(${spotId})">
            <span class="spot-dot" style="background: ${dotColor};"></span>
            <span class="spot-id">#${spotId}</span>
            <span class="spot-status">${statusText}</span>
            <span class="spot-pct" style="color: ${pctColor};">${pct}%</span>
        </div>`;
    });

    container.innerHTML = html;
}

function filterSpots() {
    const search = document.getElementById('spotSearch').value.toLowerCase();
    const filter = document.getElementById('spotFilter').value;
    const items = document.querySelectorAll('.spot-item');

    items.forEach(item => {
        const id = item.dataset.id;
        const detected = item.dataset.detected === 'true';
        const inZone = item.dataset.inzone === 'true';

        let show = true;
        if (search && !id.includes(search)) show = false;
        if (filter === 'occupied' && (!detected || !inZone)) show = false;
        if (filter === 'free' && (detected || !inZone)) show = false;
        if (filter === 'no_coverage' && inZone) show = false;

        item.style.display = show ? '' : 'none';
    });
}

function highlightSpot(spotId) {
    // Find spot and center on it
    const feature = spots.find(f => f.properties.id === spotId);
    if (!feature) return;

    const coords = feature.geometry.coordinates[0];
    let cx = 0, cy = 0;
    coords.forEach(c => { cx += c[0]; cy += c[1]; });
    cx /= coords.length;
    cy /= coords.length;

    // Center view on this spot
    const containerW = canvas.width;
    const containerH = canvas.height;
    scale = 3;
    offsetX = containerW / 2 - cx * scale;
    offsetY = containerH / 2 - cy * scale;

    hoveredSpot = spotId;
    render();
}

// ─── Canvas Rendering ─────────────────────────────────────
function render() {
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;

    // Clear
    ctx.fillStyle = '#0d1117';
    ctx.fillRect(0, 0, w, h);

    ctx.save();
    ctx.translate(offsetX, offsetY);
    ctx.scale(scale, scale);

    // Draw camera zones
    if (showCameraZones && cameras.length > 0) {
        drawCameraZones();
    }

    // Draw parking spots
    drawSpots();

    ctx.restore();

    // Draw mini-info
    drawHUD(w, h);
}

function drawCameraZones() {
    const camColors = ['rgba(255,100,100,0.08)', 'rgba(100,255,100,0.08)',
        'rgba(100,100,255,0.08)', 'rgba(255,255,100,0.08)'];
    const camBorders = ['rgba(255,100,100,0.4)', 'rgba(100,255,100,0.4)',
        'rgba(100,100,255,0.4)', 'rgba(255,255,100,0.4)'];

    cameras.forEach((cam, i) => {
        if (!cam.work_zone || cam.work_zone.length === 0) return;

        ctx.beginPath();
        cam.work_zone.forEach((pt, j) => {
            if (j === 0) ctx.moveTo(pt.x, pt.y);
            else ctx.lineTo(pt.x, pt.y);
        });
        ctx.closePath();
        ctx.fillStyle = camColors[i % camColors.length];
        ctx.fill();
        ctx.strokeStyle = camBorders[i % camBorders.length];
        ctx.lineWidth = 2 / scale;
        ctx.setLineDash([6 / scale, 4 / scale]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Label
        const pt = cam.work_zone[0];
        ctx.fillStyle = camBorders[i % camBorders.length];
        ctx.font = `${11 / scale}px Inter, sans-serif`;
        ctx.fillText(`Cam ${cam.camera_idx}`, pt.x + 5 / scale, pt.y + 14 / scale);
    });
}

function drawSpots() {
    spots.forEach(feature => {
        const coords = feature.geometry.coordinates[0];
        const spotId = feature.properties.id;
        const r = results[String(spotId)] || null;

        // Determine color
        let fillColor, strokeColor;
        if (!r) {
            fillColor = 'rgba(85,85,85,0.15)';
            strokeColor = 'rgba(85,85,85,0.4)';
        } else if (r.in_work_zone === false) {
            fillColor = 'rgba(85,85,85,0.1)';
            strokeColor = 'rgba(85,85,85,0.25)';
        } else if (r.detected) {
            const pct = r.occupancy_pct || 100;
            fillColor = occupancyFill(pct);
            strokeColor = occupancyStroke(pct);
        } else {
            fillColor = 'rgba(0,200,83,0.2)';
            strokeColor = 'rgba(0,200,83,0.6)';
        }

        // Highlight
        if (hoveredSpot === spotId) {
            fillColor = 'rgba(88,166,255,0.4)';
            strokeColor = 'rgba(88,166,255,0.9)';
        }

        // Draw polygon
        ctx.beginPath();
        coords.forEach((c, i) => {
            if (i === 0) ctx.moveTo(c[0], c[1]);
            else ctx.lineTo(c[0], c[1]);
        });
        ctx.closePath();
        ctx.fillStyle = fillColor;
        ctx.fill();
        ctx.strokeStyle = strokeColor;
        ctx.lineWidth = 1 / scale;
        ctx.stroke();

        // Spot ID text
        if (showSpotIds && scale > 0.7) {
            let cx = 0, cy = 0;
            coords.forEach(c => { cx += c[0]; cy += c[1]; });
            cx /= coords.length;
            cy /= coords.length;

            const fontSize = Math.max(7, Math.min(12, 10 / scale));
            ctx.font = `500 ${fontSize}px Inter, sans-serif`;
            ctx.fillStyle = hoveredSpot === spotId ? '#fff' : 'rgba(255,255,255,0.7)';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(String(spotId), cx, cy);
        }
    });
}

function occupancyFill(pct) {
    if (pct < 30) return 'rgba(0,200,83,0.25)';
    if (pct < 70) return 'rgba(255,193,7,0.25)';
    return 'rgba(255,23,68,0.3)';
}

function occupancyStroke(pct) {
    if (pct < 30) return 'rgba(0,200,83,0.6)';
    if (pct < 70) return 'rgba(255,193,7,0.6)';
    return 'rgba(255,23,68,0.7)';
}

function drawHUD(w, h) {
    // Zoom level
    ctx.fillStyle = 'rgba(30,30,30,0.8)';
    ctx.fillRect(w - 80, h - 28, 72, 22);
    ctx.fillStyle = '#8b949e';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(`Zoom: ${(scale * 100).toFixed(0)}%`, w - 14, h - 12);
    ctx.textAlign = 'left';
}

// ─── Canvas Interactions ──────────────────────────────────
function onMouseDown(e) {
    isDragging = true;
    dragStartX = e.clientX - offsetX;
    dragStartY = e.clientY - offsetY;
}

function onMouseMove(e) {
    if (isDragging) {
        offsetX = e.clientX - dragStartX;
        offsetY = e.clientY - dragStartY;
        render();
        return;
    }

    // Hover detection
    const rect = canvas.getBoundingClientRect();
    const mx = (e.clientX - rect.left - offsetX) / scale;
    const my = (e.clientY - rect.top - offsetY) / scale;

    let found = null;
    for (const feature of spots) {
        const coords = feature.geometry.coordinates[0];
        if (pointInPolygon(mx, my, coords)) {
            found = feature.properties.id;
            break;
        }
    }

    if (found !== hoveredSpot) {
        hoveredSpot = found;
        render();

        if (found !== null) {
            showTooltip(e.clientX, e.clientY, found);
        } else {
            hideTooltip();
        }
    } else if (found !== null) {
        moveTooltip(e.clientX, e.clientY);
    }
}

function onMouseUp() { isDragging = false; }

function onMouseLeave() {
    isDragging = false;
    hoveredSpot = null;
    hideTooltip();
    render();
}

function onWheel(e) {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    const zoomFactor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
    const newScale = Math.max(0.2, Math.min(10, scale * zoomFactor));

    // Zoom toward mouse
    offsetX = mx - (mx - offsetX) * (newScale / scale);
    offsetY = my - (my - offsetY) * (newScale / scale);
    scale = newScale;

    render();
}

// ─── Controls ─────────────────────────────────────────────
function fitToView() {
    if (spots.length === 0) return;

    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    spots.forEach(f => {
        f.geometry.coordinates[0].forEach(c => {
            minX = Math.min(minX, c[0]);
            minY = Math.min(minY, c[1]);
            maxX = Math.max(maxX, c[0]);
            maxY = Math.max(maxY, c[1]);
        });
    });

    const dataW = maxX - minX;
    const dataH = maxY - minY;
    const padding = 40;
    const canvasW = canvas.width - padding * 2;
    const canvasH = canvas.height - padding * 2;

    scale = Math.min(canvasW / dataW, canvasH / dataH);
    offsetX = padding + (canvasW - dataW * scale) / 2 - minX * scale;
    offsetY = padding + (canvasH - dataH * scale) / 2 - minY * scale;

    render();
}

function resetView() {
    fitToView();
}

function toggleCameraZones() {
    showCameraZones = !showCameraZones;
    document.getElementById('btnCamZones').classList.toggle('active', showCameraZones);
    render();
}

function toggleSpotIds() {
    showSpotIds = !showSpotIds;
    document.getElementById('btnSpotIds').classList.toggle('active', showSpotIds);
    render();
}

// ─── Tooltip ──────────────────────────────────────────────
function showTooltip(x, y, spotId) {
    const tooltip = document.getElementById('tooltip');
    const r = results[String(spotId)] || null;

    let html = `<div class="tooltip-id">Место #${spotId}</div>`;
    if (r) {
        const inZone = r.in_work_zone !== false;
        if (!inZone) {
            html += `<span class="tooltip-status no-coverage">Вне зоны покрытия</span>`;
        } else if (r.detected) {
            html += `<span class="tooltip-status occupied">Занято (${r.occupancy_pct || 0}%)</span>`;
        } else {
            html += `<span class="tooltip-status free">Свободно</span>`;
        }

        if (r.method) {
            html += `<div class="tooltip-detail">Метод: ${r.method}</div>`;
        }
        if (r.confidence !== undefined) {
            html += `<div class="tooltip-detail">Confidence: ${(r.confidence * 100).toFixed(1)}%</div>`;
        }
        if (r.num_cameras !== undefined) {
            html += `<div class="tooltip-detail">Камер: ${r.num_cameras}</div>`;
        }
    } else {
        html += `<span class="tooltip-status no-coverage">Нет данных</span>`;
    }

    tooltip.innerHTML = html;
    tooltip.style.display = 'block';
    moveTooltip(x, y);
}

function moveTooltip(x, y) {
    const tooltip = document.getElementById('tooltip');
    tooltip.style.left = (x + 15) + 'px';
    tooltip.style.top = (y - 10) + 'px';
}

function hideTooltip() {
    document.getElementById('tooltip').style.display = 'none';
}

// ─── Geometry Helpers ─────────────────────────────────────
function pointInPolygon(x, y, polygon) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i][0], yi = polygon[i][1];
        const xj = polygon[j][0], yj = polygon[j][1];
        const intersect = ((yi > y) !== (yj > y)) &&
            (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

// Initialize button states
document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('btnCamZones').classList.add('active');
    document.getElementById('btnSpotIds').classList.add('active');
});
