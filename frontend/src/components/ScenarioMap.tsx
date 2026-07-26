import { useEffect, useRef } from "react";
import L from "leaflet";

import type { GeoPoint, Polygon, Rectangle, Scenario } from "../api/scenarios";

type MapProps = {
  scenario: Scenario;
  selectedStripId?: string;
  selectedPassId?: string;
  completedStripIds?: ReadonlySet<string>;
  /** 선택된 촬영의 roll/tilt가 실제로 가리키는 지상 지점(백엔드 계산 결과). */
  attitudeTarget?: GeoPoint;
  /** attitudeTarget과 함께 쓰여, 위성이 그 순간 어디 있었는지(가장 가까운 ground track point)를 찾는 기준 시각. */
  captureTimeSec?: number;
  onSelectStrip: (stripId: string) => void;
};

const priorityColors = { red: "#ef4444", blue: "#3b82f6", background: "#94a3b8" };

function positions(geometry: Polygon): L.LatLngTuple[] {
  return geometry.vertices.map((vertex) => [vertex.lat, vertex.lon]);
}

/** 주문 geometry는 polygon이 아니라 strip을 감싸는 축 정렬 bounding box(Rectangle)다. */
function rectanglePositions(rectangle: Rectangle): L.LatLngTuple[] {
  return [
    [rectangle.min_lat, rectangle.min_lon],
    [rectangle.min_lat, rectangle.max_lon],
    [rectangle.max_lat, rectangle.max_lon],
    [rectangle.max_lat, rectangle.min_lon],
  ];
}

/** Scenario geometry와 결과 선택 상태를 읽기 전용 Leaflet 레이어로 보여 준다. */
export function ScenarioMap({
  scenario, selectedStripId, selectedPassId, completedStripIds = new Set(),
  attitudeTarget, captureTimeSec, onSelectStrip,
}: MapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = L.map(containerRef.current, { worldCopyJump: false, minZoom: 1 }).setView([15, 0], 2);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 10, attribution: "© OpenStreetMap contributors",
    }).addTo(map);
    mapRef.current = map;
    layerRef.current = L.layerGroup().addTo(map);
    return () => { map.remove(); mapRef.current = null; layerRef.current = null; };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    const layers = layerRef.current;
    if (!map || !layers) return;
    layers.clearLayers();
    const bounds: L.LatLngTuple[] = [];
    // strip/pass를 선택했을 때는 선택 대상 위주로 확대하고, 아무것도 선택하지 않았을 때만
    // 전체 주문을 기준으로 bounds를 잡는다(그렇지 않으면 항상 전체 시나리오로 fitBounds돼
    // 선택해도 확대되지 않는다).
    const hasSelection = Boolean(selectedStripId || selectedPassId);
    const orderById = new Map(scenario.orders.map((order) => [order.order_id, order]));
    const stripsByOrder = new Map<string, string[]>();
    scenario.strips.forEach((strip) => {
      stripsByOrder.set(strip.order_id, [...(stripsByOrder.get(strip.order_id) ?? []), strip.strip_id]);
    });
    function orderStatus(orderId: string): "completed" | "partial" | "missed" {
      const stripIds = stripsByOrder.get(orderId) ?? [];
      const completedCount = stripIds.filter((stripId) => completedStripIds.has(stripId)).length;
      if (completedCount === 0) return "missed";
      return completedCount === stripIds.length ? "completed" : "partial";
    }

    // 기본 레이어는 주문 윤곽만 표시해 full 규모에서 불필요한 SVG 노드를 줄인다.
    scenario.orders.forEach((order) => {
      const status = orderStatus(order.order_id);
      const polygon = L.polygon(rectanglePositions(order.geometry), {
        color: priorityColors[order.priority], weight: 1,
        fillColor: status === "completed" ? "#22c55e" : status === "partial" ? "#f59e0b" : "#475569",
        fillOpacity: status === "missed" ? 0.04 : 0.12,
      }).bindTooltip(`${order.name} · ${order.priority} · ${status === "completed" ? "완료" : status === "partial" ? "부분 완료" : "미촬영"}`);
      polygon.addTo(layers);
      if (!hasSelection) bounds.push(...rectanglePositions(order.geometry));
    });

    const visibleStrips = selectedPassId
      ? scenario.strips.filter((strip) => scenario.access_windows.some(
        (window) => window.pass_id === selectedPassId && window.strip_id === strip.strip_id,
      ))
      : selectedStripId
        ? scenario.strips.filter((strip) => strip.strip_id === selectedStripId)
        : [];
    visibleStrips.forEach((strip) => {
      const order = orderById.get(strip.order_id);
      const selected = strip.strip_id === selectedStripId;
      const completed = completedStripIds.has(strip.strip_id);
      const polygon = L.polygon(positions(strip.geometry), {
        color: selected ? "#facc15" : priorityColors[order?.priority ?? "background"],
        fillColor: selected ? "#facc15" : completed ? "#22c55e" : "#475569",
        weight: selected ? 4 : 2, fillOpacity: selected ? 0.45 : completed ? 0.3 : 0.12,
      }).bindTooltip(`${strip.strip_id}${completed ? " · 촬영 완료" : " · 미촬영"}`);
      polygon.on("click", () => onSelectStrip(strip.strip_id));
      polygon.addTo(layers);
      bounds.push(...positions(strip.geometry));
    });

    if (selectedPassId) {
      const track = scenario.ground_track_points.filter((point) => point.pass_id === selectedPassId);
      if (track.length > 1) {
        const trackPositions: L.LatLngTuple[] = track.map((point) => [point.latitude_deg, point.longitude_deg]);
        L.polyline(trackPositions, { color: "#dbeafe", weight: 2, opacity: 0.8 }).addTo(layers);
      }
      scenario.footprint_samples.filter((sample) => sample.pass_id === selectedPassId).forEach((sample) => {
        L.polygon(positions(sample.geometry), {
          color: "#14b8a6", weight: 1, opacity: 0.35, fillOpacity: 0.03,
        })
          .bindTooltip(`footprint · ${sample.time_sec.toFixed(0)}s`)
          .addTo(layers);
      });

      // 선택된 촬영이 실제로 조준한 지점(백엔드 계산)까지, 그 순간 위성 위치(가장 가까운
      // ground track point, 단순 최근접 탐색)에서 보조선을 긋는다.
      if (attitudeTarget && captureTimeSec !== undefined && track.length > 0) {
        const nearestTrackPoint = track.reduce((closest, point) =>
          Math.abs(point.time_sec - captureTimeSec) < Math.abs(closest.time_sec - captureTimeSec) ? point : closest,
        );
        const satellitePosition: L.LatLngTuple = [nearestTrackPoint.latitude_deg, nearestTrackPoint.longitude_deg];
        const targetPosition: L.LatLngTuple = [attitudeTarget.lat, attitudeTarget.lon];
        L.polyline([satellitePosition, targetPosition], { color: "#fb923c", weight: 3, dashArray: "6 4" }).addTo(layers);
        L.circleMarker(targetPosition, { radius: 6, color: "#fb923c", fillColor: "#fb923c", fillOpacity: 0.9 })
          .bindTooltip("이 촬영의 roll/tilt가 실제로 가리키는 지점")
          .addTo(layers);
        bounds.push(satellitePosition, targetPosition);
      }
    }
    if (bounds.length) map.fitBounds(bounds, { padding: [24, 24], maxZoom: selectedStripId ? 8 : 4 });
  }, [scenario, selectedStripId, selectedPassId, completedStripIds, attitudeTarget, captureTimeSec, onSelectStrip]);

  return <div className="scenario-map" ref={containerRef} aria-label="시나리오 지도" />;
}
