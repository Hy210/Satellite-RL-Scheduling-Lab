import { useEffect, useRef } from "react";
import L from "leaflet";

import type { Polygon, Scenario } from "../api/scenarios";

type MapProps = {
  scenario: Scenario;
  selectedStripId?: string;
  selectedPassId?: string;
  completedStripIds?: ReadonlySet<string>;
  onSelectStrip: (stripId: string) => void;
};

const priorityColors = { red: "#ef4444", blue: "#3b82f6", background: "#94a3b8" };

function positions(geometry: Polygon): L.LatLngTuple[] {
  return geometry.vertices.map((vertex) => [vertex.lat, vertex.lon]);
}

/** Scenario geometry와 결과 선택 상태를 읽기 전용 Leaflet 레이어로 보여 준다. */
export function ScenarioMap({
  scenario, selectedStripId, selectedPassId, completedStripIds = new Set(), onSelectStrip,
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
      const polygon = L.polygon(positions(order.geometry), {
        color: priorityColors[order.priority], weight: 1,
        fillColor: status === "completed" ? "#22c55e" : status === "partial" ? "#f59e0b" : "#475569",
        fillOpacity: status === "missed" ? 0.04 : 0.12,
      }).bindTooltip(`${order.name} · ${order.priority} · ${status === "completed" ? "완료" : status === "partial" ? "부분 완료" : "미촬영"}`);
      polygon.addTo(layers);
      bounds.push(...positions(order.geometry));
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
        L.polygon(positions(sample.geometry), { color: "#14b8a6", weight: 1, opacity: 0.35, fillOpacity: 0.03 }).addTo(layers);
      });
    }
    if (bounds.length) map.fitBounds(bounds, { padding: [24, 24], maxZoom: selectedStripId ? 8 : 4 });
  }, [scenario, selectedStripId, selectedPassId, completedStripIds, onSelectStrip]);

  return <div className="scenario-map" ref={containerRef} aria-label="시나리오 지도" />;
}
