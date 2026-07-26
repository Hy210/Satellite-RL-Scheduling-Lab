import { getJson } from "./client";

/** 시나리오 목록 화면에 필요한 가벼운 SQLite 메타데이터다. */
export type ScenarioSummary = {
  scenario_id: string;
  name: string;
  seed: number;
  created_at: string;
  updated_at: string;
};

export type Scenario = {
  scenario_id: string;
  name: string;
  seed: number;
  satellite: SatelliteConfig;
  environment: EnvironmentConfig;
  reward: RewardConfig;
  passes: OrbitPass[];
  orders: Order[];
  strips: Strip[];
  opportunities: Opportunity[];
  ground_track_points: GroundTrackPoint[];
  footprint_samples: FootprintSample[];
  access_windows: AccessWindow[];
};

export type GeoPoint = { lat: number; lon: number };
export type Polygon = { vertices: GeoPoint[] };
/** 주문 geometry는 strip polygon들을 감싸는 축 정렬 bounding box다(`rl_core/generator.py`의 `_bounding_rectangle`). */
export type Rectangle = { min_lat: number; min_lon: number; max_lat: number; max_lon: number };
export type Order = {
  order_id: string; name: string; priority: Priority; request_start_sec: number;
  request_end_sec: number; geometry: Rectangle;
};
export type Strip = { strip_id: string; order_id: string; sequence: number; geometry: Polygon };
export type Opportunity = {
  opportunity_id: string; order_id: string; strip_id: string; pass_id: string; kind: OpportunityKind;
  capture_time_sec: number; required_roll_deg: number; required_tilt_deg: number;
  source_access_window_id: string | null;
};
export type GroundTrackPoint = {
  point_id: string; pass_id: string; sample_index: number; time_sec: number;
  latitude_deg: number; longitude_deg: number;
};
export type FootprintSample = {
  footprint_id: string; pass_id: string; sample_index: number; time_sec: number;
  geometry: Polygon;
};
export type AccessWindow = {
  access_window_id: string; pass_id: string; order_id: string; strip_id: string;
  window_start_sec: number; window_end_sec: number; min_off_nadir_time_sec: number;
  footprint_ids: string[]; opportunity_ids: string[];
};

type SatelliteConfig = {
  satellite_id: string;
  roll_limit_deg: number;
  tilt_limit_deg: number;
  combined_off_nadir_limit_deg: number;
  roll_rate_deg_per_sec: number;
  tilt_rate_deg_per_sec: number;
  settling_time_sec: number;
};

type EnvironmentConfig = {
  duration_sec: number;
  imaging_duration_sec: number;
  minimum_interval_sec: number;
  max_strips: number;
  max_candidates: number;
};

type RewardConfig = {
  angle_bonus_weight: number;
  missed_penalty_weight: number;
};

export type OrbitPass = { pass_id: string; sequence: number; start_time_sec: number; end_time_sec: number };
export type Priority = "red" | "blue" | "background";
export type OpportunityKind = "early" | "min_off_nadir" | "late";

export type OrderListItem = {
  order_id: string;
  name: string;
  priority: Priority;
  request_start_sec: number;
  request_end_sec: number;
  allowed_roll_deg: { minimum: number; maximum: number };
  allowed_tilt_deg: { minimum: number; maximum: number };
  strip_count: number;
  opportunity_count: number;
};

export type StripListItem = {
  strip_id: string;
  order_id: string;
  sequence: number;
  opportunity_count: number;
};

export type OpportunityListItem = {
  opportunity_id: string;
  order_id: string;
  strip_id: string;
  pass_id: string;
  kind: OpportunityKind;
  window_start_sec: number;
  window_end_sec: number;
  capture_time_sec: number;
  required_roll_deg: number;
  required_tilt_deg: number;
  off_nadir_deg: number;
};

export type ScenarioValidation = {
  scenario_id: string;
  valid: boolean;
  issues: Array<{ code: string; location: Array<string | number>; message: string }>;
};

export type PageResponse<Item> = { items: Item[]; offset: number; limit: number; total: number };

type ScenarioListResponse = {
  items: ScenarioSummary[];
};

/** 준비된 시나리오 목록을 읽기 전용으로 조회한다. */
export function listScenarios(signal?: AbortSignal): Promise<ScenarioSummary[]> {
  return getJson<ScenarioListResponse>("/api/scenarios", signal).then(
    ({ items }) => items,
  );
}

export function getScenario(scenarioId: string, signal?: AbortSignal): Promise<Scenario> {
  return getJson<Scenario>(`/api/scenarios/${encodeURIComponent(scenarioId)}`, signal);
}

function queryPath(path: string, query: Record<string, string | number | undefined>): string {
  const parameters = new URLSearchParams();
  Object.entries(query).forEach(([key, value]) => {
    if (value !== undefined && value !== "") parameters.set(key, String(value));
  });
  const suffix = parameters.toString();
  return suffix ? `${path}?${suffix}` : path;
}

export function listOrders(
  scenarioId: string, query: { offset: number; priority?: Priority }, signal?: AbortSignal,
): Promise<PageResponse<OrderListItem>> {
  return getJson(queryPath(`/api/scenarios/${encodeURIComponent(scenarioId)}/orders`, { limit: 100, ...query }), signal);
}

export function listStrips(
  scenarioId: string, query: { offset: number; orderId?: string }, signal?: AbortSignal,
): Promise<PageResponse<StripListItem>> {
  return getJson(queryPath(`/api/scenarios/${encodeURIComponent(scenarioId)}/strips`, { limit: 100, offset: query.offset, order_id: query.orderId }), signal);
}

export function listOpportunities(
  scenarioId: string,
  query: { offset: number; orderId?: string; stripId?: string; passId?: string; kind?: OpportunityKind },
  signal?: AbortSignal,
): Promise<PageResponse<OpportunityListItem>> {
  return getJson(queryPath(`/api/scenarios/${encodeURIComponent(scenarioId)}/opportunities`, {
    limit: 100, offset: query.offset, order_id: query.orderId, strip_id: query.stripId,
    pass_id: query.passId, kind: query.kind,
  }), signal);
}

export function validateScenario(scenarioId: string, signal?: AbortSignal): Promise<ScenarioValidation> {
  return getJson<ScenarioValidation>(`/api/scenarios/${encodeURIComponent(scenarioId)}/validation`, signal);
}

/** opportunity의 roll/tilt가 실제로 가리키는 지상 지점을 backend 계산 결과 그대로 받는다.
 * 계산식(현재는 근사 모델)은 백엔드만 알고 있고, 나중에 정밀 모델로 바뀌어도 이 함수와
 * 반환 형식은 그대로 유지된다. */
export function getOpportunityAttitudeTarget(
  scenarioId: string, opportunityId: string, signal?: AbortSignal,
): Promise<GeoPoint> {
  return getJson<GeoPoint>(
    `/api/scenarios/${encodeURIComponent(scenarioId)}/opportunities/${encodeURIComponent(opportunityId)}/attitude-target`,
    signal,
  );
}
