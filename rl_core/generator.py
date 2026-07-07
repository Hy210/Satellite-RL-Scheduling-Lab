"""동일 seed에서 재현 가능한 학습용 가상 시나리오를 생성한다."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import ceil, hypot, sqrt

from rl_core.models import (
    AccessWindow,
    AngleRange,
    EnvironmentConfig,
    FootprintSample,
    GeoPoint,
    GroundTrackPoint,
    Opportunity,
    OpportunityKind,
    OrbitPass,
    Order,
    Polygon,
    Priority,
    Rectangle,
    RewardConfig,
    SatelliteConfig,
    Scenario,
    Strip,
)

GROUND_TRACK_SAMPLE_SEC = 20.0
FOOTPRINT_HALF_ALONG_DEG = 1.9
FOOTPRINT_HALF_CROSS_DEG = 2.2
STRIP_ALONG_DEG = 0.45
STRIP_CROSS_DEG = 0.12
ATTITUDE_DEG_PER_GROUND_DEG = 8.0


@dataclass(frozen=True, slots=True)
class ScenarioScale:
    """환경을 작은 문제부터 확장하기 위한 시나리오 규모 설정이다."""

    order_count: int
    pass_count: int
    max_strips_per_order: int
    max_accesses_per_strip: int


@dataclass(frozen=True, slots=True)
class AccessCandidate:
    """footprint 교차 샘플 묶음에서 opportunity를 만들기 전의 내부 후보 구간이다."""

    pass_id: str
    order_id: str
    strip_id: str
    start_time_sec: float
    end_time_sec: float
    footprints: tuple[FootprintSample, ...]


SCALES: dict[str, ScenarioScale] = {
    "tiny": ScenarioScale(5, 3, 3, 2),
    "small": ScenarioScale(20, 10, 6, 2),
    "full": ScenarioScale(100, 30, 10, 3),
}


def generate_scenario(seed: int, size: str = "tiny") -> Scenario:
    """전 세계에 분포한 재현 가능한 가상 주문과 촬영 기회를 생성한다."""

    try:
        scale = SCALES[size]
    except KeyError as exc:
        raise ValueError(f"unknown scenario size: {size}") from exc

    rng = random.Random(seed)
    environment = EnvironmentConfig()
    passes = _generate_passes(scale.pass_count, environment.duration_sec)
    ground_track_points, footprint_samples = _generate_ground_track_and_footprints(
        passes=passes,
        sample_interval=GROUND_TRACK_SAMPLE_SEC,
    )
    orders: list[Order] = []
    strips: list[Strip] = []

    for order_index in range(scale.order_count):
        strip_count = rng.randint(1, scale.max_strips_per_order)
        order_id = f"order-{order_index:03d}"
        anchor_footprint = rng.choice(footprint_samples)
        anchor_pass = next(item for item in passes if item.pass_id == anchor_footprint.pass_id)
        along, cross = _track_axes(anchor_pass.sequence)
        strip_polygons = _place_strip_polygons(
            rng=rng,
            footprint=anchor_footprint,
            strip_count=strip_count,
            along=along,
            cross=cross,
        )
        request_start = max(
            0.0,
            anchor_footprint.time_sec - rng.uniform(1_000.0, environment.duration_sec * 0.15),
        )
        request_end = min(
            environment.duration_sec,
            anchor_footprint.time_sec
            + rng.uniform(
                environment.duration_sec * 0.35,
                environment.duration_sec * 0.85,
            ),
        )
        if request_end <= request_start + environment.imaging_duration_sec:
            request_end = min(environment.duration_sec, request_start + 3_600.0)
        priority = rng.choices(
            [Priority.RED, Priority.BLUE, Priority.BACKGROUND],
            weights=[0.2, 0.35, 0.45],
            k=1,
        )[0]
        order_geometry = _bounding_rectangle(strip_polygons)
        orders.append(
            Order(
                order_id=order_id,
                name=f"Synthetic order {order_index + 1}",
                priority=priority,
                request_start_sec=request_start,
                request_end_sec=request_end,
                geometry=order_geometry,
                allowed_roll_deg=AngleRange(minimum=-30.0, maximum=30.0),
                allowed_tilt_deg=AngleRange(minimum=-30.0, maximum=30.0),
            )
        )

        for strip_index in range(strip_count):
            strip_id = f"{order_id}-strip-{strip_index:02d}"
            strip = Strip(
                strip_id=strip_id,
                order_id=order_id,
                sequence=strip_index,
                geometry=strip_polygons[strip_index],
            )
            strips.append(strip)

    access_windows, opportunities = _generate_access_windows_and_opportunities(
        rng=rng,
        passes=passes,
        orders=orders,
        strips=strips,
        footprints=footprint_samples,
        max_accesses_per_strip=scale.max_accesses_per_strip,
        imaging_duration=environment.imaging_duration_sec,
        sample_interval=GROUND_TRACK_SAMPLE_SEC,
    )

    opportunities.sort(key=lambda item: (item.capture_time_sec, item.opportunity_id))
    return Scenario(
        scenario_id=f"synthetic-{size}-{seed}",
        name=f"Synthetic {size} scenario (seed={seed})",
        seed=seed,
        satellite=SatelliteConfig(satellite_id="satellite-001"),
        environment=environment,
        reward=RewardConfig(),
        passes=passes,
        orders=orders,
        strips=strips,
        opportunities=opportunities,
        ground_track_points=ground_track_points,
        footprint_samples=footprint_samples,
        access_windows=access_windows,
    )


def _generate_passes(pass_count: int, duration: float) -> list[OrbitPass]:
    spacing = duration / pass_count
    pass_duration = min(1_200.0, spacing * 0.75)
    return [
        OrbitPass(
            pass_id=f"pass-{index:02d}",
            sequence=index,
            start_time_sec=index * spacing + (spacing - pass_duration) / 2.0,
            end_time_sec=index * spacing + (spacing + pass_duration) / 2.0,
        )
        for index in range(pass_count)
    ]


def _generate_ground_track_and_footprints(
    *,
    passes: list[OrbitPass],
    sample_interval: float,
) -> tuple[list[GroundTrackPoint], list[FootprintSample]]:
    """정밀 전파 대신 지도 검수 가능한 가상 궤적과 footprint를 만든다."""

    points: list[GroundTrackPoint] = []
    footprints: list[FootprintSample] = []
    pass_count = max(1, len(passes))
    for orbit_pass in passes:
        sample_count = int((orbit_pass.end_time_sec - orbit_pass.start_time_sec) // sample_interval)
        sample_count = max(1, sample_count)
        for sample_index in range(sample_count + 1):
            fraction = sample_index / sample_count
            latitude = _track_latitude(orbit_pass.sequence, fraction)
            longitude = _track_longitude(orbit_pass.sequence, pass_count, fraction)
            time_sec = (
                orbit_pass.start_time_sec
                + (orbit_pass.end_time_sec - orbit_pass.start_time_sec) * fraction
            )
            point_id = f"{orbit_pass.pass_id}-point-{sample_index:03d}"
            footprint_id = f"{orbit_pass.pass_id}-footprint-{sample_index:03d}"
            points.append(
                GroundTrackPoint(
                    point_id=point_id,
                    pass_id=orbit_pass.pass_id,
                    sample_index=sample_index,
                    time_sec=time_sec,
                    latitude_deg=latitude,
                    longitude_deg=longitude,
                )
            )
            footprints.append(
                FootprintSample(
                    footprint_id=footprint_id,
                    pass_id=orbit_pass.pass_id,
                    ground_track_point_id=point_id,
                    sample_index=sample_index,
                    time_sec=time_sec,
                    center_latitude_deg=latitude,
                    center_longitude_deg=longitude,
                    geometry=_oriented_rectangle(
                        center_lat=latitude,
                        center_lon=longitude,
                        along=_track_axes(orbit_pass.sequence)[0],
                        cross=_track_axes(orbit_pass.sequence)[1],
                        half_along=FOOTPRINT_HALF_ALONG_DEG,
                        half_cross=FOOTPRINT_HALF_CROSS_DEG,
                    ),
                )
            )
    return points, footprints


def _track_latitude(sequence: int, fraction: float) -> float:
    if sequence % 2 == 0:
        return -65.0 + 130.0 * fraction
    return 65.0 - 130.0 * fraction


def _track_longitude(sequence: int, pass_count: int, fraction: float) -> float:
    base = 0.0 if pass_count == 1 else -165.0 + sequence * (330.0 / (pass_count - 1))
    diagonal_drift = (fraction - 0.5) * 12.0
    return _clamp(base + diagonal_drift, -177.5, 177.5)


def _track_axes(sequence: int) -> tuple[tuple[float, float], tuple[float, float]]:
    along_lat = 130.0 if sequence % 2 == 0 else -130.0
    along_lon = 12.0
    length = sqrt(along_lat * along_lat + along_lon * along_lon)
    along = (along_lat / length, along_lon / length)
    cross = (-along[1], along[0])
    return along, cross


def _place_strip_polygons(
    *,
    rng: random.Random,
    footprint: FootprintSample,
    strip_count: int,
    along: tuple[float, float],
    cross: tuple[float, float],
) -> list[Polygon]:
    total_cross = STRIP_CROSS_DEG * strip_count
    cross_margin = max(0.0, FOOTPRINT_HALF_CROSS_DEG - total_cross / 2.0)
    along_margin = max(0.0, FOOTPRINT_HALF_ALONG_DEG - STRIP_ALONG_DEG / 2.0)
    base_cross_offset = rng.uniform(-cross_margin * 0.45, cross_margin * 0.45)
    base_along_offset = rng.uniform(-along_margin * 0.55, along_margin * 0.55)
    center_lat = footprint.center_latitude_deg + along[0] * base_along_offset
    center_lon = footprint.center_longitude_deg + along[1] * base_along_offset
    polygons: list[Polygon] = []
    for strip_index in range(strip_count):
        cross_offset = (strip_index - (strip_count - 1) / 2.0) * STRIP_CROSS_DEG
        cross_offset += base_cross_offset
        strip_center_lat = center_lat + cross[0] * cross_offset
        strip_center_lon = center_lon + cross[1] * cross_offset
        polygons.append(
            _oriented_rectangle(
                center_lat=strip_center_lat,
                center_lon=strip_center_lon,
                along=along,
                cross=cross,
                half_along=STRIP_ALONG_DEG / 2.0,
                half_cross=STRIP_CROSS_DEG / 2.0,
            )
        )
    return polygons


def _oriented_rectangle(
    *,
    center_lat: float,
    center_lon: float,
    along: tuple[float, float],
    cross: tuple[float, float],
    half_along: float,
    half_cross: float,
) -> Polygon:
    vertices: list[GeoPoint] = []
    for along_sign, cross_sign in ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0)):
        lat = center_lat + along[0] * half_along * along_sign + cross[0] * half_cross * cross_sign
        lon = center_lon + along[1] * half_along * along_sign + cross[1] * half_cross * cross_sign
        vertices.append(
            GeoPoint(
                lat=_clamp(lat, -89.9, 89.9),
                lon=_clamp(lon, -179.9, 179.9),
            )
        )
    return Polygon(vertices=vertices)


def _bounding_rectangle(polygons: list[Polygon]) -> Rectangle:
    return Rectangle(
        min_lat=min(polygon.min_lat for polygon in polygons),
        min_lon=min(polygon.min_lon for polygon in polygons),
        max_lat=max(polygon.max_lat for polygon in polygons),
        max_lon=max(polygon.max_lon for polygon in polygons),
    )


def _generate_access_windows_and_opportunities(
    *,
    rng: random.Random,
    passes: list[OrbitPass],
    orders: list[Order],
    strips: list[Strip],
    footprints: list[FootprintSample],
    max_accesses_per_strip: int,
    imaging_duration: float,
    sample_interval: float,
) -> tuple[list[AccessWindow], list[Opportunity]]:
    footprints_by_pass: dict[str, list[FootprintSample]] = {item.pass_id: [] for item in passes}
    for footprint in footprints:
        footprints_by_pass[footprint.pass_id].append(footprint)
    for pass_footprints in footprints_by_pass.values():
        pass_footprints.sort(key=lambda item: item.sample_index)

    order_by_id = {item.order_id: item for item in orders}
    access_windows: list[AccessWindow] = []
    opportunities: list[Opportunity] = []
    for strip in strips:
        order = order_by_id[strip.order_id]
        candidates = _find_access_candidates(
            strip=strip,
            order=order,
            footprints_by_pass=footprints_by_pass,
            sample_interval=sample_interval,
            imaging_duration=imaging_duration,
        )
        if len(candidates) > max_accesses_per_strip:
            candidates = sorted(
                rng.sample(candidates, k=max_accesses_per_strip),
                key=_access_sort_key,
            )
        for access_index, candidate in enumerate(candidates):
            access_window_id = f"{strip.strip_id}-{candidate.pass_id}-access-{access_index:02d}"
            derived_opportunities = _generate_access_opportunities(
                strip=strip,
                candidate=candidate,
                access_index=access_index,
                access_window_id=access_window_id,
                imaging_duration=imaging_duration,
            )
            opportunities.extend(derived_opportunities)
            access_windows.append(
                AccessWindow(
                    access_window_id=access_window_id,
                    pass_id=candidate.pass_id,
                    order_id=candidate.order_id,
                    strip_id=candidate.strip_id,
                    window_start_sec=candidate.start_time_sec,
                    window_end_sec=candidate.end_time_sec,
                    min_off_nadir_time_sec=_min_off_nadir_time(strip, candidate),
                    footprint_ids=[item.footprint_id for item in candidate.footprints],
                    opportunity_ids=[item.opportunity_id for item in derived_opportunities],
                )
            )
    return access_windows, opportunities


def _find_access_candidates(
    *,
    strip: Strip,
    order: Order,
    footprints_by_pass: dict[str, list[FootprintSample]],
    sample_interval: float,
    imaging_duration: float,
) -> list[AccessCandidate]:
    candidates: list[AccessCandidate] = []
    pass_end_by_id = {
        pass_id: max(item.time_sec for item in pass_footprints)
        for pass_id, pass_footprints in footprints_by_pass.items()
        if pass_footprints
    }
    for pass_id, footprints in footprints_by_pass.items():
        current_group: list[FootprintSample] = []
        previous_index: int | None = None
        for footprint in footprints:
            in_order_period = (
                order.request_start_sec
                <= footprint.time_sec
                <= order.request_end_sec - imaging_duration
            )
            intersects = in_order_period and _polygons_intersect(
                strip.geometry,
                footprint.geometry,
            )
            is_contiguous = previous_index is None or footprint.sample_index == previous_index + 1
            if intersects and is_contiguous:
                current_group.append(footprint)
            else:
                candidates.extend(
                    _candidate_from_group(
                        strip=strip,
                        order=order,
                        pass_id=pass_id,
                        group=current_group,
                        pass_end_sec=pass_end_by_id[pass_id],
                        sample_interval=sample_interval,
                        imaging_duration=imaging_duration,
                    )
                )
                current_group = [footprint] if intersects else []
            previous_index = footprint.sample_index if intersects else None
        candidates.extend(
            _candidate_from_group(
                strip=strip,
                order=order,
                pass_id=pass_id,
                group=current_group,
                pass_end_sec=pass_end_by_id[pass_id],
                sample_interval=sample_interval,
                imaging_duration=imaging_duration,
            )
        )
    return sorted(candidates, key=_access_sort_key)


def _candidate_from_group(
    *,
    strip: Strip,
    order: Order,
    pass_id: str,
    group: list[FootprintSample],
    pass_end_sec: float,
    sample_interval: float,
    imaging_duration: float,
) -> list[AccessCandidate]:
    if not group:
        return []
    start_time = max(group[0].time_sec, order.request_start_sec)
    end_time = min(group[-1].time_sec + sample_interval, order.request_end_sec, pass_end_sec)
    if end_time - start_time < imaging_duration:
        return []
    return [
        AccessCandidate(
            pass_id=pass_id,
            order_id=order.order_id,
            strip_id=strip.strip_id,
            start_time_sec=start_time,
            end_time_sec=end_time,
            footprints=tuple(group),
        )
    ]


def _generate_access_opportunities(
    *,
    strip: Strip,
    candidate: AccessCandidate,
    access_index: int,
    access_window_id: str,
    imaging_duration: float,
) -> list[Opportunity]:
    window_start = _quantize_time(candidate.start_time_sec)
    window_end = candidate.end_time_sec
    if window_end - window_start < imaging_duration:
        window_start = candidate.start_time_sec
    early_time = window_start
    late_time = window_end - imaging_duration
    min_angle_time = _clamp(_min_off_nadir_time(strip, candidate), early_time, late_time)

    early_roll, early_tilt = _attitude_for_time(strip, candidate, early_time)
    late_roll, late_tilt = _attitude_for_time(strip, candidate, late_time)
    min_roll, min_tilt = _attitude_for_time(strip, candidate, min_angle_time)
    best_edge_roll, best_edge_tilt = (
        (early_roll, early_tilt)
        if hypot(early_roll, early_tilt) <= hypot(late_roll, late_tilt)
        else (late_roll, late_tilt)
    )
    if hypot(min_roll, min_tilt) > hypot(best_edge_roll, best_edge_tilt):
        min_roll, min_tilt = best_edge_roll * 0.2, best_edge_tilt * 0.2
    candidates = [
        (OpportunityKind.MIN_OFF_NADIR, min_angle_time, min_roll, min_tilt),
        (OpportunityKind.EARLY, early_time, early_roll, early_tilt),
        (OpportunityKind.LATE, late_time, late_roll, late_tilt),
    ]

    result: list[Opportunity] = []
    seen_times: set[float] = set()
    for kind, capture_time, roll, tilt in candidates:
        rounded_time = round(capture_time, 6)
        if rounded_time in seen_times:
            continue
        seen_times.add(rounded_time)
        result.append(
            Opportunity(
                opportunity_id=(
                    f"{strip.strip_id}-{candidate.pass_id}-access-{access_index}-{kind.value}"
                ),
                order_id=candidate.order_id,
                strip_id=strip.strip_id,
                pass_id=candidate.pass_id,
                kind=kind,
                window_start_sec=window_start,
                window_end_sec=window_end,
                capture_time_sec=capture_time,
                required_roll_deg=roll,
                required_tilt_deg=tilt,
                source_access_window_id=access_window_id,
            )
        )
    return result


def _min_off_nadir_time(strip: Strip, candidate: AccessCandidate) -> float:
    best = min(
        candidate.footprints,
        key=lambda footprint: hypot(
            _polygon_center_lon(strip.geometry) - footprint.center_longitude_deg,
            _polygon_center_lat(strip.geometry) - footprint.center_latitude_deg,
        ),
    )
    return _clamp(best.time_sec, candidate.start_time_sec, candidate.end_time_sec - 1e-6)


def _attitude_for_time(
    strip: Strip,
    candidate: AccessCandidate,
    time_sec: float,
) -> tuple[float, float]:
    footprint = min(candidate.footprints, key=lambda item: abs(item.time_sec - time_sec))
    roll = (_polygon_center_lon(strip.geometry) - footprint.center_longitude_deg) * (
        ATTITUDE_DEG_PER_GROUND_DEG
    )
    tilt = (_polygon_center_lat(strip.geometry) - footprint.center_latitude_deg) * (
        ATTITUDE_DEG_PER_GROUND_DEG
    )
    return _clamp(roll, -27.0, 27.0), _clamp(tilt, -27.0, 27.0)


def _polygons_intersect(left: Polygon, right: Polygon) -> bool:
    axes = _polygon_axes(left) + _polygon_axes(right)
    return all(_projections_overlap(_project(left, axis), _project(right, axis)) for axis in axes)


def _polygon_axes(polygon: Polygon) -> list[tuple[float, float]]:
    axes: list[tuple[float, float]] = []
    for index, vertex in enumerate(polygon.vertices):
        next_vertex = polygon.vertices[(index + 1) % len(polygon.vertices)]
        edge_lat = next_vertex.lat - vertex.lat
        edge_lon = next_vertex.lon - vertex.lon
        normal = (-edge_lon, edge_lat)
        length = hypot(*normal)
        if length > 0.0:
            axes.append((normal[0] / length, normal[1] / length))
    return axes


def _project(polygon: Polygon, axis: tuple[float, float]) -> tuple[float, float]:
    values = [vertex.lat * axis[0] + vertex.lon * axis[1] for vertex in polygon.vertices]
    return min(values), max(values)


def _projections_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def _polygon_center_lat(polygon: Polygon) -> float:
    return sum(vertex.lat for vertex in polygon.vertices) / len(polygon.vertices)


def _polygon_center_lon(polygon: Polygon) -> float:
    return sum(vertex.lon for vertex in polygon.vertices) / len(polygon.vertices)


def _access_sort_key(candidate: AccessCandidate) -> tuple[float, str, str]:
    return candidate.start_time_sec, candidate.pass_id, candidate.strip_id


def _quantize_time(time_sec: float) -> float:
    # 같은 시각의 후보 경쟁을 유지하기 위해 기존 생성기의 10초 grid 가정을 보존한다.
    return ceil(time_sec / 10.0) * 10.0


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
