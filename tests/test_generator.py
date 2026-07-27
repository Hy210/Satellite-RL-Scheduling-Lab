from __future__ import annotations

from math import hypot

import pytest

from rl_core.generator import (
    ATTITUDE_DEG_PER_GROUND_DEG,
    SCALES,
    generate_scenario,
    resolve_attitude_look_point,
)
from rl_core.models import OpportunityKind, Polygon, Rectangle


@pytest.mark.parametrize("size", ["tiny", "small", "full"])
def test_generator_creates_expected_scale(size: str) -> None:
    scenario = generate_scenario(seed=123, size=size)

    assert len(scenario.orders) == SCALES[size].order_count
    assert len(scenario.passes) == SCALES[size].pass_count
    assert 0 < len(scenario.strips) <= scenario.environment.max_strips
    assert scenario.opportunities


@pytest.mark.parametrize("size", ["tiny", "small", "full"])
def test_generator_creates_mixed_overlapping_and_separated_orders(size: str) -> None:
    """일부 order 쌍은 의도적으로 겹치고(부분/완전/포함), 대다수는 여전히 분리돼야 한다.

    order 배치를 완전 독립 랜덤으로 두면 겹침이 통계적으로 거의 발생하지 않는다는 것을
    실측으로 확인했다(tiny 0%, small 0.53%, full 0.02% bbox 겹침, 포함관계는 세 규모
    모두 0건). 겹침 엔지니어링이 실제로 다양한 관계를 만들면서도 대다수 order는 여전히
    분리된 상태(겹침이 지배적이지 않음)로 남기는지 검증한다.
    """

    scenario = generate_scenario(seed=20260707, size=size)
    orders = scenario.orders
    strips_by_order: dict[str, list] = {}
    for strip in scenario.strips:
        strips_by_order.setdefault(strip.order_id, []).append(strip)

    total_pairs = 0
    separated_pairs = 0
    bbox_overlap_pairs = 0
    strip_level_overlap_pairs = 0
    containment_pairs = 0
    temporal_overlap_confirmed = False

    for i in range(len(orders)):
        for j in range(i + 1, len(orders)):
            order_a, order_b = orders[i], orders[j]
            total_pairs += 1
            if not _rect_overlap(order_a.geometry, order_b.geometry):
                separated_pairs += 1
                continue
            bbox_overlap_pairs += 1

            found_strip_overlap = any(
                _polygons_intersect(strip_a.geometry, strip_b.geometry)
                for strip_a in strips_by_order.get(order_a.order_id, [])
                for strip_b in strips_by_order.get(order_b.order_id, [])
            )
            if not found_strip_overlap:
                continue
            strip_level_overlap_pairs += 1

            if _rect_contains(order_a.geometry, order_b.geometry) or _rect_contains(
                order_b.geometry, order_a.geometry
            ):
                containment_pairs += 1

            if (
                order_a.request_start_sec < order_b.request_end_sec
                and order_b.request_start_sec < order_a.request_end_sec
            ):
                temporal_overlap_confirmed = True

    assert bbox_overlap_pairs > 0, f"{size}: no overlapping order pair found"
    assert strip_level_overlap_pairs > 0, f"{size}: no order pair with actual strip overlap"
    assert containment_pairs > 0, f"{size}: no containment order pair found"
    assert separated_pairs > total_pairs // 2, (
        f"{size}: overlap engineering should stay a minority, not dominate the scenario"
    )
    assert temporal_overlap_confirmed, (
        f"{size}: at least one spatially overlapping pair should also share request windows"
    )


def _rect_overlap(left: Rectangle, right: Rectangle) -> bool:
    return (
        left.min_lat <= right.max_lat
        and right.min_lat <= left.max_lat
        and left.min_lon <= right.max_lon
        and right.min_lon <= left.max_lon
    )


def _rect_contains(outer: Rectangle, inner: Rectangle) -> bool:
    return (
        outer.min_lat <= inner.min_lat
        and outer.max_lat >= inner.max_lat
        and outer.min_lon <= inner.min_lon
        and outer.max_lon >= inner.max_lon
    )


@pytest.mark.parametrize("size", ["tiny", "small", "full"])
def test_generator_is_reproducible(size: str) -> None:
    first = generate_scenario(seed=99, size=size)
    second = generate_scenario(seed=99, size=size)
    different = generate_scenario(seed=100, size=size)

    assert first.to_json() == second.to_json()
    assert first.to_json() != different.to_json()


def test_access_contains_at_most_three_unique_candidates() -> None:
    scenario = generate_scenario(seed=11, size="small")
    grouped: dict[tuple[str, str, int], list] = {}
    for opportunity in scenario.opportunities:
        access = int(opportunity.opportunity_id.split("-access-")[1].split("-")[0])
        key = (opportunity.strip_id, opportunity.pass_id, access)
        grouped.setdefault(key, []).append(opportunity)

    for opportunities in grouped.values():
        assert len(opportunities) <= 3
        assert len({item.capture_time_sec for item in opportunities}) == len(opportunities)
        min_candidates = [
            item for item in opportunities if item.kind is OpportunityKind.MIN_OFF_NADIR
        ]
        assert len(min_candidates) == 1
        assert min_candidates[0].off_nadir_deg == min(item.off_nadir_deg for item in opportunities)


def test_generated_opportunities_keep_footprint_access_evidence() -> None:
    scenario = generate_scenario(seed=7, size="tiny")
    access_by_id = {item.access_window_id: item for item in scenario.access_windows}
    footprint_by_id = {item.footprint_id: item for item in scenario.footprint_samples}
    strip_by_id = {item.strip_id: item for item in scenario.strips}

    assert scenario.ground_track_points
    assert scenario.footprint_samples
    assert scenario.access_windows
    assert all(item.source_access_window_id for item in scenario.opportunities)

    for opportunity in scenario.opportunities:
        assert opportunity.source_access_window_id is not None
        access_window = access_by_id[opportunity.source_access_window_id]
        assert opportunity.opportunity_id in access_window.opportunity_ids
        assert access_window.window_start_sec <= opportunity.window_start_sec
        assert opportunity.window_end_sec <= access_window.window_end_sec

    for access_window in scenario.access_windows:
        strip = strip_by_id[access_window.strip_id]
        assert access_window.footprint_ids
        assert any(
            _polygons_intersect(strip.geometry, footprint_by_id[footprint_id].geometry)
            for footprint_id in access_window.footprint_ids
        )


def test_generated_strip_polygons_follow_source_pass_direction() -> None:
    scenario = generate_scenario(seed=7, size="tiny")
    pass_by_id = {item.pass_id: item for item in scenario.passes}
    strip_by_id = {item.strip_id: item for item in scenario.strips}

    for access_window in scenario.access_windows:
        strip = strip_by_id[access_window.strip_id]
        orbit_pass = pass_by_id[access_window.pass_id]
        strip_axis = _longest_edge_axis(strip.geometry)
        pass_axis = _track_axis(orbit_pass.sequence)
        alignment = abs(strip_axis[0] * pass_axis[0] + strip_axis[1] * pass_axis[1])
        assert alignment > 0.99


def test_resolve_attitude_look_point_inverts_attitude_formula() -> None:
    # generator의 순방향 계산(strip 중심 -> roll/tilt)과 역방향 계산(roll/tilt -> 지점)이
    # 서로 일치하는지 확인한다 — 이 함수가 rl_core 안에서만 계산식을 알고 있어야
    # 프론트엔드가 계산식을 몰라도 되는 설계를 검증한다.
    #
    # 최근접 footprint는 반드시 opportunity를 만든 access window의 footprint 그룹
    # (`AccessWindow.footprint_ids`) 안에서만 찾아야 한다 — 같은 pass의 다른 strip이
    # 가진 footprint까지 포함해 pass 전체에서 찾으면, 시간상 우연히 더 가까운 다른
    # strip의 footprint를 잘못 골라 지점이 크게 어긋나는 실제 버그가 있었다.
    scenario = generate_scenario(seed=7, size="small")
    footprint_by_id = {sample.footprint_id: sample for sample in scenario.footprint_samples}
    window_by_id = {window.access_window_id: window for window in scenario.access_windows}

    for opportunity in scenario.opportunities:
        look_point = resolve_attitude_look_point(scenario, opportunity)

        assert opportunity.source_access_window_id is not None
        window = window_by_id[opportunity.source_access_window_id]
        group = [footprint_by_id[footprint_id] for footprint_id in window.footprint_ids]
        nearest = min(group, key=lambda sample: abs(sample.time_sec - opportunity.capture_time_sec))
        recovered_roll = (
            look_point.lon - nearest.center_longitude_deg
        ) * ATTITUDE_DEG_PER_GROUND_DEG
        recovered_tilt = (
            look_point.lat - nearest.center_latitude_deg
        ) * ATTITUDE_DEG_PER_GROUND_DEG

        # roll/tilt는 ±27도로 clamp되므로(_attitude_for_time), 그 한계에 걸린 opportunity는
        # 원래 strip 중심과 거리가 있을 수 있다 — strip 근접 여부가 아니라 "역산이
        # 순방향 공식과 정확히 맞아떨어지는가"만 확인한다.
        assert recovered_roll == pytest.approx(opportunity.required_roll_deg, abs=1e-6)
        assert recovered_tilt == pytest.approx(opportunity.required_tilt_deg, abs=1e-6)


def test_resolve_attitude_look_point_stays_within_source_access_window_group() -> None:
    # 같은 pass 안에 여러 strip의 access window가 있을 때, pass 전체 최근접이 아니라
    # 이 opportunity를 만든 access window의 footprint 그룹으로 검색 범위가 제한되는지
    # 직접 확인한다(회귀 방지: 이전에는 pass 전체에서 찾아 엉뚱한 strip의 footprint를
    # 골라 지점이 실제 strip과 무관하게 멀리 떨어지는 버그가 있었다).
    scenario = generate_scenario(seed=20260707, size="small")
    footprint_by_id = {sample.footprint_id: sample for sample in scenario.footprint_samples}
    window_by_id = {window.access_window_id: window for window in scenario.access_windows}
    strip_by_id = {strip.strip_id: strip for strip in scenario.strips}

    checked_any = False
    for opportunity in scenario.opportunities:
        assert opportunity.source_access_window_id is not None
        window = window_by_id[opportunity.source_access_window_id]
        group_ids = set(window.footprint_ids)
        same_pass_outside_group = [
            sample
            for sample in scenario.footprint_samples
            if sample.pass_id == opportunity.pass_id and sample.footprint_id not in group_ids
        ]
        if not same_pass_outside_group:
            continue
        outside_nearest = min(
            same_pass_outside_group,
            key=lambda sample: abs(sample.time_sec - opportunity.capture_time_sec),
        )
        group_nearest = min(
            (footprint_by_id[footprint_id] for footprint_id in group_ids),
            key=lambda sample: abs(sample.time_sec - opportunity.capture_time_sec),
        )
        if abs(outside_nearest.time_sec - opportunity.capture_time_sec) >= abs(
            group_nearest.time_sec - opportunity.capture_time_sec
        ):
            continue

        # pass 전체로 찾으면 그룹 밖 footprint가 더 가까운 경우를 찾았다 — 이 경우에도
        # resolve_attitude_look_point는 반드시 그룹 안의 footprint를 기준으로 계산해야 한다.
        checked_any = True
        strip = strip_by_id[opportunity.strip_id]
        centroid_lat = sum(v.lat for v in strip.geometry.vertices) / len(strip.geometry.vertices)
        centroid_lon = sum(v.lon for v in strip.geometry.vertices) / len(strip.geometry.vertices)

        look_point = resolve_attitude_look_point(scenario, opportunity)
        distance_correct = hypot(look_point.lat - centroid_lat, look_point.lon - centroid_lon)

        tilt_offset = opportunity.required_tilt_deg / ATTITUDE_DEG_PER_GROUND_DEG
        roll_offset = opportunity.required_roll_deg / ATTITUDE_DEG_PER_GROUND_DEG
        wrong_group_lat = outside_nearest.center_latitude_deg + tilt_offset
        wrong_group_lon = outside_nearest.center_longitude_deg + roll_offset
        distance_if_wrong_group = hypot(
            wrong_group_lat - centroid_lat, wrong_group_lon - centroid_lon
        )

        assert distance_correct < distance_if_wrong_group

    assert checked_any, "그룹 밖 footprint가 더 가까운 회귀 케이스를 이 seed에서 찾지 못했다"


def test_unknown_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scenario size"):
        generate_scenario(seed=1, size="huge")


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


def _longest_edge_axis(polygon: Polygon) -> tuple[float, float]:
    edges: list[tuple[float, float, float]] = []
    for index, vertex in enumerate(polygon.vertices):
        next_vertex = polygon.vertices[(index + 1) % len(polygon.vertices)]
        edge = (next_vertex.lat - vertex.lat, next_vertex.lon - vertex.lon)
        edges.append((hypot(*edge), edge[0], edge[1]))
    _, lat, lon = max(edges)
    length = hypot(lat, lon)
    return lat / length, lon / length


def _track_axis(sequence: int) -> tuple[float, float]:
    lat = 130.0 if sequence % 2 == 0 else -130.0
    lon = 12.0
    length = hypot(lat, lon)
    return lat / length, lon / length
