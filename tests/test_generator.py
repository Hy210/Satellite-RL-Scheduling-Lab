from __future__ import annotations

from math import hypot

import pytest

from rl_core.generator import SCALES, generate_scenario
from rl_core.models import OpportunityKind, Polygon


@pytest.mark.parametrize("size", ["tiny", "small", "full"])
def test_generator_creates_expected_scale(size: str) -> None:
    scenario = generate_scenario(seed=123, size=size)

    assert len(scenario.orders) == SCALES[size].order_count
    assert len(scenario.passes) == SCALES[size].pass_count
    assert 0 < len(scenario.strips) <= scenario.environment.max_strips
    assert scenario.opportunities


def test_generator_is_reproducible() -> None:
    first = generate_scenario(seed=99, size="tiny")
    second = generate_scenario(seed=99, size="tiny")
    different = generate_scenario(seed=100, size="tiny")

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
