from __future__ import annotations

import pytest

from rl_core.generator import SCALES, generate_scenario
from rl_core.models import OpportunityKind


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


def test_unknown_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scenario size"):
        generate_scenario(seed=1, size="huge")
