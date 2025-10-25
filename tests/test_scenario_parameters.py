import copy

import pytest

pd = pytest.importorskip("pandas")

from bioethanol_model import CassavaBioethanolModel
from bioethanol_model.inputs import default_input_page
from bioethanol_model.scenario import (
    ScenarioConfig,
    apply_scenario,
    scenario_parameter_catalog,
)


def test_scenario_parameter_catalog_exposes_required_parameters() -> None:
    page = default_input_page()
    catalog = scenario_parameter_catalog(page)
    assert not catalog.empty
    parameters = set(catalog["Parameter"])
    for required in [
        "Production monthly",
        "Cassava feedstock",
        "Initial Investment",
    ]:
        assert required in parameters
    cassava_row = catalog.loc[catalog["Parameter"] == "Cassava feedstock"].iloc[0]
    assert cassava_row["Base Value"] > 0


def test_apply_scenario_restores_input_tables() -> None:
    page = default_input_page()
    model = CassavaBioethanolModel(copy.deepcopy(page))
    baseline_costs = model.input_page.direct_costs_monthly.data.copy(deep=True)

    config = ScenarioConfig("Cassava Upside", {"Cassava feedstock": 750_000.0})
    result = apply_scenario(model, config)

    assert "metrics" in result
    pd.testing.assert_frame_equal(
        model.input_page.direct_costs_monthly.data.reset_index(drop=True),
        baseline_costs.reset_index(drop=True),
    )
