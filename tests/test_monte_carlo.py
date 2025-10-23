import copy

import pytest

pd = pytest.importorskip("pandas")

from bioethanol_model import CassavaBioethanolModel
from bioethanol_model.inputs import default_input_page
from bioethanol_model.sensitivity import (
    default_monte_carlo_parameters,
    monte_carlo_simulation,
)


def test_monte_carlo_simulation_restores_inputs() -> None:
    page = default_input_page()
    model = CassavaBioethanolModel(copy.deepcopy(page))
    original = model.input_page.global_inputs.data.copy(deep=True)

    params = default_monte_carlo_parameters()
    results = monte_carlo_simulation(model, params, iterations=5, random_seed=0)

    assert len(results) == 5
    pd.testing.assert_frame_equal(
        model.input_page.global_inputs.data.reset_index(drop=True),
        original.reset_index(drop=True),
    )


def test_monte_carlo_simulation_supports_multiple_distributions() -> None:
    page = default_input_page()
    model = CassavaBioethanolModel(copy.deepcopy(page))
    base = model.input_page.global_inputs.data.copy(deep=True)

    config = pd.DataFrame(
        [
            {
                "Parameter": "Corporate tax rate",
                "Distribution": "Lognormal",
                "s": 0.1,
                "scale": 0.3,
            },
            {
                "Parameter": "Investor share capital",
                "Distribution": "Multinomial",
                "n": 1,
                "pvals": "0.5,0.5",
            },
        ]
    )

    results = monte_carlo_simulation(model, config, iterations=3, random_seed=1)
    assert len(results) == 3
    assert {"Project NPV", "Project IRR", "Equity IRR"}.issubset(results.columns)
    pd.testing.assert_frame_equal(
        model.input_page.global_inputs.data.reset_index(drop=True),
        base.reset_index(drop=True),
    )


def test_monte_carlo_simulation_with_empty_configuration_returns_empty() -> None:
    page = default_input_page()
    model = CassavaBioethanolModel(copy.deepcopy(page))
    empty_config = pd.DataFrame(columns=["Parameter", "Distribution"])

    results = monte_carlo_simulation(model, empty_config, iterations=4, random_seed=2)
    assert results.empty

