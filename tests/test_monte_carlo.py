import copy

import pytest

pd = pytest.importorskip("pandas")

from bioethanol_model import CassavaBioethanolModel
from bioethanol_model.inputs import default_input_page
from bioethanol_model.sensitivity import (
    MONTE_CARLO_PARAMETER_ADAPTERS,
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
    assert {"Project NPV", "Project IRR", "Equity IRR", "DSCR (min)", "Payback Period (years)"}.issubset(results.columns)
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


def test_parameter_adapters_capture_base_values() -> None:
    page = default_input_page()
    cassava_state = MONTE_CARLO_PARAMETER_ADAPTERS["Cassava feedstock"].capture(page)
    loan_state = MONTE_CARLO_PARAMETER_ADAPTERS["Loan Schedule"].capture(page)

    assert pytest.approx(cassava_state.base_value) == 600_000
    assert pytest.approx(loan_state.base_value) == 24_000_000


def test_monte_carlo_simulation_accepts_correlation_matrix() -> None:
    page = default_input_page()
    model = CassavaBioethanolModel(copy.deepcopy(page))
    config = pd.DataFrame(
        [
            {"Parameter": "Corporate tax rate", "Distribution": "Normal", "loc": 0.28, "scale": 0.02},
            {"Parameter": "Discount rate", "Distribution": "Normal", "loc": 0.12, "scale": 0.02},
            {"Parameter": "Investor share capital", "Distribution": "Normal", "loc": 0.45, "scale": 0.03},
        ]
    )
    corr = pd.DataFrame(
        [
            [1.0, 0.6, -0.2],
            [0.6, 1.0, -0.1],
            [-0.2, -0.1, 1.0],
        ],
        index=["Corporate tax rate", "Discount rate", "Investor share capital"],
        columns=["Corporate tax rate", "Discount rate", "Investor share capital"],
    )

    results = monte_carlo_simulation(model, config, iterations=4, random_seed=7, correlation_matrix=corr)
    assert len(results) == 4
