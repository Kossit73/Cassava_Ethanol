import pytest

pd = pytest.importorskip("pandas")
pdt = pytest.importorskip("pandas.testing")

from bioethanol_model.financial_model import CassavaBioethanolModel


def test_direct_costs_preserved_across_scenarios():
    model = CassavaBioethanolModel()

    direct_rows = pd.DataFrame(
        [
            {"Month": "2025-01", "Cost Category": "Cassava Feedstock", "Amount": 575_000.0},
            {"Month": "2025-06", "Cost Category": "Enzymes & Chemicals", "Amount": 210_000.0},
            {"Month": "2025-09", "Cost Category": "Energy Cost", "Amount": 195_000.0},
        ]
    )

    table = model.input_page.direct_costs_monthly
    table.set_data(direct_rows, mark_user_input=True)

    expected = direct_rows.reset_index(drop=True)

    for scenario in model.SCENARIOS:
        prepared = model._prepare_page_for_scenario(scenario)
        result = prepared.direct_costs_monthly.data.reset_index(drop=True)

        pdt.assert_frame_equal(result, expected, check_dtype=False)
        assert not prepared.direct_costs_monthly.placeholder
