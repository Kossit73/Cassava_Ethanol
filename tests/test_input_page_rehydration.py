from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from bioethanol_model.inputs import InputLandingPage, default_input_page


def test_input_landing_page_from_dict_rehydrates_dataclass_payload() -> None:
    page = default_input_page()
    page.projection.start_year = 2026
    page.projection.end_year = 2036
    page.production_monthly.set_data(
        pd.DataFrame(
            [
                {
                    "Start Month": "2026-01",
                    "Cassava ton": 12_345.0,
                    "Ethanol litres": 2_469_000.0,
                    "Animal Feed ton": 3_395.0,
                    "Growth %": 0.01,
                }
            ]
        ),
        mark_user_input=True,
    )

    payload = asdict(page)
    rebuilt = InputLandingPage.from_dict(payload)

    assert isinstance(rebuilt, InputLandingPage)
    assert rebuilt.projection.start_year == 2026
    assert rebuilt.projection.end_year == 2036
    assert rebuilt.production_monthly.data.iloc[0]["Cassava ton"] == 12_345.0


def test_input_landing_page_from_dict_accepts_record_lists() -> None:
    page = default_input_page()
    payload = {
        "projection": {
            "start_year": page.projection.start_year,
            "end_year": page.projection.end_year,
            "planning_start": page.projection.planning_start,
        },
        "global_inputs": {
            "name": page.global_inputs.name,
            "columns": page.global_inputs.columns,
            "data": page.global_inputs.data.to_dict(orient="records"),
            "placeholder": page.global_inputs.placeholder,
        },
    }

    rebuilt = InputLandingPage.from_dict(payload)

    assert isinstance(rebuilt, InputLandingPage)
    assert list(rebuilt.global_inputs.data.columns) == page.global_inputs.columns
    assert len(rebuilt.global_inputs.data) == len(page.global_inputs.data)
