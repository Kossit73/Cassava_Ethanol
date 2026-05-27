import pytest

pd = pytest.importorskip("pandas")

from streamlit_app import _set_dataframe_cell


def test_set_dataframe_cell_upcasts_for_incompatible_editor_value():
    df = pd.DataFrame({"Start Month": [202501]})

    _set_dataframe_cell(df, 0, "Start Month", None)

    assert df.at[0, "Start Month"] is None
    assert df["Start Month"].dtype == "object"
