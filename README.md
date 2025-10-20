# Cassava_Ethanol

Cassava_Ethanol is a lightweight financial modeling toolkit for planning and
stress-testing cassava-based fuel ethanol projects. The model captures the key
production, revenue and operating drivers to generate a multi-year cash-flow
projection complete with NPV, IRR and payback calculations.

## Features

- Parameterized plant, feedstock, pricing and cost inputs defined via YAML.
- Year-by-year cash-flow breakdown including production, revenue, opex and free
  cash flow.
- Automatic computation of NPV, IRR and simple payback period.
- Scenario runner for comparing sensitivities such as feedstock prices or
  product pricing.
- Command line interface that prints formatted summaries or JSON payloads for
  downstream analysis.

## Quick start

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Run the base model using the included configuration:

```bash
python -m cassava_ethanol.cli --config examples/base_config.yaml
```

To evaluate a set of scenarios and produce JSON for further analysis:

```bash
python -m cassava_ethanol.cli \
  --config examples/base_config.yaml \
  --scenarios examples/sample_scenarios.yaml \
  --json
```

### Interactive dashboard

Launch the Streamlit experience to explore the model with sliders and
scenario uploads:

```bash
pip install -e .[app]
streamlit run streamlit_app.py
```

The app ships with the same example configuration and scenario files found in
`examples/` and allows ad-hoc overrides for any numeric assumption.

## Testing

Execute the unit test suite with pytest:

```bash
pytest
```

## Configuration reference

The YAML configuration groups assumptions into six sections:

- `plant`: name, nominal capacity, operating days and ramp-up curve.
- `feedstock`: cassava pricing, inclusion rate per liter and logistics costs.
- `pricing`: ethanol and by-product pricing with optional escalation.
- `operating_costs`: fixed, variable and maintenance expenses.
- `capital`: initial investment, depreciation schedule, salvage and tax rate.
- `financial`: discount, inflation and analysis horizon.

Overrides for scenarios use `section.field` notation, e.g. to stress the
cassava price use `feedstock.cassava_price_per_ton`.

## License

This project is provided as-is without warranty. Customize it to reflect the
specifics of your project economics.
