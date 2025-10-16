# Cassava Bioethanol Financial Model

This repository contains a Python-based toolkit that assembles a complete Cassava Bioethanol project financial model. It builds an Excel workbook featuring detailed input sheets, financial statements, key metrics dashboards, sensitivity analyses, and scenario planning utilities tailored to the FARM_ONLY, BUY_ONLY, and HYBRID feedstock sourcing strategies.

## Features
- Structured **Input Landing Page** capturing projection horizon, global inputs, CAPEX, depreciation schedules, revenue, production, opex, working capital, financing, taxation, inflation, and risk registers.
- Initial investment items now represent the combined owner and investor equity contributions, while the loan schedule accepts explicit loan amounts for each facility so debt balances follow the values you enter rather than being inferred from CAPEX totals.
- Automated **financial statements** (monthly and annual income statement, balance sheet, cash flow statement) with supporting schedules for depreciation, working capital, and debt amortisation.
- **Key metrics dashboard** reporting project NPV/IRR, equity returns, cumulative cash flows, cost breakdowns, production outlook, and debt metrics.
- Built-in **sensitivity, scenario, goal-seek, and Monte Carlo** utilities for rapid downside/upside diagnostics.
- **Break-even and payback** analytics for investment appraisal.

## Requirements
The model depends on the following Python packages:

```text
pip install -r requirements.txt
```

## Usage
Generate the Excel financial model by running:

```bash
python generate_model.py --output Cassava_Bioethanol_Financial_Model.xlsx
```

The script exports a multi-sheet Excel workbook that aligns with the requested modelling specification. Adjust inputs by editing the data tables defined in `bioethanol_model/inputs.py` or by extending the `EditableTable` structures programmatically before calling the exporter.

To explore and tweak the model interactively, launch the Streamlit dashboard:

```bash
streamlit run streamlit_app.py
```

The web app lets you edit all landing-page tables, recalculates the integrated statements on demand, and provides an instant Excel download that mirrors the command line export.

### Editing the production schedules

The **Production Monthly** table on the Input Landing Page is the source of truth for the cassava processing plan. To adjust production and have the changes flow through to the **Production Annual** roll-up:

1. Open the Streamlit app and navigate to the *Input Landing Page* tab.
2. Use either the "Modify Default Inputs & Figures" panel or the "Production Monthly" table expander to edit the **Cassava ton** values (and optional monthly growth % and start month).
3. The model immediately derives the matching ethanol litres and animal-feed tonnage using the backend conversion factors (200 L and 0.275 t per cassava ton respectively).
4. As soon as the monthly figures are updated, the production engine recomputes the compounded series across the projection horizon and refreshes the **Production Annual** table, which aggregates the updated monthly data by fiscal year.

There is no need to edit the annual table manually—the monthly entries drive both the annual totals and every downstream schedule (revenue, feedstock costs, statements, sensitivities, and exports).
