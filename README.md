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

If network egress is blocked (common in corporate or graded sandboxes), follow
the offline installation notes in ``docs/OFFLINE_SETUP.md`` to stage compatible
``numpy`` and ``pandas`` wheels locally before running the command above.

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

### Verifying the codebase

For a quick sanity check that the repository still compiles after making changes and that the scenarios remain internally
consistent, run the helper script:

```bash
python tools/run_checks.py
```

The script byte-compiles the core packages and executes the new recursive diagnostics pipeline. Each scenario is rebuilt until
its financial statements stabilise, and the resulting balance-sheet and cash reconciliations are checked for drift. If optional
dependencies such as NumPy or pandas are unavailable in the environment, the diagnostics are skipped with a warning so the
command can still finish successfully.

Behind the scenes the `CassavaBioethanolModel` now fingerprints the landing-page inputs and caches scenario builds. Repeated
recalculations from the CLI or Streamlit UI reuse cached outputs unless assumptions change, which keeps the model responsive
even when multiple tabs request the same scenario.

### Editing the production schedules

The **Production Monthly** table on the Input Landing Page is the source of truth for the cassava processing plan. To adjust production and have the changes flow through to the **Production Annual** roll-up:

1. Open the Streamlit app and navigate to the *Input Landing Page* tab.
2. Use **Quick populate from first month** (available in the Production panel, Edit Workspace, and Modify Defaults) to enter first-month **Cassava ton** plus **Monthly increment %**, then click **Propagate horizon**.
3. The model immediately derives the matching ethanol litres and animal-feed tonnage using the backend conversion factors (200 L and 0.275 t per cassava ton respectively).
4. As soon as the monthly figures are updated, the production engine recomputes the compounded series across the projection horizon and refreshes the **Production Annual** table, which aggregates the updated monthly data by fiscal year.

There is no need to edit the annual table manually—the monthly entries drive both the annual totals and every downstream schedule (revenue, feedstock costs, statements, sensitivities, and exports).

### Applying changes part-way through the projection

Most landing-page tables expose a **Change effective month/year** control directly above the data grid. To insert a new change:

1. Choose the month (or year) when the revised assumption should start from the dropdown.
2. Click **Add change** to insert a new row keyed to that period.
3. Edit the values on the inserted row—production tables will cascade the cassava tonnage and the derived ethanol/animal-feed outputs automatically, while cost and policy tables hold the new figures from that point forward.

These controls make it easy to stage CAPEX, production, cost, and working-capital adjustments later in the projection horizon without manually editing every period.
