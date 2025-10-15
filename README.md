# Cassava Bioethanol Financial Model

This repository contains a Python-based toolkit that assembles a complete Cassava Bioethanol project financial model. It builds an Excel workbook featuring detailed input sheets, financial statements, key metrics dashboards, sensitivity analyses, and scenario planning utilities tailored to the FARM_ONLY, BUY_ONLY, and HYBRID feedstock sourcing strategies.

## Features
- Structured **Input Landing Page** capturing projection horizon, global inputs, CAPEX, depreciation schedules, revenue, production, opex, working capital, financing, taxation, inflation, and risk registers.
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
