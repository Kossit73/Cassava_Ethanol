# Investor Readiness Review – Cassava Bioethanol Financial Model

## Executive view
The model already has strong structural foundations for investor conversations: integrated three-statement modelling, scenario support (FARM_ONLY/BUY_ONLY/HYBRID), debt and working-capital schedules, and core returns metrics (NPV/IRR/payback). However, several investor-critical features are either missing or not fully wired into calculations, especially downside risk quantification, debt covenant reporting, and validation guardrails for key assumptions.

## What is working well today
1. **Integrated statements and schedules** are assembled in one build path, which is what investors expect for auditability.  
2. **Scenario modeling** is native and easy to compare across feedstock sourcing strategies.  
3. **Core valuation metrics** (Project NPV/IRR, Equity IRR, Investor IRR, Owner IRR, payback) are already available.  
4. **Diagnostics** already checks accounting consistency (balance-sheet and cash reconciliation), which is a good baseline for model control.

## High-impact gaps that reduce investor confidence

### 1) Risk register is collected but not used in projections
- The input layer defines a `risk_schedule` table, but the build path does not consume it.
- Impact: investors cannot see quantified effects of operational/market risks in cash flow outcomes.

**Recommendation**
- Convert each risk row into a probabilistic cash-flow adjustment (volume, price, opex, capex, delay).
- Add a “risk-adjusted base case” and “P90 downside case” to outputs.
- Expose an expected-value adjustment bridge from unadjusted EBITDA/FCF to risk-adjusted EBITDA/FCF.

### 2) Debt bankability metrics are not surfaced as primary KPIs
- The model calculates loan schedules and free cash flow, but does not publish lender-style covenants (e.g., DSCR, LLCR, minimum cash reserve headroom) as first-class outputs.
- Impact: project can look attractive on IRR but still fail debt committee tests.

**Recommendation**
- Add monthly/annual DSCR and LLCR computations with min/avg values and covenant breach flags.
- Add debt sculpting support: debt service profile linked to projected CFADS.
- Include a dedicated “Lender Dashboard” with covenant heatmap.

### 3) Terminal value assumptions are stored but not integrated into valuation math
- Global inputs include terminal growth and capital gains tax assumptions, but key metric computation currently relies on explicit horizon cash flows without terminal value integration.
- Impact: valuation may understate (or inconsistently state) equity value versus investor expectations.

**Recommendation**
- Support optional terminal value methods: Gordon growth and exit multiple.
- Discount terminal value to present and report enterprise value bridge.
- Add sensitivity tornado for terminal growth and exit multiple.

### 4) Validation controls do not enforce completeness of critical assumptions
- Several default tables are placeholders and can resolve to empty model frames if users do not explicitly populate them.
- Impact: silent fallback to defaults (e.g., zero discount/tax in some pathways) can produce misleading returns.

**Recommendation**
- Introduce hard validation before build: block export when mandatory assumptions are missing.
- Add explicit warnings for implausible ranges (e.g., ethanol yield, tax rate, inflation, capex intensity).
- Emit an “Assumption Completeness Score” for investment memos.

### 5) Commercial realism can be improved with contract-linked revenue/cost logic
- Current revenue and feedstock costs are escalation-driven; model is less explicit on offtake terms and feedstock procurement contract structures.
- Impact: investors need visibility into contract risk and margin resilience.

**Recommendation**
- Add optional offtake contract mechanics (floor/ceiling price, take-or-pay, FX linkage).
- Add feedstock procurement mixes (spot vs contracted vs own farm), with stress on procurement shocks.
- Track gross margin by product and sourcing strategy over time.

## Priority implementation roadmap (investor-facing)

### Phase 1 (2–3 weeks): “Bankability baseline”
- Add mandatory input validation and assumption quality checks.
- Add DSCR/LLCR covenant outputs and breach flags.
- Add downside case pack (Base/P50/P90) using existing sensitivity + Monte Carlo infrastructure.

### Phase 2 (3–5 weeks): “Valuation credibility”
- Add terminal value integration and valuation bridge.
- Add debt sculpting and refinancing scenario toggle.
- Publish automated one-page investment committee summary (KPIs, risks, covenants, break-even, payback).

### Phase 3 (4–6 weeks): “Institutional-grade diligence”
- Integrate risk register directly into stochastic cash-flow drivers.
- Add ESG/carbon monetisation module (if applicable in your target market) and policy incentive assumptions.
- Implement scenario governance: locked assumptions, change log, and versioned outputs.

## Suggested investor KPI set to add immediately
- Equity NPV @ target hurdle rate.
- DSCR (min, average, years below covenant).
- LLCR and PLCR.
- EBITDA margin and cash conversion ratio.
- Breakeven utilisation (% capacity).
- Value-at-Risk on equity IRR (e.g., 10th percentile IRR).
- Time-to-distribution for equity investors.

## Final recommendation
For investor attraction, reposition the model from a **planning tool** to a **bankability and diligence tool**. The quickest win is to add covenant analytics, quantified downside cases, and hard input validation; those three upgrades materially improve confidence in both returns quality and execution realism.
