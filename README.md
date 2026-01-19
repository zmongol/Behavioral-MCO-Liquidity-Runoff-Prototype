# Behavioral MCO Liquidity Runoff Prototype



**Clean • Reproducible • Tail-Focused**  
Open-source prototype demonstrating a modern, two-stage behavioral **Maximum Cumulative Outflow (MCO)** model for retail & SME transactional deposit balances.

This is an **educational and research-oriented prototype** — intentionally scoped to behavioral outflows only.  
It is **not** production software and should not be used directly for regulatory reporting, capital/liquidity planning, or live risk decisions without major extensions and rigorous validation.



## Strengths & What It Does Well

- **Very clean, reproducible end-to-end pipeline** — from synthetic data generation to multi-horizon training and rich reporting  
- **Strong focus on tail risk** — tail capture %, P90 coverage, top contributors (what really matters in liquidity stress)  
- **Industry-standard two-stage modeling** — runoff probability (classifier) + conditional severity (P50/P90 regressors)  
- **Multi-entity & multi-currency ready** — automatic FX conversion and per-entity breakdowns out-of-the-box  
- **High-quality documentation** — clear explanations of concepts, limitations, and realistic next steps  
- **Easy to run & experiment** — works instantly with small or large synthetic datasets  
- **Efficient & GPU-aware** — scales reasonably to thousands of customers on consumer hardware  

This prototype fills a real gap: a simple, well-documented, open behavioral MCO reference that many teams and researchers still lack.



## Current Limitations (Known & Intentional)

- **Behavioral-only** — does **not** include contractual maturities (term deposits, wholesale funding, scheduled outflows)  
- **Severity predictions** are point estimates (no built-in uncertainty intervals or conformal prediction yet)  
- **Simple train/test split** — no out-of-time validation or historical stress backtesting  
- **Stylized synthetic data** — real-world data will be noisier and require additional preprocessing/feature engineering  
- **XGBoost only** — no model comparisons (LightGBM, CatBoost, neural nets, etc.) or ensembling  
- **No explainability layer** — no SHAP, partial dependence, or feature importance reporting yet  
- **No regulatory mapping** — no direct LCR/NSFR runoff factor comparison or HQLA integration  

These choices keep the prototype focused, understandable, and easy to extend — but they are deliberate gaps, not oversights.



## Recommended Next Steps & High-Impact Extensions

The following would significantly increase real-world usefulness:

1. **Add contractual maturity layer** (highest priority) — deterministic base + behavioral overlay  
2. **Integrate explainability** — SHAP values, global/local feature importance  
3. **Add uncertainty quantification** — quantile regression, conformal intervals for severity  
4. **Include proper validation** — out-of-time splits, backtesting against historical stress periods  
5. **Regulatory alignment** — map to LCR/NSFR buckets and compare modeled vs regulatory runoff  
6. **Model benchmarking** — compare XGBoost with other tabular models  
7. **Product extensions** — early redemption/breakage modeling, non-retail segments, intraday liquidity  

Pull requests, issues, and ideas in these areas are very welcome!



## Quick Start

```bash
# Setup (one time)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Quick smoke test (needs existing mco_mock_data.zip)
./run_all.sh small

# Realistic scale run (auto-generates synthetic data)
./run_all.sh big

# Your own data
./run_all.sh custom /path/to/your-data.zip
```



**Output structure** (new timestamped folder each run):
```
runs/20260120_134500_big/
├── metadata/     # run parameters & snapshot
├── outputs/      # extracted CSVs, labels, predictions, models, metrics
├── reports/      # summaries, tail capture, top contributors, entity views
└── logs.txt      # full execution log
```



## Data Requirements

Four CSV files inside the ZIP (schema reference: [generate_big_dataset.py](generate_big_dataset.py)):

- **CUSTOMER_MASTER.csv** – customer metadata
- **BALANCE_DAILY.csv** – daily balances
- **TXN_LEDGER.csv** – inflows/outflows
- **FX_RATES_DAILY.csv** – for base currency conversion



## Core Concepts & Terminology

| Term | Definition in this prototype |
|------|-----|
| **MCO** | max(0, maximum cumulative net outflow over horizon) |
| **Runoff** | Material balance drop (absolute + relative threshold) — explicit classification target |
| **Rundown** | Gradual decline — captured implicitly via cumulative net outflows |
| **Behavioral** | Customer-driven, discretionary flows (this prototype) vs contractual maturities (not modeled) |
| **Tail capture** | % of actual worst outflows correctly ranked in model's predicted worst cases |
| **P90 coverage** | Proportion of actual MCO events falling below model's predicted P90 severity |
| **Base currency** | All metrics in SGD (configurable via `BASE_CCY`) |



## ## Behavioral vs Contractual Outflows

This prototype focuses exclusively on **behavioral outflows** (customer-driven, discretionary flows).

In production MCO/LCR/NSFR modeling, the industry-standard pattern is:

1. **Contractual outflows first** (deterministic base layer — easy to implement, low modeling risk)
2. **Behavioral outflows overlaid on top** (incremental/discretionary component — modeled, higher uncertainty)


**Common integration methods:**

- **Additive:** Total = Contractual + Modeled Behavioral
- **Adjusted:** Apply behavioral breakage rates to contractual maturities

→ **Adding a contractual layer is the most natural and high-impact next step for this prototype.**

## License

MIT — free to use, modify, fork, and share.

Built as a clean, open reference for behavioral liquidity risk modeling.

Feedback, contributions, and forks are enthusiastically encouraged!

---
*Last updated: January 2026*