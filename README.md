# SuryaSetu — Community Energy Resilience Platform

> **Seva Innovation Challenge 2026**  
> A software-only decision-support prototype for community renewable energy planning.

---

> ⚠️ **All data in this application are SYNTHETIC and illustrative.**  
> No real measurements, pilot results, cost savings, government partnerships or field validation are claimed.  
> This prototype does not control physical equipment and must not be used for engineering design, procurement or safety-critical decisions without independent expert review.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [Features & Module Architecture](#3-features--module-architecture)
4. [AI/ML vs Rule-Based vs Optimisation](#4-aiml-vs-rule-based-vs-optimisation)
5. [Installation — Windows PowerShell](#5-installation--windows-powershell)
6. [Running the Application](#6-running-the-application)
7. [Using Each Screen](#7-using-each-screen)
8. [Dataset Fields & Synthetic-Data Assumptions](#8-dataset-fields--synthetic-data-assumptions)
9. [Model Evaluation](#9-model-evaluation)
10. [Testing](#10-testing)
11. [Limitations, Safety & Future Work](#11-limitations-safety--future-work)

---

## 1. Problem Statement

Rural and peri-urban communities in low- and middle-income countries depend on essential services — primary health centres, drinking-water facilities, and schools — that are vulnerable to unreliable electricity supply. Renewable solar energy offers a path to greater resilience, but planning when and how much energy will be available, and ensuring that critical loads are always prioritised, is a complex coordination problem.

Community managers and local authorities lack accessible planning tools that:
- Forecast solar generation and demand together
- Enforce transparent rules for critical-service protection
- Create actionable weekly schedules for flexible loads
- Allow "what-if" scenario testing before committing resources

## 2. Solution Overview

SuryaSetu ("Solar Bridge" in Hindi) is a web-based energy planning platform built with Python and Streamlit. It:

1. **Forecasts** hourly solar generation and electricity demand for seven days ahead using a baseline historical mean and a machine-learning model.
2. **Protects** critical services (health centres, water facilities) through configurable hard-constraint rules that are never silently violated.
3. **Plans** a rolling weekly schedule optimised to shift flexible loads into solar-surplus windows.
4. **Compares** three scheduling approaches (fixed, rule-based, LP-optimised) on identical scenario inputs.
5. **Tests** user-defined scenarios including reduced solar, increased demand, battery degradation and grid outages.
6. **Exports** results as CSV and PDF for offline review.

The platform runs entirely offline after dependency installation. No paid APIs, live sensors or grid connections are required.

## 3. Features & Module Architecture

```
suryasetu/
├── app.py                        # Streamlit application (7 pages)
├── requirements.txt
├── README.md
├── data/
│   ├── synthetic_data_generator.py   # Generates 90-day synthetic dataset
│   └── synthetic_energy_data.csv     # Auto-generated on first run
├── src/
│   ├── data_loader.py            # Load, validate, index dataset
│   ├── validation.py             # Input/schema validation helpers
│   ├── forecasting.py            # Module A: Forecasting pipeline
│   ├── essential_services.py     # Module B: Service-protection rules
│   ├── weekly_planner.py         # Module C: 7-day rolling planner
│   ├── optimiser.py              # Module D: PuLP optimisation engine
│   ├── scenarios.py              # Module E: Scenario laboratory
│   ├── metrics.py                # Energy balance + anomaly detection
│   └── reporting.py              # CSV and PDF export
└── tests/
    ├── conftest.py
    ├── test_data_loader.py
    ├── test_forecasting.py
    ├── test_essential_services.py
    ├── test_optimiser.py
    ├── test_weekly_planner.py
    ├── test_scenarios.py
    └── test_reporting.py
```

### Module A — Solar Generation & Demand Forecasting (`forecasting.py`)

- **Baseline**: Same-hour-of-day historical mean, computed from training data only.
- **ML model**: `sklearn.ensemble.RandomForestRegressor` with cyclic hour/day features, lagged values and weather proxies.
- **Chronological split**: Last 14 days held out as test set; no future data leaks into training features.
- **Evaluation**: MAE and RMSE on held-out test set, clearly labelled as synthetic.
- **7-day projection**: Uncertainty bands (±1 σ from ensemble of trees) are propagated forward.

### Module B — Essential Services Protection Layer (`essential_services.py`)

- **Three priority tiers**: Critical → Important → Flexible.
- **Hard minimums**: Critical loads (health centre, water facility) carry `min_energy_kwh_day` constraints.
- **Daily coverage evaluation**: Available solar energy is allocated to critical loads first. Shortfalls are reported explicitly — never silently hidden.
- **Water volume estimation**: Converts kWh → litres using an explicit pump-efficiency assumption (kWh per kL). Clearly labelled as illustrative; real values require hydraulic engineering review.

### Module C — Seven-Day Energy Planner (`weekly_planner.py`)

- Rolling 7-day hourly plan with solar, demand, storage state-of-charge and grid import.
- Battery storage simulation with round-trip efficiency (η = 0.9).
- Grid-outage mode: sets grid capacity to zero.
- Flexible-load recommendations: identifies hours with forecast solar surplus within each facility's scheduling window.
- Shortage warnings emitted whenever unmet demand > 0.1 kWh on any day.

### Module D — Energy Optimisation Engine (`optimiser.py`)

Uses **PuLP** linear programming with CBC solver.

| Method | Description |
|--------|-------------|
| 1. Fixed Schedule | Loads run at their configured operating hours regardless of solar availability. |
| 2. Rule-Based | Flexible loads shifted proportionally to surplus-solar hours. |
| 3. Optimised (PuLP LP) | Minimise `Σ unmet_demand + 0.5 × Σ grid_usage` subject to capacity, operating-window and critical-load constraints. |

If the LP is infeasible (e.g., critical demand exceeds supply), the conflict is exposed and the system falls back to the rule-based result with a warning.

### Module E — Scenario Laboratory (`scenarios.py`)

Six presets, all editable:

| Scenario | Key Changes |
|----------|-------------|
| Normal (baseline) | solar ×1.0, demand ×1.0 |
| Reduced Solar −30% | solar ×0.70 |
| Increased Demand +20% | demand ×1.20 |
| Reduced Storage | storage = 10 kWh |
| Grid Outage | grid_available = False |
| Critical Shortage | solar ×0.40, demand ×1.30, no grid, 10 kWh storage |

Users can also define fully custom scenarios. All presets are labelled as illustrative defaults.

### Module F — Anomaly Detection (`metrics.py`)

Simple threshold-based check: flags values more than 3 standard deviations from a rolling 24-hour mean. Clearly labelled as statistical only; does not detect real equipment faults.

## 4. AI/ML vs Rule-Based vs Optimisation

| Component | Approach | Justification |
|-----------|----------|---------------|
| Solar + demand forecast | RandomForestRegressor (ML) + hour-mean baseline | ML adds value when features (lags, cyclic encodings) improve over the mean; baseline provides an interpretable comparison. |
| Essential-service protection | Transparent priority rules (Critical → Important → Flexible) | Rules are auditable, configurable and explainable to community managers without technical training. |
| Weekly scheduling | PuLP LP optimisation | Mathematical optimisation finds a feasible, near-optimal schedule across 168 hourly decisions simultaneously. |
| Anomaly detection | Rolling z-score threshold | Simple and explainable; no ML needed for basic outlier flagging on a small dataset. |

## 5. Installation — Windows PowerShell

### Prerequisites

- Python 3.9 or later (Python 3.14 tested)
- pip

### Steps

```powershell
# 1. Navigate to the project directory
cd C:\Users\Jainisha\Desktop\Suryasetu

# 2. (Recommended) Create a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate the synthetic dataset (runs automatically on first app launch,
#    but can also be run manually)
python data/synthetic_data_generator.py
```

If you see a PowerShell execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 6. Running the Application

```powershell
streamlit run app.py
```

The application opens in your default browser at `http://localhost:8501`.

To run on a specific port:
```powershell
streamlit run app.py --server.port 8080
```

The application runs completely offline after dependencies are installed.

## 7. Using Each Screen

All screens are accessible from the sidebar navigation. The sidebar also contains global scenario sliders that affect all views.

### 🏠 Overview
Shows the weekly energy outlook, essential-service status, forecast warnings and key metrics for the selected scenario. All values update when sidebar sliders change.

### 🏘️ Community Profile
Add, edit or remove illustrative facilities. Configure:
- Priority category (critical / important / flexible)
- Minimum daily energy requirement
- Operating hours
- Peak demand and flexibility settings
- Water-facility assumptions (tank capacity, pump efficiency)

Click **Save changes** after editing each facility.

### 📈 Forecasts
- **7-Day Forecast tab**: Hourly solar and demand charts with uncertainty bands (±1 σ).
- **Baseline vs ML tab**: Side-by-side comparison of the two models.
- **Evaluation Metrics tab**: MAE and RMSE on the synthetic test set; threshold-based anomaly detection.

### 🔒 Essential Services
Shows daily coverage per facility. Red rows indicate critical shortages. Coverage percentages are computed after allocating available solar supply in priority order (critical first). Shortage details are always shown — never suppressed.

### 📅 Weekly Plan
Displays the 7-day rolling plan with:
- Hourly energy balance chart (solar, demand, grid, storage SoC, unmet)
- Daily summary table
- Flexible-load scheduling recommendations
- Shortage warnings

Click **Regenerate Plan** to refresh after changing sidebar settings.

### 🔬 Scenario Lab
Select a preset or build a custom scenario. Click **Run Scenario** to execute. Results show:
- Key metrics (shortage, renewable fraction, grid energy)
- Side-by-side comparison of all three scheduling methods
- Essential-service coverage for the scenario
- Daily plan chart for the optimised schedule

### 📄 Reports
Download:
- **Forecast CSV**: 7-day hourly forecast with synthetic data label.
- **Weekly Plan CSV**: Hourly and daily plan with assumptions header.
- **Service Coverage CSV**: Daily essential-service coverage table.
- **PDF Report** (or plain-text if ReportLab unavailable): Full summary with methodology, disclaimer, energy balance, coverage table and scheduling comparison.

## 8. Dataset Fields & Synthetic-Data Assumptions

The generator (`data/synthetic_data_generator.py`) creates 90 days × 24 hours = **2,160 rows** of synthetic hourly data.

| Column | Unit | Description |
|--------|------|-------------|
| `timestamp` | Datetime | Hourly timestamp starting 2026-01-01 |
| `hour` | 0–23 | Hour of day |
| `day_of_week` | 0–6 | 0 = Monday |
| `day_of_year` | 1–365 | Day of year |
| `cloud_factor` | 0–1 | Synthetic daily cloud cover (1 = clear) |
| `solar_irradiance_factor` | 0–1 | Hour-of-day × cloud-cover irradiance proxy |
| `solar_generation_kw` | kW | Synthetic PV output (15 kW total capacity) |
| `health_centre_demand_kw` | kW | Synthetic PHC demand (24 h operation) |
| `water_facility_demand_kw` | kW | Synthetic pump demand (06:00–20:00) |
| `school_demand_kw` | kW | Synthetic school demand (07:00–17:00 weekdays) |
| `total_demand_kw` | kW | Sum of all facility demands |
| `net_balance_kw` | kW | solar_generation − total_demand |
| `solar_lag1h` | kW | Solar output one hour prior |
| `demand_lag1h` | kW | Total demand one hour prior |
| `solar_lag24h` | kW | Solar output 24 hours prior |
| `demand_lag24h` | kW | Total demand 24 hours prior |

### Key Assumptions (illustrative, not validated)

- **Solar capacity**: 15 kW total PV for the community.
- **Irradiance model**: Sinusoidal sunrise–sunset with seasonal shift (±1 h) and stochastic cloud factor.
- **Cloud factor**: Monsoon-season dip (July–September, ~DOY 180–270) is represented as a smooth sinusoidal reduction. Not based on real meteorological data.
- **Health centre**: 24 h load, peaking 09:00–17:00 at ~5 kW clinic hours.
- **Water facility**: Pumps run 06:00–20:00; peak morning/evening fill cycles at ~5.5 kW.
- **School**: Weekday 07:00–17:00 operation; midday cooking peak at ~4 kW; weekend community use at ~0.8 kW.
- **Water volume**: 1 kWh pumps 1/0.5 = 2 kL = 2 000 L at the illustrative 0.5 kWh/kL efficiency. Real pump curves depend on head pressure, pipe diameter and groundwater depth.
- **All values are synthetic and have not been validated against real site measurements.**

## 9. Model Evaluation

Evaluation is performed on a **chronological** held-out test set (last 14 days of the 90-day dataset). No future data is used in training features.

Reported metrics:
- **MAE (Mean Absolute Error)** in kW
- **RMSE (Root Mean Square Error)** in kW

Expected approximate results on synthetic data (may vary slightly):

| Target | Model | MAE (kW) | RMSE (kW) |
|--------|-------|----------|-----------|
| Solar generation | Baseline (hour mean) | ~0.4–0.8 | ~0.6–1.2 |
| Solar generation | RandomForest | ~0.3–0.7 | ~0.5–1.0 |
| Total demand | Baseline (hour mean) | ~0.3–0.7 | ~0.5–0.9 |
| Total demand | RandomForest | ~0.2–0.5 | ~0.3–0.7 |

The ML model is expected to be competitive with the baseline on synthetic data. On real data, the improvement could be larger due to genuine weather and behavioural patterns.

> ⚠️ These metrics are computed on **synthetic test data only** and do not represent performance on real measurements.

## 10. Testing

### Running tests

```powershell
# From the project root
python -m pytest tests/ -v
```

### Test coverage by module

| Test file | What it tests |
|-----------|---------------|
| `test_data_loader.py` | Schema validation, column checks, negative-value detection, facility config and scenario param validation |
| `test_forecasting.py` | No-leakage feature engineering, chronological split ordering, baseline/ML correctness, evaluation metrics |
| `test_essential_services.py` | Critical shortage visibility, priority ordering, water volume units, shortage warning generation |
| `test_optimiser.py` | Energy balance, scheduling-window constraints, infeasible-scenario exposure, three-method comparison |
| `test_weekly_planner.py` | Plan generation, response to scenario changes, grid outage, storage effects, shortage warnings |
| `test_scenarios.py` | All preset scenarios run, critical shortage scenario exposes unmet demand, reproducibility |
| `test_reporting.py` | CSV parseable, synthetic disclaimer present, PDF byte generation, handling of empty sections |

### Key safety test

`test_essential_services.py::test_critical_shortage_is_visible` — asserts that when solar supply is only 10% of normal, the system **explicitly reports** critical shortages rather than hiding them. This test will **fail** if the system silently suppresses shortfalls.

## 11. Limitations, Safety & Future Work

### Current Limitations

- **Synthetic data only**: All results shown are based on a deterministic synthetic model with plausible but unvalidated parameters.
- **No real-time data**: The platform does not connect to live sensors, grid APIs or weather feeds.
- **No cost modelling**: Tariffs and operational costs are not included unless the user configures them (currently not implemented in this MVP).
- **Water safety**: Water-volume estimates use a single pump-efficiency assumption. Real water supply systems require expert hydraulic, groundwater and public-health engineering assessment.
- **Infrastructure not modelled**: Electricity sharing between facilities requires physical wiring, protection equipment and regulatory approval that this tool cannot assess.
- **Storage model**: A simplified round-trip efficiency (η = 0.9) is used. Real battery systems have non-linear SoC, temperature and ageing effects.
- **Grid modelling**: Grid supply is represented as a constant-capacity import limit only. Grid tariffs, outage patterns and export constraints are not modelled.

### Safety Disclaimers

- SuryaSetu is a **decision-support prototype**, not a certified energy-control system.
- It **cannot guarantee uninterrupted power** to any facility.
- It **does not control physical equipment**.
- Results displayed must not be used to make commitments about energy supply, water safety or public health outcomes.
- Any real deployment requires review by qualified electrical engineers, water engineers, and health and safety professionals.

### Future Work

- Integrate real weather API (e.g., Open-Meteo, which is free and open) for live forecasts.
- Add time-series cross-validation for more robust ML evaluation.
- Extend optimisation to include battery degradation, tariff structures and demand-response incentives.
- Add multi-community modelling for shared-grid scenarios.
- Implement a mobile-friendly interface for field use.
- Validate synthetic assumptions against real community energy audit data.
- Add proper uncertainty quantification (e.g., conformal prediction intervals).
- Support multiple languages for community managers.

---

*SuryaSetu — Seva Innovation Challenge 2026*  
*Built with Python, Streamlit, scikit-learn, PuLP and Plotly.*  
*All data synthetic. All values illustrative.*
