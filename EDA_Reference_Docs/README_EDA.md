# EDA Project: Agentic AI Interconnection Risk Screener
## Exploratory Data Analysis Complete ✓

---

## 🎯 Project Overview

This project develops an **Agentic AI Interconnection Risk Screener** that predicts the probability of renewable energy projects being accepted or rejected from the power grid interconnection queue.

The Exploratory Data Analysis (EDA) phase has been completed to understand the data and prepare for model development.

---

## 📊 Dataset Summary

| Metric | Value |
|--------|-------|
| **Records** | 38,202 interconnection requests |
| **Time Period** | 2000-2025 (25 years) |
| **Geographic Scope** | 7 ISO/RTOs + Non-ISO regions |
| **Features** | 30 variables |
| **Target Variable** | Queue Status (operational, active, withdrawn, suspended) |

### Key Statistics

- **Success Rate**: 34.4% projects reach operational status
- **Failure Rate**: 38.0% of projects are withdrawn from the queue
- **Active/Pending**: 24.7% still in queue
- **Suspended**: 2.9% temporarily paused

---

## 📁 Deliverable Files

### 1. **EDA_SUMMARY.md** (759 lines)
Comprehensive summary document covering:
- Data quality assessment (missing values, data types)
- Target variable analysis
- Distribution analysis for all major features
- Temporal trends and patterns
- What the data can/cannot tell us
- ML model development recommendations
- SQL database schema design
- RAG component requirements
- Agent tool architecture

**Read this first** for high-level findings and recommendations.

### 2. **FEATURE_REFERENCE.md** (332 lines)
Technical reference guide with:
- Column definitions and metadata
- Feature engineering strategies with code
- Feature importance rankings
- Categorical encoding approaches
- Missing data handling strategies
- Outlier detection methods
- Data quality flags
- Preprocessing pipeline templates

**Use this** as a technical reference during model development.

### 3. **Notebooks/EDA.ipynb**
Jupyter notebook with:
- Data loading and exploration
- Data quality checks
- Statistical analysis
- Visualizations (distributions, trends, comparisons)
- Feature engineering examples
- Correlation analysis

**Run this** to reproduce the analysis or explore the data interactively.

---

## 🔍 Key Findings

### Strongest Predictive Signals

1. **Resource Type** ⭐⭐⭐⭐⭐ (Strongest)
   - Solar: 39% operational success
   - Wind: 35% operational success
   - Battery: 15% operational success (⚠️ High withdrawal risk)
   - Clear differentiation → Use in ML models

2. **Geographic Region** ⭐⭐⭐⭐ (Strong)
   - Significant variation across ISO/RTOs (CAISO, ERCOT, SPP, PJM, MISO)
   - Different regulatory environments and queue dynamics
   - → Recommend regional models or regional features

3. **Service Type** ⭐⭐⭐ (Moderate)
   - ERIS (Energy Resource): 42% withdrawal rate
   - NRIS (Network Resource): 37% withdrawal rate
   - Different approval paths → Include in model

4. **Project Capacity** ⭐⭐ (Weak)
   - Median: 50 MW
   - Similar distributions across all outcomes
   - Useful for stratification but not primary predictor

5. **Time Dimension** ⭐⭐ (Weak)
   - Recent projects (2020+) show slightly different patterns
   - Interconnection rules evolved over 25-year period
   - Include year as feature to capture temporal shifts

### Data Quality Assessment

**Strengths** ✓:
- Large sample size (38,202 records)
- Low missingness in critical fields (<5%): status, type, capacity, region
- 25-year historical coverage enables trend analysis
- Clear, well-defined target variable

**Challenges** ⚠️:
- High missingness in metadata (73% for project_name, 69% for developer)
- All columns stored as object type (requires conversion)
- Class imbalance: Suspended (2.9%) heavily underrepresented
- Temporal dependencies: Recent projects may not have complete outcomes

**Limitations** ❌:
- No explanation for WHY projects are withdrawn
- No developer financial/experience data
- No technical feasibility scores
- No market economics (prices, incentives)
- No detailed site constraints (solar/wind resources, permitting)
- Data spans major regulatory changes (different rules 2000 vs 2025)

---

## 🤖 Machine Learning Recommendations

### Recommended Approach

**Model Type**: Gradient Boosting Machines
- **Recommended**: XGBoost, LightGBM
- **Alternative**: Random Forest
- **Why**: Best performance for tabular data, handles mixed features, feature importance built-in

**Target Variable**: Binary Classification
- **Option A (Recommended)**: `Withdrawn (1)` vs `Not Withdrawn (0)`
  - Class balance: 38% vs 62%
  - Direct match to "rejection from queue" goal
  
- **Option B**: `Operational (1)` vs `Not Operational (0)`
  - Class balance: 34.4% vs 65.6%
  - Alternative success metric

### Recommended Features

**Categorical** (Must include):
- `type_1` (Resource type) - Strongest signal
- `region` (ISO/RTO) - Strong signal
- `service` (ERIS vs NRIS) - Moderate signal

**Numerical** (Important):
- `total_mw` (Project capacity)
- `q_year` (Queue submission year)

**Engineered** (Optional):
- `is_hybrid` (Hybrid/co-located binary)
- `capacity_class` (Size bins)
- `queue_to_ia_days` (Processing time)

### Training Strategy

```python
# Data split: Time-based (most realistic)
train = df[(df['q_status'].isin(['operational', 'withdrawn', 'suspended'])) &
           (df['q_year'] <= 2022)]
test = df[(df['q_status'].isin(['operational', 'withdrawn', 'suspended'])) &
          (df['q_year'] >= 2023)]

# Cross-validation: Stratified K-Fold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Class weights: Handle imbalance
class_weights = {0: 1.0, 1: 1.2}  # Slight weight to minority

# Evaluation: Stratified by region and type
for region in df['region'].unique():
    evaluate_on_region(region)
```

### Key Metrics

- **Primary**: Recall (minimize missed rejections)
- **Secondary**: Precision, F1-Score
- **Regional**: Separate evaluation by ISO/RTO
- **Type-specific**: Separate metrics for Solar/Wind/Battery

---

## 🗄️ SQL Database Architecture

### Core Tables

```sql
-- Main fact table: interconnection_requests
- request_id (PK)
- region_id (FK)
- queue_date, proposed_date, ia_date, withdrawal_date
- status (operational, active, withdrawn, suspended)
- capacity_mw
- type_1, type_2, type_3 (resource types)
- service (ERIS, NRIS)
- project_type (Generation, Surplus, Upgrade, Replacement)
- is_hybrid (boolean)

-- Supporting tables
- regions (lookup)
- entities (lookup)
- time_dimensions (date hierarchy)
```

### Performance Views

- `v_active_projects` - Current queue
- `v_withdrawn_projects` - Historical failures
- `v_time_to_ia` - Timeline statistics
- `v_regional_summary` - Regional aggregations

---

## 🔗 RAG Component: Essential Data Sources

To complement ML predictions, integrate these information sources:

### Regulatory Documents
- FERC Orders (Order 2023, 2020-A, historical)
- NERC Reliability Standards
- ISO/RTO Tariffs and Operating Procedures
- State Renewable Energy Policies

### Market Intelligence
- Electricity Price Forecasts
- Renewable Energy Incentive Programs (ITC, PTC)
- Grid Capacity Reports
- Transmission Constraint Analyses

### Technical References
- POI Technical Data
- Transmission Network Maps
- Grid Upgrade Cost Estimates
- Environmental Review Documents

---

## 🤖 Agent Tool Architecture

### Three-Part System

**1. SQL Tool** (Historical Data & Benchmarks)
- Query recent projects by type/region
- Calculate historical approval rates
- Timeline statistics
- Regional risk profiles

**2. ML Tool** (Predictive Model)
- Predict approval probability
- Estimate timeline to IA
- Identify key risk factors
- Feature importance explanations

**3. RAG Tool** (Regulatory & Market Context)
- Retrieve relevant policies
- Grid capacity information
- Market condition updates
- Technical constraint data

### Example Query Flow

```
User: "What's the approval probability for a 100 MW solar project in CAISO?"

Agent Process:
1. [SQL] Find comparable projects → 73% approval rate for similar projects
2. [ML] Predict using model → 71.3% approval probability
3. [RAG] Get CAISO updates → "Recent policy X accelerates solar approvals"
4. Synthesize → "Your project has 71% approval probability, timeline 18-24 months"
```

---

## 🚀 Next Steps (Priority Order)

### Week 1-2: Data Engineering
- [ ] Design and create SQL database
- [ ] Load Berkeley Lab data
- [ ] Implement data validation
- [ ] Create indexed views

### Week 2-3: Feature Engineering
- [ ] Build preprocessing pipeline
- [ ] Handle missing values
- [ ] Create derived features
- [ ] Implement train/test split

### Week 3-5: Model Development
- [ ] Train baseline models
- [ ] Develop XGBoost/LightGBM models
- [ ] Hyperparameter tuning
- [ ] Cross-validation & evaluation
- [ ] Feature importance analysis

### Week 4-6: RAG Component
- [ ] Collect regulatory documents
- [ ] Build document index
- [ ] Implement semantic search
- [ ] Create retrieval pipeline

### Week 6-8: Agent Integration
- [ ] Develop SQL query tool
- [ ] Develop ML prediction tool
- [ ] Develop RAG retrieval tool
- [ ] Build agent orchestration
- [ ] End-to-end testing

---

## 📊 Quick Reference: Feature Summary

| Feature | Type | Missing % | Importance | Notes |
|---------|------|-----------|-----------|-------|
| q_status | Category | 0% | ⭐⭐⭐⭐⭐ | **TARGET** |
| type_1 | Category | 0.5% | ⭐⭐⭐⭐⭐ | Strongest signal |
| region | Category | 0% | ⭐⭐⭐⭐ | Strong variation |
| total_mw | Numeric | 2% | ⭐⭐ | Good for stratification |
| service | Category | 1% | ⭐⭐⭐ | Moderate signal |
| q_year | Integer | 0% | ⭐⭐ | Temporal feature |
| is_hybrid | Boolean | ~1% | ⭐⭐ | Complexity indicator |
| q_date | Date | 0% | ⭐⭐ | Timeline feature |
| ia_date | Date | 44.7% | ⭐⭐ | For time-to-event |
| project_type | Category | 1% | ⭐ | Weak signal |

---

## 📚 How to Use These Documents

1. **Start with EDA_SUMMARY.md**
   - Read sections 1-3 for data overview
   - Read section 8 for ML recommendations
   - Read section 14 for architecture

2. **Reference FEATURE_REFERENCE.md** during development
   - Use encoding strategies (Section 3)
   - Use preprocessing templates (Section 12)
   - Use missing data handling (Section 6)

3. **Run Notebooks/EDA.ipynb** to:
   - Explore the data yourself
   - Verify findings
   - Generate additional visualizations
   - Create custom analyses

---

## ✅ Checklist for Model Development

- [ ] Read EDA_SUMMARY.md (Executive Summary)
- [ ] Review FEATURE_REFERENCE.md (Technical Details)
- [ ] Run Notebooks/EDA.ipynb (Reproduce analysis)
- [ ] Set up SQL database (Data Engineering)
- [ ] Implement feature pipeline (Feature Engineering)
- [ ] Train baseline model (Establish baseline)
- [ ] Train advanced models (XGBoost/LightGBM)
- [ ] Evaluate on test set (Validation)
- [ ] Create RAG component (Regulatory data)
- [ ] Integrate with agent (Final assembly)

---

## 📞 Questions & Next Steps

For questions about:
- **Data structure**: See FEATURE_REFERENCE.md
- **Analysis findings**: See EDA_SUMMARY.md
- **Model recommendations**: See EDA_SUMMARY.md sections 8-9
- **Database design**: See EDA_SUMMARY.md section 9
- **Feature engineering**: See FEATURE_REFERENCE.md sections 2-3

---

## 📋 Version Information

- **EDA Date**: September 25, 2026
- **Dataset Version**: Berkeley Lab 2025 (38,202 records)
- **Analysis Scope**: 2000-2025 historical data
- **Python Version**: 3.9+
- **Key Libraries**: pandas, scikit-learn, matplotlib, seaborn

---

**Status**: ✅ EDA COMPLETE - Ready for Model Development Phase

Next: See documentation files for detailed analysis and recommendations.
