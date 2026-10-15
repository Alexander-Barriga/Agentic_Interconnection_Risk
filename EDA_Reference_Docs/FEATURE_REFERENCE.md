# Feature Reference Guide
## Berkeley Lab Interconnection Queue Dataset

---

## Column Definitions

| Column | Type | Missing % | Definition | Key Insights |
|--------|------|-----------|-----------|--------------|
| **q_id** | String | 0% | Queue ID (unique with entity) | Composite key with entity |
| **q_status** | Category | 0% | Queue status | **TARGET**: operational, active, withdrawn, suspended |
| **q_date** | Date | 0% | Queue submission date | 2000-2025 range |
| **q_year** | Integer | 0% | Queue year (derived) | Use for temporal stratification |
| **prop_date** | Date | 68.2% | Proposed date | High missingness, use carefully |
| **prop_year** | Integer | 92.1% | Proposed year | Too sparse, not recommended |
| **ia_date** | Date | 44.7% | Interconnection agreement date | Only for projects reaching IA |
| **wd_date** | Date | 45.2% | Withdrawal date | Only for withdrawn projects |
| **on_date** | Date | 65.5% | Online/operational date | Only for operational projects |
| **IA_phase_raw** | String | 42.9% | IA phase (raw values) | Use IA_phase_clean instead |
| **IA_phase_clean** | String | 42.9% | IA phase (standardized) | Shows study phase at last update |
| **region** | Category | 0% | ISO/RTO region | **STRONG SIGNAL**: High variation in outcomes |
| **entity** | Category | 0% | Balancing authority | Can use but high cardinality (50+) |
| **utility** | String | 0% | Utility name | Similar to entity |
| **state** | String | 0% | US State code | Use for geographic clustering |
| **county** | String | 2% | County name | Potential for GIS analysis |
| **fips_code** | Integer | 2% | FIPS county code | For database joins |
| **poi_name** | String | 0% | Point of interconnection | Use for mapping to transmission network |
| **type_1** | Category | 0.5% | Primary resource type | **STRONG SIGNAL**: Solar/Wind/Battery/etc. |
| **type_2** | Category | 85.2% | Secondary resource type | For hybrid/co-located projects |
| **type_3** | Category | 99.9% | Tertiary resource type | Rarely used |
| **type_clean** | Category | 0.5% | Cleaned resource types | Alternative to type_1 for hybrids |
| **mw_1** | Numeric (as Object) | 0.5% | Primary capacity in MW | Convert to float |
| **mw_2** | Numeric (as Object) | 85.2% | Secondary capacity in MW | For hybrid projects |
| **mw_3** | Numeric (as Object) | 99.9% | Tertiary capacity in MW | Rarely used |
| **is_hybrid** | Boolean (engineered) | N/A | Is hybrid or co-located? | Derived: type_2.notna() OR type_3.notna() |
| **total_mw** | Numeric (engineered) | ~2% | Total capacity | mw_1 + mw_2 + mw_3 |
| **service** | Category | 1% | Interconnection service type | ERIS vs. NRIS - shows different approval rates |
| **project_type** | Category | 1% | Project classification | Generation, Surplus, Upgrade, Replacement |
| **developer** | String | 69% | Developer/owner name | High missingness, inconsistent formatting |
| **project_name** | String | 73.5% | Project name | High missingness |
| **cluster** | String | 50% | Queue cluster grouping | For regional grouping |

---

## Feature Engineering Recommendations

### Derived Features to Create

```python
# Time-based features
df['queue_to_ia_days'] = (df['ia_date'] - df['q_date']).dt.days
df['queue_to_wd_days'] = (df['wd_date'] - df['q_date']).dt.days
df['prop_to_queue_days'] = (df['q_date'] - df['prop_date']).dt.days

# Binary indicators
df['is_withdrawn'] = (df['q_status'] == 'withdrawn').astype(int)
df['is_operational'] = (df['q_status'] == 'operational').astype(int)
df['is_active'] = (df['q_status'] == 'active').astype(int)
df['is_hybrid'] = (df['type_2'].notna() | df['type_3'].notna()).astype(int)
df['has_wd_date'] = df['wd_date'].notna().astype(int)
df['has_ia_date'] = df['ia_date'].notna().astype(int)

# Capacity binning
df['capacity_class'] = pd.cut(df['total_mw'], 
                              bins=[0, 50, 500, 2000, 10000],
                              labels=['Small', 'Medium', 'Large', 'Mega'])

# Time-period features
df['queue_year_group'] = pd.cut(df['q_year'], 
                                bins=[2000, 2010, 2015, 2020, 2025],
                                labels=['2000-2010', '2010-2015', '2015-2020', '2020-2025'])

# Target encoding by group
type_approval_rate = df.groupby('type_1')['is_operational'].mean()
df['type_approval_rate'] = df['type_1'].map(type_approval_rate)

region_approval_rate = df.groupby('region')['is_operational'].mean()
df['region_approval_rate'] = df['region'].map(region_approval_rate)

# Missing data indicators
df['missing_capacity'] = df['mw_1'].isna().astype(int)
df['missing_dev_info'] = df['developer'].isna().astype(int)
```

---

## Feature Importance Rankings

### For Predicting "Withdrawn" Status (Rejection Risk)

| Rank | Feature | Signal Strength | Notes |
|------|---------|-----------------|-------|
| 1 | **type_1** | ★★★★★ Strong | Battery 68% withdrawn, Wind 37%, Solar 39% |
| 2 | **region** | ★★★★☆ Strong | CAISO vs ERCOT have different rates |
| 3 | **service** | ★★★☆☆ Moderate | ERIS 42% vs NRIS 37% withdrawal |
| 4 | **total_mw** | ★★☆☆☆ Weak | Distributions similar across status |
| 5 | **is_hybrid** | ★★☆☆☆ Weak | Hybrid status indicates complexity |
| 6 | **q_year** | ★★☆☆☆ Weak | Recent years show temporal trends |
| 7 | **project_type** | ★☆☆☆☆ Very Weak | Limited variation across categories |

### For Predicting "Operational" Status (Success)

| Rank | Feature | Signal Strength | Notes |
|------|---------|-----------------|-------|
| 1 | **type_1** | ★★★★★ Strong | Same ranking as withdrawal (inverse) |
| 2 | **region** | ★★★★☆ Strong | Regional policy/capacity differences |
| 3 | **service** | ★★★☆☆ Moderate | Different approval paths |
| 4 | **total_mw** | ★★☆☆☆ Weak | All size classes have similar rates |
| 5 | **q_year** | ★★☆☆☆ Weak | Temporal effects |
| 6 | **is_hybrid** | ★★☆☆☆ Weak | Complexity indicator |
| 7 | **project_type** | ★☆☆☆☆ Very Weak | Minimal variation |

---

## Encoding Strategies

### Resource Type (type_1)

**Recommended**: One-Hot Encoding
```python
type1_dummies = pd.get_dummies(df['type_1'], prefix='type')
# Creates: type_Solar, type_Wind, type_Battery, type_Gas, etc.
```

**Or**: Target Encoding (if high cardinality)
```python
# Encode with approval rate by type
type_approval_rate = df.groupby('type_1')['is_operational'].mean()
df['type_1_encoded'] = df['type_1'].map(type_approval_rate)
```

### Region (region)

**Recommended**: Target Encoding + Frequency
```python
# Encode with regional approval rate
region_approval = df.groupby('region')['is_operational'].mean()
df['region_target'] = df['region'].map(region_approval)

# Optional: Add frequency-based weight
region_counts = df['region'].value_counts()
df['region_count'] = df['region'].map(region_counts)
```

### Service Type (service)

**Recommended**: One-Hot Encoding
```python
service_dummies = pd.get_dummies(df['service'], prefix='service')
# Creates: service_ERIS, service_NRIS, service_ERIS/NRIS, service_Other
```

### State (state)

**Recommended**: Target Encoding or Grouping
```python
# If high cardinality, group to regions
state_grouping = {
    'CA': 'West', 'OR': 'West', 'WA': 'West',
    'TX': 'South', 'OK': 'South', 'LA': 'South',
    # ... etc
}
df['state_region'] = df['state'].map(state_grouping)
```

---

## Missing Data Handling Strategy

| Feature | Missing % | Strategy |
|---------|-----------|----------|
| **mw_1** | 0.5% | Drop or impute with median by type_1 |
| **mw_2, mw_3** | 85%+ | Create boolean "is_hybrid" flag |
| **ia_date** | 44.7% | Create "has_ia_date" flag, impute with median group |
| **wd_date** | 45.2% | Expected missingness (only withdrawn projects) |
| **on_date** | 65.5% | Expected missingness (only operational projects) |
| **prop_date/year** | 68-92% | Drop or create "missing_proposal" flag |
| **developer** | 69% | Drop or create "missing_developer" flag |
| **project_name** | 73.5% | Drop (too sparse) |
| **IA_phase_raw** | 42.9% | Use IA_phase_clean or create "missing_phase" flag |
| **county** | 2% | Impute with mode by state, or drop |

---

## Outlier Detection

### Capacity (total_mw)

```python
# Identify outliers
Q1 = df['total_mw'].quantile(0.25)
Q3 = df['total_mw'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = Q3 + 1.5 * IQR  # ~240 MW

outliers = df[df['total_mw'] > upper_bound]
# 42 projects > 240 MW (2.5% of data)
# Max: 5,800 MW (mega-project)

# Options:
# 1. Keep outliers (legitimate mega-projects)
# 2. Cap at 95th percentile (450 MW)
# 3. Log-transform for model
```

### Timeline (queue_to_ia_days)

```python
# Identify unrealistic timelines
df[(df['queue_to_ia_days'] < 0) | (df['queue_to_ia_days'] > 10000)]
# Some records may have data entry errors (negative days)
# Solutions: Remove, flag, or investigate

# Distribution:
# Median: 850 days
# P99: 3,500 days
```

---

## Data Quality Flags

Create boolean columns for data quality issues:

```python
df['flag_missing_capacity'] = df['mw_1'].isna()
df['flag_missing_dates'] = df['ia_date'].isna() & (df['q_status'] != 'active')
df['flag_missing_developer'] = df['developer'].isna()
df['flag_unrealistic_timeline'] = (df['queue_to_ia_days'] < 0) | (df['queue_to_ia_days'] > 8000)
df['flag_missing_phase'] = df['IA_phase_clean'].isna()
df['flag_recent_project'] = df['q_year'] >= 2023
```

Then filter training data:
```python
# Remove records with critical issues
training_df = df[
    ~df['flag_unrealistic_timeline'] & 
    (df['q_status'].isin(['operational', 'withdrawn', 'suspended']))
].copy()
```

---

## Class Distribution Summary

### Target Variable: q_status

```
Operational: 13,156 (34.4%)  ← Success
Active:       9,427 (24.7%)  ← Still in queue
Withdrawn:   14,524 (38.0%)  ← Failure/rejection
Suspended:    1,095 (2.9%)   ← Paused
```

**Imbalance Ratio**: 14:1 (Operational to Suspended)

**Handling Strategies**:
1. **Class Weights**: `{operational: 1.0, withdrawn: 1.0, suspended: 3.0}`
2. **SMOTE**: Oversample minority class
3. **Threshold Adjustment**: Tune decision threshold for business needs
4. **Multi-class to Binary**: Combine classes (e.g., Success vs. Failure)

---

## Recommended Train/Test Split Strategy

```python
# Time-based split (most realistic for interconnection context)
train_data = df[(df['q_status'].isin(['operational', 'withdrawn', 'suspended'])) &
                (df['q_year'] <= 2022)]
test_data = df[(df['q_status'].isin(['operational', 'withdrawn', 'suspended'])) &
               (df['q_year'] >= 2023)]

# Or: Region-based split (if building regional models)
regions = df['region'].unique()
for region in regions:
    region_data = df[df['region'] == region]
    region_train = region_data[region_data['q_year'] <= 2022]
    region_test = region_data[region_data['q_year'] >= 2023]

# Validation: 5-fold Stratified K-Fold
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(train_data, train_data['q_status']):
    # Train and validate
```

---

## Example Preprocessing Pipeline

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Define feature groups
numeric_features = ['total_mw', 'queue_to_ia_days', 'q_year']
categorical_features = ['type_1', 'region', 'service']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), 
         categorical_features)
    ])

# Create pipeline
from xgboost import XGBClassifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(n_estimators=100, max_depth=6))
])

# Fit
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
```

---

## Version History

- **v1.0** (Sept 25, 2026): Initial EDA based on Berkeley Lab 2025 dataset
- Dataset: 38,202 records, 30 features, 2000-2025 timespan
- Data Quality: Acceptable for ML (low missingness in critical features)
- Predictive Power: Strong signals from type and region, moderate from service/year
