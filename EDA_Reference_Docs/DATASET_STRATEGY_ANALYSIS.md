# Dataset Strategy Analysis: Berkeley Lab vs CASIO for CA-Focused AI Screener

**Date:** September 25, 2026  
**Analysis:** Comparing two EDA approaches for the Agentic AI Interconnection Risk Screener  
**Question:** Train on national Berkeley Lab data, validate on CA CASIO data?

---

## Executive Summary

**TL;DR:** Using Berkeley Lab (national) as training + CASIO CA-only for testing is **LOGICALLY SOUND** and **IMPLEMENTATIONALLY FEASIBLE**, but **only if your final goal is truly CA-focused**. However, there's a critical trade-off: using the full Berkeley dataset enables more powerful predictions but sacrifices CA-specific interpretability.

**Recommendation:** If simplicity and CA-focus are paramount → use CASIO exclusively. If robust national patterns → use Berkeley with CA fine-tuning.

---

## Part 1: Does This Make Logical Sense?

### ✅ YES - Here's Why

#### 1. **Transfer Learning is Valid Here**
The interconnection approval process has universal principles:
- **Physics doesn't change by region:** Power quality requirements are similar across CAISO, ERCOT, SPP, PJM
- **Regulatory patterns are similar:** All regions follow FERC interconnection procedures
- **Technical risks correlate:** Project size, voltage level, and timeline urgency affect approvals everywhere
- **Berkeley can learn generalizable patterns:** The national dataset teaches the model "what makes projects succeed"

**Example:** A 100 MW solar project with uncertain timeline risks approval in all regions. Patterns learned from Texas projects in Berkeley should transfer to California projects in CASIO.

#### 2. **CAISO Data is a Valid Test Set**
Using CASIO Cluster 15 for validation makes sense because:
- ✅ **Similar features:** Both datasets have MW, voltage, technology type, dates
- ✅ **Same domain:** Both are interconnection queues, not different types of projects
- ✅ **CA-specific nuances:** Tests if national model captures CAISO-specific approval patterns
- ✅ **Adequate sample size:** 86 CASIO projects is sufficient for meaningful validation

#### 3. **Regional Variation Actually Strengthens the Model**
From the EDA findings:
- **Berkeley shows regional variation:** CAISO has ~8,000+ requests vs ERCOT's different withdrawal patterns
- **CAISO-specific signals exist:** Median 950 days for solar in CAISO (different from other regions)
- **Model learns: "Region matters"** → The national model will learn to adjust predictions for CAISO
- **CASIO validates:** "Does the learned regional adjustment work for our specific CA data?"

#### 4. **No Data Leakage Risk**
- ✅ CASIO Cluster 15 is SEPARATE from Berkeley Lab data (different sources)
- ✅ No overlap in projects → clean train/test split
- ✅ CASIO data wasn't used during Berkeley model training

---

### ⚠️ BUT - Here Are the Caveats

#### 1. **Berkeley Has Outcomes, CASIO Doesn't**
- **Berkeley:** 38,202 projects with **q_status** (target variable: operational/withdrawn/suspended/active)
- **CASIO Cluster 15:** 86 projects with **NO outcome labels**

**Problem:** You can't validate model predictions on CASIO if you don't know the ground truth!

**Solution:** CASIO can be used for:
- ✅ Feature distribution comparison ("Is a 100 MW solar project in CA similar to Berkeley's 100 MW solar projects?")
- ✅ Confidence interval testing ("Is the model uncertain on CA-specific projects?")
- ❌ NOT for accuracy metrics (precision/recall/confusion matrix)

#### 2. **CAISO is Only 64% of CASIO's Data**
From Cluster 15 findings:
- California (CAISO): 55 projects, 64% of dataset
- Texas (ERCOT): 31 projects, 36% of dataset

**Problem:** If you want "CA-only" focus, CASIO Cluster 15 has TX projects mixed in.

**Solution:** Filter to CA projects only → ~55 projects remaining (too small for robust testing)

#### 3. **Cluster 15 is Just One Cluster**
- **CASIO has multiple clusters** (Cluster 14, prior clusters, etc.)
- **Cluster 15 = current/active projects** (NOT historical outcomes)
- **Other CASIO clusters may have historical outcomes** (investigate!)

**Implication:** If other CASIO clusters have outcome labels, you should use those instead of Berkeley!

#### 4. **Berkeley is 25 Years of Data, CASIO is 9 Years**
- **Berkeley:** 2000-2025 (comprehensive historical patterns)
- **CASIO Cluster 15:** 2015-2024 (recent projects only)

**Problem:** Different time periods → approval timelines have changed

**Impact:** National patterns learned from 2000-2010 may not apply to 2020+ projects; vice versa

---

## Part 2: Can You Actually Do This Project with These Datasets?

### Scenario 1: Train on Berkeley, Test on CASIO (Current Proposal)

| Aspect | Assessment | Notes |
|--------|-----------|-------|
| **Logical Soundness** | ✅ Yes | Transfer learning valid for interconnection domain |
| **Feature Compatibility** | ✅ Yes | Both datasets have capacity, voltage, technology, dates |
| **Target Variable** | ⚠️ Partial | Berkeley has outcomes; CASIO doesn't |
| **Validation Metrics** | ⚠️ Limited | Can't measure accuracy without CASIO ground truth |
| **CA Specificity** | ❌ No | Berkeley is national; loses CA nuances |
| **Sample Size** | ✅ Adequate | 38,202 training → 55 CA test projects is valid |
| **Implementability** | ✅ Yes | Technically feasible |

### Scenario 2: Use Berkeley ONLY (Recommended if CA-focus not critical)

| Aspect | Assessment | Notes |
|--------|-----------|-------|
| **Logical Soundness** | ✅ Yes | Clear national patterns; robust dataset |
| **Feature Compatibility** | ✅ Yes | Designed for this purpose |
| **Target Variable** | ✅ Yes | 38,202 projects with outcomes |
| **Validation Metrics** | ✅ Yes | Full metrics available (train/test split) |
| **CA Specificity** | ✅ Yes | Can easily filter to CAISO subset |
| **Sample Size** | ✅ Excellent | 8,000+ CAISO projects in Berkeley |
| **Implementability** | ✅ Easiest | Straightforward, no integration needed |

### Scenario 3: Use CASIO ONLY (If CA-focus is PARAMOUNT)

| Aspect | Assessment | Notes |
|--------|-----------|-------|
| **Logical Soundness** | ❌ No | No outcome labels for supervised learning |
| **Feature Compatibility** | ✅ Yes | Good feature set for clustering |
| **Target Variable** | ❌ Missing | This is the blocker |
| **Validation Metrics** | ❌ Impossible | Can't validate without outcomes |
| **CA Specificity** | ✅ 100% | Pure CA data (subset of CASIO) |
| **Sample Size** | ❌ Too Small | ~55 CA projects insufficient for ML |
| **Implementability** | ✅ Simple | But requires outcome data first |

### Scenario 4: Combine Both (BEST if outcomes exist in other CASIO clusters)

| Aspect | Assessment | Notes |
|--------|-----------|-------|
| **Logical Soundness** | ✅ Yes | Leverages both national + CA patterns |
| **Feature Compatibility** | ✅ Yes | Identical schemas |
| **Target Variable** | ✅ Yes | If other clusters have outcomes |
| **Validation Metrics** | ✅ Yes | Full metrics on CA test set |
| **CA Specificity** | ✅ Yes | CA-focused model from CA+national data |
| **Sample Size** | ✅ Excellent | Berkeley + CASIO historical clusters |
| **Implementability** | ⚠️ Complex | Requires integrating both datasets |

---

## Part 3: Detailed Comparison Table

### Berkeley Lab Dataset (National)

```
Size: 38,202 interconnection requests
Period: 2000-2025 (25 years)
Geographic: National (CAISO: ~8,000; ERCOT: varies; SPP, PJM, MISO, other)
Target Variable: q_status (operational, withdrawn, suspended, active)
Status: ✅ COMPLETE - Ready for supervised learning
```

**Strengths:**
- ✅ Has outcome labels for supervised learning
- ✅ 8,000+ CAISO projects for CA analysis
- ✅ 25 years of historical patterns
- ✅ Large enough for robust ML model training
- ✅ Regional variation enables learning "region matters"
- ✅ EDA complete and documented

**Weaknesses:**
- ❌ National focus (dilutes CA-specific signals)
- ❌ Approval rates vary by region (CAISO ≠ ERCOT patterns)
- ❌ Historical bias (2000-2010 projects ≠ 2020+ projects)
- ❌ May not capture recent CA policy changes

**Ideal For:**
- National interconnection risk screener
- Understanding general approval patterns
- Building baseline model for fine-tuning
- Understanding cross-regional variation

---

### CASIO Cluster 15 (CA/TX - Current Queue)

```
Size: 86 interconnection requests
Period: 2015-2024 (9 years)
Geographic: California 55 (CAISO), Texas 31 (ERCOT)
Target Variable: ❌ MISSING
Status: ⚠️ INCOMPLETE - No outcomes (current queue projects)
```

**Strengths:**
- ✅ 55 pure CA projects (perfect for CA focus)
- ✅ Recent projects (2015-2024, reflects current market)
- ✅ High-quality features (clean data, 99.74% complete)
- ✅ Perfect for CA-specific inference
- ✅ EDA complete and documented

**Weaknesses:**
- ❌ No outcome labels (blocks supervised learning)
- ❌ Only 55 CA projects (too small for independent training)
- ❌ All projects are "active/pending" (no completed outcomes)
- ❌ Can't validate model accuracy on CA data
- ❌ No historical patterns (recent data only)

**Ideal For:**
- Testing model on current projects
- CA-specific inference and scoring
- Understanding CA queue characteristics
- Feature importance validation on CA projects

---

## Part 4: Practical Recommendation

### IF Your Goal is: **"Simple, CA-focused AI Screener"**

**Use CASIO exclusively** BUT with this limitation:
```
❌ STOP: You cannot build supervised ML model without outcomes
⚠️ WORKAROUND: Use clustering + anomaly detection instead
✅ DO: Get outcomes from other CASIO clusters or Cluster 14
```

**Action:**
1. Check if other CASIO clusters have outcome labels
2. Combine Cluster 15 (current projects) with Cluster 14 (historical) for outcomes
3. Train on combined CASIO historical data
4. Use Cluster 15 for inference on current projects
5. Result: Pure CA model, CA-specific insights

---

### IF Your Goal is: **"Robust, Production-Ready Screener (any region, including CA)"**

**Use Berkeley Lab as primary** with CASIO for validation:
```
✅ TRAIN: 38,202 national projects (Berkeley)
✅ VALIDATE: 55 CA projects (CASIO Cluster 15 filtered)
✅ FINAL: Deploy model nationally, with CA-specific parameters
```

**Action:**
1. Train XGBoost on Berkeley Lab (q_status → outcomes)
2. Extract CAISO subset (8,000 projects from Berkeley for pre-validation)
3. Test learned CA patterns on CASIO Cluster 15
4. Fine-tune region-specific parameters
5. Result: Robust model with CA specialty layer

---

### IF Your Goal is: **"Best of Both Worlds"**

**Combine Berkeley + CASIO clusters** (RECOMMENDED):
```
✅ TRAIN: Berkeley Lab (38,202 national) + other CASIO clusters with outcomes
✅ VALIDATE: Mix of Berkeley test set + CASIO Cluster 15
✅ CA LAYER: CASIO data trains region-specific features
✅ FINAL: National model with strong CA focus
```

**Action:**
1. Check CASIO Cluster 14 and prior for outcomes → merge with Berkeley
2. Train ensemble: Berkeley (pre-training) + CASIO history (fine-tuning)
3. Use Cluster 15 for production inference
4. Result: CA-optimal model with national robustness

---

## Part 5: Critical Questions to Ask

Before proceeding with the Berkeley + CASIO approach:

### Question 1: Do other CASIO clusters have outcome labels?
```
Current Status: CASIO Cluster 15 = Active projects (NO outcomes)
Need to Check: Cluster 14, Cluster 13, ... Cluster 1
If Found: ✅ Use combined CASIO data for stronger CA signal
If Not Found: ⚠️ Must use Berkeley as training source
```

### Question 2: What does "CA-focused" mean for your use case?
```
Option A: Model primarily serves CA users
         → Use Berkeley + fine-tune on CASIO outcomes
         → Deploy nationally but emphasize CA accuracy

Option B: Model ONLY handles CA projects
         → Get CASIO outcome data first
         → Train exclusively on CA projects
         → Skip Berkeley entirely

Option C: Model handles all regions but specializes in CA
         → Use Berkeley + CASIO as described
         → Learn national patterns + CA nuances
```

### Question 3: Are 55 CA projects enough to validate on?
```
Yes if: Model doesn't need perfect CA accuracy (directional guidance)
No if: CA accuracy must be >90% (need 200+ projects)

Calculation:
- Berkeley has ~8,000 CAISO projects
- CASIO has ~55 CA projects
- Ratio: ~145:1 (Berkeley has 145x more CA-region data)
- Prediction: Model learns CA patterns well from Berkeley
- Validation: 55 projects can test if patterns transfer
```

### Question 4: Can you obtain ground truth for CASIO Cluster 15?
```
Current Problem: Cluster 15 projects don't have outcomes (still pending)
Possible Solutions:
- Get other clusters with historical outcomes
- Track Cluster 15 projects over time (future)
- Use Berkeley to predict Cluster 15 outcomes (circular validation)
```

---

## Part 6: Final Recommendation Matrix

| Your Priority | Best Dataset | Why | Caveats |
|---|---|---|---|
| **CA Accuracy** | Combined CASIO + Berkeley | CA outcomes + national patterns | Need other CASIO clusters with outcomes |
| **Simplicity** | Berkeley only | Largest, most complete, well-documented | Less CA-specific |
| **Pure CA Focus** | CASIO only | 100% CA data, interpretable | Must get outcomes from other clusters FIRST |
| **National + CA** | Berkeley (train) + CASIO (test) | Transfer learning is valid | Can't measure CA accuracy without outcomes |
| **Hybrid (Best)** | Berkeley pre-train + CASIO fine-tune | Robust + CA-specific | Implementation complexity |

---

## Part 7: Technical Feasibility Assessment

### Can You Train on Berkeley + Test on CASIO?

✅ **YES - Technical Implementation**

```python
# 1. LOAD & PREPARE
berkeley = pd.read_excel('Berkeley_LabQueue_Data_2025.xlsx')
casio = pd.read_excel('CASIO_Cluster_15.xlsx')

# 2. HARMONIZE FEATURES
# Map Berkeley features to CASIO features
berkeley_features = ['total_mw', 'voltage', 'type_1', 'region', 'q_year']
casio_features = ['NET MW POI', 'Voltage kV', 'Generation/Fuel 1', 'PTO', 'Queue_Year']

# 3. TRAIN on Berkeley (has outcomes)
from sklearn.ensemble import XGBClassifier
model = XGBClassifier()
model.fit(berkeley[berkeley_features], berkeley['q_status'])

# 4. INFERENCE on CASIO (no outcomes needed for prediction)
predictions = model.predict(casio[casio_features])
confidence = model.predict_proba(casio[casio_features])

# 5. ANALYZE (without ground truth)
# - Feature distribution alignment
# - Confidence level on CA projects
# - Region-specific prediction patterns
# - Uncertainty quantification
```

✅ **Implementable** - Code is straightforward

⚠️ **Validation Challenge** - Can't measure accuracy metrics

### Can You Validate Without CASIO Ground Truth?

✅ **Partially - Use Alternative Validation**

| Validation Method | Feasibility | Purpose |
|---|---|---|
| Confusion matrix (require outcomes) | ❌ No | Skip |
| Feature distribution comparison | ✅ Yes | Check similarity |
| Confidence calibration | ✅ Yes | Is model uncertain on CA data? |
| Prediction range analysis | ✅ Yes | Do CA projects cluster differently? |
| Domain expert review | ✅ Yes | Do predictions "make sense"? |
| Prediction stability | ✅ Yes | Do small input changes cause large output changes? |
| Temporal consistency | ✅ Yes | Do patterns align over time? |

---

## Conclusion & Recommendation

### Best Path Forward:

**Option A (Recommended): Investigate Other CASIO Clusters First**
```
1. Check Cluster 14, 13, ... for historical outcomes
2. If found: Combine CASIO history + Berkeley for training
3. Result: CA-focused model with national robustness
```

**Option B (If CASIO History Unavailable): Berkeley + CASIO Transfer Learning**
```
1. Train on Berkeley (38,202 national projects)
2. Validate patterns on CASIO Cluster 15 (without accuracy metrics)
3. Deploy nationally with CA specialization
4. Plan: Track Cluster 15 projects for future outcome data
```

**Option C (If Simplicity Paramount): Berkeley Only**
```
1. Use Berkeley's 8,000+ CAISO projects
2. Train national model
3. Deploy with CA specialization parameters
4. Skip CASIO integration (simpler implementation)
```

---

### Key Takeaway:

**Does Berkeley + CASIO make logical sense?** ✅ **Yes** - Transfer learning is valid.

**Does it make implementation sense?** ⚠️ **Partially** - Implementation is feasible, but validation is limited without CASIO outcomes.

**Should you do it?** ✅ **Yes, if:**
- CASIO outcomes unavailable in current cluster
- You understand validation limitations
- You have plan to obtain outcomes eventually
- Your CA-focus tolerance is directional (not precision-critical)

**Better alternative:** Get CASIO historical cluster outcomes → combine with Berkeley → train on pooled dataset → validate on Cluster 15

---

**Document Date:** September 25, 2026  
**Analysis Complete:** YES  
**Recommendation Status:** Ready to implement  
**Next Step:** Check for CASIO Cluster 14+ outcome data
