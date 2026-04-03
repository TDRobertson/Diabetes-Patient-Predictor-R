# Diabetes Patient Prediction Model

A comparative machine learning pipeline for predicting diabetes diagnosis in female patients of Pima Indian heritage — built in R, covering two imputation strategies, four classification models, and rigorous evaluation across sensitivity, specificity, and balanced accuracy.

---

## About The Project

The Diabetes Patient Prediction Model is a supervised machine learning pipeline developed for CSC-3220 at Tennessee Tech University. The project addresses a real clinical problem: given a set of health measurements from female patients of Pima Indian heritage, can we reliably predict whether a patient has diabetes? Early detection matters — catching diabetes before complications arise shifts treatment from reactive to preventive, improving patient outcomes and reducing long-term healthcare costs.

The dataset originates from the National Institute of Diabetes and Digestive and Kidney Diseases (NIDDK) and is available via Kaggle. It contains 768 patient records, each with eight clinical predictor variables and one binary outcome. The dataset presented two significant challenges: missing values were encoded as biologically impossible zeros rather than `NA`, and the classes are imbalanced — only 35% of patients (268/768) are diabetic.

The pipeline compares two imputation strategies (MICE and cluster-based insulin imputation), evaluates four classification algorithms (logistic regression, LASSO + SMOTE, SVM, and decision tree), and uses McNemar's test alongside ROC-AUC and balanced accuracy to surface the true performance picture beyond raw accuracy alone.

### Repository

- **GitHub**: [CSC-3220-Group-1-Diabetes-Patients](https://github.com/TDRobertson/CSC-3220-Group-1-Diabetes-Patients)

---

## Dataset

**Source**: [Kaggle — Diabetes Dataset (NIDDK)](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)

### Features

| # | Variable | Description | Units |
|---|---|---|---|
| 1 | `Pregnancies` | Number of times pregnant | Count |
| 2 | `Glucose` | Plasma glucose concentration (2-hr oral glucose tolerance test) | mg/dL |
| 3 | `BloodPressure` | Diastolic blood pressure | mm Hg |
| 4 | `SkinThickness` | Triceps skin fold thickness | mm |
| 5 | `Insulin` | 2-Hour serum insulin | mu U/ml |
| 6 | `BMI` | Body mass index (weight in kg / height in m²) | kg/m² |
| 7 | `DiabetesPedigreeFunction` | Genetic predisposition score based on family history | Score |
| 8 | `Age` | Age | Years |
| 9 | `Outcome` | **Target variable** — 1 = diabetic, 0 = non-diabetic | Binary |

### Dataset Characteristics

- **Total records**: 768 patients
- **Class distribution**: 268 diabetic (34.9%) / 500 non-diabetic (65.1%)
- **Missing value encoding**: Zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` represent missing data — a value of zero is biologically implausible for these measurements in a living patient
- **Demographic scope**: Female patients of Pima Indian heritage, aged 21 and older
- **Outliers identified (post-zero-removal)**: `Insulin` — 24, `BloodPressure` — 14, `BMI` — 8, `SkinThickness` — 3

---

## Core Methodology

### Data Preprocessing

**Step 1 — Zero-to-NA Conversion**

Before any imputation, all zero values in the five affected columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) are converted to `NA`. The `Pregnancies` and `Outcome` columns retain their zero values, which are valid.

**Step 2 — Outlier Detection**

Outliers are identified using the IQR method (values beyond 1.5 × IQR from the quartiles). Depending on the script:
- Non-insulin outliers are capped at the IQR boundary
- Insulin outliers may be removed entirely (rows dropped) due to the high number of extreme values (24 outliers)

### Imputation Strategies

Two imputation strategies are implemented and compared across downstream models:

#### MICE — Multivariate Imputation by Chained Equations

MICE imputes missing values by modeling each incomplete variable as a function of all other variables in the dataset. This preserves the correlation structure between features rather than replacing missing values with a fixed statistic.

**Implementation:**
- Method: `pmm` (Predictive Mean Matching) — draws actual observed values as replacements, preserving the variable's distributional shape
- Imputations: `m = 5` (five complete datasets generated; first is used)
- Iterations: `maxit = 50`
- Seed: `123` (reproducibility)
- Package: `mice`

**Why MICE:** The biological variables in this dataset are interdependent. Insulin levels correlate with glucose; BMI correlates with skin thickness. Mean imputation destroys these relationships. MICE models them explicitly, producing more clinically plausible fill values.

#### Cluster-Based Insulin Imputation

For datasets where the simpler imputation approach is used, insulin is imputed via a glucose-tier clustering strategy rather than global mean replacement.

**Implementation:**
- Patients are grouped into four glucose categories: Low (< 100), Moderate (100–125), High (125–150), Very High (≥ 150)
- Within each category, the mean insulin value (from non-missing records) is calculated
- Missing insulin values are replaced with the cluster mean for their glucose group
- All other columns use global mean imputation

**Why cluster-based for insulin:** Insulin and glucose have a well-documented biological relationship — higher glucose generally coincides with higher insulin. Imputing insulin using a single global mean ignores this, assigning identical values to low-glucose and high-glucose patients. Cluster-based imputation respects the relationship.

### Class Imbalance Handling — SMOTE

The dataset is imbalanced (35% diabetic). A model trained without correction can achieve ~65% accuracy by predicting "no diabetes" for every patient — which is clinically useless.

**SMOTE (Synthetic Minority Over-sampling Technique)** generates synthetic samples for the minority class (diabetic) to balance the training set before model fitting.

**Implementation:**
- Package: `themis` (tidymodels extension)
- Recipe step: `step_smote(Outcome, over_ratio = 1)` — creates synthetic diabetic samples until classes are balanced 1:1
- Applied only to the training set; test set remains the original distribution

### Feature Engineering

Applied in `SMOTE_regression.R` to test whether engineered features improve the LASSO model:

- **Interaction term**: `Glucose_BMI` — captures the combined effect of two of the strongest predictors
- **Polynomial feature**: `Glucose²` — tests whether the glucose-diabetes relationship has a non-linear component
- **LASSO regularization** (`cv.glmnet`, `alpha = 1`) performs automatic feature selection, shrinking unimportant coefficients to zero and identifying which features drive prediction most

---

## Models

### 1. Logistic Regression (Baseline)

The standard binary classification model. Predicts the log-odds of diabetes as a linear combination of predictor variables, outputting probabilities in the (0, 1) range. Serves as the performance baseline against which all other models are compared.

- **Script**: `MICE_Imputation_Logistic_Regression.R`
- **Framework**: `glm()` (base R)
- **Split**: 80/20 train/test
- **Imputation**: MICE
- **AUC**: 0.8406

### 2. Logistic Regression (LASSO + SMOTE)

Extends the baseline with two additions: LASSO regularization for feature selection (penalizes and removes low-signal predictors) and SMOTE for minority class balancing.

- **Script**: `SMOTE_regression.R`
- **Regularization**: `cv.glmnet()`, `alpha = 1` (L1 / LASSO), cross-validated lambda selection
- **Class balancing**: SMOTE via `themis::step_smote()`
- **Feature engineering**: `Glucose_BMI` interaction + `Glucose²` polynomial
- **AUC**: 0.8343

### 3. Support Vector Machine (Linear Kernel)

SVM identifies the optimal hyperplane separating diabetic from non-diabetic patients. The linear kernel was selected after evaluation; normalization of features is applied as a preprocessing step because SVM is distance-based and sensitive to feature scale.

- **Script**: `SVM_analysis.R`
- **Package**: `e1071`
- **Kernel**: Linear
- **Cost parameter**: 0.01
- **Split**: 70/30 train/test
- **Preprocessing**: Feature normalization via tidymodels recipe
- **Imputation**: Cluster-based (glucose-tier for insulin, mean for others)

### 4. Decision Tree (RPart)

Tree-based binary classifier trained using the RPart engine inside a tidymodels workflow. Recursively splits data at the feature value that best separates classes. Most interpretable model — the resulting tree diagram shows exactly which decision rules drive each classification.

- **Script**: `Classification Tree With ROC Curve.R`
- **Engine**: `rpart` via tidymodels
- **Split**: 80/20 train/test
- **Imputation**: MICE
- **Probability threshold**: 0.5 for class assignment
- **Visualization**: `rpart.plot` tree diagram + ROC curve

### 5. K-Means Clustering (Exploratory)

Unsupervised clustering used to analyze the structural relationship between glucose and insulin in the dataset — not for classification. Implemented from scratch in base R to demonstrate the algorithm mechanics.

- **Scripts**: `k-means.R`, `kmeans-imputed.R`
- **Clusters**: `k = 3`
- **Feature space**: Glucose vs. Insulin (2D)
- **Algorithm**: Manual Euclidean-distance assignment + centroid update, max 100 iterations
- **Finding**: Imputation-cleaned data produced tighter, more distinct clusters than raw mean-imputed data

---

## Results

### Model Performance Comparison

| Model | Accuracy | Sensitivity | Specificity | Balanced Acc | AUC | McNemar's p |
|---|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 77.32% | 86.78% | 58.87% | 72.82% | 0.8406 | 0.0041 |
| Logistic Regression (LASSO + SMOTE) | 76.47% | 80.20% | 69.23% | **74.71%** | 0.8343 | 0.6171 |
| SVM (linear kernel) | 75.16% | **90.00%** | 47.17% | 68.58% | — | 0.0058 |
| Decision Tree | 74.68% | 80.41% | 64.91% | 72.66% | — | — |

### Metric Definitions

- **Sensitivity (Recall)**: % of actual diabetic patients correctly identified — high sensitivity reduces missed diagnoses (false negatives)
- **Specificity**: % of actual non-diabetic patients correctly identified — high specificity reduces false alarms (false positives)
- **Balanced Accuracy**: Average of sensitivity and specificity; accounts for class imbalance
- **McNemar's p**: Tests whether false positive and false negative rates are balanced; p > 0.05 indicates balanced errors

### Recommendations

| Use Case | Recommended Model | Reason |
|---|---|---|
| General clinical screening | **LASSO + SMOTE** | Highest balanced accuracy (74.71%), best error balance (p=0.617) |
| Minimize missed diagnoses | **SVM** | Highest sensitivity (90.00%) — catches the most diabetic patients |
| Interpretable clinical decision support | **Decision Tree** | Visual rule-based output, clinician-readable |

---

## Project Structure

```
Diabetes-Patient-Predictor-R/
├── diabetes.csv                               # Original NIDDK dataset (768 rows × 9 columns)
├── cleaned_data                               # Preprocessed dataset output
├── CSC-3220-Team-1-Diabetes-Final-Report.pdf  # Full project report with methodology and results
├── README.md                                  # This file
└── Analytics_scripts/
    ├── MICE_Cleaner.R                         # MICE imputation pipeline (standalone cleaner)
    ├── MICE_Imputation_Logistic_Regression.R  # MICE → logistic regression → ROC/AUC
    ├── SMOTE_regression.R                     # MICE → SMOTE → LASSO logistic regression
    ├── ClusterImputation_LogisticRegression.R # Cluster imputation → logistic regression
    ├── SVM_analysis.R                         # SVM classifier (linear kernel)
    ├── Classification Tree With ROC Curve.R   # Decision tree with ROC analysis
    ├── clusterCleaner.R                       # Cluster-based imputation cleaner
    ├── k-means.R                              # From-scratch k-means (raw data)
    ├── kmeans-imputed.R                       # From-scratch k-means (MICE-imputed data)
    ├── SVM_Code_Documentation.md              # SVM implementation notes and lessons learned
    └── kmeans_summary.md                      # K-means methodology documentation
```

---

## Script Reference

### `MICE_Cleaner.R`
Standalone MICE imputation preprocessor. Converts biologically invalid zeros to `NA`, runs MICE with PMM (5 imputations, 50 iterations, seed 123), and extracts the first complete imputed dataset. Use this to generate a clean dataset before passing to any downstream model.

**Libraries:** `mice`

---

### `MICE_Imputation_Logistic_Regression.R`
Full pipeline: MICE imputation → distribution visualization → correlation matrix → 80/20 train/test split → logistic regression → ROC curve + AUC. Produces the baseline logistic regression model (AUC 0.8406).

**Libraries:** `tidyverse`, `readxl`, `caret`, `pROC`, `mice`, `corrplot`, `ggplot2`

**Output:** ROC curve, correlation heatmap, confusion matrix, AUC score

---

### `SMOTE_regression.R`
Advanced pipeline: MICE imputation → IQR outlier visualization → SMOTE class balancing → feature engineering (Glucose_BMI interaction, Glucose² polynomial) → LASSO regularization via cross-validated `cv.glmnet` → ROC curve. Produces the highest balanced-accuracy model (74.71%).

**Libraries:** `tidyverse`, `mice`, `caret`, `corrplot`, `ggplot2`, `glmnet`, `themis`, `pROC`

**Output:** Histogram grid, ROC curve, LASSO coefficient plot, confusion matrix, AUC score

---

### `ClusterImputation_LogisticRegression.R`
Cluster-based imputation pipeline → logistic regression. Uses glucose-tier mean insulin imputation (four glucose categories) instead of MICE, then trains a standard logistic regression model for comparison against the MICE-based baseline.

**Libraries:** `tidyverse`, `caret`, `pROC`

---

### `SVM_analysis.R`
Support Vector Machine classifier with tidymodels workflow. Applies cluster-based imputation and feature normalization, then trains a linear-kernel SVM (`e1071`) with cost = 0.01. Uses a 70/30 split. Produces the highest sensitivity model (90.00%).

**Libraries:** `e1071`, `caret`, `tidyverse`, `tidymodels`

**Output:** Confusion matrix, accuracy metrics

---

### `Classification Tree With ROC Curve.R`
Decision tree classifier using tidymodels with the RPart engine. MICE-imputed inputs, 80/20 split, probability-based classification at 0.5 threshold. Outputs a visual tree diagram and ROC curve. Most interpretable model in the pipeline.

**Libraries:** `rpart.plot`, `tidyverse`, `tidymodels`, `mice`, `caret`, `corrplot`, `ggplot2`

**Output:** Visual classification tree diagram, ROC curve, confusion matrix

---

### `k-means.R`
From-scratch k-means implementation in base R, applied to cluster-cleaned data. Groups patients into 3 clusters based on Glucose and Insulin values using manual Euclidean distance assignment and centroid updates over max 100 iterations. Visualizes cluster assignments and reports convergence iteration.

**Libraries:** None (base R only)

---

### `kmeans-imputed.R`
Same from-scratch k-means algorithm as `k-means.R`, but applied to MICE-imputed data. Produces tighter, more distinct clusters — demonstrating that imputation quality directly affects downstream unsupervised analysis.

**Libraries:** `mice`, `caret`, `corrplot`, `ggplot2`

---

### `clusterCleaner.R`
Data cleaning utility implementing cluster-based imputation. Performs mean imputation on non-insulin columns, then applies glucose-tier cluster mean imputation specifically for insulin. Includes IQR-based outlier detection and capping. Used as the preprocessing step for `SVM_analysis.R` and `k-means.R`.

**Libraries:** None (base R only)

---

## R Package Requirements

Install all required packages before running any script:

```r
install.packages(c(
  "tidyverse",   # Data manipulation and visualization
  "mice",        # Multivariate Imputation by Chained Equations
  "caret",       # Model training framework and confusion matrices
  "pROC",        # ROC curve generation and AUC calculation
  "glmnet",      # LASSO / Ridge / Elastic Net regularization
  "e1071",       # Support Vector Machines
  "themis",      # SMOTE and class imbalance handling (tidymodels)
  "tidymodels",  # Unified model workflow framework
  "corrplot",    # Correlation matrix visualization
  "ggplot2",     # Data visualization
  "rpart.plot",  # Decision tree visualization
  "readxl"       # Excel/CSV data import
))
```

**R version requirement:** R 4.0 or higher

---

## Usage

### Recommended Execution Order

Run scripts in this order for a complete end-to-end analysis:

**1. Generate clean dataset (MICE)**

```r
source("Analytics_scripts/MICE_Cleaner.R")
```

**2. Baseline logistic regression**

```r
source("Analytics_scripts/MICE_Imputation_Logistic_Regression.R")
```

**3. LASSO + SMOTE logistic regression**

```r
source("Analytics_scripts/SMOTE_regression.R")
```

**4. Cluster imputation baseline (for comparison)**

```r
source("Analytics_scripts/clusterCleaner.R")
source("Analytics_scripts/ClusterImputation_LogisticRegression.R")
```

**5. SVM classifier**

```r
source("Analytics_scripts/SVM_analysis.R")
```

**6. Decision tree**

```r
source("Analytics_scripts/Classification Tree With ROC Curve.R")
```

**7. K-means clustering analysis**

```r
# On cluster-cleaned data
source("Analytics_scripts/k-means.R")

# On MICE-imputed data (for comparison)
source("Analytics_scripts/kmeans-imputed.R")
```

> **Note:** Each script is self-contained — it loads and processes the dataset internally. Running `MICE_Cleaner.R` first is only necessary if you want to inspect the cleaned dataset as a standalone object.

---

## Key Findings

### Strongest Predictors

Based on correlation analysis and LASSO feature selection, the two most predictive features are:

- **Glucose** — strongest correlation with diabetes outcome (r ≈ 0.50); directly tied to the biological mechanism of the disease
- **BMI** — second strongest predictor; correlates with insulin resistance

### Notable Correlations in the Dataset

- `Glucose` ↔ `Outcome`: r ≈ 0.50 (moderately strong — expected, given glucose's direct role in diabetes diagnosis)
- `BMI` ↔ `SkinThickness`: r ≈ 0.65 (strong — skin fold thickness is a proxy for body fat, which links to BMI)
- `Insulin` ↔ `Glucose`: r ≈ 0.60 (strong — high glucose typically accompanies elevated insulin)
- `Age` ↔ `Pregnancies`: r ≈ 0.54 (moderate — older patients are more likely to have had more pregnancies)

### Imputation Strategy Comparison

MICE-imputed models outperformed cluster-imputed models on most evaluation metrics. The cluster-based approach produced slightly better SVM results in some configurations, but MICE provided a more consistent baseline overall. The glucose distribution remained stable before and after MICE imputation, confirming the PMM method preserved the original distributional shape.

### Clinical Interpretation

- A model optimized purely for accuracy will systematically underperform on the minority class (diabetic patients) — the population where missed predictions matter most
- McNemar's test confirmed the LASSO + SMOTE model had the most balanced error distribution (p = 0.617), making it the most reliable for general screening
- The SVM's 90% sensitivity makes it the best choice when the cost of a false negative (missed diabetes) exceeds the cost of a false positive (unnecessary follow-up)

---

## Authors

**Thomas D. Robertson** — Project Lead, Lead Data Scientist
- Led the design of all models, imputation strategies, and optimization methods
- Authored: `MICE_Cleaner.R`, `MICE_Imputation_Logistic_Regression.R`, `SMOTE_regression.R`, `kmeans-imputed.R` (cleaner component)
- Directly guided all team members through their implementations
- Led primary analysis and assisted with report writing

**Tania Perdomo Flores** — Model Validation, Report Coordinator
- `ClusterImputation_LogisticRegression.R`, model validation, report writing
- Primary responsibility was report coordination and ensuring the accuracy and clarity of all documented findings.

**Logan Bolton** — Data Preprocessing, K-Means
- `clusterCleaner.R`, `k-means.R`, `kmeans-imputed.R` (model component)

**Fengjun Han** — SVM Implementation
- `SVM_analysis.R`, `SVM_Code_Documentation.md`

**Kristian Obrusanszki** — Decision Tree
- `Classification Tree With ROC Curve.R`

*CSC-3220 — Machine Learning, Tennessee Tech University — Fall 2024*
*Course Professor: Dr. William Eberle*

---

*Full methodology, detailed statistical analysis, and comprehensive results are documented in `CSC-3220-Team-1-Diabetes-Final-Report.pdf`.*
