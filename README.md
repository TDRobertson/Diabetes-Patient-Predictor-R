# Diabetes Patient Prediction Model

## Project Overview

This project implements a comprehensive machine learning pipeline to predict diabetes diagnosis in patients using various clinical and demographic features. The analysis explores multiple imputation strategies, feature engineering techniques, and classification algorithms to create an accurate and interpretable prediction model.

## Problem Statement

Diabetes affects millions of people worldwide, and early detection is crucial for effective treatment. This project aims to:

- **Predict diabetes diagnosis** based on clinical measurements and patient demographics
- **Compare multiple imputation strategies** for handling missing data
- **Evaluate various classification models** to find the optimal prediction approach
- **Provide interpretable results** for medical decision-making

## Dataset

**Source**: [Kaggle Diabetes Dataset](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)

**Features**:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration (mg/dL)
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)²)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age in years
- **Outcome**: Target variable (1 = diabetes, 0 = no diabetes)

**Dataset Characteristics**:

- 768 patients
- 8 predictor variables
- Binary classification problem
- Contains missing values requiring imputation

## Methodology

### Data Preprocessing

1. **Missing Value Analysis**

   - Identified missing values in multiple features
   - Implemented various imputation strategies

2. **Imputation Strategies Tested**
   - **Mean/Median Imputation**: Simple statistical replacement
   - **K-Means Clustering Imputation**: Group-based replacement using cluster centroids
   - **MICE (Multiple Imputation by Chained Equations)**: Advanced iterative imputation
   - **SMOTE**: Synthetic Minority Over-sampling Technique for class balancing

### Feature Engineering

- **Statistical Features**: Mean, median, standard deviation of clinical measurements
- **Interaction Terms**: Cross-feature relationships
- **Normalization**: Standard scaling for numerical features
- **Feature Selection**: Correlation analysis and importance ranking

### Model Development

#### Classification Algorithms

1. **Logistic Regression**

   - Baseline linear model
   - LASSO regularization for feature selection
   - SMOTE optimization for class imbalance

2. **Support Vector Machine (SVM)**

   - Kernel-based classification
   - Hyperparameter tuning for optimal performance
   - RBF and linear kernel comparison

3. **Decision Trees**

   - Interpretable tree-based classification
   - ROC curve analysis
   - Pruning for optimal tree depth

4. **K-Means Clustering**
   - Unsupervised patient segmentation
   - Cluster-based feature engineering
   - Validation of cluster assignments

### Evaluation Metrics

- **Accuracy**: Overall prediction correctness
- **Sensitivity (Recall)**: True positive rate for diabetes detection
- **Specificity**: True negative rate for non-diabetes cases
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve
- **Precision**: Positive predictive value
- **F1-Score**: Harmonic mean of precision and recall

## Project Structure

```
diabetes-patient-prediction/
├── diabetes.csv                           # Original dataset
├── cleaned_data                           # Preprocessed dataset
├── CSC-3220-Team-1-Diabetes-Final-Report.pdf  # Comprehensive project report
├── README.md                              # This file
└── Analytics_scripts/                     # R analysis scripts
    ├── ClusterImputation_LogisticRegression.R
    ├── MICE_Imputation_Logistic_Regression.R
    ├── SVM_analysis.R
    ├── Classification Tree With ROC Curve.R
    ├── k-means.R
    ├── kmeans-imputed.R
    ├── clusterCleaner.R
    ├── MICE_Cleaner.R
    ├── SMOTE_regression.R
    ├── SVM_Code_Documentation.md
    └── kmeans_summary.md
```

## Key Findings

### Model Performance Comparison

| Model                         | Accuracy | Sensitivity | Specificity | ROC-AUC |
| ----------------------------- | -------- | ----------- | ----------- | ------- |
| Logistic Regression (MICE)    | 78.5%    | 75.2%       | 81.8%       | 0.82    |
| SVM (RBF Kernel)              | 76.8%    | 73.1%       | 80.5%       | 0.79    |
| Decision Tree                 | 74.2%    | 71.8%       | 76.6%       | 0.77    |
| Logistic Regression (K-Means) | 77.1%    | 74.5%       | 79.7%       | 0.80    |

### Critical Insights

1. **MICE Imputation Superior**: Multiple imputation by chained equations provided the best results for handling missing data
2. **Feature Importance**: Glucose levels and BMI were the most predictive features
3. **Class Imbalance**: SMOTE significantly improved model performance on minority class
4. **Model Interpretability**: Logistic regression provided the best balance of performance and interpretability

### Clinical Relevance

- **Early Detection**: Model can identify high-risk patients for preventive measures
- **Resource Allocation**: Helps prioritize screening efforts
- **Patient Education**: Identifies key risk factors for patient counseling

## Technical Implementation

### R Scripts Overview

1. **ClusterImputation_LogisticRegression.R**

   - K-means clustering for missing value imputation
   - Logistic regression with cluster-based features

2. **MICE_Imputation_Logistic_Regression.R**

   - Multiple imputation by chained equations
   - LASSO regularization and SMOTE optimization

3. **SVM_analysis.R**

   - Support Vector Machine implementation
   - Kernel selection and hyperparameter tuning

4. **Classification Tree With ROC Curve.R**

   - Decision tree classification
   - ROC curve analysis and visualization

5. **k-means.R & kmeans-imputed.R**
   - Patient segmentation analysis
   - Cluster-based feature engineering

### Statistical Methods

- **Cross-validation**: 5-fold cross-validation for robust model evaluation
- **Bootstrap sampling**: Confidence interval estimation
- **Statistical testing**: Significance testing for model comparisons
- **Resampling**: SMOTE for handling class imbalance

## Usage Instructions

### Prerequisites

- R 4.0+
- Required R packages: `caret`, `mice`, `DMwR`, `e1071`, `rpart`, `ROCR`

### Running the Analysis

1. **Data Preparation**:

   ```r
   # Load and preprocess data
   source("Analytics_scripts/MICE_Cleaner.R")
   ```

2. **Model Training**:

   ```r
   # Run MICE imputation with logistic regression
   source("Analytics_scripts/MICE_Imputation_Logistic_Regression.R")

   # Run SVM analysis
   source("Analytics_scripts/SVM_analysis.R")
   ```

3. **Results Analysis**:
   ```r
   # Generate ROC curves and performance metrics
   source("Analytics_scripts/Classification Tree With ROC Curve.R")
   ```

## Future Work

- **Deep Learning Models**: Neural network implementations for improved performance
- **Feature Engineering**: Advanced feature selection and engineering techniques
- **Real-time Deployment**: Web application for clinical use
- **Multi-modal Data**: Integration of additional clinical data sources
- **Interpretability**: SHAP values and model explanation tools

## Team Members

- **Thomas D. Robertson** - Project coordination and analysis
- **Tania Perdomo Flores** - Model validation and report coordinator
- **Logan Bolton** - Data preprocessing and feature engineering
- **Fengjun Han** - Model development and optimization
- **Kristian Obrusanszki** - Auxillary Support

## Results Documentation

Comprehensive project results, methodology details, and statistical analysis are available in:

- **CSC-3220-Team-1-Diabetes-Final-Report.pdf**: Complete project report with detailed findings
- **Analytics_scripts/**: Individual R scripts with detailed comments

---

_This project demonstrates comprehensive machine learning skills including data preprocessing, multiple modeling approaches, statistical validation, and practical application to healthcare analytics._
