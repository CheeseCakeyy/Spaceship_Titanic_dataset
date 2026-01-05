# 🚀 Spaceship Titanic – Iterative Machine Learning Approach

## 📌 Project Overview
This project tackles the **Kaggle Spaceship Titanic classification problem** using a structured, iterative machine learning workflow.  
Each iteration builds on insights from the previous one, focusing on **better preprocessing, feature engineering, model selection, and validation strategy**.

The goal was not just leaderboard performance, but **correct ML methodology and learning**.

---

## 🧪 Iteration 1: Baseline Models & Classical ML

### 🔧 Approach
- Basic preprocessing (imputation, encoding)
- Train–validation split (no cross-validation)
- Classical models:
  - Naive Bayes
  - Logistic Regression (with regularization)
  - Random Forest

### 📊 Key Observations
- Logistic Regression performed better after tuning `C`
- Random Forest outperformed linear models
- Model performance was sensitive to random state
- Feature engineering not yet explored

### 🏁 Results
- **Validation Accuracy:** ~0.77–0.79  
- **Kaggle Score:** _(fill in)_  
- **Leaderboard Rank:** _(fill in)_

### ✅ Takeaway
Solid baseline established, but:
- No CV → unstable estimates
- Feature interactions not captured well

---

## 🔁 Iteration 2: Feature Engineering + Cross-Validation + Feature Selection

### 🔧 Approach
- Proper **ColumnTransformer pipelines**
- Features grouped into:
  - Numerical
  - Boolean
  - Categorical
  - Count / engineered features
- New engineered features:
  - `Total_spend`
  - `Has_spent`
  - Spend-based ratios
- Feature selection using:
  - L1 (Lasso / Logistic Regression)
- **5-Fold Cross-Validation**
- Models evaluated:
  - RF + LR-based feature selection (**RF_lr**)
  - RF + RF-based feature selection
  - LR baselines

### 📊 Key Observations
- RF with LR feature selection was the **most stable**
- CV reduced randomness issues
- Feature selection improved generalization
- Accuracy gains were incremental but consistent

### 🏁 Results
- **Best CV Accuracy (RF_lr):** ~0.800  
- **Test Accuracy:** ~0.796  
- **Kaggle Score:** slight improvement (+0.001)  
- **Leaderboard Rank:** _(fill in)_

### ✅ Takeaway
- Feature engineering does **not always guarantee gains**
- Stability + correctness > chasing accuracy
- RF_lr became the **strong classical ML baseline**

---

## ⚡ Iteration 3: Gradient Boosting with XGBoost

### 🔧 Approach
- Introduced **XGBoost (XGBClassifier)**
- Same clean preprocessing pipeline
- Cross-validation retained
- Baseline XGBoost (minimal tuning):
  - Controlled depth
  - Moderate learning rate
  - No aggressive regularization

### 📊 Key Observations
- XGBoost outperformed RF and feature-selection pipelines
- Higher mean accuracy but slightly higher variance
- Indicates lower bias but higher sensitivity
- No evidence of data leakage or overfitting

### 🏁 Results
- **CV Accuracy:** ~0.804–0.805  
- **Kaggle Score:** ~0.805  
- **Leaderboard Rank:** _(fill in)_

### ✅ Takeaway
- XGBoost extracted additional non-linear signal
- Performance close to the **practical ceiling** for this dataset
- Further gains likely require:
  - Domain-heavy feature engineering
  - Ensembling / stacking

---

## 📈 Final Comparison Summary

| Iteration | Model | CV / Val Accuracy | Kaggle Score | Rank |
|--------|------|------------------|-------------|------|
| Iter 1 | RF / LR | ~0.77–0.79 | _(fill)_ | _(fill)_ |
| Iter 2 | RF + LR FS | ~0.80 | _(fill)_ | _(fill)_ |
| Iter 3 | XGBoost | **~0.805** | _(fill)_ | _(fill)_ |

---

## 🧠 Key Learnings
- Cross-validation is **mandatory** for trustworthy evaluation
- Feature engineering helps only when it adds **new information**
- PCA is not suitable for tree-based models here
- Boosting reduces bias but must be variance-controlled
- Small leaderboard gains can still reflect **major learning progress**

---

## 📌 Conclusion
This project demonstrates a **correct and disciplined ML workflow**, moving from baseline models to advanced boosting while maintaining reproducibility and methodological rigor.

> Achieving ~80.5% accuracy with clean pipelines and CV reflects strong modeling rather than overfitting tricks.

---

## 🔮 Future Work
- Advanced domain-driven feature engineering
- Probability-based ensembling (RF + XGBoost)
- Calibration and threshold tuning
- Try LightGBM / CatBoost for categorical handling
