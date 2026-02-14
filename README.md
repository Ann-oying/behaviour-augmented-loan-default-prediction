# Behaviour-augmented-loan-default-prediction

### Momentum ’26 Datathon Project

## Overview

Financial institutions traditionally rely on **static borrower attributes** such as income, employment history, and credit records to predict loan default.
However, these variables alone do not fully capture **real-world financial behaviour**.

This project integrates:

* **Structural credit data** (income, employment, loan details, credit history)
* **Transactional behavioural data** (spending patterns, volatility, timing, and frequency)

to build a **behaviour-augmented loan default prediction framework** and evaluate whether behavioural signals improve predictive performance beyond traditional credit scoring.

---

## Problem Statement

**Goal:**
Improve the accuracy and interpretability of **loan default risk prediction**.

**Core Hypothesis:**
Borrower **transaction behaviour reveals hidden financial stress** that is not captured by static demographic or credit variables.


---

## Dataset

The dataset used in this project was provided during the **Momentum ’26 Datathon**.


The data consists of:

1. **Loan Dataset (Structural Credit Profile)**

   * Income
   * Employment length
   * Loan amount & interest rate
   * Credit history length
   * Default status

2. **Transaction Dataset (Behavioural Activity)**

   * Transaction amount
   * Frequency & timing
   * Sequential spending behaviour
   * Spending volatility

Both datasets are **merged at the user level** to form a **multimodal borrower representation**.

---

## End-to-End ML Pipeline

### 1. Data Cleaning & Preprocessing

* Aggregated transaction data → **total spend, average spend, transaction count**
* Merged transactional and loan datasets on **user_id**
* Handled missing values and ensured type consistency
* Checked **class imbalance (~14% default rate)**
* Applied **feature scaling** for linear models

Produced a **model-ready feature matrix**.

---

### 2. Exploratory Data Analysis (EDA)

Key structural and behavioural insights:

* **Income strongly stratifies default risk**

  * Low income ≈ 24% default
  * High income ≈ 6% default

* **Behaviour varies across borrowers** and evolves over time.

* Confirms **predictive signal exists in both structure and behaviour**.

---

### 3. Feature Engineering & Selection

We transformed raw data into **interpretable behavioural risk signals**:

#### Financial Stress Indicators

* High loan-to-income ratio
* High average spending
* **Combined high-risk flag**

#### Temporal Behaviour Features

* 24-hour transaction bursts
* Early vs late spending change
* First transaction timing

#### Volatility Metrics

* **Spending Volatility Index (SVI)**
* Volatility-based borrower clustering

Converts **raw behavioural activity → predictive financial signals**.

---

### 4. Model Building

We evaluated both **interpretable** and **non-linear** learners:

* **Logistic Regression** → transparent, explainable baseline
* **Random Forest** → captures potential non-linear behavioural patterns

---

### 5. Model Evaluation

Performance metrics:

* **ROC-AUC ≈ 0.87** → strong discrimination
* **Cross-validation mean ≈ 0.867** with very low variance
* **Repeated train-test splits → stable performance**
* **Ablation testing → temporal features add limited AUC gain**

**Key Scientific Insight:**
Structural **income dominates prediction**, while behavioural features mainly improve **interpretability and monitoring**, not raw accuracy.

---

### 6. Behavioural Risk Analysis

#### Spending Volatility Findings

Default rates by SVI group:

* Stable ≈ **15.6%**
* Moderate ≈ **12.1% (lowest risk)**
* Highly volatile ≈ **15.0%**

Statistical validation:

* **Chi-square significant (p < 0.05)**
* **Cramér’s V small but real effect**

Behaviour **does influence default risk**,
but **less strongly than structural income**.

---

### 7. Stability & Robustness Checks

* Consistent **ROC-AUC across CV folds and random seeds**
* **ANOVA confirms income-risk differences are statistically significant**
* Feature ablation verifies **limited marginal gain from temporal features**

Confirms **model reliability and scientific validity**.

---

## Key Insights

* **Income segmentation is the strongest predictor** of loan default.
* **Behavioural transaction signals provide meaningful but smaller effects.**
* Combining **structure + behaviour** improves:

  * interpretability
  * monitoring capability
  * early-risk detection

---

## Business Impact

This framework enables:

* **Risk-based borrower segmentation**
* **Behavioural early warning systems**
* **Automated credit approval & pricing strategies**


---

## Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **Matplotlib & statistical testing**
* **Jupyter Notebook**

---

## Outcome
This project demonstrates the impact of transactional intelligence and feature engineering in credit risk modeling. It highlights how combining static borrower attributes with behavioral signals leads to more reliable and interpretable predictive systems.

This repository showcases practical machine learning workflow, financial analytics, and feature engineering techniques applied to real-world structured datasets.
