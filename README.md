# Usage-Based Insurance (UBI) Risk Scoring using Tweedie Ensemble

### 🚀 **Executive Summary**

This project implements a **Stacking Ensemble (XGBoost + LightGBM + CatBoost)** to predict insurance claim severity (Pure Premium) for a Usage-Based Insurance (UBI) portfolio.

By modeling the compound distribution of claim frequency and severity using the **Tweedie Distribution**, we evolved from a simple linear baseline to a state-of-the-art ensemble, improving the **Gini Coefficient from 0.150 to 0.165**. This lift directly translates to more accurate pricing and better risk segmentation.

**Key Results:**
| Model | Gini Coefficient | Lift over GLM |
| :--- | :--- | :--- |
| **Tweedie GLM (Baseline)** | 0.1500 | - |
| **XGBoost (Baseline)** | 0.1584 | +5.6% |
| **XGBoost (Optuna Tuned)** | 0.1621 | +8.0% |
| **Ensemble (XGB+LGBM+Cat)** | **0.1650** | **+10.0%** |

---

### 🧮 **The Mathematical Framework**

Standard regression models (MSE, RMSE) fail in insurance because claim data is **Zero-Inflated** (90%+ of drivers have 0 claims) and **Heavy-Tailed** (rare accidents cost millions). To solve this, we use the **Tweedie Distribution**.

#### 1. The Tweedie Distribution ($p=1.5$)

The Tweedie distribution models the **Pure Premium** ($Y$) as a Compound Poisson-Gamma process:

$$ Y = \sum\_{i=1}^{N} X_i $$

Where:

- $N \sim \text{Poisson}(\lambda)$: The number of claims (Frequency).
- $X_i \sim \text{Gamma}(\alpha, \beta)$: The cost of each claim (Severity).
- $p=1.5$: The variance power parameter optimal for insurance loss modeling.

#### 2. The Objective Function (Tweedie Deviance)

Instead of minimizing Squared Error, our gradient boosting models minimize the **Tweedie Deviance**:

$$ D(y, \hat{y}) = 2 \sum\_{i} w_i \left( y_i \frac{y_i^{1-p} - \hat{y}^{1-p}}{1-p} - \frac{y_i^{2-p} - \hat{y}^{2-p}}{2-p} \right) $$

This properly handles the zero-mass at $y=0$ while strictly penalizing large errors in the tail.

---

### 🛠 **Modeling Strategy**

1.  **Feature Engineering**:
    - **`Power_Age_Ratio`**: Interaction between vehicle power and driver age.
    - **`Young_Urban`**: Risk flag for inexperienced drivers in dense cities.
    - **`LogDensity`**: Linearization of population density.

2.  **Baselines**:
    - **GLM**: `sklearn.linear_model.TweedieRegressor` (One-Hot Encoded).
    - **XGBoost**: Standard `XGBRegressor` with `objective='reg:tweedie'`.

3.  **Optimization**:
    - Used **Optuna** to tune learning rates, tree depth, and regularization for XGBoost, LightGBM, and CatBoost independently.

4.  **Ensemble**:
    - Weighted average of the three optimized models:
      $$ \hat{y}_{final} = 0.34 \cdot \hat{y}_{XGB} + 0.33 \cdot \hat{y}_{LGBM} + 0.33 \cdot \hat{y}_{CAT} $$

---

### 💻 **Project Structure**

| Script                             | Description                                                                   |
| :--------------------------------- | :---------------------------------------------------------------------------- |
| **`data_eda.py`**                  | Data fetching from OpenML and initial actuarial EDA.                          |
| **`preprocessing_engineering.py`** | Cleaning, capping claims, and feature creation.                               |
| **`model_tests.ipynb`**            | **Experimentation Lab**: GLM vs. XGB vs. Ensemble comparison + Optuna tuning. |
| **`model_training_ensemble.py`**   | **Production Pipeline**: Trains and saves the final 3-model ensemble.         |
| **`actuarial_validation.py`**      | Generates Lift Charts, Lorenz Curves, and Loss Ratios.                        |
| **`explainability.py`**            | SHAP Beeswarm and Waterfall plots for model transparency.                     |

---

### 👨‍💻 **Author**

**Will Trevarthen**
_Data Science & Actuarial Modeling_
