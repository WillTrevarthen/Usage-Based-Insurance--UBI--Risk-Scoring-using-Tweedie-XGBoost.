# Usage-Based Insurance (UBI) Risk Scoring

### 🚀 **Executive Summary**

This project implements a **Tweedie-XGBoost** model to predict insurance claim severity (Pure Premium) for a Usage-Based Insurance (UBI) portfolio. By modeling the compound distribution of claim frequency and severity, the final model achieved a **Gini Coefficient of 0.1622** and successfully identified a high-risk segment with a loss ratio **1.8x higher** than the low-risk segment.

---

### 🧮 **The Mathematical Framework**

Standard regression models (MSE, RMSE) fail in insurance because claim data is **Zero-Inflated** (most people have 0 claims) and **Heavy-Tailed** (rare accidents cost millions). To solve this, we use the **Tweedie Distribution**.

#### 1. The Tweedie Distribution ($p=1.5$)

The Tweedie distribution models the **Pure Premium** ($Y$) as a Compound Poisson-Gamma process:

$$Y = \sum_{i=1}^{N} X_i$$

Where:

- $N \sim \text{Poisson}(\lambda)$: The number of claims (Frequency).
- $X_i \sim \text{Gamma}(\alpha, \beta)$: The cost of each claim (Severity).
- $1 < p < 2$: The power parameter. We selected $p=1.5$ to balance frequency and severity.

#### 2. The Objective Function (Tweedie Deviance)

Instead of minimizing Squared Error, XGBoost minimizes the **Tweedie Deviance**:

$$D(y, \hat{y}) = 2 \sum_{i} w_i \left( y_i \frac{y_i^{1-p} - \hat{y}^{1-p}}{1-p} - \frac{y_i^{2-p} - \hat{y}^{2-p}}{2-p} \right)$$

This loss function penalizes errors correctly even when the actual value $y_i$ is exactly zero.

#### 3. Why Gini instead of R-Squared?

The **Gini Coefficient** measures the area between the **Lorenz Curve** (cumulative actual loss) and the line of equality.

$$Gini = 1 - 2 \int_{0}^{1} L(p) dp$$

- **0.00:** Random sorting (useless model).
- **Our Score (0.162):** Indicates robust segmentation capability without overfitting.

---

### 🛠 **Feature Engineering Strategy**

We moved beyond raw inputs by creating interaction features that capture non-linear risk signals:

- **`Power_Age_Ratio`**: $\frac{\text{Vehicle Power}}{\text{Driver Age} + 1}$
- **`Young_Urban`**: Interaction of inexperienced drivers in high-density traffic zones.
- **`LogDensity`**: $\log(\text{Density})$ to linearize population effects.

---

### 💻 **Project Structure**

| Script                     | Description                                          |
| :------------------------- | :--------------------------------------------------- |
| **`01_data_loading.py`**   | Data fetching and 99th percentile capping.           |
| **`02_preprocessing.py`**  | Feature engineering and ColumnTransformer setup.     |
| **`03_model_training.py`** | XGBoost Tweedie training (Optimized at Gini 0.1622). |
| **`04_validation.py`**     | Gini calculation and Lorenz/Lift charts.             |
| **`05_explainability.py`** | SHAP Interpretability and Waterfall plots.           |

---

### 👨‍💻 **Author**

**Will Trevarthen**
_Data Science & Actuarial Modeling_
