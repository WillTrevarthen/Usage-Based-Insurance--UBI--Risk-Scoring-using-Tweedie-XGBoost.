# Usage-Based Insurance Risk Scoring with Tweedie Boosting

This project builds an actuarial risk scoring workflow for motor insurance using the French Motor Third-Party Liability dataset from OpenML. It combines policy exposure, driver and vehicle attributes, claim counts, and claim severity into an end-to-end pure premium modeling case study.

The current modeling track uses Tweedie objectives to handle the insurance loss-cost shape: many zero-claim policies, a small number of positive claims, and a heavy right tail. The strongest model is a weighted blend of XGBoost, LightGBM, and CatBoost Tweedie regressors.

## Results

| Model | Gini Coefficient | Lift over GLM |
| :--- | ---: | ---: |
| Tweedie GLM baseline | 0.1500 | - |
| XGBoost baseline | 0.1584 | +5.6% |
| XGBoost with Optuna tuning | 0.1621 | +8.0% |
| XGBoost + LightGBM + CatBoost blend | **0.1650** | **+10.0%** |

These results should be read as project benchmark results rather than a fully locked production estimate. The project still needs a separate validation split or final holdout protocol before the headline Gini can be treated as a clean out-of-sample score.

## Why Tweedie?

Insurance claim cost is not well matched to ordinary least squares assumptions. Most policies have zero loss, while a small number have large positive losses. A Tweedie compound Poisson-Gamma model is a practical fit for this setting because it can represent:

- claim frequency through a Poisson process;
- claim severity through a Gamma distribution;
- aggregate loss cost, or pure premium, in one objective.

The current models use a Tweedie variance power of `1.5`, a common starting point for loss-cost modeling:

```text
Y = X_1 + X_2 + ... + X_N

N ~ Poisson(lambda)
X_i ~ Gamma(alpha, beta)
```

Boosting models are trained with exposure as `sample_weight`, so policies with more earned exposure have more influence on the fitted objective.

## Modeling Workflow

1. **Data ingestion**
   - Downloads the French motor frequency and severity datasets from OpenML.
   - Aggregates claim severity to policy level.
   - Merges policy characteristics, exposure, claim count, and claim amount.
   - Caps claim amount at the 99th percentile for model stability.

2. **Feature engineering**
   - `Power_Age_Ratio`: vehicle power relative to driver age.
   - `LogDensity`: transformed population density.
   - `Is_New_Car`: flag for vehicles up to one year old.
   - `Young_Urban`: flag for young drivers in high-density areas.

3. **Modeling**
   - Tweedie GLM baseline.
   - XGBoost Tweedie baseline.
   - Optuna-tuned XGBoost.
   - Weighted blend of XGBoost, LightGBM, and CatBoost Tweedie regressors.

4. **Validation and interpretation**
   - Gini coefficient for ranking power.
   - Lift and Lorenz-style actuarial validation.
   - SHAP-based model explainability.
   - Streamlit app for interactive risk scoring.

## Streamlit Risk App

`src/app.py` provides a lightweight Streamlit interface for testing the trained risk engine interactively. The app loads the saved ensemble model from `pickles/tweedie_model.joblib` and the fitted preprocessing bundle from `pickles/processed_step_2.pkl`.

The sidebar lets a user enter a sample policyholder profile:

- area, region, vehicle brand, and fuel type;
- vehicle power and vehicle age;
- driver age;
- local population density.

The app applies the same feature engineering used during training, including `Power_Age_Ratio`, `LogDensity`, `Is_New_Car`, and `Young_Urban`, then returns a predicted annual pure premium and a simple low, medium, or high risk tier.

Run it after training the model:

```bash
streamlit run src/app.py
```

## Repository Structure

| Path | Purpose |
| :--- | :--- |
| `src/data_eda.py` | Fetches OpenML data, merges policy and claim tables, caps claim amounts, and performs initial EDA. |
| `src/preprocessing_engineering.py` | Builds engineered features, preprocessing transformers, train/test split, target, and exposure weights. |
| `src/model_training.py` | Trains earlier baseline models. |
| `src/model_training_ensemble.py` | Trains the current XGBoost, LightGBM, and CatBoost weighted blend. |
| `src/actuarial_validation.py` | Produces validation views such as lift charts, Lorenz curves, and loss-ratio style checks. |
| `src/explainability.py` | Builds SHAP explainability outputs. |
| `src/app.py` | Streamlit demo for scoring driver and vehicle profiles with the saved Tweedie blend. |
| `src/model_tests.ipynb` | Experiment notebook for model comparison and tuning. |
| `src/evaluation.ipynb` | Evaluation notebook for final metrics and visual diagnostics. |
| `plan.md` | Review notes and roadmap for improving actuarial rigor and reproducibility. |

## Running the Project

Create a Python environment and install the core packages used by the scripts:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna shap streamlit matplotlib seaborn joblib openml
```

Then run the pipeline from the project root:

```bash
mkdir -p pickles
python src/data_eda.py
python src/preprocessing_engineering.py
python src/model_training_ensemble.py
streamlit run src/app.py
```

Note: `src/data_eda.py` currently writes `pickles/processed_step_1_test.pkl`, while `src/preprocessing_engineering.py` expects `pickles/processed_step_1.pkl`. Rename the generated file or update the script path before running the full pipeline.

## Current Limitations

This repository is a strong modeling case study, but several items should be tightened before presenting it as production-grade actuarial work:

- The target is currently `ClaimAmount_Capped` with `Exposure` as `sample_weight`; a cleaner annualized pure premium target should be reviewed.
- The test split is used during model fitting and early stopping in parts of the workflow, so a true train/validation/holdout split is still needed.
- Claim capping is useful for stability but should be documented with the cap value, number of affected claims, and total loss removed.
- The ensemble is a weighted blend, not true stacking. True stacking would require out-of-fold base model predictions and a meta-model.
- Model classes and artifact names are duplicated across scripts and should be centralized.
- A pinned `requirements.txt` or `pyproject.toml` is still needed for reproducibility.

## Roadmap

Priority improvements:

- Add a consistent pure premium target definition and exposure treatment.
- Add train/validation/holdout splitting for clean final reporting.
- Move shared metrics and the `TweedieEnsemble` class into reusable modules.
- Standardize artifact names under `pickles/` or a dedicated `models/` directory.
- Add `requirements.txt`, saved metrics, and reproducible run commands.
- Add calibration by decile and segment-level actual-versus-expected checks.
- Compare direct Tweedie modeling with separate frequency and severity models.

## Author

Will Trevarthen  
Data Science & Actuarial Modeling
