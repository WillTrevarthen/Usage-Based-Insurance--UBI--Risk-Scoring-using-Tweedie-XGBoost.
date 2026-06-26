# Project Review and Deepening Plan

## Current Project at a Glance

This project builds a usage-based insurance risk scoring model on the French motor third-party liability dataset from OpenML. It merges policy-level frequency data with claim severity, engineers a small set of pricing features, trains Tweedie models with exposure weights, compares GLM, XGBoost, tuned XGBoost, and a three-model gradient boosting ensemble, then presents validation plots, SHAP explainability, and a Streamlit demo.

The strongest parts are:

- The modeling objective is directionally appropriate for insurance loss cost: Tweedie is a sensible compound Poisson-Gamma framework for zero-heavy pure premium data.
- The project is end-to-end rather than just a notebook: ingestion, preprocessing, modeling, validation, explainability, and app files all exist.
- The business framing is clear: Gini, lift charts, Lorenz curves, risk tiers, and pure premium are more relevant than generic RMSE.
- The project already contains a GLM benchmark, which is important in actuarial work because interpretability and governance matter.

The main opportunity is to make the work more actuarially rigorous, reproducible, and credible as a production-grade modeling case study.

## Critical Issues to Fix First

### 1. Target and exposure handling need a careful actuarial review

The code models `ClaimAmount_Capped` directly while passing `Exposure` as `sample_weight`. That may not be the intended pure premium formulation.

For annualized pure premium modeling, the target should usually be either:

- loss cost per unit exposure, `ClaimAmount / Exposure`, with `Exposure` as sample weight; or
- aggregate claim amount with exposure handled as an offset or equivalent exposure-aware objective.

At present, a policy with 0.1 exposure and EUR 100 claim amount is treated as a EUR 100 outcome, not a EUR 1,000 annualized loss cost. The validation later divides actual and predicted sums by exposure, but the fitted target itself is not clearly exposure-normalized.

Recommended fix:

- Create explicit targets:
  - `claim_count = ClaimNb`
  - `claim_amount = uncapped total claim amount`
  - `claim_amount_capped`
  - `pure_premium = claim_amount_capped / Exposure`
- Decide and document the chosen modeling target.
- Re-run GLM and boosting comparisons under a consistent exposure treatment.
- Add tests that prove predictions are interpretable as annual pure premium.

### 2. The test set is being used for model selection

The Optuna notebook tunes hyperparameters by evaluating directly on `X_test`, and early stopping also uses the test split. This turns the reported test Gini into a validation score rather than a clean out-of-sample estimate.

Recommended fix:

- Split data into train, validation, and final holdout sets.
- Use validation for Optuna and early stopping.
- Touch the holdout only once for final model reporting.
- Prefer cross-validation by policy group or region if the project wants a stronger generalization story.

### 3. Claim capping is under-justified and may distort the insurance story

The project caps claim amount at the 99th percentile. This stabilizes modeling but removes exactly the tail behavior that insurance severity models are supposed to understand.

Recommended fix:

- Report the cap value, number of affected claims, total loss removed, and impact on pure premium.
- Compare uncapped, 99.5th percentile capped, 99th percentile capped, and robust/two-part alternatives.
- If capped losses are used for modeling, add a separate tail load or large-loss adjustment discussion.

### 4. Gini calculation should be verified and standardized

The Gini function sorts predictions ascending and computes `1 - 2 * area`. This can be valid depending on convention, but it should be checked against a known normalized Gini implementation and documented. Insurance lift/Gini comparisons can be sensitive to ordering, weighting, and whether claim amount or loss ratio is used.

Recommended fix:

- Implement `weighted_lorenz`, `weighted_gini`, and `normalized_gini` in one shared metrics module.
- Add unit tests with small hand-calculated examples.
- Report both raw Gini and normalized Gini if useful.
- Include confidence intervals via bootstrap.

### 5. Production code and experiment code are inconsistent

There are several mismatches:

- `data_eda.py` saves `pickles/processed_step_1_test.pkl`, while `preprocessing_engineering.py` expects `pickles/processed_step_1.pkl`.
- `explainability.py` loads `pickles/tuned_tweedie_ensemble_model.joblib`, while `app.py` loads `pickles/tweedie_model.joblib`.
- The `TweedieEnsemble` class is duplicated in `model_training_ensemble.py`, `actuarial_validation.py`, `explainability.py`, `app.py`, and notebooks.
- The README reports an ensemble Gini of `0.1650`, while the experiment notebook includes an `Ensemble Gini: 0.16366` printout and the evaluation notebook prints `0.1650`.
- Notebooks contain absolute local paths from an older project location.

Recommended fix:

- Create a package-like structure:
  - `src/data.py`
  - `src/features.py`
  - `src/models.py`
  - `src/metrics.py`
  - `src/validation.py`
  - `src/config.py`
- Move `TweedieEnsemble` into `src/models.py`.
- Use a single artifact naming convention.
- Replace absolute paths with project-relative paths.
- Add a `Makefile` or simple CLI commands for the full pipeline.

### 6. Reproducibility is incomplete

There is no `requirements.txt`, `pyproject.toml`, environment file, data versioning note, or command sequence that recreates the reported results from a fresh clone.

Recommended fix:

- Add `pyproject.toml` or `requirements.txt`.
- Add a reproducible pipeline:
  - `python -m src.data`
  - `python -m src.features`
  - `python -m src.train`
  - `python -m src.evaluate`
- Store model metrics in `reports/metrics.json`.
- Store plots in `reports/figures/`.
- Add random seeds for Optuna samplers, XGBoost, LightGBM, CatBoost, and splitting.

## Modeling Improvements

### 1. Build a proper actuarial baseline suite

The current GLM baseline is good, but it can be deepened.

Add:

- Frequency GLM: Poisson or negative binomial on `ClaimNb` with log exposure offset.
- Severity GLM: Gamma on positive claims only.
- Pure premium model: frequency times severity.
- Tweedie GLM: direct aggregate loss cost model.
- Gradient boosted Tweedie: current approach.

This gives a strong actuarial comparison: direct Tweedie versus explicit frequency-severity decomposition.

### 2. Tune the Tweedie variance power

The project fixes `p = 1.5`. That is plausible, but not proven.

Recommended work:

- Estimate or tune `tweedie_variance_power` over a constrained range such as `1.1` to `1.9`.
- Compare deviance, Gini, calibration, and lift.
- Explain the chosen value in the README.

### 3. Use native categorical handling where appropriate

The current preprocessing one-hot encodes categorical variables for all models. That is acceptable for XGBoost and GLM, but CatBoost can handle categorical features natively, and LightGBM can too with care.

Recommended work:

- Keep one-hot for GLM.
- Compare one-hot versus native categorical handling for CatBoost and LightGBM.
- Add target encoding only if done inside cross-validation to avoid leakage.

### 4. Improve ensemble methodology

The ensemble is a simple weighted average with fixed weights. This is fine as a first pass, but the project calls it a stacking ensemble even though there is no out-of-fold meta-model.

Recommended work:

- Rename current method to weighted blending, or implement true stacking.
- Generate out-of-fold predictions for XGBoost, LightGBM, and CatBoost.
- Train a constrained meta-model, such as non-negative linear regression, on validation folds.
- Compare equal weights, optimized weights, and true stacking on final holdout.

### 5. Add calibration analysis

A pricing model must not only rank risk; it must produce sensible expected cost levels.

Add:

- Calibration by decile: actual versus predicted pure premium.
- Calibration by key segments: age band, region, vehicle power, density band, vehicle age.
- Actual-to-expected ratios.
- Bootstrap confidence bands on decile charts.
- A check that total predicted loss approximately equals total actual loss on holdout.

## UBI-Specific Deepening

The current dataset is not truly telematics-rich; it is more of a traditional motor pricing dataset with a UBI framing. To deepen the project, add synthetic or external telematics-like features carefully, or reframe the project as a motor pure premium model with a UBI extension.

Potential UBI features:

- Annual mileage or kilometers driven.
- Night-driving share.
- Harsh braking events per 100 km.
- Harsh acceleration events per 100 km.
- Speeding percentage.
- Urban driving share.
- Trip regularity or commute concentration.
- Phone distraction proxy.
- Weather or road-risk exposure.

Best next step:

- Add a second modeling track called `telematics_extension`.
- Simulate telematics features conditional on existing variables, with transparent assumptions.
- Show how telematics changes ranking, lift, fairness, and interpretability.
- Clearly label synthetic telematics as synthetic, not observed.

Stronger version:

- Find a real telematics or driving-behavior dataset and join at an aggregate segment level.
- Build a separate frequency model using driving exposure as the core denominator.

## Explainability and Governance

### 1. Improve SHAP reliability

Current SHAP uses `shap.Explainer(model.predict, background_data)` over the ensemble. That works as a model-agnostic approximation, but it is slow and can be noisy.

Recommended work:

- Use model-specific TreeExplainer for individual tree models.
- Explain the ensemble by averaging component SHAP values when possible.
- Add segment-level SHAP summaries, not only global beeswarm.
- Add local explanations for low, medium, and high risk examples.

### 2. Add model documentation

Create a model card covering:

- Intended use.
- Data source and limitations.
- Target definition.
- Exposure treatment.
- Capping policy.
- Metrics.
- Bias and fairness considerations.
- Known failure modes.
- Monitoring requirements.

### 3. Add fairness and regulatory checks

The dataset includes age, region, vehicle type, and density. Even if protected classes are absent, proxies can matter.

Add:

- Performance by driver age band.
- Calibration by region and urban density.
- Monotonicity checks for driver age and vehicle age where business intuition suggests structure.
- Sensitivity tests for young drivers and high-density areas.
- A statement about what features should or should not be used in regulated pricing contexts.

## Engineering Improvements

### 1. Refactor into reusable modules

Suggested structure:

```text
src/
  ubi_model/
    __init__.py
    config.py
    data.py
    features.py
    metrics.py
    models.py
    train.py
    evaluate.py
    explain.py
    app.py
tests/
reports/
  figures/
  metrics.json
artifacts/
  models/
  preprocessors/
```

### 2. Add tests

Start with focused tests:

- Data merge preserves one row per policy.
- Missing claims become zero.
- Exposure is always positive.
- Feature engineering matches between training and app inference.
- Preprocessor output column count matches feature names.
- Gini implementation matches known examples.
- Saved model can reload and predict positive values.

### 3. Use sklearn pipelines

Bundle preprocessing and model prediction together where possible:

- `Pipeline([("features", preprocessor), ("model", model)])`
- Save one inference artifact instead of requiring separate model and preprocessor loading.
- This reduces app/training mismatch risk.

### 4. Add CI-style quality checks

Add:

- `ruff` for linting.
- `pytest` for tests.
- Optional `nbstripout` or notebook output cleaning.
- A small smoke test that runs on a data sample.

## README and Portfolio Improvements

The README is polished, but it should become more defensible.

Add:

- Exact data source links and dataset IDs.
- Reproduction commands.
- Environment setup.
- Clear target definition.
- Train/validation/test methodology.
- Final holdout metrics only.
- A table for deviance, Gini, calibration error, and lift.
- A limitations section.
- Screenshots of validation charts and Streamlit app.

Tone adjustment:

- Replace “state-of-the-art ensemble” with a more precise claim, such as “gradient boosting blend”.
- Avoid saying the lift “directly translates” to better pricing unless calibration and business simulation are shown.
- Make the project sound rigorous rather than over-claimed.

## Suggested Priority Roadmap

### Phase 1: Make it reproducible and internally consistent

- Fix artifact names and paths.
- Add dependency file.
- Move duplicated ensemble class into one module.
- Add one command to rebuild data, train model, and evaluate.
- Save all metrics to a report file.
- Update README to match actual outputs.

### Phase 2: Correct methodology

- Redefine target and exposure treatment.
- Add train/validation/holdout split.
- Move Optuna tuning off the final test set.
- Standardize Gini and lift metrics.
- Add calibration checks.

### Phase 3: Deepen actuarial modeling

- Add frequency-severity decomposition.
- Tune Tweedie variance power.
- Compare uncapped, capped, and tail-adjusted approaches.
- Add bootstrap confidence intervals.
- Add segment-level actual-versus-expected analysis.

### Phase 4: Make it feel like a serious UBI project

- Add telematics extension with transparent synthetic assumptions or real driving data.
- Evaluate telematics lift over traditional rating factors.
- Add fairness, governance, and regulatory discussion.
- Improve Streamlit app to show actual-to-expected, risk drivers, and scenario comparison.

### Phase 5: Polish as a portfolio artifact

- Add model card.
- Add tests and CI-quality checks.
- Add plots under `reports/figures/`.
- Clean notebook outputs or convert experiments into scripts.
- Add a concise project architecture diagram.

## Highest-Value Next Implementation Step

The single best next move is to rebuild the pipeline around a correct, explicit target:

```text
pure_premium = ClaimAmount_Capped / Exposure
sample_weight = Exposure
```

Then create a proper train/validation/holdout split and re-run:

- Tweedie GLM
- XGBoost Tweedie
- LightGBM Tweedie
- CatBoost Tweedie
- weighted blend or true stacking

This will either confirm the current headline Gini under a cleaner methodology or reveal that the reported improvement was partly caused by target/exposure handling and test-set reuse. Either outcome makes the project much stronger.
