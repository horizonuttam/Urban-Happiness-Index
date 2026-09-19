#!/usr/bin/env python3
"""
Happiness analysis

- Normalizes features to 1-10 (including GDP and Healthy Life Expectancy)
- Fits OLS (statsmodels) and reports p-values
- Trains an XGBoost regressor and compares metrics
- Constructs a final Urban Happiness Index using p-value derived weights

Usage:
    python happiness_analysis.py [path/to/merged_whr_data_2011-2023_excl_2013.csv]

If required libraries are missing, the script will print installation instructions.
"""
import sys
from pathlib import Path
import math

try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import statsmodels.api as sm
    from math import log10
    HAVE_LIBS = True
except Exception as e:
    HAVE_LIBS = False
    print("Missing Python packages:", e)

try:
    import xgboost as xgb
    import joblib
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False


def normalize_series_1_10(s: pd.Series) -> pd.Series:
    s = s.copy()
    mask = s.notna()
    if mask.sum() == 0:
        return s
    vals = s[mask].astype(float)
    vmin = vals.min()
    vmax = vals.max()
    if math.isclose(vmin, vmax):
        s.loc[mask] = 5.5
    else:
        s.loc[mask] = 1 + 9 * (vals - vmin) / (vmax - vmin)
    return s


def pvalue_weights(pvals: pd.Series, eps=1e-12) -> pd.Series:
    # Convert p-values to scores where smaller p -> larger weight
    p = pvals.clip(lower=eps)
    scores = -np.log10(p)
    if scores.sum() == 0:
        return pd.Series(np.repeat(1.0 / len(scores), len(scores)), index=p.index)
    return scores / scores.sum()


def main(csv_path=None):
    if not HAVE_LIBS:
        print("Please install required Python packages first. Create a virtualenv and run:")
        print("    pip install -r requirements.txt")
        return

    if csv_path is None:
        csv_path = Path('merged_whr_data_2011-2023_excl_2013.csv')
    else:
        csv_path = Path(csv_path)

    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Define target and features
    target = 'life_evaluation'
    features = [
        'gdp_per_capita',
        'healthy_life_expectancy',
        'pm2_5',
        'intentional_homicide_rate',
        'degree_of_urbanization',
        'sdg_index_score'
    ]

    # Normalize features to 1-10 and add suffix _n
    for f in features:
        col_n = f + '_n'
        df[col_n] = normalize_series_1_10(df[f])

    normalized_cols = [f + '_n' for f in features]

    # Prepare dataset for modelling: drop rows missing target or all Xs
    model_df = df[[target] + normalized_cols].dropna()
    if model_df.empty:
        print("No complete rows available for modelling after dropping NaNs.")
        return

    X = model_df[normalized_cols]
    y = model_df[target]

    # OLS regression (statsmodels) to get p-values
    X_const = sm.add_constant(X)
    ols_model = sm.OLS(y, X_const).fit()
    print("\nOLS Regression Summary:\n")
    print(ols_model.summary())

    pvals = ols_model.pvalues.drop('const')
    coefs = ols_model.params.drop('const')
    print("\nP-values for features:")
    print(pvals)

    # Compare p-values of all vars with GDP and Healthy Life Expectancy
    compare_base = ['gdp_per_capita_n', 'healthy_life_expectancy_n']
    print('\nComparison vs GDP and Healthy Life Expectancy (p-value smaller -> more significant):')
    comp_df = pd.DataFrame({'pvalue': pvals, 'coef': coefs}).sort_values('pvalue')
    print('\nFeatures ranked by p-value (ascending):')
    print(comp_df)

    # Select features with alpha=0.05 and run PCA to derive weights
    alpha = 0.05
    selected = pvals[pvals < alpha].index.tolist()
    print(f'\nSelected features with p < {alpha}: {selected}')

    if selected:
        pca = PCA(n_components=1)
        X_selected = X[selected]
        pca.fit(X_selected)
        loadings = pd.Series(pca.components_[0], index=selected)
        abs_loadings = loadings.abs()
        pca_weights = abs_loadings / abs_loadings.sum()
        print('\nPCA first-component loadings and derived weights:')
        print(pd.DataFrame({'loading': loadings, 'weight': pca_weights}))

        pca_index_raw = X_selected.dot(pca_weights)
        pca_index_scaled = normalize_series_1_10(pca_index_raw)
        model_df['pca_happiness_index_raw'] = pca_index_raw
        model_df['pca_happiness_index'] = pca_index_scaled

        sustainable_raw = pca_index_raw
        sustainable_scaled = normalize_series_1_10(sustainable_raw)
        model_df['sustainable_happiness_index_raw'] = sustainable_raw
        model_df['sustainable_happiness_index'] = sustainable_scaled
        print('\nCreated Sustainable Happiness Index using PCA weights and normalized values.')
    else:
        print('No features selected by alpha threshold; skipping PCA index creation.')

    # Train/Test split for predictive comparison
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # OLS predictions for test set (re-fit by sklearn-like for prediction stability)
    ols_pred = ols_model.predict(sm.add_constant(X_test))
    ols_rmse = math.sqrt(mean_squared_error(y_test, ols_pred))
    ols_r2 = r2_score(y_test, ols_pred)

    xgb_results = None
    if HAVE_XGB:
        xgbr = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, random_state=42, verbosity=0)
        xgbr.fit(X_train, y_train)
        xgb_pred = xgbr.predict(X_test)
        xgb_rmse = math.sqrt(mean_squared_error(y_test, xgb_pred))
        xgb_r2 = r2_score(y_test, xgb_pred)
        xgb_results = {'rmse': xgb_rmse, 'r2': xgb_r2}

        deployment_model = xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42,
            verbosity=0,
        )
        deployment_model.fit(X, y)
        artifacts_dir = Path(__file__).parent / 'artifacts'
        artifacts_dir.mkdir(exist_ok=True)
        deployment_model.save_model(str(artifacts_dir / 'happiness_xgb.json'))
        joblib.dump(
            {
                'features': features,
                'min_values': df[features].min().to_dict(),
                'max_values': df[features].max().to_dict(),
            },
            artifacts_dir / 'preprocessing.joblib',
        )
        print(f'Model artifacts saved to: {artifacts_dir}')

    print('\nPrediction performance on holdout test set:')
    print(f"OLS    - RMSE: {ols_rmse:.4f}, R2: {ols_r2:.4f}")
    if xgb_results:
        print(f"XGBoost- RMSE: {xgb_results['rmse']:.4f}, R2: {xgb_results['r2']:.4f}")
    else:
        print('XGBoost not available (install xgboost to run).')

    # Construct Urban Happiness Index using p-value derived weights for four urban factors
    # Assumption: the four factors (A,B,C,D) are: pm2_5, intentional_homicide_rate, degree_of_urbanization, sdg_index_score
    uh_factors = ['pm2_5_n', 'intentional_homicide_rate_n', 'degree_of_urbanization_n', 'sdg_index_score_n']
    missing = [c for c in uh_factors if c not in pvals.index]
    if missing:
        print(f"Cannot compute Urban Happiness Index: missing p-values for {missing}")
    else:
        uh_pvals = pvals[uh_factors]
        weights = pvalue_weights(uh_pvals)
        print('\nUrban factors p-values and derived weights:')
        print(pd.DataFrame({'pvalue': uh_pvals, 'weight': weights}))

        # Compute an additional PCA-based urban index so degree_of_urbanization_n has explicit loadings and weights.
        uh_pca = PCA(n_components=1)
        X_uh = X[uh_factors]
        uh_pca.fit(X_uh)
        uh_loadings = pd.Series(uh_pca.components_[0], index=uh_factors)
        uh_abs_loadings = uh_loadings.abs()
        uh_pca_weights = uh_abs_loadings / uh_abs_loadings.sum()
        print('\nUrban factor PCA loadings and derived weights:')
        print(pd.DataFrame({'loading': uh_loadings, 'weight': uh_pca_weights}))

        # compute weighted sum using normalized feature values from the model_df index
        # align weights to columns in original df
        df_index = model_df.index
        uh_index = (df.loc[df_index, uh_factors] * weights.values).sum(axis=1)
        uh_index_scaled = normalize_series_1_10(uh_index)
        uh_index_pca_raw = (df.loc[df_index, uh_factors] * uh_pca_weights.values).sum(axis=1)
        uh_index_pca_scaled = normalize_series_1_10(uh_index_pca_raw)
        out_df = df.loc[df_index].copy()
        out_df['urban_happiness_index_raw'] = uh_index
        out_df['urban_happiness_index'] = uh_index_scaled
        out_df['urban_happiness_index_pca_raw'] = uh_index_pca_raw
        out_df['urban_happiness_index_pca'] = uh_index_pca_scaled
        if 'pca_happiness_index' in model_df.columns:
            out_df['pca_happiness_index_raw'] = model_df.loc[df_index, 'pca_happiness_index_raw']
            out_df['pca_happiness_index'] = model_df.loc[df_index, 'pca_happiness_index']
        if 'sustainable_happiness_index' in model_df.columns:
            out_df['sustainable_happiness_index_raw'] = model_df.loc[df_index, 'sustainable_happiness_index_raw']
            out_df['sustainable_happiness_index'] = model_df.loc[df_index, 'sustainable_happiness_index']

        # Save outputs
        out_file = csv_path.with_name('happiness_analysis_results.csv')
        out_df.to_csv(out_file, index=False)
        print(f"\nSaved results with Urban Happiness Index to: {out_file}")


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    main(arg)
