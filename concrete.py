import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.cluster import FeatureAgglomeration
import openml

# Optional: xgboost may need installation; handle gracefully
try:
    from xgboost import XGBRegressor
    has_xgb = True
except Exception:
    has_xgb = False


def load_openml_dataset(dataset_id=46917, target_name='ConcreteCompressiveStrength'):
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=target_name, dataset_format='dataframe'
    )
    feature_names = list(X.columns)
    return X, y, feature_names


def summarize_target(y):
    # Return a dict of basic distribution stats without plotting
    y_arr = y.values if hasattr(y, 'values') else np.asarray(y)
    summary = {
        'count': int(y_arr.shape[0]),
        'mean': float(np.mean(y_arr)),
        'std': float(np.std(y_arr, ddof=1)) if y_arr.shape[0] > 1 else 0.0,
        'min': float(np.min(y_arr)),
        '25%': float(np.percentile(y_arr, 25)),
        '50%': float(np.percentile(y_arr, 50)),
        '75%': float(np.percentile(y_arr, 75)),
        'max': float(np.max(y_arr)),
    }
    return summary


def rank_features_random_forest(X, y, feature_names):
    rf = RandomForestRegressor()
    rf.fit(X, y)
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    ranked = [feature_names[i] for i in order]
    return ranked, importances


def rank_features_xgboost(X, y, feature_names):
    if not has_xgb:
        # Fallback to RF if XGB not available
        return rank_features_random_forest(X, y, feature_names)
    xgb = XGBRegressor()
    xgb.fit(X.values, y.values)
    booster = xgb.get_booster()
    # Prefer 'gain' importance
    score_map = booster.get_score(importance_type='gain')
    gains = np.zeros(len(feature_names), dtype=float)
    for i in range(len(feature_names)):
        gains[i] = score_map.get(f"f{i}", 0.0)
    order = np.argsort(gains)[::-1]
    ranked = [feature_names[i] for i in order]
    return ranked, gains


def rank_features_hvgs(X, feature_names):
    variances = np.var(X.values, axis=0)
    order = np.argsort(variances)[::-1]
    ranked = [feature_names[i] for i in order]
    return ranked, variances


def rank_features_spearman(X, y, feature_names):
    corrs = np.zeros(len(feature_names), dtype=float)
    for i, fname in enumerate(feature_names):
        rho, _ = spearmanr(X[fname].values, y.values)
        corrs[i] = abs(rho)
    order = np.argsort(corrs)[::-1]
    ranked = [feature_names[i] for i in order]
    return ranked, corrs


def rank_features_feature_agglomeration(X, feature_names):
    # Create multiple clusters and compute cluster-level variance scores,
    # then assign that score to member features and rank globally.
    n_features = X.shape[1]
    n_clusters = max(2, min(n_features // 2, n_features))
    fa = FeatureAgglomeration(n_clusters=n_clusters)
    fa.fit(X.values)
    labels = fa.labels_
    cluster_vars = {}
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        cluster_signal = np.mean(X.values[:, members], axis=1)
        cluster_vars[c] = np.var(cluster_signal)
    scores = np.array([cluster_vars[labels[i]] for i in range(n_features)], dtype=float)
    order = np.argsort(scores)[::-1]
    ranked = [feature_names[i] for i in order]
    return ranked, scores


def cross_validate_model(X, y, features, model, n_splits):
    Xsub = X[features]
    cv = KFold(n_splits=n_splits, shuffle=False)
    scores = cross_val_score(model, Xsub, y, cv=cv, scoring='r2')
    return float(np.mean(scores))


def process_method(method_name, rank_func, X, y, feature_names, cv_models):
    # Compute full ranking
    if method_name in ['random_forest', 'xgboost', 'spearman']:
        ranked_full, _ = rank_func(X, y, feature_names)
    else:
        ranked_full, _ = rank_func(X, feature_names)

    # set1: top 6 features from full set
    top6 = ranked_full[:6]
    model_for_cv = cv_models[method_name]
    cv6_set1 = cross_validate_model(X, y, top6, model_for_cv, n_splits=6)
    cv5_set1 = cross_validate_model(X, y, top6, model_for_cv, n_splits=5)

    # Build set2 (not reported in CSV, but carried out per instructions)
    highest = ranked_full[0]
    X_reduced = X.drop(columns=[highest])
    feature_names_reduced = [f for f in feature_names if f != highest]
    if method_name in ['random_forest', 'xgboost', 'spearman']:
        ranked_reduced, _ = rank_func(X_reduced, y, feature_names_reduced)
    else:
        ranked_reduced, _ = rank_func(X_reduced, feature_names_reduced)
    top5 = ranked_reduced[:5]
    # Perform CV for set2 (results not included in CSV)
    _ = cross_validate_model(X, y, top5, model_for_cv, n_splits=6)
    _ = cross_validate_model(X, y, top5, model_for_cv, n_splits=5)

    return {
        'method': method_name,
        'CV6 accuracy': cv6_set1,
        'CV5 accuracy': cv5_set1,
        'top 6 feature rankings': json.dumps(top6),
        'top 5 feature rankings': json.dumps(top5)
    }


def main():
    # Load data
    X, y, feature_names = load_openml_dataset(46917, 'ConcreteCompressiveStrength')

    # Show shape and target distribution (printed to console)
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    target_summary = summarize_target(y)
    print("Target distribution summary (no plot):")
    print(json.dumps(target_summary, indent=2))

    # Define methods and CV models
    methods = {
        'random_forest': rank_features_random_forest,
        'xgboost': rank_features_xgboost,
        'feature_agglomeration': rank_features_feature_agglomeration,
        'hvgs': rank_features_hvgs,
        'spearman': rank_features_spearman
    }

    cv_models = {
        'random_forest': RandomForestRegressor(),
        'xgboost': XGBRegressor() if has_xgb else RandomForestRegressor(),
        'feature_agglomeration': RandomForestRegressor(),
        'hvgs': RandomForestRegressor(),
        'spearman': RandomForestRegressor()
    }

    rows = []
    for name, func in methods.items():
        rows.append(process_method(name, func, X, y, feature_names, cv_models))

    # Create summary table as specified (only CV6 and CV5 on set1)
    df = pd.DataFrame(rows, columns=[
        'method', 'CV6 accuracy', 'CV5 accuracy', 'top 6 feature rankings', 'top 5 feature rankings'
    ])

    # Save to result.csv
    df.to_csv('result.csv', index=False)
    print('Saved result.csv')


if __name__ == '__main__':
    main()
