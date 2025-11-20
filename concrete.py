import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.model_selection import cross_val_score
from scipy.stats import spearmanr
from xgboost import XGBRegressor
import openml
import warnings
warnings.filterwarnings('ignore')

# Load dataset from OpenML
dataset = openml.datasets.get_dataset(46917)
X, y, categorical_indicator, attribute_names = dataset.get_data(
    target=dataset.default_target_attribute)

# Convert to numpy arrays if needed
if hasattr(X, 'values'):
    X = X.values
if hasattr(y, 'values'):
    y = y.values

# Show shape of dataset
print(f"Dataset shape: X = {X.shape}, y = {y.shape}")
print(f"Feature names: {attribute_names}")

# Define function to get feature importance for each method
def get_feature_importance(X, y, method):
    if method == 'random_forest':
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        importance = model.feature_importances_
        
    elif method == 'xgboost':
        model = XGBRegressor(random_state=42)
        model.fit(X, y)
        importance = model.feature_importances_
        
    elif method == 'feature_agglomeration':
        # For feature agglomeration, we'll use correlation with target as a measure
        importance = np.array([abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])])
        
    elif method == 'highly_variable':
        # For highly variable gene selection, we'll use variance as importance
        importance = np.var(X, axis=0)
        
    elif method == 'spearman':
        # Calculate Spearman correlation with target
        importance = np.array([abs(spearmanr(X[:, i], y)[0]) for i in range(X.shape[1])])
    
    return importance

# Calculate feature importance for all methods
methods = ['random_forest', 'xgboost', 'feature_agglomeration', 'highly_variable', 'spearman']
importances = {}
rankings = {}

for method in methods:
    imp = get_feature_importance(X, y, method)
    importances[method] = imp
    rankings[method] = np.argsort(-imp)  # Sort indices in descending order of importance

# Create set1 (top 6 features) and set2 (top 5 features after removing the highest feature)
sets = {}
for method in methods:
    # Set 1: Top 6 features
    top_features = rankings[method][:6]
    sets[f"{method}_set1"] = X[:, top_features]
    
    # Set 2: Remove highest feature and select top 5
    reduced_X = np.delete(X, rankings[method][0], axis=1)
    reduced_ranking = get_feature_importance(reduced_X, y, method)
    top_reduced_features = np.argsort(-reduced_ranking)[:5]
    sets[f"{method}_set2"] = reduced_X[:, top_reduced_features]

# Cross-validation
results = {}
for method in methods:
    if method in ['random_forest', 'xgboost']:
        if method == 'random_forest':
            model = RandomForestRegressor(random_state=42)
        else:  # xgboost
            model = XGBRegressor(random_state=42)
    else:
        # For other methods, use RandomForest for cross-validation
        model = RandomForestRegressor(random_state=42)
    
    # Cross-validate set1
    cv_scores_set1 = cross_val_score(model, sets[f"{method}_set1"], y, cv=6, scoring='r2')
    
    # Cross-validate set2
    cv_scores_set2 = cross_val_score(model, sets[f"{method}_set2"], y, cv=5, scoring='r2')
    
    results[method] = {
        'CV6_R2': np.mean(cv_scores_set1),
        'CV5_R2': np.mean(cv_scores_set2),
        'Top_6_Features': [attribute_names[i] for i in rankings[method][:6]],
        'Top_5_Features_Reduced': [
            attribute_names[i if i < rankings[method][0] else i+1] 
            for i in np.argsort(-get_feature_importance(np.delete(X, rankings[method][0], axis=1), y, method))[:5]
        ]
    }

# Create summary table
summary_table = []
for method in methods:
    method_name = method.replace('_', ' ').title()
    summary_table.append({
        'Method': method_name,
        'CV6 R2 Score': round(results[method]['CV6_R2'], 4),
        'CV5 R2 Score': round(results[method]['CV5_R2'], 4),
        'Top 6 Feature Rankings': ', '.join(results[method]['Top_6_Features']),
        'Top 5 Feature Rankings (Reduced)': ', '.join(results[method]['Top_5_Features_Reduced'])
    })

summary_df = pd.DataFrame(summary_table)
summary_df.to_csv('result.csv', index=False)

# Display summary
print("\nResults Summary:")
print(summary_df)
