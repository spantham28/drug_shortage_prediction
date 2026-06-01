"""
Comprehensive ML Pipeline for Predicting Net Income from multiple features
with Stratified Sampling and Multiple Regression Models.

Target: Net Income in dollars (no transform). Hyperparameter tuning minimizes sMAPE.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, make_scorer, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LassoCV
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_mape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error (sMAPE) to handle negative values"""
    # Use symmetric MAPE which works better with negative values
    # sMAPE = mean(200 * |y_true - y_pred| / (|y_true| + |y_pred|))
    denominator = np.abs(y_true) + np.abs(y_pred)
    mask = denominator > 0  # Avoid division by zero
    if mask.sum() == 0:
        return np.nan
    return np.mean(200 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])

def calculate_adjusted_r2(y_true, y_pred, n_features):
    """Adjusted R² with p = number of predictors."""
    n = len(y_true)
    if n <= n_features + 1:
        return np.nan
    r2 = r2_score(y_true, y_pred)
    return 1 - (1 - r2) * (n - 1) / (n - n_features - 1)

def create_stratified_splits(df):
    """
    Create stratified splits based on:
    Quantiles of Net Income (4 quantiles)
    """
    # Create stratification labels based only on Net Income quantiles
    target_col = 'Net Income'
    n_samples = len(df)
    
    # Create 4 quantiles of the target variable
    if n_samples >= 8:  # At least 2 samples per quantile after splits
        try:
            quantiles = pd.qcut(df[target_col], q=4, labels=False, duplicates='drop')
            df['stratum'] = [f"Q{q}" for q in quantiles]
        except ValueError:
            # If quantiles can't be created (e.g., too many duplicates), use fewer quantiles
            try:
                quantiles = pd.qcut(df[target_col], q=2, labels=False, duplicates='drop')
                df['stratum'] = [f"Q{q}" for q in quantiles]
            except ValueError:
                # If still can't create, just use a single stratum
                df['stratum'] = 'All'
    else:
        # For very small datasets, use a single stratum
        df['stratum'] = 'All'
    
    # Handle any remaining NaN values
    df['stratum'] = df['stratum'].fillna('Unknown')
    
    # Merge strata that have too few samples (less than 3) with similar ones
    stratum_counts = df['stratum'].value_counts()
    small_strata = stratum_counts[stratum_counts < 3].index
    
    if len(small_strata) > 0:
        # For small strata, merge them with the nearest quantile
        for small_stratum in small_strata:
            # Find the largest stratum to merge with
            large_strata = stratum_counts[stratum_counts >= 3].index
            if len(large_strata) > 0:
                merge_target = large_strata[0]
                df.loc[df['stratum'] == small_stratum, 'stratum'] = merge_target
            else:
                # If all are small, just use 'All'
                df.loc[df['stratum'] == small_stratum, 'stratum'] = 'All'
    
    return df

def stratified_train_val_test_split(df, test_size=0.2, val_size=0.2, random_state=42):
    """
    Split data into train, validation, and test sets maintaining stratification
    """
    # Check stratum sizes and handle small ones
    stratum_counts = df['stratum'].value_counts()
    
    # For strata with only 1 sample, we'll assign them deterministically
    single_sample_strata = stratum_counts[stratum_counts == 1].index
    multi_sample_strata = stratum_counts[stratum_counts > 1].index
    
    # Initialize indices
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Handle multi-sample strata with stratified splitting
    if len(multi_sample_strata) > 0:
        multi_sample_df = df[df['stratum'].isin(multi_sample_strata)].copy()
        
        # First split: separate test set (20%)
        try:
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_val_idx, test_idx = next(sss1.split(multi_sample_df, multi_sample_df['stratum']))
            
            train_val_df = multi_sample_df.iloc[train_val_idx].copy()
            test_df = multi_sample_df.iloc[test_idx].copy()
            
            # Second split: separate validation set from train+val
            val_size_adjusted = val_size / (1 - test_size)  # 0.2 / 0.8 = 0.25
            
            # Check if we can do stratified split for validation
            stratum_counts_val = train_val_df['stratum'].value_counts()
            if (stratum_counts_val >= 2).all():
                sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_state+1)
                train_idx, val_idx = next(sss2.split(train_val_df, train_val_df['stratum']))
                
                train_indices.extend(train_val_df.iloc[train_idx].index.tolist())
                val_indices.extend(train_val_df.iloc[val_idx].index.tolist())
            else:
                # Fall back to random split if some strata are too small
                train_idx, val_idx = train_test_split(
                    train_val_df.index, 
                    test_size=val_size_adjusted, 
                    random_state=random_state+1
                )
                train_indices.extend(train_idx.tolist())
                val_indices.extend(val_idx.tolist())
            
            test_indices.extend(test_df.index.tolist())
        except ValueError:
            # If stratified split fails, fall back to random split
            train_idx, temp_idx = train_test_split(
                multi_sample_df.index,
                test_size=test_size,
                random_state=random_state
            )
            val_idx, test_idx = train_test_split(
                temp_idx,
                test_size=val_size / (test_size + val_size),  # Adjust for remaining data
                random_state=random_state+1
            )
            train_indices.extend(train_idx.tolist())
            val_indices.extend(val_idx.tolist())
            test_indices.extend(test_idx.tolist())
    
    # Handle single-sample strata - assign them proportionally
    if len(single_sample_strata) > 0:
        single_sample_df = df[df['stratum'].isin(single_sample_strata)].copy()
        n_single = len(single_sample_df)
        n_train = int(n_single * 0.6)
        n_val = int(n_single * 0.2)
        n_test = n_single - n_train - n_val
        
        # Shuffle and assign
        single_indices = single_sample_df.index.tolist()
        np.random.seed(random_state)
        np.random.shuffle(single_indices)
        
        train_indices.extend(single_indices[:n_train])
        val_indices.extend(single_indices[n_train:n_train+n_val])
        test_indices.extend(single_indices[n_train+n_val:])
    
    # Create final dataframes
    train_df = df.loc[train_indices].copy()
    val_df = df.loc[val_indices].copy()
    test_df = df.loc[test_indices].copy()
    
    return train_df, val_df, test_df

def get_model_configs():
    """Define models and their hyperparameter grids with reduced complexity to prevent overfitting"""
    return {
        'DecisionTreeRegressor': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {
                'max_depth': [3, 5, 7, 10],  # Reduced max depth
                'min_samples_split': [10, 20, 30],  # Increased to prevent overfitting
                'min_samples_leaf': [5, 10, 15]  # Increased to prevent overfitting
            }
        },
        'RandomForestRegressor': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100],  # Reduced
                'max_depth': [5, 10, 15],  # Reduced max depth
                'min_samples_split': [10, 20],  # Increased
                'min_samples_leaf': [5, 10]  # Increased
            }
        },
        'GradientBoostingRegressor': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100],  # Reduced
                'learning_rate': [0.05, 0.1],  # Reduced learning rate
                'max_depth': [3, 5],  # Reduced max depth
                'min_samples_split': [10, 20],  # Increased
                'subsample': [0.8, 0.9]  # Added subsampling
            }
        },
        'Ridge': {
            'model': Ridge(random_state=42),
            'params': {
                'alpha': [1.0, 10.0, 100.0, 1000.0, 10000.0]  # Increased regularization range
            }
        },
        'Lasso': {
            'model': Lasso(random_state=42),
            'params': {
                'alpha': [1.0, 10.0, 100.0, 1000.0, 10000.0]  # Increased regularization range
            }
        },
        'ElasticNet': {
            'model': ElasticNet(random_state=42),
            'params': {
                'alpha': [1.0, 10.0, 100.0, 1000.0],  # Increased regularization
                'l1_ratio': [0.3, 0.5, 0.7]  # Focused range
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            }
        },
        'KNeighborsRegressor': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': [3, 5, 7, 10, 15],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        },
        'MLPRegressor': {
            'model': MLPRegressor(random_state=42, max_iter=500),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        },
        'AdaBoostRegressor': {
            'model': AdaBoostRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
                'loss': ['linear', 'square', 'exponential']
            }
        },
        'ExtraTreesRegressor': {
            'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100],  # Reduced
                'max_depth': [5, 10, 15],  # Reduced max depth
                'min_samples_split': [10, 20],  # Increased
                'min_samples_leaf': [5, 10]  # Increased
            }
        },
        'XGBRegressor': {
            'model': XGBRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100],  # Reduced
                'max_depth': [3, 5],  # Reduced max depth
                'learning_rate': [0.05, 0.1],  # Reduced learning rate
                'subsample': [0.8, 0.9],  # Added subsampling
                'colsample_bytree': [0.8, 0.9],  # Added feature subsampling
                'reg_alpha': [0.1, 1.0],  # Added L1 regularization
                'reg_lambda': [1.0, 10.0]  # Added L2 regularization
            }
        }
    }

def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """Train and evaluate models on Net Income in dollars; tune hyperparameters with sMAPE."""
    model_configs = get_model_configs()
    results = []
    best_models = {}
    test_predictions = {}  # Store test predictions for each model
    n_features = len(feature_names)
    
    print("=" * 80)
    print("Training and Evaluating Models")
    print("=" * 80)
    
    # Scale features for models that need it (fit on training data only)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    for model_name, config in model_configs.items():
        print(f"\n{'='*80}")
        print(f"Training {model_name}...")
        print(f"{'='*80}")
        
        model = config['model']
        param_grid = config['params']
        
        # Determine if model needs scaling
        needs_scaling = model_name in ['Ridge', 'Lasso', 'ElasticNet', 'SVR', 
                                       'KNeighborsRegressor', 'MLPRegressor']
        
        if needs_scaling:
            X_train_use = X_train_scaled
            X_val_use = X_val_scaled
            X_test_use = X_test_scaled
        else:
            X_train_use = X_train
            X_val_use = X_val
            X_test_use = X_test
        
        # Grid search with cross-validation - minimize sMAPE on dollar Net Income
        mape_scorer = make_scorer(calculate_mape, greater_is_better=False)
        
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=5, 
            scoring=mape_scorer,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_use, y_train)
        best_model = grid_search.best_estimator_
        best_models[model_name] = {
            'model': best_model,
            'scaler': scaler if needs_scaling else None,
            'needs_scaling': needs_scaling,
            'feature_names': feature_names
        }
        
        y_train_pred = best_model.predict(X_train_use)
        y_val_pred = best_model.predict(X_val_use)
        y_test_pred = best_model.predict(X_test_use)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = calculate_rmse(y_train, y_train_pred)
        train_mape = calculate_mape(y_train, y_train_pred)
        
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = calculate_rmse(y_val, y_val_pred)
        val_mape = calculate_mape(y_val, y_val_pred)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        test_mape = calculate_mape(y_test, y_test_pred)
        
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_adj_r2 = calculate_adjusted_r2(y_train, y_train_pred, n_features)
        val_adj_r2 = calculate_adjusted_r2(y_val, y_val_pred, n_features)
        test_adj_r2 = calculate_adjusted_r2(y_test, y_test_pred, n_features)
        
        # Overfitting indicators
        overfitting_gap_rmse = train_rmse - test_rmse
        overfitting_gap_mape = train_mape - test_mape
        
        # Store results
        result = {
            'Model': model_name,
            'Best_Params': grid_search.best_params_,
            'Train_MSE': train_mse,
            'Train_RMSE': train_rmse,
            'Train_MAPE': train_mape,
            'Train_R2': train_r2,
            'Train_Adjusted_R2': train_adj_r2,
            'Val_MSE': val_mse,
            'Val_RMSE': val_rmse,
            'Val_MAPE': val_mape,
            'Val_R2': val_r2,
            'Val_Adjusted_R2': val_adj_r2,
            'Test_MSE': test_mse,
            'Test_RMSE': test_rmse,
            'Test_MAPE': test_mape,
            'Test_R2': test_r2,
            'Test_Adjusted_R2': test_adj_r2,
            'Overfitting_Gap_RMSE': overfitting_gap_rmse,
            'Overfitting_Gap_MAPE': overfitting_gap_mape
        }
        results.append(result)
        
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Train - MAPE: {train_mape:.2f}% (tuning metric), RMSE: {train_rmse:.2f}, MSE: {train_mse:.2f}")
        print(f"Val   - MAPE: {val_mape:.2f}%, RMSE: {val_rmse:.2f}, MSE: {val_mse:.2f}")
        print(f"Test  - MAPE: {test_mape:.2f}%, RMSE: {test_rmse:.2f}, MSE: {test_mse:.2f}")
        print(f"        R²: {test_r2:.4f}, Adjusted R²: {test_adj_r2:.4f}")
        print(f"Overfitting Gap (Train-Test MAPE): {overfitting_gap_mape:.2f}%")
        print(f"Overfitting Gap (Train-Test RMSE): {overfitting_gap_rmse:.2f}")
        
        # Store test predictions
        test_predictions[model_name] = y_test_pred
    
    return pd.DataFrame(results), best_models, test_predictions

def get_feature_importance(best_models, feature_names):
    """Extract feature importance from models that support it"""
    importance_data = []
    
    for model_name, model_info in best_models.items():
        model = model_info['model']
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for i, feature in enumerate(feature_names):
                importance_data.append({
                    'Model': model_name,
                    'Feature': feature,
                    'Importance': importances[i]
                })
        elif hasattr(model, 'coef_'):
            # For linear models
            coef = model.coef_
            if coef.ndim == 1:
                for i, feature in enumerate(feature_names):
                    importance_data.append({
                        'Model': model_name,
                        'Feature': feature,
                        'Importance': abs(coef[i])
                    })
    
    return pd.DataFrame(importance_data)

# Additional predictors beyond the filtered analysis pipeline
EXTRA_PREDICTORS = ['Profit']

# Isolation Forest contamination: ~2% of rows are multivariate extremes on this dataset
# (aligns with 1st-99th percentile tail on Net Income; see remove_outliers_isolation_forest)
OUTLIER_CONTAMINATION = 0.02

def _encode_profit_flag(series, fill_value='Loss'):
    """Encode Profit/Loss flag as 1/0 for modeling."""
    profit_map = {'Profit': 1, 'Loss': 0}
    return series.astype(str).map(profit_map).fillna(profit_map[fill_value]).astype(float).values

def prepare_selected_features(train_df, val_df, test_df, df_original):
    """
    Prepare features based on nonlinear_correlations.csv
    Automatically determines which features to use and what transformations to apply.
    """
    # Load feature lists and transformations from CSV file
    print("   Loading features and transformations from nonlinear_correlations.csv...")
    try:
        nonlinear_df = pd.read_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/nonlinear_correlations.csv')
        final_features = nonlinear_df['Feature'].tolist()
        print(f"   Loaded {len(final_features)} features from nonlinear_correlations.csv")
        
        # Extract transformations from the CSV
        # Map CSV transformation names to code transformation types
        transform_map = {
            'Square': 'square',
            'Sqrt': 'sqrt',
            'Original': 'none',
            'Log': 'log'  # In case it's used in the future
        }
        
        transformations = {}
        for _, row in nonlinear_df.iterrows():
            feature = row['Feature']
            csv_transform = row['Best_Transformation']
            transformations[feature] = transform_map.get(csv_transform, 'none')
            print(f"     {feature}: {csv_transform} -> {transformations[feature]}")
        
        for feat in EXTRA_PREDICTORS:
            if feat not in final_features:
                final_features.append(feat)
                transformations[feat] = 'none'
                print(f"     {feat}: (extra predictor) -> none")
        
    except FileNotFoundError:
        print("   WARNING: nonlinear_correlations.csv not found. Using default features.")
        final_features = [
            'Total Bed Days Available',
            'Total Liabilities',
            'Land',
            'Land Improvements',
            'Number of Beds',
            'Accounts Receivable',
            'Total Current Assets',
            'Fixed Equipment'
        ]
        
        # Default transformations based on typical patterns
        transformations = {
            'Total Bed Days Available': 'square',
            'Total Liabilities': 'square',
            'Land': 'none',
            'Land Improvements': 'square',
            'Number of Beds': 'sqrt',
            'Accounts Receivable': 'sqrt',
            'Total Current Assets': 'sqrt',
            'Fixed Equipment': 'sqrt',
        }
        for feat in EXTRA_PREDICTORS:
            if feat not in final_features:
                final_features.append(feat)
                transformations[feat] = 'none'
    
    # Prepare feature dictionaries
    train_feat_dict = {}
    val_feat_dict = {}
    test_feat_dict = {}
    feature_names = []
    
    print("   Applying transformations to features...")
    for feat in final_features:
        if feat not in train_df.columns:
            print(f"   WARNING: Feature '{feat}' not found in dataset, skipping...")
            continue
        
        # Get transformation type (default to 'none' if not specified)
        transform_type = transformations.get(feat, 'none')
        
        if feat == 'Profit':
            train_feat_dict[feat] = _encode_profit_flag(train_df[feat])
            val_feat_dict[feat] = _encode_profit_flag(val_df[feat])
            test_feat_dict[feat] = _encode_profit_flag(test_df[feat])
            feature_names.append(feat)
            continue
        
        # Get raw values
        train_vals = train_df[feat].fillna(train_df[feat].median()).values
        val_vals = val_df[feat].fillna(train_df[feat].median()).values  # Use train median to avoid leakage
        test_vals = test_df[feat].fillna(train_df[feat].median()).values
        
        # Apply transformation
        if transform_type == 'sqrt':
            # Handle negative values for sqrt
            offset = abs(np.min(train_vals)) if np.min(train_vals) < 0 else 0
            train_feat_dict[feat] = np.sqrt(np.abs(train_vals) + offset)
            val_feat_dict[feat] = np.sqrt(np.abs(val_vals) + offset)
            test_feat_dict[feat] = np.sqrt(np.abs(test_vals) + offset)
            feature_names.append(feat)
        elif transform_type == 'square':
            train_feat_dict[feat] = train_vals ** 2
            val_feat_dict[feat] = val_vals ** 2
            test_feat_dict[feat] = test_vals ** 2
            feature_names.append(feat)
        elif transform_type == 'log':
            # Handle non-positive values for log
            offset = abs(np.min(train_vals)) + 1 if np.min(train_vals) <= 0 else 0
            train_feat_dict[feat] = np.log(np.abs(train_vals) + offset)
            val_feat_dict[feat] = np.log(np.abs(val_vals) + offset)
            test_feat_dict[feat] = np.log(np.abs(test_vals) + offset)
            feature_names.append(feat)
        elif transform_type == 'none':
            train_feat_dict[feat] = train_vals
            val_feat_dict[feat] = val_vals
            test_feat_dict[feat] = test_vals
            feature_names.append(feat)
    
    print(f"   Prepared {len(feature_names)} features")
    return train_feat_dict, val_feat_dict, test_feat_dict, feature_names

def remove_outliers_isolation_forest(
    X_train, y_train, X_val, y_val, X_test, y_test,
    train_df, val_df, test_df,
    target_col='Net Income',
    contamination=OUTLIER_CONTAMINATION,
    random_state=42,
):
    """
    Remove multivariate outliers using Isolation Forest fit on the training set only.

    Standardizes modeling features (fit on train), appends the target, and fits
    IsolationForest on train rows. The same scaler and forest score val/test rows
    so removal thresholds are learned without peeking at holdout labels' distribution.

    Chosen over univariate IQR/MAD because Net Income is heavily right-skewed
    (skew ~4.5): IQR/MAD flag ~8% of rows, including many legitimate large hospitals.
    Percentile trimming is simpler but ignores unusual feature-target combinations.
    """
    print("=" * 80)
    print("Outlier Detection and Removal (Isolation Forest)")
    print("=" * 80)
    print(f"   Method: Isolation Forest on standardized features + target")
    print(f"   Contamination (expected outlier rate): {contamination:.1%}")
    print(f"   Fit on: training set only ({len(y_train)} rows)")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    train_xy = np.hstack([X_train_scaled, y_train.reshape(-1, 1)])
    val_xy = np.hstack([X_val_scaled, y_val.reshape(-1, 1)])
    test_xy = np.hstack([X_test_scaled, y_test.reshape(-1, 1)])

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=200,
    )
    iso.fit(train_xy)

    train_inlier = iso.predict(train_xy) == 1
    val_inlier = iso.predict(val_xy) == 1
    test_inlier = iso.predict(test_xy) == 1

    def _summarize(split_name, mask, y_split, df_split):
        n_removed = (~mask).sum()
        print(f"   {split_name}: removed {n_removed} / {len(mask)} rows ({100 * n_removed / len(mask):.1f}%)")
        if n_removed > 0:
            removed_y = y_split[~mask]
            print(f"      Removed {target_col} range: {removed_y.min():,.0f} to {removed_y.max():,.0f}")
        return n_removed

    n_train_removed = _summarize('Train', train_inlier, y_train, train_df)
    n_val_removed = _summarize('Validation', val_inlier, y_val, val_df)
    n_test_removed = _summarize('Test', test_inlier, y_test, test_df)

    outlier_rows = []
    for split_name, mask, df_split in [
        ('train', train_inlier, train_df),
        ('validation', val_inlier, val_df),
        ('test', test_inlier, test_df),
    ]:
        removed = df_split.loc[~mask].copy()
        if len(removed) > 0:
            removed = removed.assign(Split=split_name)
            outlier_rows.append(removed)

    if outlier_rows:
        outlier_report = pd.concat(outlier_rows, axis=0)
        report_path = (
            '/Users/janakipantham/Desktop/drug_shortage_platform/'
            'drug_shortage_cost_prediction/removed_outliers_report.csv'
        )
        outlier_report.to_csv(report_path, index=True)
        print(f"   Saved removed rows: removed_outliers_report.csv")

    return (
        X_train[train_inlier], y_train[train_inlier], train_df.iloc[train_inlier].copy(),
        X_val[val_inlier], y_val[val_inlier], val_df.iloc[val_inlier].copy(),
        X_test[test_inlier], y_test[test_inlier], test_df.iloc[test_inlier].copy(),
        {
            'train_removed': n_train_removed,
            'val_removed': n_val_removed,
            'test_removed': n_test_removed,
        },
    )

def main():
    """Main execution function"""
    print("=" * 80)
    print("Drug Shortage Cost Prediction - ML Pipeline")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/hospital_ops_updated.csv')
    print(f"   Loaded {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Identify target and features
    target_col = 'Net Income'
    exclude_cols = ['Provider CCN', target_col, 'Unnamed: 0']
    
    # Get feature columns (all except Provider CCN, target, and index column).
    # Profit is used only as an extra predictor in prepare_selected_features, not in bulk dropna.
    feature_cols = [col for col in df.columns if col not in exclude_cols + ['Profit']]
    # Explicitly remove 'Unnamed: 0' if it somehow got included
    if 'Unnamed: 0' in feature_cols:
        feature_cols.remove('Unnamed: 0')
    print(f"\n   Target variable: {target_col}")
    print(f"   Feature variables: {len(feature_cols)} features")
    
    # Check for missing values
    print("\n2. Data quality check...")
    check_cols = [target_col] + feature_cols + EXTRA_PREDICTORS
    missing_counts = df[check_cols].isnull().sum()
    print(f"   Missing values:\n{missing_counts[missing_counts > 0]}")
    
    # Remove rows with missing target, filtered features, or extra predictors
    df = df.dropna(subset=[target_col] + feature_cols + EXTRA_PREDICTORS)
    print(f"   After removing missing values: {len(df)} rows")
    
    # Create stratified splits
    print("\n3. Creating stratified splits...")
    df = create_stratified_splits(df)
    print(f"   Number of unique strata: {df['stratum'].nunique()}")
    print(f"   Strata distribution:\n{df['stratum'].value_counts().head(10)}")
    
    # Split data
    print("\n4. Splitting data into train/validation/test sets...")
    train_df, val_df, test_df = stratified_train_val_test_split(df)
    print(f"   Train set: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Validation set: {len(val_df)} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test set: {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    # Prepare selected features with transformations
    print("\n5. Preparing selected features with transformations...")
    
    # Load original dataframe for final output
    df_original = pd.read_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/hospital_ops_updated.csv')
    
    # Prepare features using the selected features from analysis
    train_feat_dict, val_feat_dict, test_feat_dict, feature_names = prepare_selected_features(
        train_df, val_df, test_df, df_original
    )
    
    # Create feature matrices
    X_train = np.column_stack([train_feat_dict[col] for col in feature_names])
    X_val = np.column_stack([val_feat_dict[col] for col in feature_names])
    X_test = np.column_stack([test_feat_dict[col] for col in feature_names])
    y_train = train_df[target_col].values
    y_val = val_df[target_col].values
    y_test = test_df[target_col].values
    
    # Final check for any remaining NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e10, neginf=-1e10)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=1e10, neginf=-1e10)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e10, neginf=-1e10)
    
    # Remove multivariate outliers (Isolation Forest fit on train only)
    print("\n5b. Outlier detection and removal...")
    (
        X_train, y_train, train_df,
        X_val, y_val, val_df,
        X_test, y_test, test_df,
        outlier_stats,
    ) = remove_outliers_isolation_forest(
        X_train, y_train, X_val, y_val, X_test, y_test,
        train_df, val_df, test_df,
        target_col=target_col,
    )
    print(f"   Total rows removed: {sum(outlier_stats.values())}")
    
    print(f"\n   Target: {target_col} (dollars, no transform)")
    print(f"   Train mean: {y_train.mean():,.0f}, std: {y_train.std():,.0f}")
    
    print(f"\n   Final features for prediction: {len(feature_names)}")
    print(f"   Feature shape: {X_train.shape}")
    print(f"   Features: {feature_names}")
    
    # Train and evaluate models
    print("\n6. Training and evaluating all models...")
    results_df, best_models, test_predictions = train_and_evaluate_models(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    )
    
    # Get best model (lowest test sMAPE — aligned with tuning objective)
    best_model_row = results_df.loc[results_df['Test_MAPE'].idxmin()]
    best_model_name = best_model_row['Model']
    
    print(f"\n{'='*80}")
    print("BEST MODEL (Selected by lowest Test MAPE)")
    print(f"{'='*80}")
    print(f"  Model: {best_model_name}")
    print(f"  Test MAPE: {best_model_row['Test_MAPE']:.2f}% (primary metric)")
    print(f"  Test RMSE: {best_model_row['Test_RMSE']:.2f}")
    print(f"  Test MSE: {best_model_row['Test_MSE']:.2f}")
    print(f"  Test R²: {best_model_row['Test_R2']:.4f}")
    print(f"  Test Adjusted R²: {best_model_row['Test_Adjusted_R2']:.4f}")
    
    # Save performance metrics
    print("\n7. Saving results...")
    results_df.to_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/model_performance_results.csv', index=False)
    print(f"   Saved: model_performance_results.csv")
    
    # Get and save feature importance
    feature_importance_df = get_feature_importance(best_models, feature_names)
    feature_importance_df.to_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/feature_importance_results.csv', index=False)
    print(f"   Saved: feature_importance_results.csv")
    
    # Save test predictions for best model
    best_predictions = test_predictions[best_model_name]
    
    # Calculate sign prediction accuracy (positive vs negative Net Income)
    print(f"\n{'='*80}")
    print("SIGN PREDICTION ACCURACY (Positive vs Negative Net Income)")
    print(f"{'='*80}")
    
    actual_signs = np.sign(y_test)
    predicted_signs = np.sign(best_predictions)
    
    # Calculate accuracy
    sign_accuracy = np.mean(actual_signs == predicted_signs) * 100
    
    # Calculate confusion matrix components
    true_positives = np.sum((actual_signs > 0) & (predicted_signs > 0))
    true_negatives = np.sum((actual_signs < 0) & (predicted_signs < 0))
    false_positives = np.sum((actual_signs < 0) & (predicted_signs > 0))
    false_negatives = np.sum((actual_signs > 0) & (predicted_signs < 0))
    zeros_actual = np.sum(actual_signs == 0)
    zeros_predicted = np.sum(predicted_signs == 0)
    
    print(f"  Sign Prediction Accuracy: {sign_accuracy:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"    Actual Positive, Predicted Positive: {true_positives}")
    print(f"    Actual Negative, Predicted Negative: {true_negatives}")
    print(f"    Actual Negative, Predicted Positive: {false_positives}")
    print(f"    Actual Positive, Predicted Negative: {false_negatives}")
    if zeros_actual > 0:
        print(f"    Actual Zero: {zeros_actual}")
    if zeros_predicted > 0:
        print(f"    Predicted Zero: {zeros_predicted}")
    
    # Distribution statistics
    actual_positive_pct = np.sum(actual_signs > 0) / len(actual_signs) * 100
    actual_negative_pct = np.sum(actual_signs < 0) / len(actual_signs) * 100
    predicted_positive_pct = np.sum(predicted_signs > 0) / len(predicted_signs) * 100
    predicted_negative_pct = np.sum(predicted_signs < 0) / len(predicted_signs) * 100
    
    print(f"\n  Distribution:")
    print(f"    Actual:   {actual_positive_pct:.1f}% positive, {actual_negative_pct:.1f}% negative")
    print(f"    Predicted: {predicted_positive_pct:.1f}% positive, {predicted_negative_pct:.1f}% negative")
    
    # Add sign accuracy to results
    results_df['Sign_Prediction_Accuracy'] = None
    for model_name in results_df['Model']:
        if model_name in test_predictions:
            model_pred_signs = np.sign(test_predictions[model_name])
            model_sign_accuracy = np.mean(actual_signs == model_pred_signs) * 100
            results_df.loc[results_df['Model'] == model_name, 'Sign_Prediction_Accuracy'] = model_sign_accuracy
    
    # Re-save results with sign accuracy
    results_df.to_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/model_performance_results.csv', index=False)
    
    # Calculate errors
    absolute_error = np.abs(y_test - best_predictions)
    percentage_error = np.where(
        np.abs(y_test) + np.abs(best_predictions) > 0,
        200 * absolute_error / (np.abs(y_test) + np.abs(best_predictions)),
        0
    )
    
    # Create prediction results dataframe
    prediction_results = pd.DataFrame({
        'Actual_Net_Income': y_test,
        'Predicted_Net_Income': best_predictions,
        'Absolute_Error': absolute_error,
        'Percentage_Error': percentage_error,
        'Squared_Error': (y_test - best_predictions) ** 2
    })
    
    # Add index from test_df if available
    if hasattr(test_df, 'index'):
        prediction_results.index = test_df.index
    
    prediction_results.to_csv('test_set_predictions_and_errors.csv', index=True)
    print(f"   Saved: test_set_predictions_and_errors.csv")
    
    # Create test set with predictions and errors for error analysis
    print("\n8. Creating test set with predictions and errors...")
    
    # test_df already has all original columns from hospital_ops_updated.csv
    # (it was split from df which came from the filtered original dataset)
    test_set_output = test_df.copy()
    
    print(f"   Test set size: {len(test_set_output)} rows")
    print(f"   Test set columns: {len(test_set_output.columns)} columns")
    
    # Add predictions and errors to test set
    # best_predictions already contains test set predictions from the best model
    test_set_output['Predicted_Net_Income'] = best_predictions
    test_set_output['Prediction_Error'] = y_test - best_predictions
    
    # Save test set with all original columns plus predictions and errors
    test_set_output.to_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/full_dataset_with_predictions.csv', index=False)
    print(f"   Saved: full_dataset_with_predictions.csv")
    print(f"   Contains: {len(test_set_output)} test set rows")
    print(f"   Columns: All original columns from hospital_ops_updated.csv + 'Predicted_Net_Income' + 'Prediction_Error'")
    
    # Additional statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"\nTarget Variable ({target_col}) Statistics:")
    print(f"  Overall - Mean: {df[target_col].mean():.2f}, Std: {df[target_col].std():.2f}")
    print(f"  Train   - Mean: {train_df[target_col].mean():.2f}, Std: {train_df[target_col].std():.2f}")
    print(f"  Val     - Mean: {val_df[target_col].mean():.2f}, Std: {val_df[target_col].std():.2f}")
    print(f"  Test    - Mean: {test_df[target_col].mean():.2f}, Std: {test_df[target_col].std():.2f}")
    
    print("\n" + "=" * 80)
    print("Pipeline completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()

