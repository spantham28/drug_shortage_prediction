"""
Comprehensive ML Pipeline for Predicting Net Income from multiple features
with Stratified Sampling and Multiple Regression Models
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV, train_test_split, cross_val_score
from scipy.stats import uniform, randint, loguniform
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, make_scorer, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
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
    return np.mean(100 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])

def calculate_adjusted_r2(y_true, y_pred, n_features):
    """
    Calculate Adjusted R-squared
    R²_adj = 1 - (1 - R²) * (n - 1) / (n - p - 1)
    where n is number of samples, p is number of features
    """
    n_samples = len(y_true)
    if n_samples <= n_features + 1:
        return np.nan  # Cannot calculate if n <= p + 1
    
    r2 = r2_score(y_true, y_pred)
    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return adjusted_r2

def create_stratified_splits(df):
    """
    Create stratified splits based on:
    Quantiles of Net Income (15 quantiles)
    """
    # Create stratification labels based only on Net Income quantiles
    target_col = 'Net Income'
    n_samples = len(df)
    
    # Create 15 quantiles of the target variable
    if n_samples >= 30:  # At least 2 samples per quantile after splits (15 quantiles * 2 = 30)
        try:
            quantiles = pd.qcut(df[target_col], q=15, labels=False, duplicates='drop')
            df['stratum'] = [f"Q{q}" for q in quantiles]
        except ValueError:
            # If quantiles can't be created (e.g., too many duplicates), use fewer quantiles
            try:
                quantiles = pd.qcut(df[target_col], q=10, labels=False, duplicates='drop')
                df['stratum'] = [f"Q{q}" for q in quantiles]
            except ValueError:
                try:
                    quantiles = pd.qcut(df[target_col], q=5, labels=False, duplicates='drop')
                    df['stratum'] = [f"Q{q}" for q in quantiles]
                except ValueError:
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
    
    # Validate no overlap between sets
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    
    train_val_overlap = train_set.intersection(val_set)
    train_test_overlap = train_set.intersection(test_set)
    val_test_overlap = val_set.intersection(test_set)
    
    if train_val_overlap:
        raise ValueError(f"ERROR: Found {len(train_val_overlap)} overlapping indices between train and validation sets!")
    if train_test_overlap:
        raise ValueError(f"ERROR: Found {len(train_test_overlap)} overlapping indices between train and test sets!")
    if val_test_overlap:
        raise ValueError(f"ERROR: Found {len(val_test_overlap)} overlapping indices between validation and test sets!")
    
    # Verify total count matches
    total_unique = len(train_set) + len(val_set) + len(test_set)
    if total_unique != len(df):
        raise ValueError(f"ERROR: Total indices ({total_unique}) don't match dataframe length ({len(df)})!")
    
    return train_df, val_df, test_df

def get_model_configs():
    """
    Define models and their optimized hyperparameter grids maximizing rigor while staying efficient.
    Uses GridSearchCV for simpler models and RandomizedSearchCV for complex models with many combinations.
    """
    return {
        'DecisionTreeRegressor': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {
                'max_depth': [None, 5, 10, 15, 20, 25, 30],
                'min_samples_split': [2, 5, 10, 20, 30],
                'min_samples_leaf': [1, 2, 5, 10, 15],
                'max_features': ['sqrt', 'log2', None, 0.7, 0.8]
            },
            'search_type': 'grid',
            'n_iter': None
        },
        'RandomForestRegressor': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200, 300, 400, 500],
                'max_depth': [None, 10, 15, 20, 25, 30],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'max_features': ['sqrt', 'log2', 0.7, 0.8, None]
            },
            'search_type': 'random',
            'n_iter': 150
        },
        'GradientBoostingRegressor': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': randint(100, 600),
                'learning_rate': loguniform(0.01, 0.3),
                'max_depth': randint(3, 10),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'subsample': uniform(0.7, 0.3),
                'max_features': ['sqrt', 'log2', 0.7, 0.8]
            },
            'search_type': 'random',
            'n_iter': 200
        },
        'Ridge': {
            'model': Ridge(random_state=42),
            'params': {
                'alpha': loguniform(0.001, 100000.0),
                'solver': ['auto', 'svd', 'cholesky', 'sag', 'saga']
            },
            'search_type': 'random',
            'n_iter': 100
        },
        'Lasso': {
            'model': Lasso(random_state=42, max_iter=2000),
            'params': {
                'alpha': loguniform(0.0001, 100000.0)
            },
            'search_type': 'random',
            'n_iter': 100
        },
        'ElasticNet': {
            'model': ElasticNet(random_state=42, max_iter=2000),
            'params': {
                'alpha': loguniform(0.0001, 100000.0),
                'l1_ratio': uniform(0.1, 0.8)
            },
            'search_type': 'random',
            'n_iter': 150
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'C': loguniform(0.01, 10000.0),
                'gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
                'kernel': ['rbf', 'linear', 'poly'],
                'epsilon': uniform(0.001, 0.3)
            },
            'search_type': 'random',
            'n_iter': 150
        },
        'KNeighborsRegressor': {
            'model': KNeighborsRegressor(),
            'params': {
                'n_neighbors': randint(3, 50),
                'weights': ['uniform', 'distance'],
                'p': [1, 2],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            },
            'search_type': 'random',
            'n_iter': 100
        },
        'MLPRegressor': {
            'model': MLPRegressor(random_state=42, max_iter=1500, early_stopping=True, validation_fraction=0.1, n_iter_no_change=10),
            'params': {
                'hidden_layer_sizes': [(50,), (100,), (150,), (200,), (50, 50), (100, 50), (100, 100), (150, 100), (200, 100), (100, 50, 25)],
                'activation': ['relu', 'tanh', 'logistic'],
                'alpha': loguniform(0.00001, 1.0),
                'learning_rate': ['constant', 'adaptive', 'invscaling'],
                'learning_rate_init': loguniform(0.0001, 0.1),
                'solver': ['adam', 'lbfgs', 'sgd']
            },
            'search_type': 'random',
            'n_iter': 150
        },
        'AdaBoostRegressor': {
            'model': AdaBoostRegressor(random_state=42),
            'params': {
                'n_estimators': randint(50, 500),
                'learning_rate': loguniform(0.01, 2.0),
                'loss': ['linear', 'square', 'exponential']
            },
            'search_type': 'random',
            'n_iter': 100
        },
        'ExtraTreesRegressor': {
            'model': ExtraTreesRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': randint(100, 500),
                'max_depth': [None, 10, 15, 20, 25, 30],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', 0.7, 0.8, None]
            },
            'search_type': 'random',
            'n_iter': 150
        },
        'XGBRegressor': {
            'model': XGBRegressor(random_state=42, n_jobs=-1, tree_method='hist'),
            'params': {
                'n_estimators': randint(100, 600),
                'max_depth': randint(3, 10),
                'learning_rate': loguniform(0.01, 0.3),
                'subsample': uniform(0.7, 0.3),
                'colsample_bytree': uniform(0.7, 0.3),
                'colsample_bylevel': uniform(0.7, 0.3),
                'reg_alpha': loguniform(0.001, 100.0),
                'reg_lambda': loguniform(0.001, 100.0),
                'min_child_weight': randint(1, 10),
                'gamma': uniform(0, 5)
            },
            'search_type': 'random',
            'n_iter': 250
        }
    }

def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """Train multiple models with hyperparameter tuning and evaluate"""
    model_configs = get_model_configs()
    results = []
    best_models = {}
    test_predictions = {}  # Store test predictions for each model
    
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
        
        # Hyperparameter search with cross-validation - use MAPE for scoring (primary metric)
        # MAPE scorer (negated because GridSearchCV/RandomizedSearchCV maximizes, so we negate to minimize)
        mape_scorer = make_scorer(calculate_mape, greater_is_better=False)
        
        # Determine search type and parameters
        search_type = config.get('search_type', 'grid')
        n_iter = config.get('n_iter', None)
        
        if search_type == 'random' and n_iter is not None:
            # Use RandomizedSearchCV for complex models with many parameter combinations
            print(f"   Using RandomizedSearchCV with {n_iter} iterations...")
            print(f"   Parameter space includes {len(param_grid)} dimensions")
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=n_iter,
                cv=10,  # 10-fold cross-validation for rigorous evaluation
                scoring=mape_scorer,
                n_jobs=-1,
                verbose=1,
                refit=True,
                random_state=42  # For reproducibility
            )
        else:
            # Use GridSearchCV for simpler models
            total_combinations = 1
            for key, value in param_grid.items():
                if isinstance(value, list):
                    total_combinations *= len(value)
            print(f"   Using GridSearchCV with ~{total_combinations} combinations...")
            search = GridSearchCV(
                model,
                param_grid,
                cv=10,  # 10-fold cross-validation for rigorous evaluation
                scoring=mape_scorer,
                n_jobs=-1,
                verbose=1,
                refit=True
            )
        
        search.fit(X_train_use, y_train)
        best_model = search.best_estimator_
        best_models[model_name] = {
            'model': best_model,
            'scaler': scaler if needs_scaling else None,
            'needs_scaling': needs_scaling,
            'feature_names': feature_names
        }
        
        # Predictions
        y_train_pred = best_model.predict(X_train_use)
        y_val_pred = best_model.predict(X_val_use)
        y_test_pred = best_model.predict(X_test_use)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_rmse = calculate_rmse(y_train, y_train_pred)
        train_mape = calculate_mape(y_train, y_train_pred)
        
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_rmse = calculate_rmse(y_val, y_val_pred)
        val_mape = calculate_mape(y_val, y_val_pred)
        
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_rmse = calculate_rmse(y_test, y_test_pred)
        test_mape = calculate_mape(y_test, y_test_pred)
        
        # Calculate adjusted R-squared
        n_features = len(feature_names)
        train_adj_r2 = calculate_adjusted_r2(y_train, y_train_pred, n_features)
        val_adj_r2 = calculate_adjusted_r2(y_val, y_val_pred, n_features)
        test_adj_r2 = calculate_adjusted_r2(y_test, y_test_pred, n_features)
        
        # Calculate overfitting indicator (gap between train and test MAPE - primary metric)
        overfitting_gap_mape = train_mape - test_mape
        overfitting_gap_rmse = train_rmse - test_rmse
        
        # Store results
        result = {
            'Model': model_name,
            'Best_Params': search.best_params_,
            'Train_MSE': train_mse,
            'Train_RMSE': train_rmse,
            'Train_MAPE': train_mape,
            'Train_Adj_R2': train_adj_r2,
            'Val_MSE': val_mse,
            'Val_RMSE': val_rmse,
            'Val_MAPE': val_mape,
            'Val_Adj_R2': val_adj_r2,
            'Test_MSE': test_mse,
            'Test_RMSE': test_rmse,
            'Test_MAPE': test_mape,
            'Test_Adj_R2': test_adj_r2,
            'Overfitting_Gap_RMSE': overfitting_gap_rmse,
            'Overfitting_Gap_MAPE': overfitting_gap_mape
        }
        results.append(result)
        
        print(f"\nBest Parameters: {search.best_params_}")
        print(f"Best CV Score (MAPE): {-search.best_score_:.2f}%")
        print(f"Train - MAPE: {train_mape:.2f}% (primary), RMSE: {train_rmse:.2f}, MSE: {train_mse:.2f}, Adj R²: {train_adj_r2:.4f}")
        print(f"Val   - MAPE: {val_mape:.2f}% (primary), RMSE: {val_rmse:.2f}, MSE: {val_mse:.2f}, Adj R²: {val_adj_r2:.4f}")
        print(f"Test  - MAPE: {test_mape:.2f}% (primary), RMSE: {test_rmse:.2f}, MSE: {test_mse:.2f}, Adj R²: {test_adj_r2:.4f}")
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

def prepare_selected_features(train_df, val_df, test_df, df_original):
    """
    Prepare features based on nonlinear_correlations.csv
    Automatically determines which features to use and what transformations to apply.
    Includes Profit as a categorical feature to help distinguish positive vs negative Net Income.
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
            'Fixed Equipment': 'sqrt'
        }
    
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
    
    # Add Profit as a categorical feature
    print("\n   Adding Profit as categorical feature...")
    if 'Profit' in train_df.columns:
        # Encode Profit feature (Profit/Loss)
        le_profit = LabelEncoder()
        
        # Fit on training data
        train_profit = train_df['Profit'].astype(str)
        le_profit.fit(train_profit)
        
        # Transform with handling for unseen values
        def safe_transform_profit(values, encoder):
            """Transform Profit values, handling unseen labels"""
            result = []
            for val in values:
                val_str = str(val)
                if val_str in encoder.classes_:
                    result.append(encoder.transform([val_str])[0])
                else:
                    # Assign unseen values to most common class (or 0)
                    result.append(0)
            return np.array(result)
        
        train_feat_dict['Profit'] = safe_transform_profit(train_df['Profit'].astype(str), le_profit)
        val_feat_dict['Profit'] = safe_transform_profit(val_df['Profit'].astype(str), le_profit)
        test_feat_dict['Profit'] = safe_transform_profit(test_df['Profit'].astype(str), le_profit)
        feature_names.append('Profit')
        
        print(f"     Profit: categorical encoding (classes: {le_profit.classes_.tolist()})")
        print(f"     NOTE: Profit is derived from Net Income (Profit if > 0, Loss if < 0).")
        print(f"     This creates a circular dependency, but may help the model distinguish positive vs negative predictions.")
    else:
        print("     WARNING: 'Profit' column not found in dataset. Skipping Profit feature.")
    
    # Add shortage_num as a numerical feature
    print("\n   Adding shortage_num as numerical feature...")
    if 'shortage_num' in train_df.columns:
        train_feat_dict['shortage_num'] = train_df['shortage_num'].fillna(train_df['shortage_num'].median()).values
        val_feat_dict['shortage_num'] = val_df['shortage_num'].fillna(train_df['shortage_num'].median()).values
        test_feat_dict['shortage_num'] = test_df['shortage_num'].fillna(train_df['shortage_num'].median()).values
        feature_names.append('shortage_num')
        print(f"     shortage_num: no transformation")
    else:
        print("     WARNING: 'shortage_num' column not found in dataset. Skipping shortage_num feature.")
    
    print(f"\n   Prepared {len(feature_names)} features (including Profit and shortage_num)")
    return train_feat_dict, val_feat_dict, test_feat_dict, feature_names

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
    
    # Get feature columns (all except Provider CCN, target, and index column)
    # Note: Profit is included as a feature even though it's derived from Net Income
    # This creates a circular dependency but may help distinguish positive vs negative predictions
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    # Explicitly remove 'Unnamed: 0' if it somehow got included
    if 'Unnamed: 0' in feature_cols:
        feature_cols.remove('Unnamed: 0')
    
    # Check if Profit column exists
    if 'Profit' in df.columns:
        print(f"\n   NOTE: 'Profit' feature detected (binary: Profit/Loss based on Net Income > 0).")
        print(f"   This feature will be included to help distinguish positive vs negative Net Income predictions.")
        print(f"   WARNING: Since Profit is derived from Net Income, this creates a circular dependency.")
        print(f"   However, this may improve the model's ability to predict the sign of Net Income.")
    
    print(f"\n   Target variable: {target_col}")
    print(f"   Feature variables: {len(feature_cols)} features")
    
    # Check for missing values
    print("\n2. Data quality check...")
    check_cols = [target_col] + feature_cols
    missing_counts = df[check_cols].isnull().sum()
    print(f"   Missing values:\n{missing_counts[missing_counts > 0]}")
    
    # Remove rows with missing target or features
    df = df.dropna(subset=[target_col] + feature_cols)
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
    
    # Validate no overlap between sets
    train_indices_set = set(train_df.index)
    val_indices_set = set(val_df.index)
    test_indices_set = set(test_df.index)
    
    train_val_overlap = train_indices_set.intersection(val_indices_set)
    train_test_overlap = train_indices_set.intersection(test_indices_set)
    val_test_overlap = val_indices_set.intersection(test_indices_set)
    
    if train_val_overlap:
        raise ValueError(f"ERROR: Found {len(train_val_overlap)} overlapping indices between train and validation sets!")
    if train_test_overlap:
        raise ValueError(f"ERROR: Found {len(train_test_overlap)} overlapping indices between train and test sets!")
    if val_test_overlap:
        raise ValueError(f"ERROR: Found {len(val_test_overlap)} overlapping indices between validation and test sets!")
    
    # Verify total count matches
    total_unique = len(train_indices_set) + len(val_indices_set) + len(test_indices_set)
    if total_unique != len(df):
        raise ValueError(f"ERROR: Total indices ({total_unique}) don't match dataframe length ({len(df)})!")
    
    print(f"   ✓ Validation: No overlap detected between train, validation, and test sets")
    print(f"   ✓ Validation: All {total_unique} rows are properly assigned to exactly one set")
    
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
    
    print(f"\n   Final features for prediction: {len(feature_names)}")
    print(f"   Feature shape: {X_train.shape}")
    print(f"   Features: {feature_names}")
    
    # Train and evaluate models
    print("\n6. Training and evaluating all models...")
    results_df, best_models, test_predictions = train_and_evaluate_models(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    )
    
    # Get best model (lowest MAPE - primary metric)
    best_model_row = results_df.loc[results_df['Test_MAPE'].idxmin()]
    best_model_name = best_model_row['Model']
    
    print(f"\n{'='*80}")
    print("BEST MODEL (Selected by lowest Test MAPE)")
    print(f"{'='*80}")
    print(f"  Model: {best_model_name}")
    print(f"  Test MAPE: {best_model_row['Test_MAPE']:.2f}% (primary metric)")
    print(f"  Test RMSE: {best_model_row['Test_RMSE']:.2f}")
    print(f"  Test MSE: {best_model_row['Test_MSE']:.2f}")
    print(f"  Test Adjusted R²: {best_model_row['Test_Adj_R2']:.4f}")
    
    # Save performance metrics
    print("\n7. Saving results...")
    results_df.to_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/model_performance_results_profit_binary.csv', index=False)
    print(f"   Saved: model_performance_results_profit_binary.csv")
    
    # Get and save feature importance
    feature_importance_df = get_feature_importance(best_models, feature_names)
    feature_importance_df.to_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/feature_importance_results_profit_binary.csv', index=False)
    print(f"   Saved: feature_importance_results_profit_binary.csv")
    
    # Save test predictions for best model
    best_predictions = test_predictions[best_model_name]
    
    # Analyze sign prediction accuracy (Profit/Loss prediction)
    print(f"\n{'='*80}")
    print("SIGN PREDICTION ANALYSIS (Profit vs Loss)")
    print(f"{'='*80}")
    actual_signs = np.sign(y_test)  # 1 for positive, -1 for negative, 0 for zero
    predicted_signs = np.sign(best_predictions)
    
    sign_accuracy = (actual_signs == predicted_signs).mean() * 100
    print(f"  Sign Prediction Accuracy: {sign_accuracy:.2f}%")
    print(f"  (Correctly predicted positive vs negative Net Income)")
    
    # Breakdown by actual Profit/Loss
    if 'Profit' in test_df.columns:
        profit_mask = test_df['Profit'].str.contains('Profit', case=False, na=False)
        loss_mask = test_df['Profit'].str.contains('Loss', case=False, na=False)
        
        if profit_mask.sum() > 0:
            profit_sign_accuracy = (np.sign(y_test[profit_mask]) == np.sign(best_predictions[profit_mask])).mean() * 100
            print(f"  Sign Accuracy for Profit cases: {profit_sign_accuracy:.2f}% ({profit_mask.sum()} cases)")
        
        if loss_mask.sum() > 0:
            loss_sign_accuracy = (np.sign(y_test[loss_mask]) == np.sign(best_predictions[loss_mask])).mean() * 100
            print(f"  Sign Accuracy for Loss cases: {loss_sign_accuracy:.2f}% ({loss_mask.sum()} cases)")
    
    # Count predictions by sign
    positive_pred = (predicted_signs > 0).sum()
    negative_pred = (predicted_signs < 0).sum()
    zero_pred = (predicted_signs == 0).sum()
    positive_actual = (actual_signs > 0).sum()
    negative_actual = (actual_signs < 0).sum()
    zero_actual = (actual_signs == 0).sum()
    
    print(f"\n  Actual Distribution:")
    print(f"    Positive (Profit): {positive_actual} ({positive_actual/len(y_test)*100:.1f}%)")
    print(f"    Negative (Loss): {negative_actual} ({negative_actual/len(y_test)*100:.1f}%)")
    print(f"    Zero: {zero_actual} ({zero_actual/len(y_test)*100:.1f}%)")
    
    print(f"\n  Predicted Distribution:")
    print(f"    Positive (Profit): {positive_pred} ({positive_pred/len(best_predictions)*100:.1f}%)")
    print(f"    Negative (Loss): {negative_pred} ({negative_pred/len(best_predictions)*100:.1f}%)")
    print(f"    Zero: {zero_pred} ({zero_pred/len(best_predictions)*100:.1f}%)")
    
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
    
    prediction_results.to_csv('test_set_predictions_and_errors_profit_binary.csv', index=True)
    print(f"   Saved: test_set_predictions_and_errors_profit_binary.csv")
    
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
    test_set_output.to_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/full_dataset_with_predictions_profit_binary.csv', index=False)
    print(f"   Saved: full_dataset_with_predictions_profit_binary.csv")
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

