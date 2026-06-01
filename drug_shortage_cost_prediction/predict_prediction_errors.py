"""
ML Pipeline for Predicting Prediction Error from filtered features
with Stratified Sampling and Multiple Regression Models
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, make_scorer
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
    return np.mean(200 * np.abs(y_true[mask] - y_pred[mask]) / denominator[mask])

def create_stratified_splits(df, target_col='Prediction_Error'):
    """
    Create stratified splits based on:
    Quantiles of Prediction_Error (4 quantiles)
    """
    # Create stratification labels based on Prediction_Error quantiles
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
    60% training, 20% validation, 20% testing
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
    
    # Handle single-sample strata - assign them proportionally (60/20/20)
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
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [10, 20, 30],
                'min_samples_leaf': [5, 10, 15]
            }
        },
        'RandomForestRegressor': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, 15],
                'min_samples_split': [10, 20],
                'min_samples_leaf': [5, 10]
            }
        },
        'GradientBoostingRegressor': {
            'model': GradientBoostingRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5],
                'min_samples_split': [10, 20],
                'subsample': [0.8, 0.9]
            }
        },
        'Ridge': {
            'model': Ridge(random_state=42),
            'params': {
                'alpha': [1.0, 10.0, 100.0, 1000.0, 10000.0]
            }
        },
        'Lasso': {
            'model': Lasso(random_state=42),
            'params': {
                'alpha': [1.0, 10.0, 100.0, 1000.0, 10000.0]
            }
        },
        'ElasticNet': {
            'model': ElasticNet(random_state=42),
            'params': {
                'alpha': [1.0, 10.0, 100.0, 1000.0],
                'l1_ratio': [0.3, 0.5, 0.7]
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
                'n_estimators': [50, 100],
                'max_depth': [5, 10, 15],
                'min_samples_split': [10, 20],
                'min_samples_leaf': [5, 10]
            }
        },
        'XGBRegressor': {
            'model': XGBRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8, 0.9],
                'colsample_bytree': [0.8, 0.9],
                'reg_alpha': [0.1, 1.0],
                'reg_lambda': [1.0, 10.0]
            }
        }
    }

def train_and_evaluate_models(X_train, y_train, X_val, y_val, X_test, y_test, feature_names):
    """Train multiple models with hyperparameter tuning and evaluate"""
    model_configs = get_model_configs()
    results = []
    best_models = {}
    test_predictions = {}
    
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
        
        # Grid search with cross-validation - use RMSE for scoring (primary metric)
        rmse_scorer = make_scorer(calculate_rmse, greater_is_better=False)
        
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=5, 
            scoring=rmse_scorer,
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
        
        # Calculate overfitting indicator (gap between train and test RMSE)
        overfitting_gap_rmse = train_rmse - test_rmse
        overfitting_gap_mape = train_mape - test_mape
        
        # Store results
        result = {
            'Model': model_name,
            'Best_Params': grid_search.best_params_,
            'Train_MSE': train_mse,
            'Train_RMSE': train_rmse,
            'Train_MAPE': train_mape,
            'Val_MSE': val_mse,
            'Val_RMSE': val_rmse,
            'Val_MAPE': val_mape,
            'Test_MSE': test_mse,
            'Test_RMSE': test_rmse,
            'Test_MAPE': test_mape,
            'Overfitting_Gap_RMSE': overfitting_gap_rmse,
            'Overfitting_Gap_MAPE': overfitting_gap_mape
        }
        results.append(result)
        
        print(f"\nBest Parameters: {grid_search.best_params_}")
        print(f"Train - RMSE: {train_rmse:.2f} (primary), MAPE: {train_mape:.2f}%, MSE: {train_mse:.2f}")
        print(f"Val   - RMSE: {val_rmse:.2f} (primary), MAPE: {val_mape:.2f}%, MSE: {val_mse:.2f}")
        print(f"Test  - RMSE: {test_rmse:.2f} (primary), MAPE: {test_mape:.2f}%, MSE: {test_mse:.2f}")
        print(f"Overfitting Gap (Train-Test RMSE): {overfitting_gap_rmse:.2f}")
        print(f"Overfitting Gap (Train-Test MAPE): {overfitting_gap_mape:.2f}%")
        
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

def prepare_selected_features(train_df, val_df, test_df):
    """
    Prepare features based on nonlinear_correlations_errors.csv
    Uses the filtered features with their optimal transformations from error prediction analysis.
    """
    # Load feature lists from CSV files
    print("   Loading feature lists from error analysis files...")
    try:
        nonlinear_df = pd.read_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/nonlinear_correlations_errors.csv')
        final_features = nonlinear_df['Feature'].tolist()
        print(f"   Loaded {len(final_features)} features from nonlinear_correlations_errors.csv")
    except FileNotFoundError:
        print("   WARNING: nonlinear_correlations_errors.csv not found. Using default features.")
        final_features = [
            'Total Current Assets',
            'Predicted_Net_Income',
            'Total Current Liabilities',
            'Accounts Receivable',
            'Hospital Number of Beds For Adults & Peds',
            'Total Assets',
            'Total Fixed Assets',
        ]
    
    # Define transformations for specific features based on nonlinear_correlations_errors.csv
    transformations = {
        'Total Current Assets': 'square',
        'Predicted_Net_Income': 'square',
        'Total Current Liabilities': 'square',
        'Accounts Receivable': 'none',
        'Hospital Number of Beds For Adults & Peds': 'square',
        'Total Assets': 'sqrt',
        'Total Fixed Assets': 'square'
    }
    
    # Prepare feature dictionaries
    train_feat_dict = {}
    val_feat_dict = {}
    test_feat_dict = {}
    feature_names = []
    
    print("   Applying transformations to features...")
    for feat in final_features:
        # Skip Net Income (data leakage)
        if feat == 'Net Income':
            print(f"   Skipping 'Net Income' (data leakage)")
            continue
        
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
            train_vals = np.sqrt(np.maximum(train_vals, 0))
            val_vals = np.sqrt(np.maximum(val_vals, 0))
            test_vals = np.sqrt(np.maximum(test_vals, 0))
        elif transform_type == 'square':
            train_vals = train_vals ** 2
            val_vals = val_vals ** 2
            test_vals = test_vals ** 2
        elif transform_type == 'none':
            # No transformation
            pass
        
        train_feat_dict[feat] = train_vals
        val_feat_dict[feat] = val_vals
        test_feat_dict[feat] = test_vals
        feature_names.append(feat)
    
    print(f"\n   Prepared {len(feature_names)} features")
    return train_feat_dict, val_feat_dict, test_feat_dict, feature_names

def main():
    """Main execution function"""
    print("=" * 80)
    print("Prediction Error Prediction - ML Pipeline")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/full_dataset_with_predictions_profit_binary.csv')
    print(f"   Loaded {len(df)} rows")
    
    # Identify target and features
    target_col = 'Prediction_Error'
    exclude_cols = ['Provider CCN', 'Net Income', target_col, 'Unnamed: 0']
    
    print(f"\n   Target variable: {target_col}")
    print(f"   Note: 'Net Income' excluded from features (data leakage)")
    print(f"   Note: Features and transformations loaded from nonlinear_correlations_errors.csv")
    
    # Check for missing values
    print("\n2. Data quality check...")
    if target_col not in df.columns:
        print(f"   ERROR: Target column '{target_col}' not found in dataset.")
        print(f"   Please ensure full_dataset_with_predictions_profit_binary.csv contains 'Prediction_Error' column.")
        return
    
    # Remove rows with missing target
    df = df.dropna(subset=[target_col])
    print(f"   After removing missing target values: {len(df)} rows")
    
    # Create stratified splits based on Prediction_Error quantiles
    print("\n3. Creating stratified splits based on Prediction_Error quantiles...")
    df = create_stratified_splits(df, target_col=target_col)
    print(f"   Number of unique strata: {df['stratum'].nunique()}")
    print(f"   Strata distribution:\n{df['stratum'].value_counts().head(10)}")
    
    # Split data (60% train, 20% val, 20% test)
    print("\n4. Splitting data into train/validation/test sets (60/20/20)...")
    train_df, val_df, test_df = stratified_train_val_test_split(df)
    print(f"   Train set: {len(train_df)} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Validation set: {len(val_df)} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test set: {len(test_df)} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    # Prepare selected features from error analysis
    print("\n5. Preparing selected features from error analysis...")
    train_feat_dict, val_feat_dict, test_feat_dict, feature_names = prepare_selected_features(
        train_df, val_df, test_df
    )
    
    if len(feature_names) == 0:
        print("   ERROR: No features found. Please run test_nonlinear_correlations_errors.py first.")
        return
    
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
    print(f"   Features: {feature_names[:10]}..." if len(feature_names) > 10 else f"   Features: {feature_names}")
    
    # Train and evaluate models
    print("\n6. Training and evaluating all models...")
    results_df, best_models, test_predictions = train_and_evaluate_models(
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names
    )
    
    # Get best model (lowest RMSE - primary metric)
    best_model_row = results_df.loc[results_df['Test_RMSE'].idxmin()]
    best_model_name = best_model_row['Model']
    
    print(f"\n{'='*80}")
    print("BEST MODEL (Selected by lowest Test RMSE)")
    print(f"{'='*80}")
    print(f"  Model: {best_model_name}")
    print(f"  Test RMSE: {best_model_row['Test_RMSE']:.2f} (primary metric)")
    print(f"  Test MAPE: {best_model_row['Test_MAPE']:.2f}%")
    print(f"  Test MSE: {best_model_row['Test_MSE']:.2f}")
    
    # Save performance metrics
    print("\n7. Saving results...")
    results_df.to_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/model_performance_results_errors.csv', index=False)
    print(f"   Saved: model_performance_results_errors.csv")
    
    # Get and save feature importance
    feature_importance_df = get_feature_importance(best_models, feature_names)
    feature_importance_df.to_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/feature_importance_results_errors.csv', index=False)
    print(f"   Saved: feature_importance_results_errors.csv")
    
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

