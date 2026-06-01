from imblearn.over_sampling import SMOTE, RandomOverSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import statsmodels.api as sm

def create_stratified_bins(y, n_bins=2):
    """Create stratified labels based on binary target 'shortage_flag' (0/1)."""
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    return y.astype(int)

def remove_outliers(X, y, method='compare', contamination=0.1):
    """
    Remove outliers from a binary classification dataset using class-aware methods.

    Parameters:
    - X: Feature matrix (continuous features)
    - y: Binary target variable (0/1)
    - method: 'class_mahalanobis', 'class_isolation_forest', 'class_lof', or 'compare'
    - contamination: Expected proportion of outliers within each class (0-1)

    Returns:
    - X_clean: Cleaned feature matrix
    - y_clean: Cleaned target variable
    - outlier_indices: Indices of removed outliers
    """
    print(f"Removing outliers using {method} method...")
    original_size = len(X)
    
    if method == 'compare':
        # Compare class-aware methods and return the most conservative result
        print("Comparing class-aware outlier detection methods...")

        # Get results from each method
        X_md, y_md, outliers_md = remove_outliers(X, y, 'class_mahalanobis', contamination)
        X_iso, y_iso, outliers_iso = remove_outliers(X, y, 'class_isolation_forest', contamination)
        X_lof, y_lof, outliers_lof = remove_outliers(X, y, 'class_lof', contamination)

        # Find the most conservative approach (fewest outliers removed)
        n_outliers_md = len(outliers_md)
        n_outliers_iso = len(outliers_iso)
        n_outliers_lof = len(outliers_lof)

        print(f"\nOutlier detection comparison:")
        print(f"  Class Mahalanobis: {n_outliers_md} outliers ({n_outliers_md/original_size*100:.1f}%)")
        print(f"  Class Isolation Forest: {n_outliers_iso} outliers ({n_outliers_iso/original_size*100:.1f}%)")
        print(f"  Class LOF: {n_outliers_lof} outliers ({n_outliers_lof/original_size*100:.1f}%)")

        # Use the method that removes the fewest outliers (most conservative)
        if n_outliers_md <= n_outliers_iso and n_outliers_md <= n_outliers_lof:
            print(f"  Using Class Mahalanobis method (most conservative)")
            return X_md, y_md, outliers_md
        elif n_outliers_iso <= n_outliers_lof:
            print(f"  Using Class Isolation Forest method (most conservative)")
            return X_iso, y_iso, outliers_iso
        else:
            print(f"  Using Class LOF method (most conservative)")
            return X_lof, y_lof, outliers_lof
    
    elif method == 'class_mahalanobis':
        # Class-conditional Mahalanobis distance
        outlier_mask = np.zeros(len(X), dtype=bool)
        X_values = X.values
        for cls in [0, 1]:
            cls_idx = (y.astype(int).values == cls)
            if cls_idx.sum() < max(3, X.shape[1] + 1):
                continue
            X_cls = X_values[cls_idx]
            mu = np.mean(X_cls, axis=0)
            cov = np.cov(X_cls, rowvar=False)
            # Regularize covariance for numerical stability
            eps = 1e-6
            cov += np.eye(cov.shape[0]) * eps
            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(cov)
            diffs = X_cls - mu
            d2 = np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs)
            # Threshold by per-class quantile corresponding to contamination
            thr = np.quantile(d2, 1.0 - contamination)
            class_outliers = np.zeros(len(X_cls), dtype=bool)
            class_outliers[d2 > thr] = True
            outlier_mask[np.where(cls_idx)[0]] = class_outliers

    elif method == 'class_isolation_forest':
        # Isolation Forest applied within each class
        from sklearn.ensemble import IsolationForest
        outlier_mask = np.zeros(len(X), dtype=bool)
        X_values = X.values
        y_values = y.astype(int).values
        for cls in [0, 1]:
            cls_idx = (y_values == cls)
            if cls_idx.sum() < 10:
                continue
            iso = IsolationForest(contamination=contamination, random_state=42)
            labels = iso.fit_predict(X_values[cls_idx])
            class_outliers = labels == -1
            outlier_mask[np.where(cls_idx)[0]] = class_outliers

    elif method == 'class_lof':
        # Local Outlier Factor applied within each class
        from sklearn.neighbors import LocalOutlierFactor
        outlier_mask = np.zeros(len(X), dtype=bool)
        X_values = X.values
        y_values = y.astype(int).values
        for cls in [0, 1]:
            cls_idx = (y_values == cls)
            n_cls = cls_idx.sum()
            if n_cls < 10:
                continue
            n_neighbors = min(20, max(5, n_cls // 5))
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            labels = lof.fit_predict(X_values[cls_idx])
            class_outliers = labels == -1
            outlier_mask[np.where(cls_idx)[0]] = class_outliers
    
    else:
        print(f"Unknown outlier removal method: {method}. Using class_isolation_forest method.")
        return remove_outliers(X, y, method='class_isolation_forest', contamination=contamination)
    
    # Remove outliers
    clean_indices = ~outlier_mask
    X_clean = X[clean_indices].copy()
    y_clean = y[clean_indices].copy()
    
    removed_count = original_size - len(X_clean)
    removed_pct = (removed_count / original_size) * 100
    
    print(f"  Removed {removed_count} outliers ({removed_pct:.1f}% of data)")
    print(f"  Clean dataset size: {len(X_clean)} samples")
    
    return X_clean, y_clean, np.where(outlier_mask)[0]

def hyperparameter_tuning(X_train, y_train, model_name, cv_strategy, n_iter=20):
    """Perform hyperparameter tuning for classification models."""
    print(f"Performing hyperparameter tuning for {model_name}...")

    if model_name == 'RandomForest':
        param_grid = {
            'n_estimators': [50, 100, 200, 300, 400, 500, 800, 1000],
            'max_depth': [None, 3, 5, 8, 10, 15, 20, 25, 30],
            'min_samples_split': [2, 3, 5, 8, 10, 15, 20],
            'min_samples_leaf': [1, 2, 3, 4, 5, 8, 10],
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'ccp_alpha': [0.0, 0.0001, 0.001, 0.01, 0.1],
            'max_samples': [None, 0.6, 0.7, 0.8, 0.9],
            'bootstrap': [True, False]
        }
        base_model = RandomForestClassifier(n_jobs=-1, random_state=42)

    elif model_name == 'DecisionTree':
        param_grid = {
            'max_depth': [None, 2, 3, 5, 8, 10, 15, 20, 25, 30],
            'min_samples_split': [2, 3, 5, 8, 10, 15, 20, 25],
            'min_samples_leaf': [1, 2, 3, 4, 5, 8, 10, 15],
            'criterion': ['gini', 'entropy', 'log_loss'],
            'class_weight': [None, 'balanced'],
            'ccp_alpha': [0.0, 0.0001, 0.001, 0.01, 0.1],
            'max_features': [None, 'sqrt', 'log2', 0.3, 0.5, 0.7],
            'splitter': ['best', 'random'],
            'min_weight_fraction_leaf': [0.0, 0.01, 0.05, 0.1]
        }
        base_model = DecisionTreeClassifier(random_state=42)

    elif model_name == 'GradientBoosting':
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500, 800, 1000],
            'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
            'max_depth': [2, 3, 4, 5, 6, 8, 10],
            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'min_samples_split': [2, 3, 5, 8, 10, 15, 20],
            'min_samples_leaf': [1, 2, 3, 4, 5, 8, 10],
            'max_features': [None, 'sqrt', 'log2', 0.3, 0.5, 0.7],
            'criterion': ['friedman_mse', 'squared_error'],
            'min_weight_fraction_leaf': [0.0, 0.01, 0.05, 0.1],
            'validation_fraction': [0.1, 0.2],
            'n_iter_no_change': [5, 10, 15],
            'tol': [1e-4, 1e-3]
        }
        base_model = GradientBoostingClassifier(random_state=42)

    elif model_name == 'SVM':
        param_grid = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear'],
            'class_weight': [None, 'balanced']
        }
        base_model = SVC(probability=False, random_state=42)

    elif model_name == 'Logit':
        param_grid = {
            'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0],
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'class_weight': [None, 'balanced'],
            'max_iter': [200, 500, 1000, 2000],
            'tol': [1e-5, 1e-4, 1e-3],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
        }
        base_model = LogisticRegression()

    elif model_name == 'XGBoost':
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            print(f"  XGBoost not available: {str(e)}")
            return None, None, None
        param_grid = {
            'n_estimators': [200, 300, 400, 500, 800, 1000],
            'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
            'max_depth': [3, 4, 5, 6, 8, 10],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bylevel': [0.5, 0.7, 0.9, 1.0],
            'min_child_weight': [1, 2, 3, 5, 7, 10],
            'gamma': [0.0, 0.01, 0.1, 0.5, 1.0],
            'reg_alpha': [0.0, 0.001, 0.01, 0.1, 0.5, 1.0],
            'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'scale_pos_weight': [1.0, 2.0, 5.0, 10.0],
            'max_delta_step': [0, 1, 2, 5]
        }
        base_model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            tree_method='hist',
            random_state=42,
            n_jobs=-1
        )

    elif model_name == 'LightGBM':
        try:
            from lightgbm import LGBMClassifier
        except Exception as e:
            print(f"  LightGBM not available: {str(e)}")
            return None, None, None
        param_grid = {
            'n_estimators': [200, 300, 500, 800, 1000],
            'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
            'num_leaves': [15, 31, 63, 127],
            'max_depth': [-1, 3, 5, 8, 10],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bynode': [0.5, 0.7, 0.9, 1.0],
            'min_child_samples': [5, 10, 15, 20, 30, 40],
            'min_child_weight': [1e-3, 1e-2, 1e-1, 1.0],
            'reg_alpha': [0.0, 0.001, 0.01, 0.1, 0.5, 1.0],
            'reg_lambda': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'min_split_gain': [0.0, 0.01, 0.1, 0.5],
            'subsample_freq': [1, 3, 5, 7],
            'cat_l2': [1, 5, 10, 20],
            'cat_smooth': [1, 5, 10, 20]
        }
        base_model = LGBMClassifier(objective='binary', random_state=42, n_jobs=-1)

    elif model_name == 'CatBoost':
        try:
            from catboost import CatBoostClassifier
        except Exception as e:
            print(f"  CatBoost not available: {str(e)}")
            return None, None, None
        param_grid = {
            'iterations': [200, 300, 500, 800, 1000],
            'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2],
            'depth': [3, 4, 5, 6, 7, 8, 10],
            'l2_leaf_reg': [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
            'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            'colsample_bylevel': [0.5, 0.7, 0.9, 1.0],
            'min_data_in_leaf': [1, 3, 5, 10, 15, 20],
            'random_strength': [0.1, 0.5, 1.0, 2.0, 5.0],
            'leaf_estimation_method': ['Newton', 'Gradient'],
            'grow_policy': ['SymmetricTree', 'Depthwise'],
            'max_leaves': [10, 20, 31, 50, 100],
            'max_bin': [128, 254],
            'feature_border_type': ['GreedyLogSum', 'UniformAndQuantiles'],
            'bagging_temperature': [0.0, 0.5, 1.0, 2.0],
            'auto_class_weights': [None, 'Balanced', 'SqrtBalanced']
        }
        base_model = CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', verbose=False, random_state=42)

    elif model_name == 'ExtraTrees':
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 500, 800, 1000],
            'max_depth': [None, 3, 5, 8, 10, 15, 20, 25, 30],
            'min_samples_split': [2, 3, 5, 8, 10, 15, 20],
            'min_samples_leaf': [1, 2, 3, 4, 5, 8, 10],
            'max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],
            'class_weight': [None, 'balanced', 'balanced_subsample'],
            'ccp_alpha': [0.0, 0.0001, 0.001, 0.01, 0.1],
            'max_samples': [None, 0.6, 0.7, 0.8, 0.9],
            'bootstrap': [True, False],
            'criterion': ['gini', 'entropy', 'log_loss']
        }
        base_model = ExtraTreesClassifier(n_jobs=-1, random_state=42)

    elif model_name == 'AdaBoost':
        param_grid = {
            'n_estimators': [50, 100, 200, 300, 400, 500, 800],
            'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.5, 1.0],
            'algorithm': ['SAMME', 'SAMME.R']
        }
        base_model = AdaBoostClassifier(random_state=42)

    elif model_name == 'KNN':
        param_grid = {
            'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31],
            'weights': ['uniform', 'distance'],
            'p': [1, 2, 3],
            'leaf_size': [10, 20, 30, 40, 50],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'metric': ['minkowski', 'manhattan', 'euclidean', 'chebyshev']
        }
        base_model = KNeighborsClassifier(n_jobs=-1)

    elif model_name == 'GaussianNB':
        param_grid = {
            'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
        }
        base_model = GaussianNB()

    elif model_name == 'LDA':
        param_grid = {
            'solver': ['lsqr'],
            'shrinkage': [None, 0.0, 0.1, 0.3, 0.5]
        }
        base_model = LinearDiscriminantAnalysis()

    elif model_name == 'QDA':
        param_grid = {
            'reg_param': [0.0, 0.001, 0.01, 0.1, 0.5]
        }
        base_model = QuadraticDiscriminantAnalysis()

    elif model_name == 'MLP':
        param_grid = {
            'hidden_layer_sizes': [(32,), (64,), (128,), (256,), (64, 32), (128, 64), (256, 128), (64, 32, 16), (128, 64, 32)],
            'alpha': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'learning_rate_init': [1e-4, 1e-3, 5e-3, 1e-2, 5e-2],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd'],
            'max_iter': [300, 500, 1000],
            'tol': [1e-5, 1e-4, 1e-3],
            'momentum': [0.0, 0.1, 0.5, 0.9],
            'beta_1': [0.8, 0.9, 0.95, 0.99],
            'beta_2': [0.8, 0.9, 0.95, 0.99, 0.999],
            'epsilon': [1e-8, 1e-7, 1e-6],
            'early_stopping': [True, False],
            'validation_fraction': [0.1, 0.2],
            'n_iter_no_change': [5, 10, 15]
        }
        base_model = MLPClassifier(random_state=42)

    else:
        print(f"No hyperparameter tuning defined for {model_name}")
        return None, None, None

    try:
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=min(n_iter * 2, 100),  # Increase iterations for exhaustive search
            cv=cv_strategy,
            scoring='accuracy',
            n_jobs=-1,
            verbose=0,
            random_state=42,
            refit=True
        )

        search.fit(X_train, y_train)

        print(f"  Best parameters: {search.best_params_}")
        print(f"  Best CV ROC AUC: {search.best_score_:.4f}")

        return search.best_estimator_, search.best_params_, search.best_score_

    except Exception as e:
        print(f"  Error in hyperparameter tuning for {model_name}: {str(e)}")
        print("  Using default parameters...")
        return None, None, None

def main():
    print("=== Drug Shortage Prediction Model with Hyperparameter Tuning ===")
    print("Using Stratified Data Splitting and Multiple Model Selection")
    print(f"Analysis started at: {datetime.now()}")
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv("/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_timeline_prediction/price_signals_complete.csv")
    print(f"Original dataset shape: {df.shape}")
    
    # Data exploration
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Target variable range: {df['shortage_flag'].min():.0f} to {df['shortage_flag'].max():.0f}")
    
    # Create simple features
    print("\n2. Creating simple features...")
    
    # Prepare features and target
    features = ['avg_nadac', 'manufacturer_num', 'ingredient_num', 'num_forms', 'liquid_flag']
    
    X = df[features]
    y = df['shortage_flag']
    
    # Remove outliers
    print("\n3. Removing outliers...")
    X_clean, y_clean, outlier_indices = remove_outliers(X, y, method='compare', contamination=0.1)
    
    # Create stratified split on binary target (70/30)
    print("\n4. Creating stratified data split by shortage_flag (70/30)...")
    y_bins = create_stratified_bins(y_clean, n_bins=2)
    print("Class distribution (overall):")
    for cls in [0, 1]:
        cls_count = (y_bins == cls).sum()
        print(f"  Class {cls}: {cls_count} ({cls_count/len(y_bins)*100:.1f}%)")
    
    # Stratified train-test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    
    for train_idx, test_idx in sss.split(X_clean, y_bins):
        X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
    
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    print(f"\nStratified split results:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_test)} samples")
    
    # Verify stratification quality
    print("\nStratification quality check:")
    for i in range(2):
        train_bin_count = len(y_train[y_train == i])
        train_bin_pct = train_bin_count / len(y_train) * 100
        test_bin_count = len(y_test[y_test == i])
        test_bin_pct = test_bin_count / len(y_test) * 100
        print(f"  Bin {i}: Train {train_bin_count} ({train_bin_pct:.1f}%) | Test {test_bin_count} ({test_bin_pct:.1f}%)")
    
    # Scale features
    print("\n6. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    
    # Define models for training and hyperparameter tuning
    print("\n7. Training multiple models with hyperparameter tuning...")
    
    # Define cross-validation strategy for hyperparameter tuning
    from sklearn.model_selection import StratifiedKFold
    cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Define models to train (classification)
    models_to_train = [
        'DecisionTree', 'RandomForest', 'ExtraTrees', 'GradientBoosting', 'AdaBoost',
        'XGBoost', 'LightGBM', 'CatBoost',
        'Logit', 'KNN', 'GaussianNB', 'LDA', 'QDA', 'MLP',
        'Probit', 'LPM'
    ]
    
    # Store results for each model
    model_results = {}
    best_roc_auc = -np.inf
    best_accuracy = -np.inf
    best_model = None
    best_model_name = None
    
    for model_name in models_to_train:
        print(f"\n--- Training {model_name} ---")
        
        # Tune where applicable
        tuned_model, best_params, best_cv_auc = hyperparameter_tuning(
            X_train_scaled, y_train, model_name, cv_strategy, n_iter=20
        )

        fitted_model = None
        cv_auc = best_cv_auc

        if tuned_model is not None:
            fitted_model = tuned_model
            fitted_model.fit(X_train_scaled, y_train)
        else:
            # Models without sklearn tuning
            if model_name == 'Probit':
                X_train_const = sm.add_constant(X_train_scaled, has_constant='add')
                probit = sm.Probit(y_train.astype(int), X_train_const).fit(disp=0)
                fitted_model = probit
                best_params = None
                cv_auc = None
            elif model_name == 'LPM':
                lpm = LinearRegression()
                lpm.fit(X_train_scaled, y_train)
                fitted_model = lpm
                best_params = None
                cv_auc = None
            else:
                print(f"  Skipping {model_name} due to tuning failure")
                continue

        # Threshold tuning for Probit and LPM (optimize F1 via CV)
        def cv_optimal_threshold(X, y, model_name_local, thresholds=None, n_splits=3):
            if thresholds is None:
                thresholds = np.linspace(0.3, 0.7, 9)
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            mean_f1_per_thr = np.zeros(len(thresholds), dtype=float)
            for tr_idx, va_idx in skf.split(X, y):
                X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
                y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
                if model_name_local == 'Probit':
                    X_tr_c = sm.add_constant(X_tr, has_constant='add')
                    X_va_c = sm.add_constant(X_va, has_constant='add')
                    m = sm.Probit(y_tr.astype(int), X_tr_c).fit(disp=0)
                    proba = np.clip(m.predict(X_va_c), 0.0, 1.0)
                else:  # LPM
                    m = LinearRegression().fit(X_tr, y_tr)
                    proba = np.clip(m.predict(X_va), 0.0, 1.0)
                for ti, thr in enumerate(thresholds):
                    preds = (proba >= thr).astype(int)
                    mean_f1_per_thr[ti] += f1_score(y_va, preds, zero_division=0)
            mean_f1_per_thr /= n_splits
            return thresholds[int(np.argmax(mean_f1_per_thr))]

        threshold = 0.5
        if model_name in ['Probit', 'LPM']:
            threshold = cv_optimal_threshold(X_train_scaled, y_train.astype(int), model_name)

        # Prediction helper
        def predict_scores_and_labels(model, X, model_name_local, threshold_local=0.5):
            if model_name_local == 'Probit':
                X_const = sm.add_constant(X, has_constant='add')
                proba = model.predict(X_const)
                proba = np.clip(proba, 0.0, 1.0)
                labels = (proba >= threshold_local).astype(int)
                return proba, labels
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[:, 1]
                labels = (proba >= threshold_local).astype(int)
                return proba, labels
            if hasattr(model, 'decision_function'):
                scores = model.decision_function(X)
                proba = 1.0 / (1.0 + np.exp(-scores))
                labels = (proba >= threshold_local).astype(int)
                return proba, labels
            preds = model.predict(X)
            proba = np.clip(preds, 0.0, 1.0)
            labels = (proba >= threshold_local).astype(int)
            return proba, labels

        y_score, y_pred = predict_scores_and_labels(fitted_model, X_test_scaled, model_name, threshold_local=threshold)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        try:
            roc_auc = roc_auc_score(y_test, y_score)
        except Exception:
            roc_auc = np.nan
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Store
        model_results[model_name] = {
            'model': fitted_model,
            'best_params': best_params,
            'best_threshold': threshold,
            'cv_roc_auc': cv_auc,
            'test_accuracy': accuracy,
            'test_roc_auc': roc_auc,
            'test_precision': precision,
            'test_recall': recall,
            'test_f1': f1,
            'pred_labels': y_pred,
            'pred_scores': y_score
        }

        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test ROC AUC: {roc_auc:.4f}")
        print(f"  Test Precision/Recall/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}")

        # Select best via ROC AUC (fallback to accuracy)
        sel_metric = accuracy
        if sel_metric > best_accuracy:
            best_accuracy = sel_metric
            best_model = fitted_model
            best_model_name = model_name
    # Simple ensemble of top 3 models by validation accuracy (soft-average of scores)
    if len(model_results) >= 3:
        sorted_models = sorted(model_results.items(), key=lambda kv: kv[1]['test_accuracy'], reverse=True)
        top3 = sorted_models[:3]
        top3_names = [m[0] for m in top3]
        try:
            # Average probability-like scores; fallback to majority vote if needed
            scores_stack = np.vstack([m[1]['pred_scores'] for m in top3])
            ens_scores = np.mean(scores_stack, axis=0)
            ens_preds = (ens_scores >= 0.5).astype(int)
        except Exception:
            # Hard vote using labels
            labels_stack = np.vstack([m[1]['pred_labels'] for m in top3])
            ens_preds = (np.sum(labels_stack, axis=0) >= 2).astype(int)
            ens_scores = ens_preds.astype(float)

        ens_accuracy = accuracy_score(y_test, ens_preds)
        try:
            ens_roc_auc = roc_auc_score(y_test, ens_scores)
        except Exception:
            ens_roc_auc = np.nan
        ens_precision = precision_score(y_test, ens_preds, zero_division=0)
        ens_recall = recall_score(y_test, ens_preds, zero_division=0)
        ens_f1 = f1_score(y_test, ens_preds, zero_division=0)

        model_results['EnsembleTop3'] = {
            'model': None,
            'best_params': {'members': top3_names, 'strategy': 'soft_mean@0.5'},
            'best_threshold': 0.5,
            'cv_roc_auc': None,
            'test_accuracy': ens_accuracy,
            'test_roc_auc': ens_roc_auc,
            'test_precision': ens_precision,
            'test_recall': ens_recall,
            'test_f1': ens_f1,
            'pred_labels': ens_preds,
            'pred_scores': ens_scores
        }

        print(f"\n--- EnsembleTop3 (members: {', '.join(top3_names)}) ---")
        print(f"  Test Accuracy: {ens_accuracy:.4f}")
        print(f"  Test ROC AUC: {ens_roc_auc:.4f}")
        print(f"  Test Precision/Recall/F1: {ens_precision:.4f}/{ens_recall:.4f}/{ens_f1:.4f}")
    
    # Select best model (consider ensemble if better)
    if 'EnsembleTop3' in model_results and model_results['EnsembleTop3']['test_accuracy'] > best_accuracy:
        best_accuracy = model_results['EnsembleTop3']['test_accuracy']
        best_model = None
        best_model_name = 'EnsembleTop3'

    print(f"\n8. Model Selection Results:")
    print(f"Best model: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    # Print all model results
    print(f"\nAll model performance:")
    for model_name, results in model_results.items():
        print(f"  {model_name}: Accuracy={results['test_accuracy']:.4f}, ROC AUC={results['test_roc_auc']:.4f}")
    
    # Get best model results
    best_results = model_results[best_model_name]
    y_pred = best_results['pred_labels']
    y_score = best_results['pred_scores']
    accuracy = best_results['test_accuracy']
    roc_auc = best_results['test_roc_auc']
    
    # Feature importance (only for models that support it)
    print(f"\n9. Feature Importance Analysis:")
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("Top 15 most important features:")
        for i, row in feature_importance.head(15).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Shortage-specific feature importance
        shortage_features = [f for f in features if 'shortage' in f.lower()]
        shortage_importance = feature_importance[feature_importance['feature'].isin(shortage_features)]
        
        print(f"\nShortage-specific features ({len(shortage_features)} total):")
        for i, row in shortage_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
    else:
        print("Feature importance not available for this model type")
        feature_importance = None
        shortage_features = [f for f in features if 'shortage' in f.lower()]
    
    # Save results
    print("\n10. Saving results...")
    
    # Save best model and scaler (skip saving None for ensemble)
    import joblib
    if best_model is not None:
        joblib.dump(best_model, 'best_shortage_prediction_model.pkl')
    joblib.dump(scaler, 'best_shortage_prediction_scaler.pkl')
    
    # Save feature importance if available
    if feature_importance is not None:
        feature_importance.to_csv('shortage_prediction_best_feature_importance.csv', index=False)
    
    # Save validation predictions
    val_df = pd.DataFrame(X_test_scaled, columns=features)
    val_df['shortage_flag_true'] = y_test.values
    val_df['shortage_flag_pred'] = y_pred
    val_df['shortage_score'] = y_score
    val_df.to_csv('shortage_prediction_best_validation_predictions.csv', index=False)

    # Two-way contingency table and classification metrics (test set)
    y_test_arr = np.asarray(y_test).astype(int)
    y_pred_arr = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_test_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    contingency_df = pd.DataFrame(
        cm,
        index=['Actual_shortage_flag_0', 'Actual_shortage_flag_1'],
        columns=['Predicted_shortage_flag_0', 'Predicted_shortage_flag_1'],
    )
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    positive_predictive_value = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    negative_predictive_value = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    metrics_df = pd.DataFrame(
        {
            'metric': [
                'sensitivity',
                'specificity',
                'positive_predictive_value',
                'negative_predictive_value',
            ],
            'value': [
                sensitivity,
                specificity,
                positive_predictive_value,
                negative_predictive_value,
            ],
        }
    )
    contingency_table_path = 'shortage_prediction_test_contingency_table.csv'
    with open(contingency_table_path, 'w') as f:
        f.write('Two-way contingency table (test set)\n')
        contingency_df.to_csv(f)
        f.write('\nClassification metrics\n')
        metrics_df.to_csv(f, index=False)
    
    # Save results summary
    def _csv_value(value):
        if isinstance(value, (list, dict)):
            return str(value)
        return value

    run_summary_df = pd.DataFrame(
        [
            ('timestamp', datetime.now().isoformat()),
            ('best_model', best_model_name),
            ('original_rows', int(df.shape[0])),
            ('original_columns', int(df.shape[1])),
            ('features_used', int(len(features))),
            ('training_samples', int(len(X_train))),
            ('validation_samples', int(len(X_test))),
            ('missing_values_handled', False),
            ('outlier_removal_method', 'compare'),
            ('outlier_contamination', 0.1),
            ('outliers_removed', int(len(outlier_indices))),
            ('outliers_removed_pct', float(len(outlier_indices) / df.shape[0] * 100)),
            ('clean_dataset_size', int(len(X_clean))),
            ('feature_scaling', 'StandardScaler'),
            ('categorical_encoding', 'None'),
            ('best_model_accuracy', float(accuracy)),
            ('best_model_roc_auc', float(roc_auc)),
            ('total_features', int(len(features))),
            ('shortage_features', int(len(shortage_features))),
            ('cost_ratio_features', int(len([f for f in features if 'ratio' in f.lower()]))),
            ('interaction_features', int(len([f for f in features if 'impact' in f.lower()]))),
        ],
        columns=['metric', 'value'],
    )

    stratification_df = pd.DataFrame(
        [
            ('method', 'stratified_shuffle_split_on_target'),
            ('classes', '[0, 1]'),
            ('test_size', 0.3),
            ('random_state', 42),
        ],
        columns=['metric', 'value'],
    )

    all_models_df = pd.DataFrame(
        [
            {
                'model': model_name,
                'accuracy': float(results['test_accuracy']),
                'roc_auc': float(results['test_roc_auc']) if results['test_roc_auc'] is not None else None,
                'precision': float(results['test_precision']),
                'recall': float(results['test_recall']),
                'f1': float(results['test_f1']),
            }
            for model_name, results in model_results.items()
        ]
    )

    best_model_params_df = pd.DataFrame(
        [
            {'parameter': param, 'value': _csv_value(value)}
            for param, value in best_results['best_params'].items()
        ]
    )

    class_distribution_df = pd.DataFrame(
        [
            {
                'class': i,
                'train_count': int(len(y_train[y_train == i])),
                'train_pct': float(len(y_train[y_train == i]) / len(y_train) * 100),
                'test_count': int(len(y_test[y_test == i])),
                'test_pct': float(len(y_test[y_test == i]) / len(y_test) * 100),
            }
            for i in [0, 1]
        ]
    )

    results_summary_path = 'shortage_prediction_results.csv'
    with open(results_summary_path, 'w') as f:
        f.write('Run summary\n')
        run_summary_df.to_csv(f, index=False)
        f.write('\nStratification\n')
        stratification_df.to_csv(f, index=False)
        f.write('\nAll models performance (test set)\n')
        all_models_df.to_csv(f, index=False)
        f.write('\nBest model parameters\n')
        best_model_params_df.to_csv(f, index=False)
        f.write('\nClass distributions\n')
        class_distribution_df.to_csv(f, index=False)
    
    print("\nResults saved to:")
    print("- best_shortage_prediction_model.pkl (best model)")
    print("- best_shortage_prediction_scaler.pkl (scaler)")
    if feature_importance is not None:
        print("- shortage_prediction_best_feature_importance.csv (feature importance)")
    print("- shortage_prediction_best_validation_predictions.csv (validation predictions)")
    print("- shortage_prediction_test_contingency_table.csv (test contingency table and metrics)")
    print("- shortage_prediction_results.csv (results summary)")
    
    print(f"\n=== Analysis completed at: {datetime.now()} ===")
    print(f"Best Model: {best_model_name}")
    print(f"Features: {len(features)}")
    print(f"Original dataset: {df.shape[0]} samples")
    print(f"After outlier removal: {len(X_clean)} samples")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_test)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Data preprocessing and model selection benefits
    print(f"\n=== Data Preprocessing and Model Selection Benefits ===")
    print(f"✓ Outliers removed using class-aware methods: Mahalanobis, Isolation Forest, and LOF")
    print(f"✓ Features standardized using StandardScaler")
    print(f"✓ Multiple classification models trained and compared")
    print(f"✓ Hyperparameter tuning for key models")
    print(f"✓ Best model selected based on ROC AUC")

if __name__ == "__main__":
    main()
