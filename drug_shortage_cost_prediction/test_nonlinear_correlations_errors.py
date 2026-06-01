"""
Non-linear Correlation Testing Script for Prediction Error Analysis
Tests log, square, and square root transformations of features against prediction error
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def main():
    """Main execution function"""
    print("=" * 80)
    print("Non-linear Correlation Testing for Prediction Error Analysis")
    print("=" * 80)
    
    # Load final features from feature_target_correlations_errors.csv
    print("\n1. Loading final features (for error prediction)...")
    final_features_df = pd.read_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/feature_target_correlations_errors.csv')
    final_features = final_features_df['Feature'].tolist()
    print(f"   Loaded {len(final_features)} final features")
    
    # Load data with predictions
    print("\n2. Loading data with predictions...")
    df = pd.read_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/full_dataset_with_predictions_profit_binary.csv')
    target_col = 'Prediction_Error'
    
    # Remove rows with missing target
    df_clean = df.dropna(subset=[target_col])
    y = df_clean[target_col].values
    
    print(f"   Loaded {len(df_clean)} rows")
    print(f"   Note: 'Net Income' excluded from features (data leakage)")
    
    # Prepare feature data
    print("\n3. Preparing feature data...")
    feature_data = {}
    categorical_features = set()
    
    # Identify date columns from feature names
    date_cols = set()
    for feat in final_features:
        if '_year' in feat or '_month' in feat:
            original_col = feat.replace('_year', '').replace('_month', '')
            date_cols.add(original_col)
    
    # Process features
    for feat in final_features:
        if '_year' in feat or '_month' in feat:
            # Date-derived feature
            original_col = feat.replace('_year', '').replace('_month', '')
            if original_col in df_clean.columns:
                dates = pd.to_datetime(df_clean[original_col], errors='coerce')
                if '_year' in feat:
                    values = dates.dt.year.fillna(dates.dt.year.median())
                else:
                    values = dates.dt.month.fillna(dates.dt.month.median())
                feature_data[feat] = values.values
        else:
            # Regular feature - skip 'Net Income' (data leakage)
            if feat == 'Net Income':
                print(f"   Skipping 'Net Income' (data leakage)")
                continue
            if feat in df_clean.columns:
                if df_clean[feat].dtype == 'object' or df_clean[feat].dtype.name == 'category':
                    # Categorical feature - skip transformations
                    le = LabelEncoder()
                    encoded_values = le.fit_transform(df_clean[feat].astype(str))
                    feature_data[feat] = encoded_values
                    categorical_features.add(feat)
                else:
                    # Numerical feature
                    feature_data[feat] = df_clean[feat].fillna(df_clean[feat].median()).values
    
    print(f"   Prepared {len(feature_data)} features")
    print(f"   Categorical features (will be skipped): {len(categorical_features)}")
    
    # Test non-linear correlations
    print("\n4. Testing non-linear correlations with Prediction_Error...")
    correlation_results = []
    
    for feat in final_features:
        if feat not in feature_data:
            continue
        
        # Skip categorical features
        if feat in categorical_features:
            continue
        
        # Get original feature values
        feat_values = feature_data[feat].copy()
        
        # Align lengths and remove NaN
        valid_mask = ~(np.isnan(feat_values) | np.isnan(y))
        feat_values_clean = feat_values[valid_mask]
        y_clean = y[valid_mask]
        
        if len(feat_values_clean) < 2:
            continue
        
        # Calculate original correlation
        corr_original = np.corrcoef(feat_values_clean, y_clean)[0, 1]
        
        # Transformation 1: Log
        # Handle negative values and zeros
        feat_values_for_log = feat_values_clean.copy()
        min_val = feat_values_for_log.min()
        if min_val <= 0:
            # Shift to make all values positive
            offset = abs(min_val) + 1
            feat_values_for_log = feat_values_for_log + offset
        else:
            offset = 0
        
        feat_log = np.log(feat_values_for_log)
        corr_log = np.corrcoef(feat_log, y_clean)[0, 1]
        
        # Transformation 2: Square
        feat_square = feat_values_clean ** 2
        corr_square = np.corrcoef(feat_square, y_clean)[0, 1]
        
        # Transformation 3: Square Root
        # Handle negative values
        feat_values_for_sqrt = feat_values_clean.copy()
        if feat_values_for_sqrt.min() < 0:
            # Shift to make all values non-negative
            offset_sqrt = abs(feat_values_for_sqrt.min())
            feat_values_for_sqrt = feat_values_for_sqrt + offset_sqrt
        else:
            offset_sqrt = 0
        
        feat_sqrt = np.sqrt(feat_values_for_sqrt)
        corr_sqrt = np.corrcoef(feat_sqrt, y_clean)[0, 1]
        
        # Store results
        correlation_results.append({
            'Feature': feat,
            'Original_Correlation': corr_original,
            'Abs_Original_Correlation': abs(corr_original),
            'Log_Correlation': corr_log,
            'Abs_Log_Correlation': abs(corr_log),
            'Square_Correlation': corr_square,
            'Abs_Square_Correlation': abs(corr_square),
            'Sqrt_Correlation': corr_sqrt,
            'Abs_Sqrt_Correlation': abs(corr_sqrt),
            'Best_Transformation': max([
                ('Original', abs(corr_original)),
                ('Log', abs(corr_log)),
                ('Square', abs(corr_square)),
                ('Sqrt', abs(corr_sqrt))
            ], key=lambda x: x[1])[0],
            'Best_Correlation': max([
                abs(corr_original),
                abs(corr_log),
                abs(corr_square),
                abs(corr_sqrt)
            ])
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(correlation_results)
    results_df = results_df.sort_values('Best_Correlation', ascending=False)
    
    print(f"\n   Tested {len(results_df)} features")
    print(f"\n   Correlation Statistics (with Prediction_Error):")
    print(f"     Original - Min: {results_df['Abs_Original_Correlation'].min():.6f}, "
          f"Median: {results_df['Abs_Original_Correlation'].median():.6f}, "
          f"Max: {results_df['Abs_Original_Correlation'].max():.6f}")
    print(f"     Log      - Min: {results_df['Abs_Log_Correlation'].min():.6f}, "
          f"Median: {results_df['Abs_Log_Correlation'].median():.6f}, "
          f"Max: {results_df['Abs_Log_Correlation'].max():.6f}")
    print(f"     Square   - Min: {results_df['Abs_Square_Correlation'].min():.6f}, "
          f"Median: {results_df['Abs_Square_Correlation'].median():.6f}, "
          f"Max: {results_df['Abs_Square_Correlation'].max():.6f}")
    print(f"     Sqrt     - Min: {results_df['Abs_Sqrt_Correlation'].min():.6f}, "
          f"Median: {results_df['Abs_Sqrt_Correlation'].median():.6f}, "
          f"Max: {results_df['Abs_Sqrt_Correlation'].max():.6f}")
    
    # Count best transformations
    best_trans_counts = results_df['Best_Transformation'].value_counts()
    print(f"\n   Best Transformation Counts:")
    for trans, count in best_trans_counts.items():
        print(f"     {trans}: {count}")
    
    # Show top 10 features by best correlation
    print(f"\n   Top 10 features by best correlation (with Prediction_Error):")
    for idx, row in results_df.head(10).iterrows():
        print(f"     {row['Feature']}: {row['Best_Transformation']} ({row['Best_Correlation']:.6f})")
    
    # Save results
    results_df.to_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/nonlinear_correlations_errors.csv', index=False)
    print(f"\n   Saved: nonlinear_correlations_errors.csv")
    
    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nFeatures Tested: {len(results_df)}")
    print(f"Features with improved correlation after transformation:")
    print(f"  Log improves: {(results_df['Abs_Log_Correlation'] > results_df['Abs_Original_Correlation']).sum()}")
    print(f"  Square improves: {(results_df['Abs_Square_Correlation'] > results_df['Abs_Original_Correlation']).sum()}")
    print(f"  Sqrt improves: {(results_df['Abs_Sqrt_Correlation'] > results_df['Abs_Original_Correlation']).sum()}")
    
    print("\n" + "=" * 80)
    print("Analysis completed successfully!")
    print("=" * 80)
    print("\nOutput file:")
    print("  nonlinear_correlations_errors.csv - Non-linear correlation measures for each feature (with Prediction_Error)")

if __name__ == "__main__":
    main()

