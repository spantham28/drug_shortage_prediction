"""
Prediction Error Analysis Script
Measures Information Gain and Correlation for all features in predicting prediction error
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def main():
    """Main execution function"""
    print("=" * 80)
    print("Prediction Error Analysis - Information Gain and Correlation")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/full_dataset_with_predictions_profit_binary.csv')
    print(f"   Loaded {len(df)} rows")
    print(f"   Columns: {df.columns.tolist()}")
    
    # Identify target and features
    target_col = 'Prediction_Error'
    # Exclude Provider CCN, Net Income (original target), Prediction_Error (target), Predicted_Net_Income (derived from prediction), and index column
    # Note: We keep Predicted_Net_Income as a feature since it might help predict the error
    exclude_cols = ['Provider CCN', 'Net Income', target_col, 'Unnamed: 0']
    
    # Get feature columns (all except excluded columns)
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    # Explicitly remove 'Unnamed: 0' if it somehow got included
    if 'Unnamed: 0' in feature_cols:
        feature_cols.remove('Unnamed: 0')
    
    print(f"\n   Target variable: {target_col}")
    print(f"   Number of features: {len(feature_cols)}")
    print(f"   Note: 'Net Income' excluded from features (data leakage)")
    print(f"   Note: 'Predicted_Net_Income' included as feature (may help predict error)")
    
    # Check for missing values
    print("\n2. Data quality check...")
    check_cols = [target_col] + feature_cols
    missing_counts = df[check_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print(f"   Missing values:\n{missing_counts[missing_counts > 0]}")
    
    # Remove rows with missing target or features
    df_clean = df.dropna(subset=[target_col] + feature_cols)
    print(f"   After removing missing values: {len(df_clean)} rows (removed {len(df) - len(df_clean)} rows)")
    
    # Prepare target
    y = df_clean[target_col].values
    
    # Prepare features
    print("\n3. Preparing features...")
    
    # Identify date columns
    date_cols = [col for col in feature_cols if 'date' in col.lower() or 'Date' in col]
    print(f"   Detected {len(date_cols)} date columns: {date_cols}")
    
    # Process date columns - convert to datetime and extract year/month
    feature_data = {}
    all_feature_names = []
    
    for col in feature_cols:
        if col in date_cols:
            # Convert to datetime and extract features
            dates = pd.to_datetime(df_clean[col], errors='coerce')
            year_col = f'{col}_year'
            month_col = f'{col}_month'
            
            df_clean[year_col] = dates.dt.year
            df_clean[month_col] = dates.dt.month
            
            # Fill NaN with median
            year_median = df_clean[year_col].median()
            month_median = df_clean[month_col].median()
            df_clean[year_col] = df_clean[year_col].fillna(year_median)
            df_clean[month_col] = df_clean[month_col].fillna(month_median)
            
            feature_data[year_col] = df_clean[year_col].values
            feature_data[month_col] = df_clean[month_col].values
            all_feature_names.extend([year_col, month_col])
        else:
            # Check if column is categorical
            if df_clean[col].dtype == 'object' or df_clean[col].dtype.name == 'category':
                # Encode categorical variables
                le = LabelEncoder()
                encoded_values = le.fit_transform(df_clean[col].astype(str))
                feature_data[col] = encoded_values
                all_feature_names.append(col)
            else:
                # Numerical feature
                feature_data[col] = df_clean[col].values
                all_feature_names.append(col)
    
    # Create feature matrix
    X = np.column_stack([feature_data[col] for col in all_feature_names])
    
    # Clean NaN/Inf values
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    print(f"   Total features after processing: {len(all_feature_names)}")
    print(f"   Feature matrix shape: {X.shape}")
    
    # Calculate Information Gain
    print("\n4. Calculating Information Gain (Mutual Information) for Prediction Error...")
    mi_scores = mutual_info_regression(X, y, random_state=42)
    
    # Create information gain results dataframe
    ig_results = pd.DataFrame({
        'Feature': all_feature_names,
        'Information_Gain': mi_scores
    }).sort_values('Information_Gain', ascending=False)
    
    print(f"\n   Information Gain Statistics:")
    print(f"     Min: {mi_scores.min():.6f}")
    print(f"     Median: {np.median(mi_scores):.6f}")
    print(f"     Max: {mi_scores.max():.6f}")
    print(f"     Mean: {mi_scores.mean():.6f}")
    print(f"\n   Top 10 features by Information Gain (for predicting error):")
    for idx, row in ig_results.head(10).iterrows():
        print(f"     {row['Feature']}: {row['Information_Gain']:.6f}")
    
    # Save information gain results
    ig_results.to_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/information_gain_analysis_errors.csv', index=False)
    print(f"\n   Saved: information_gain_analysis_errors.csv")
    
    # Filter features for correlation analysis: only those with IG >= median
    ig_median = np.median(mi_scores)
    selected_features_mask = mi_scores >= ig_median
    selected_feature_names = [all_feature_names[i] for i in range(len(all_feature_names)) if selected_features_mask[i]]
    selected_feature_indices = np.where(selected_features_mask)[0]
    
    print(f"\n   Filtering features for correlation analysis:")
    print(f"     Median Information Gain: {ig_median:.6f}")
    print(f"     Features with IG >= median: {len(selected_feature_names)}")
    print(f"     Features excluded from correlation analysis: {len(all_feature_names) - len(selected_feature_names)}")
    
    # Calculate Correlation Matrix (only for selected features)
    print("\n5. Calculating Correlation Matrix (for features with IG >= median)...")
    X_selected = X[:, selected_feature_indices]
    corr_matrix = np.corrcoef(X_selected.T)
    
    # Create correlation results dataframe
    # This will be a long format with Feature1, Feature2, Correlation
    correlation_data = []
    for i in range(len(selected_feature_names)):
        for j in range(i+1, len(selected_feature_names)):  # Only upper triangle to avoid duplicates
            correlation_data.append({
                'Feature1': selected_feature_names[i],
                'Feature2': selected_feature_names[j],
                'Correlation': corr_matrix[i, j],
                'Abs_Correlation': abs(corr_matrix[i, j])
            })
    
    corr_results = pd.DataFrame(correlation_data)
    corr_results = corr_results.sort_values('Abs_Correlation', ascending=False)
    
    print(f"\n   Correlation Statistics (for {len(selected_feature_names)} features):")
    if len(corr_matrix) > 1:
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        print(f"     Min: {upper_triangle.min():.6f}")
        print(f"     Median: {np.median(upper_triangle):.6f}")
        print(f"     Max: {upper_triangle.max():.6f}")
        print(f"     Mean: {np.mean(upper_triangle):.6f}")
    
    # Count high correlations
    high_corr_threshold = 0.7
    high_corr_count = (corr_results['Abs_Correlation'] > high_corr_threshold).sum()
    print(f"\n   Feature pairs with |correlation| > {high_corr_threshold}: {high_corr_count}")
    print(f"   Top 10 highest correlations:")
    for idx, row in corr_results.head(10).iterrows():
        print(f"     {row['Feature1']} <-> {row['Feature2']}: {row['Correlation']:.4f}")
    
    # Save correlation results
    corr_results.to_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/correlation_matrix_analysis_errors.csv', index=False)
    print(f"\n   Saved: correlation_matrix_analysis_errors.csv")
    
    # Also create a square correlation matrix (easier to read)
    corr_matrix_df = pd.DataFrame(corr_matrix, index=selected_feature_names, columns=selected_feature_names)
    corr_matrix_df.to_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/correlation_matrix_square_errors.csv')
    print(f"   Saved: correlation_matrix_square_errors.csv (square matrix format)")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nTotal Features Analyzed: {len(all_feature_names)}")
    print(f"\nInformation Gain (for predicting Prediction_Error):")
    print(f"  Features with IG > 0.3: {(ig_results['Information_Gain'] > 0.3).sum()}")
    print(f"  Features with IG > 0.2: {(ig_results['Information_Gain'] > 0.2).sum()}")
    print(f"  Features with IG > 0.1: {(ig_results['Information_Gain'] > 0.1).sum()}")
    print(f"  Features with IG >= median ({ig_median:.6f}): {len(selected_feature_names)}")
    print(f"\nCorrelation (analyzed for {len(selected_feature_names)} features with IG >= median):")
    print(f"  Feature pairs with |r| > 0.8: {(corr_results['Abs_Correlation'] > 0.8).sum()}")
    print(f"  Feature pairs with |r| > 0.7: {(corr_results['Abs_Correlation'] > 0.7).sum()}")
    print(f"  Feature pairs with |r| > 0.6: {(corr_results['Abs_Correlation'] > 0.6).sum()}")
    
    print("\n" + "=" * 80)
    print("Analysis completed successfully!")
    print("=" * 80)
    print("\nOutput files:")
    print("  1. information_gain_analysis_errors.csv - Information gain for each feature (predicting error)")
    print("  2. correlation_matrix_analysis_errors.csv - Correlation pairs (long format)")
    print("  3. correlation_matrix_square_errors.csv - Full correlation matrix (square format)")

if __name__ == "__main__":
    main()

