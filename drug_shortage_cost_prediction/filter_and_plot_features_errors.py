"""
Feature Filtering and Visualization Script for Prediction Error Analysis
Filters features by correlation using pre-computed error analysis, then plots each feature against prediction error
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def main():
    """Main execution function"""
    print("=" * 80)
    print("Feature Filtering and Visualization for Prediction Error Analysis")
    print("=" * 80)
    
    # Load pre-computed information gain results for error prediction
    print("\n1. Loading pre-computed Information Gain results (for error prediction)...")
    ig_results = pd.read_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/information_gain_analysis_errors.csv')
    print(f"   Loaded IG scores for {len(ig_results)} features")
    
    # Calculate median and filter features with IG >= median
    ig_median = ig_results['Information_Gain'].median()
    selected_features_ig = ig_results[ig_results['Information_Gain'] >= ig_median].copy()
    selected_feature_names = selected_features_ig['Feature'].tolist()
    
    print(f"   Median Information Gain: {ig_median:.6f}")
    print(f"   Features with IG >= median: {len(selected_feature_names)}")
    
    # Create IG dictionary for quick lookup
    ig_dict = {row['Feature']: row['Information_Gain'] for _, row in ig_results.iterrows()}
    
    # Load pre-computed correlation matrix results for error prediction
    print("\n2. Loading pre-computed Correlation Matrix results (for error prediction)...")
    corr_results = pd.read_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/correlation_matrix_analysis_errors.csv')
    print(f"   Loaded correlation pairs for {len(corr_results)} feature pairs")
    
    # Filter correlation results to only include features with IG >= median
    corr_filtered = corr_results[
        (corr_results['Feature1'].isin(selected_feature_names)) & 
        (corr_results['Feature2'].isin(selected_feature_names))
    ].copy()
    print(f"   Correlation pairs for features with IG >= median: {len(corr_filtered)}")
    
    # Filter features by correlation
    print("\n3. Filtering features by correlation (removing lower IG feature when |r| >= 0.7)...")
    features_to_remove = set()
    corr_threshold = 0.7
    
    # Iterate through correlation pairs
    for _, row in corr_filtered.iterrows():
        feat1 = row['Feature1']
        feat2 = row['Feature2']
        corr_value = row['Correlation']
        
        # Skip if either feature is already marked for removal
        if feat1 in features_to_remove or feat2 in features_to_remove:
            continue
        
        # Check if correlation is above threshold
        if abs(corr_value) >= corr_threshold:
            ig1 = ig_dict.get(feat1, 0)
            ig2 = ig_dict.get(feat2, 0)
            
            # Remove feature with lower IG
            if ig1 >= ig2:
                features_to_remove.add(feat2)
            else:
                features_to_remove.add(feat1)
    
    # Get final filtered features
    final_features = [f for f in selected_feature_names if f not in features_to_remove]
    
    print(f"   Removed {len(features_to_remove)} features due to high correlation")
    print(f"   Final features after correlation filtering: {len(final_features)}")
    
    if features_to_remove:
        print(f"   Removed features: {list(features_to_remove)[:10]}...")  # Show first 10
    
    # Load data with predictions for plotting
    print("\n4. Loading data with predictions for plotting...")
    df = pd.read_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/full_dataset_with_predictions_profit_binary.csv')
    target_col = 'Prediction_Error'
    
    # Remove rows with missing target
    df_clean = df.dropna(subset=[target_col])
    y = df_clean[target_col].values
    
    print(f"   Loaded {len(df_clean)} rows for plotting")
    print(f"   Note: 'Net Income' excluded from features (data leakage)")
    
    # Prepare feature data for plotting (minimal preprocessing)
    print("\n5. Preparing feature data for plotting...")
    feature_data = {}
    categorical_features = set()
    
    # Identify date columns from feature names
    date_cols = set()
    for feat in final_features:
        if '_year' in feat or '_month' in feat:
            # Extract original date column name
            original_col = feat.replace('_year', '').replace('_month', '')
            date_cols.add(original_col)
    
    # Process features needed for plotting
    for feat in final_features:
        if '_year' in feat or '_month' in feat:
            # This is a date-derived feature
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
                    # Categorical feature
                    le = LabelEncoder()
                    encoded_values = le.fit_transform(df_clean[feat].astype(str))
                    feature_data[feat] = encoded_values
                    categorical_features.add(feat)
                else:
                    # Numerical feature
                    feature_data[feat] = df_clean[feat].fillna(df_clean[feat].median()).values
            else:
                print(f"   WARNING: Feature '{feat}' not found in dataframe, skipping...")
    
    print(f"   Prepared {len(feature_data)} features for plotting")
    print(f"   Categorical features: {len(categorical_features)}")
    
    # Calculate correlation with target for each final feature
    print("\n6. Calculating correlation with Prediction_Error for each feature...")
    correlation_results = []
    
    for feat in final_features:
        if feat not in feature_data:
            continue
        
        # Get feature values
        feat_values = feature_data[feat]
        
        # Align lengths (in case of missing values)
        valid_mask = ~(np.isnan(feat_values) | np.isnan(y))
        if valid_mask.sum() < 2:
            continue
        
        feat_values_clean = feat_values[valid_mask]
        y_clean = y[valid_mask]
        
        # Calculate correlation with target
        corr_with_target = np.corrcoef(feat_values_clean, y_clean)[0, 1]
        
        # Calculate other statistics
        ig_value = ig_dict.get(feat, 0)
        
        correlation_results.append({
            'Feature': feat,
            'Information_Gain': ig_value,
            'Correlation_with_Prediction_Error': corr_with_target,
            'Abs_Correlation_with_Prediction_Error': abs(corr_with_target),
            'Is_Categorical': feat in categorical_features
        })
    
    correlation_df = pd.DataFrame(correlation_results)
    correlation_df = correlation_df.sort_values('Abs_Correlation_with_Prediction_Error', ascending=False)
    
    print(f"\n   Correlation with Prediction_Error Statistics:")
    print(f"     Min: {correlation_df['Correlation_with_Prediction_Error'].min():.6f}")
    print(f"     Median: {correlation_df['Correlation_with_Prediction_Error'].median():.6f}")
    print(f"     Max: {correlation_df['Correlation_with_Prediction_Error'].max():.6f}")
    print(f"     Mean: {correlation_df['Correlation_with_Prediction_Error'].mean():.6f}")
    
    # Save correlation results
    correlation_df.to_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/feature_target_correlations_errors.csv', index=False)
    print(f"\n   Saved: feature_target_correlations_errors.csv")
    
    # Create plots directory
    plots_dir = '/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/feature_plots_errors'
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot each feature against target (skip categorical features)
    print("\n7. Creating plots (skipping categorical features)...")
    plot_count = 0
    skipped_count = 0
    
    for feat in final_features:
        if feat not in feature_data:
            continue
        
        if feat in categorical_features:
            skipped_count += 1
            continue
        
        # Get feature values
        feat_values = feature_data[feat]
        
        # Align lengths and remove NaN
        valid_mask = ~(np.isnan(feat_values) | np.isnan(y))
        feat_values_clean = feat_values[valid_mask]
        y_clean = y[valid_mask]
        
        if len(feat_values_clean) < 2:
            continue
        
        # Get correlation value
        corr_val = correlation_df[correlation_df['Feature'] == feat]['Correlation_with_Prediction_Error'].values[0]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(feat_values_clean, y_clean, alpha=0.5, s=20)
        plt.xlabel(feat, fontsize=12)
        plt.ylabel(target_col, fontsize=12)
        plt.title(f'{feat} vs {target_col}\nCorrelation: {corr_val:.4f}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot (sanitize filename)
        safe_filename = feat.replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_').replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_').replace('|', '_')
        plot_path = os.path.join(plots_dir, f'{safe_filename}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        plot_count += 1
        if plot_count % 10 == 0:
            print(f"   Created {plot_count} plots...")
    
    print(f"\n   Created {plot_count} plots")
    print(f"   Skipped {skipped_count} categorical features")
    print(f"   Plots saved to: {plots_dir}")
    
    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print(f"\nFeatures with IG >= median ({ig_median:.6f}): {len(selected_feature_names)}")
    print(f"Features removed due to correlation: {len(features_to_remove)}")
    print(f"Final Features: {len(final_features)}")
    print(f"\nPlots Created: {plot_count}")
    print(f"Categorical Features Skipped: {skipped_count}")
    
    print("\n" + "=" * 80)
    print("Analysis completed successfully!")
    print("=" * 80)
    print("\nOutput files:")
    print("  1. feature_target_correlations_errors.csv - Correlation measures for each feature (with Prediction_Error)")
    print(f"  2. {plot_count} plot files in: feature_plots_errors/")

if __name__ == "__main__":
    main()

