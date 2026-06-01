"""
Comprehensive analysis of target variable (Net Income) distribution
Generates histograms, boxplots, QQ plots, and statistical summaries
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def analyze_target_distribution(df, target_col='Net Income', output_dir='/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction'):
    """
    Comprehensive analysis of target variable distribution
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    output_dir : str
        Directory to save plots (default: current directory)
    """
    print("=" * 80)
    print("TARGET VARIABLE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Extract target variable
    y = df[target_col].dropna().values
    n_samples = len(y)
    
    print(f"\nDataset: {target_col}")
    print(f"Number of samples: {n_samples:,}")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    mean_val = np.mean(y)
    median_val = np.median(y)
    std_val = np.std(y)
    min_val = np.min(y)
    max_val = np.max(y)
    range_val = max_val - min_val
    
    print(f"  Mean:       {mean_val:,.2f}")
    print(f"  Median:     {median_val:,.2f}")
    print(f"  Std Dev:    {std_val:,.2f}")
    print(f"  Min:        {min_val:,.2f}")
    print(f"  Max:        {max_val:,.2f}")
    print(f"  Range:      {range_val:,.2f}")
    
    # Percentiles
    print(f"\nPercentiles:")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(y, p)
        print(f"  {p:2d}th percentile: {val:,.2f}")
    
    # Distribution shape
    print(f"\nDistribution Shape:")
    skewness = stats.skew(y)
    kurtosis = stats.kurtosis(y)
    print(f"  Skewness:   {skewness:.4f} (normal=0, highly skewed if |skew| > 2)")
    print(f"  Kurtosis:   {kurtosis:.4f} (normal=0, heavy tails if |kurtosis| > 2)")
    
    # Special values
    print(f"\nSpecial Values:")
    n_zero = (y == 0).sum()
    n_negative = (y < 0).sum()
    n_positive = (y > 0).sum()
    print(f"  Zeros:      {n_zero} ({n_zero/n_samples*100:.2f}%)")
    print(f"  Negative:   {n_negative} ({n_negative/n_samples*100:.2f}%)")
    print(f"  Positive:   {n_positive} ({n_positive/n_samples*100:.2f}%)")
    
    # Outlier detection using IQR method
    print(f"\n{'='*80}")
    print("OUTLIER DETECTION (IQR Method)")
    print(f"{'='*80}")
    
    q1 = np.percentile(y, 25)
    q3 = np.percentile(y, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = y[(y < lower_bound) | (y > upper_bound)]
    n_outliers = len(outliers)
    
    print(f"  Q1 (25th percentile):     {q1:,.2f}")
    print(f"  Q3 (75th percentile):     {q3:,.2f}")
    print(f"  IQR (Q3 - Q1):            {iqr:,.2f}")
    print(f"  Lower bound (Q1 - 1.5*IQR): {lower_bound:,.2f}")
    print(f"  Upper bound (Q3 + 1.5*IQR): {upper_bound:,.2f}")
    print(f"  Number of outliers:       {n_outliers} ({n_outliers/n_samples*100:.2f}%)")
    if n_outliers > 0:
        print(f"  Outlier range:            {np.min(outliers):,.2f} to {np.max(outliers):,.2f}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Histogram with linear y-axis
    ax1 = plt.subplot(2, 3, 1)
    n_bins = min(50, int(np.sqrt(n_samples)))  # Reasonable number of bins
    ax1.hist(y, bins=n_bins, edgecolor='black', alpha=0.7)
    ax1.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:,.0f}')
    ax1.axvline(median_val, color='g', linestyle='--', linewidth=2, label=f'Median: {median_val:,.0f}')
    ax1.set_xlabel(target_col)
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram (Linear Y-axis)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram with logarithmic y-axis
    ax2 = plt.subplot(2, 3, 2)
    counts, bins, patches = ax2.hist(y, bins=n_bins, edgecolor='black', alpha=0.7)
    ax2.set_yscale('log')
    ax2.axvline(mean_val, color='r', linestyle='--', linewidth=2, label=f'Mean: {mean_val:,.0f}')
    ax2.axvline(median_val, color='g', linestyle='--', linewidth=2, label=f'Median: {median_val:,.0f}')
    ax2.set_xlabel(target_col)
    ax2.set_ylabel('Frequency (log scale)')
    ax2.set_title('Histogram (Logarithmic Y-axis)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Boxplot
    ax3 = plt.subplot(2, 3, 3)
    bp = ax3.boxplot([y], vert=True, patch_artist=True, 
                     boxprops=dict(facecolor='lightblue', alpha=0.7))
    ax3.set_ylabel(target_col)
    ax3.set_title('Boxplot')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add outlier count to boxplot
    outlier_positions = bp['fliers'][0].get_data()[1]
    n_plot_outliers = len(outlier_positions)
    ax3.text(1, np.percentile(y, 99), f'Outliers: {n_plot_outliers}', 
             ha='center', va='bottom', fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. QQ plot against normal distribution
    ax4 = plt.subplot(2, 3, 4)
    stats.probplot(y, dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot (vs Normal Distribution)')
    ax4.grid(True, alpha=0.3)
    
    # 5. Boxplot without outliers (zoomed in)
    ax5 = plt.subplot(2, 3, 5)
    # Filter out outliers for this view
    y_filtered = y[(y >= lower_bound) & (y <= upper_bound)]
    if len(y_filtered) > 0:
        bp2 = ax5.boxplot([y_filtered], vert=True, patch_artist=True,
                          boxprops=dict(facecolor='lightgreen', alpha=0.7))
        ax5.set_ylabel(target_col)
        ax5.set_title(f'Boxplot (Excluding {n_outliers} Outliers)')
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, 'No data after outlier removal', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Boxplot (Excluding Outliers)')
    
    # 6. Histogram of absolute values
    ax6 = plt.subplot(2, 3, 6)
    y_abs = np.abs(y)
    ax6.hist(y_abs, bins=n_bins, edgecolor='black', alpha=0.7, color='orange')
    ax6.axvline(np.mean(y_abs), color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(y_abs):,.0f}')
    ax6.axvline(np.median(y_abs), color='g', linestyle='--', linewidth=2, 
                label=f'Median: {np.median(y_abs):,.0f}')
    ax6.set_xlabel(f'|{target_col}|')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Histogram (Absolute Values)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = f'/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/target_distribution_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Plots saved to: {output_path}")
    print(f"{'='*80}")
    
    # Additional statistical tests
    print(f"\n{'='*80}")
    print("STATISTICAL TESTS FOR NORMALITY")
    print(f"{'='*80}")
    
    # Initialize statistics dictionary for saving
    stats_dict = {
        'Metric': [],
        'Value': [],
        'Interpretation': []
    }
    
    # Add basic statistics
    stats_dict['Metric'].extend([
        'Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Range',
        'Skewness', 'Kurtosis', 'Sample Size'
    ])
    stats_dict['Value'].extend([
        f'{mean_val:,.2f}', f'{median_val:,.2f}', f'{std_val:,.2f}',
        f'{min_val:,.2f}', f'{max_val:,.2f}', f'{range_val:,.2f}',
        f'{skewness:.4f}', f'{kurtosis:.4f}', f'{n_samples:,}'
    ])
    stats_dict['Interpretation'].extend([
        'Average value',
        'Middle value (50th percentile)',
        'Standard deviation',
        'Minimum value',
        'Maximum value',
        'Range (Max - Min)',
        f'{"Highly skewed" if abs(skewness) > 2 else "Moderately skewed" if abs(skewness) > 1 else "Approximately symmetric"} (normal=0)',
        f'{"Heavy tails" if abs(kurtosis) > 2 else "Normal tails"} (normal=0)',
        'Number of samples'
    ])
    
    # Add percentiles
    for p in percentiles:
        val = np.percentile(y, p)
        stats_dict['Metric'].append(f'{p}th Percentile')
        stats_dict['Value'].append(f'{val:,.2f}')
        stats_dict['Interpretation'].append(f'{p}% of values are below this')
    
    # Add special values
    stats_dict['Metric'].extend(['Zeros', 'Negative Values', 'Positive Values'])
    stats_dict['Value'].extend([
        f'{n_zero} ({n_zero/n_samples*100:.2f}%)',
        f'{n_negative} ({n_negative/n_samples*100:.2f}%)',
        f'{n_positive} ({n_positive/n_samples*100:.2f}%)'
    ])
    stats_dict['Interpretation'].extend([
        'Number (percentage) of zero values',
        'Number (percentage) of negative values',
        'Number (percentage) of positive values'
    ])
    
    # Add outlier statistics
    stats_dict['Metric'].extend([
        'Q1 (25th percentile)', 'Q3 (75th percentile)', 'IQR',
        'Lower Bound (Q1 - 1.5*IQR)', 'Upper Bound (Q3 + 1.5*IQR)',
        'Number of Outliers', 'Outlier Percentage'
    ])
    stats_dict['Value'].extend([
        f'{q1:,.2f}', f'{q3:,.2f}', f'{iqr:,.2f}',
        f'{lower_bound:,.2f}', f'{upper_bound:,.2f}',
        f'{n_outliers}', f'{n_outliers/n_samples*100:.2f}%'
    ])
    stats_dict['Interpretation'].extend([
        'First quartile', 'Third quartile', 'Interquartile range',
        'Lower outlier threshold', 'Upper outlier threshold',
        'Number of outliers (IQR method)',
        'Percentage of outliers'
    ])
    if n_outliers > 0:
        stats_dict['Metric'].extend(['Outlier Min', 'Outlier Max'])
        stats_dict['Value'].extend([
            f'{np.min(outliers):,.2f}', f'{np.max(outliers):,.2f}'
        ])
        stats_dict['Interpretation'].extend([
            'Minimum outlier value', 'Maximum outlier value'
        ])
    
    # Shapiro-Wilk test (for n < 5000)
    shapiro_stat = None
    shapiro_p = None
    if n_samples < 5000:
        try:
            shapiro_stat, shapiro_p = stats.shapiro(y)
            print(f"  Shapiro-Wilk test:")
            print(f"    Statistic: {shapiro_stat:.4f}")
            print(f"    p-value:   {shapiro_p:.4e}")
            if shapiro_p < 0.05:
                print(f"    → Distribution is NOT normal (p < 0.05)")
                shapiro_interp = 'Distribution is NOT normal (p < 0.05)'
            else:
                print(f"    → Distribution appears normal (p >= 0.05)")
                shapiro_interp = 'Distribution appears normal (p >= 0.05)'
            
            stats_dict['Metric'].append('Shapiro-Wilk Statistic')
            stats_dict['Value'].append(f'{shapiro_stat:.6f}')
            stats_dict['Interpretation'].append('Test statistic')
            
            stats_dict['Metric'].append('Shapiro-Wilk p-value')
            stats_dict['Value'].append(f'{shapiro_p:.6e}')
            stats_dict['Interpretation'].append(shapiro_interp)
        except:
            print(f"  Shapiro-Wilk test: Not applicable (sample size)")
    
    # D'Agostino's K² test (more robust for larger samples)
    dagostino_stat = None
    dagostino_p = None
    try:
        dagostino_stat, dagostino_p = stats.normaltest(y)
        print(f"\n  D'Agostino's K² test:")
        print(f"    Statistic: {dagostino_stat:.4f}")
        print(f"    p-value:   {dagostino_p:.4e}")
        if dagostino_p < 0.05:
            print(f"    → Distribution is NOT normal (p < 0.05)")
            dagostino_interp = 'Distribution is NOT normal (p < 0.05)'
        else:
            print(f"    → Distribution appears normal (p >= 0.05)")
            dagostino_interp = 'Distribution appears normal (p >= 0.05)'
        
        stats_dict['Metric'].append('D\'Agostino K² Statistic')
        stats_dict['Value'].append(f'{dagostino_stat:.6f}')
        stats_dict['Interpretation'].append('Test statistic')
        
        stats_dict['Metric'].append('D\'Agostino K² p-value')
        stats_dict['Value'].append(f'{dagostino_p:.6e}')
        stats_dict['Interpretation'].append(dagostino_interp)
    except:
        pass
    
    # Kolmogorov-Smirnov test against normal distribution
    ks_stat = None
    ks_p = None
    try:
        # Standardize the data
        y_standardized = (y - mean_val) / std_val
        ks_stat, ks_p = stats.kstest(y_standardized, 'norm')
        print(f"\n  Kolmogorov-Smirnov test (vs normal):")
        print(f"    Statistic: {ks_stat:.4f}")
        print(f"    p-value:   {ks_p:.4e}")
        if ks_p < 0.05:
            print(f"    → Distribution significantly differs from normal (p < 0.05)")
            ks_interp = 'Distribution significantly differs from normal (p < 0.05)'
        else:
            print(f"    → Distribution similar to normal (p >= 0.05)")
            ks_interp = 'Distribution similar to normal (p >= 0.05)'
        
        stats_dict['Metric'].append('Kolmogorov-Smirnov Statistic')
        stats_dict['Value'].append(f'{ks_stat:.6f}')
        stats_dict['Interpretation'].append('Test statistic')
        
        stats_dict['Metric'].append('Kolmogorov-Smirnov p-value')
        stats_dict['Value'].append(f'{ks_p:.6e}')
        stats_dict['Interpretation'].append(ks_interp)
    except:
        pass
    
    # Save statistics to CSV
    stats_df = pd.DataFrame(stats_dict)
    stats_csv_path = f'/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/target_distribution_statistics.csv'
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"\n{'='*80}")
    print(f"Statistics saved to: {stats_csv_path}")
    print(f"{'='*80}")
    
    plt.show()
    
    return {
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'n_outliers': n_outliers,
        'outlier_percentage': n_outliers/n_samples*100,
        'n_negative': n_negative,
        'n_positive': n_positive,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'dagostino_stat': dagostino_stat,
        'dagostino_p': dagostino_p,
        'ks_stat': ks_stat,
        'ks_p': ks_p
    }

def main():
    """Main execution function"""
    # Load data
    print("Loading data...")
    df = pd.read_csv('/Users/janakipantham/Desktop/drug_shortage_platform/drug_shortage_cost_prediction/hospital_ops_updated.csv')
    
    target_col = 'Net Income'
    
    # Check if target column exists
    if target_col not in df.columns:
        print(f"ERROR: Target column '{target_col}' not found in dataset")
        print(f"Available columns: {df.columns.tolist()}")
        return
    
    # Run analysis
    stats_dict = analyze_target_distribution(df, target_col=target_col)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
