export const modelSynopsis = {
  shortage: {
    title: "Drug Shortage Classification Model",
    subtitle: "Predicts whether a generic drug will experience a shortage (shortage_flag = 1)",
    pipeline: [
      {
        step: "Data source",
        detail:
          "5,745 drug products from NADAC pricing signals merged with ASHP shortage records (price_signals_complete.csv).",
      },
      {
        step: "Features (5)",
        detail:
          "avg_nadac (acquisition cost), manufacturer_num, ingredient_num, num_forms, and liquid_flag (1 = injectable/liquid, 0 = solid).",
      },
      {
        step: "Outlier removal",
        detail:
          "Class-aware comparison of Mahalanobis distance, Isolation Forest, and Local Outlier Factor (10% contamination per class). The most conservative method is selected — 574 rows removed (10.0%), leaving 5,171 clean samples.",
      },
      {
        step: "Train / test split",
        detail:
          "Stratified 70/30 split on shortage_flag (random_state=42). Training set balanced with SMOTE oversampling to 6,178 samples (50/50 classes).",
      },
      {
        step: "Scaling",
        detail: "StandardScaler fit on SMOTE-augmented training data, applied to the held-out test set.",
      },
      {
        step: "Model selection",
        detail:
          "16 classifiers tuned with RandomizedSearchCV (3-fold stratified CV, accuracy scoring). Top 3 models — ExtraTrees, RandomForest, GradientBoosting — combined into EnsembleTop3 via soft probability averaging at threshold 0.5.",
      },
      {
        step: "Feature importance",
        detail:
          "Permutation importance computed on the held-out test set for EnsembleTop3 (shortage_prediction_best_feature_importance.csv). Native tree importances are not available for the soft-vote ensemble.",
      },
    ],
    results: [
      { metric: "Best model", value: "EnsembleTop3" },
      { metric: "Test accuracy", value: "90.08%" },
      { metric: "ROC AUC", value: "94.29%" },
      { metric: "Precision", value: "63.03%" },
      { metric: "Recall (sensitivity)", value: "78.51%" },
      { metric: "F1 score", value: "69.92%" },
      { metric: "Specificity", value: "92.07%" },
    ],
    featureImportance: [
      {
        feature: "avg_nadac",
        label: "Drug Acquisition Cost (avg NADAC)",
        permutationImportance: 0.2299,
        permutationStd: 0.0083,
        rank: 1,
      },
      {
        feature: "num_forms",
        label: "Number of Dosage Forms",
        permutationImportance: 0.1911,
        permutationStd: 0.0083,
        rank: 2,
      },
      {
        feature: "ingredient_num",
        label: "Number of Ingredients",
        permutationImportance: 0.1282,
        permutationStd: 0.0089,
        rank: 3,
      },
      {
        feature: "manufacturer_num",
        label: "Number of Manufacturers",
        permutationImportance: 0.0981,
        permutationStd: 0.0061,
        rank: 4,
      },
      {
        feature: "liquid_flag",
        label: "Injectable / Liquid Form",
        permutationImportance: 0.0838,
        permutationStd: 0.0079,
        rank: 5,
      },
    ],
    featureImportanceSource: "shortage_prediction_best_feature_importance.csv",
  },
  income: {
    title: "Hospital Net Income Regression Model",
    subtitle: "Predicts hospital net income in dollars from operational and financial features",
    pipeline: [
      {
        step: "Data source",
        detail:
          "~2,500 hospital cost reports (hospital_ops_updated.csv) with CMS financial and operational variables.",
      },
      {
        step: "Information gain feature screening",
        detail:
          "Mutual information (information gain) ranked all candidate predictors against Net Income. Top features included Total Days, Less Total Operating Expense, Inventory, Total Bed Days Available, and Total Assets (IG up to 0.31 for utilization days).",
      },
      {
        step: "Cross-correlation analysis",
        detail:
          "Pearson correlation matrix identified highly collinear hospital utilization pairs (r > 0.97 between discharge and day counts). Nonlinear correlation analysis selected 14 features with optimal transforms (Original, Square, Sqrt, or Log) to maximize |correlation| with Net Income — e.g. Total Liabilities → Square (0.44), Hospital Beds → Log (0.33).",
      },
      {
        step: "Feature set (15)",
        detail:
          "14 correlation-selected features plus Profit/Loss flag (encoded 1/0). Transforms applied per nonlinear_correlations.csv before modeling.",
      },
      {
        step: "Outlier removal",
        detail:
          "Isolation Forest on standardized features + target (2% contamination, fit on training only). Removes multivariate extremes while preserving legitimate large hospitals — superior to univariate IQR on heavily right-skewed Net Income.",
      },
      {
        step: "Train / val / test split",
        detail:
          "Stratified 60/20/20 split on Net Income quantile strata to preserve income distribution across splits.",
      },
      {
        step: "Model selection",
        detail:
          "12 regressors tuned with GridSearchCV minimizing sMAPE on dollar Net Income. RandomForestRegressor selected as best performer (lowest test sMAPE per model_performance_results.csv).",
      },
    ],
    results: [
      { metric: "Best model", value: "RandomForestRegressor" },
      { metric: "Test sMAPE", value: "64.11%" },
      { metric: "Test RMSE", value: "$26,060,227" },
      { metric: "Test R²", value: "0.686" },
      { metric: "Test Adjusted R²", value: "0.676" },
      { metric: "Sign prediction accuracy", value: "100%" },
      { metric: "Outliers removed (train)", value: "~2% via Isolation Forest" },
    ],
    topInformationGain: [
      { feature: "Total Days (V + XVIII + XIX + Unknown)", ig: "0.314" },
      { feature: "Less Total Operating Expense", ig: "0.290" },
      { feature: "Less Contractual Allowance and Discounts", ig: "0.290" },
      { feature: "Inventory", ig: "0.276" },
      { feature: "Total Bed Days Available", ig: "0.257" },
      { feature: "Total Assets", ig: "0.254" },
    ],
    topCorrelations: [
      { feature: "Total Assets", transform: "Original", r: "0.543" },
      { feature: "Inventory", transform: "Original", r: "0.528" },
      { feature: "Less Total Operating Expense", transform: "Original", r: "0.498" },
      { feature: "Total Liabilities", transform: "Square", r: "0.444" },
      { feature: "Hospital Number of Beds", transform: "Log", r: "0.328" },
      { feature: "Total Current Assets", transform: "Sqrt", r: "0.314" },
    ],
  },
};
