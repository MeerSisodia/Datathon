# Datathon

# Toronto Real Estate Price Prediction

## Overview

This Python script (`datathon.ipnyb`) is designed to predict real estate prices in Toronto using a dataset of housing information. The script performs Exploratory Data Analysis (EDA), data preprocessing, feature engineering, and implements several machine learning models to predict house prices. The models used include Random Forest Regressor, Extra Trees Regressor, XGBoost Regressor, and a Feed Forward Neural Network, along with an ensemble model that averages the predictions of these individual models.

This project aims to provide insights into factors influencing housing prices in Toronto and to build a predictive model that can estimate house prices based on a set of features.

**AI Assistance:** It's important to note that Artificial Intelligence tools were utilized in the code development process for efficiency and code generation, particularly in structuring some parts of the data cleaning and model fitting sections.

## Data Source

The dataset used for this project is `real-estate-data.csv`, which should be placed in the same directory as the script or accessible via the Google Drive path specified in the script if running on Google Colab. The dataset contains features related to real estate properties and their corresponding prices.

## Methodology

The script follows these key steps:

1.  **Exploratory Data Analysis (EDA):**
    *   Loads the dataset and displays the first few rows (`df.head()`).
    *   Visualizes the distribution of house prices using a histogram (`plt.hist(df['price'])`).
    *   Generates a pairplot (`sns.pairplot(df)`) to examine pairwise correlations between features.
    *   Identifies missing values in the dataset (`df.isnull().sum()`).

2.  **Data Preprocessing:**
    *   **Data Cleaning Pipeline:** A `Pipeline` named `data_pipeline` is constructed to handle data cleaning and imputation:
        *   **`RealEstateDataCleaner` Transformer:**
            *   Drops the `id_` column.
            *   Removes rows with missing `price` values.
            *   Encodes categorical columns (`size`, `exposure`, `DEN`, `parking`, `ward`) into numerical representations using predefined mappings (`size_mapping`, `exposure_mapping`, `den_mapping`, `parking_mapping`, `ward_mapping`).
        *   **`SeparateImputer` Transformer:**
            *   Imputes missing values in categorical columns (`beds`, `size`) using `KNNCategoricalImputer` (KNN-based categorical imputation).
            *   Imputes missing values in numerical columns (`maint`, `D_mkt`) using `IterativeImputer` (MICE - Multiple Imputation by Chained Equations).
    *   **Data Refinement:**
        *   Removes rows where `D_mkt` is not an integer after imputation to ensure data consistency.
        *   Creates a scatter plot of 'size' vs 'price' to visually inspect their relationship.
        *   Creates a 'bed\_bath\_ratio' feature, visualizes its correlation with 'price' and 'maint', and then drops this ratio feature.
        *   Visualizes house prices geographically on a Toronto map using `plotly.express`, categorizing prices into 'Low', 'Mid', and 'High' quantiles. This visualization uses latitude (`lt`) and longitude (`lg`) data.

3.  **Feature Engineering:**
    *   **Local Average Price Feature:**
        *   The function `compute_local_avg_price` calculates the average price of houses within a 700-meter radius (`radius_km = 0.7`) for each house, considering houses with the same number of bedrooms (`beds`).
        *   It uses a `BallTree` for efficient nearest neighbor searches based on latitude (`lt`) and longitude (`lg`).
        *   Two new features are created: `local_avg_price` (the computed average price) and `local_neighbor_count` (the number of neighbors used for the average).
    *   Latitude (`lt`) and longitude (`lg`) columns are dropped after feature engineering as they are no longer directly used in the models.

4.  **Model Fitting and Evaluation:**
    *   **Data Splitting:** The cleaned and engineered dataset is split into training and testing sets (`train_test_split`) with an 80/20 ratio.
    *   **Feature Scaling:** Numerical features in both training and testing sets are scaled using `StandardScaler`.
    *   **Regression Models:** Four regression models are trained and evaluated:
        *   **Random Forest Regressor (`RandomForestRegressor`)**
        *   **Extra Trees Regressor (`ExtraTreesRegressor`)**
        *   **XGBoost Regressor (`XGBRegressor`)**
        *   **Feed Forward Neural Network (`Sequential` from Keras):** A simple neural network with ReLU activation functions and an output layer for regression.
    *   **Ensemble Model:** An ensemble prediction is created by averaging the predictions of all four models.
    *   **Evaluation Metric:** Root Mean Squared Error (RMSE) is used to evaluate the performance of each model and the ensemble model.
    *   **Custom Accuracy Metric:** A custom "accuracy" metric is calculated, defining a prediction as "accurate" if it falls within 15% of the actual price.
    *   **Results Visualization:**
        *   For each model (Random Forest, Extra Trees, XGBoost, Neural Network, Ensemble), scatter plots of "Predicted Price vs Actual Price" and histograms of "Error Distribution" are generated to visually assess model performance.
        *   A summary table (`results`) is printed, comparing the RMSE of all models.

## Key Variables

*   **`df`**: Initial Pandas DataFrame loaded from `real-estate-data.csv`.
*   **`cleaned_df`**: Pandas DataFrame after applying the `data_pipeline` for cleaning and imputation.
*   **`data_pipeline`**: Scikit-learn `Pipeline` object encapsulating data cleaning and imputation steps.
*   **`X_train`, `y_train`, `X_test`, `y_test`**: Training and testing feature matrices and target vectors (price).
*   **`X_train_scaled`, `X_test_scaled`**: Scaled training and testing feature matrices after applying `StandardScaler`.
*   **`rf`, `et`, `xgb`, `nn_model`**: Trained machine learning models (Random Forest, Extra Trees, XGBoost, Neural Network).
*   **`rf_pred`, `et_pred`, `xgb_pred`, `nn_pred`**: Price predictions from each individual model on the test set.
*   **`ensemble_pred`**: Ensemble price predictions, averaged from individual model predictions.
*   **`rf_rmse`, `et_rmse`, `xgb_rmse`, `loss` (NN Loss), `ensemble_rmse`**: Root Mean Squared Error values for each model.
*   **`accuracy`**: Custom accuracy metric for the Random Forest model (percentage of predictions within 15% error).
*   **`errors`, `errors_et`, `errors_xgb`, `errors_ensemble`, `errors_rf`**: Error arrays (Predicted - Actual prices) for each model used for error distribution plots.

## Model Performance

The script outputs a table comparing the RMSE for each model and the ensemble model. The Ensemble model generally shows the lowest RMSE, indicating better predictive performance compared to individual models in terms of RMSE. However, when considering the custom accuracy metric (predictions within 15% error), the Random Forest Regressor shows a competitive result. Visualizations are provided to further analyze the performance of each model, showing predicted vs. actual prices and error distributions.

## Dependencies

*   `pandas`
*   `math`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `sklearn` (scikit-learn): `BaseEstimator`, `TransformerMixin`, `Pipeline`, `IterativeImputer`, `NearestNeighbors`, `train_test_split`, `StandardScaler`, `mean_squared_error`, `RandomForestRegressor`, `ExtraTreesRegressor`
*   `scipy`: `stats.mode`
*   `xgboost`
*   `keras`

To install the required packages, you can use pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost keras
