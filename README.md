# Datathon

# Toronto Real Estate Price Prediction Project

## Overview

This repository contains a Python script (`datathon.py`) designed to predict real estate prices in Toronto. The project leverages a dataset of Toronto housing information to perform comprehensive Exploratory Data Analysis (EDA), rigorous data preprocessing, insightful feature engineering, and employs a suite of machine learning models to accurately estimate house prices.

The predictive models implemented include:

*   **Random Forest Regressor**
*   **Extra Trees Regressor**
*   **XGBoost Regressor**
*   **Feed Forward Neural Network**

Furthermore, an **ensemble model** is constructed to average the predictions from these individual models, aiming to enhance overall prediction accuracy and robustness.

This project provides a detailed analysis of factors influencing Toronto's housing market and delivers a robust predictive tool for estimating property values based on a variety of features.

**AI Assistance in Development:**  In the interest of development efficiency and code optimization, Artificial Intelligence tools played a role in structuring and generating certain segments of the codebase. This was particularly beneficial in streamlining the data cleaning processes and model fitting routines, allowing for a more focused approach on the core analytical and predictive aspects of the project.

## Repository Structure

The repository is organized as follows:

*   `.idea/`: Contains IntelliJ IDEA project files (IDE configuration).
*   `SDSS Datathon Cases/`: Potentially contains case-related data or documents from the SDSS Datathon (context-specific folder, may not be directly relevant to running the script).
*   `01_hierarchical_clustering.ipynb`: Jupyter Notebook exploring Hierarchical Clustering techniques. While investigated, Hierarchical Clustering was **not directly used** in the final price prediction model implemented in `datathon.py`. Associated data output from this exploration might have been saved as `data_clusters.csv`.
*   `EDA.ipynb`, `EDA2.ipynb`, `EDA3.ipynb`: Jupyter Notebooks containing Exploratory Data Analysis (EDA) carried out on the dataset. These notebooks informed the data cleaning, preprocessing, and feature engineering steps implemented in the `datathon.py` script.
*   `README.md`: This file, providing an overview of the project.
*   `cleaned_data.csv`: Likely an intermediate or pre-processed version of the dataset, potentially generated during the EDA phase. Note: The primary input data should be `real-estate-data.csv`.
*   `data_clusters.csv`: Data file potentially containing cluster assignments or results from Hierarchical Clustering exploration (not used in the final prediction model).
*   `datathon.py`: The main Python script containing the complete data processing, feature engineering, model training, and evaluation pipeline for real estate price prediction.
*   `real-estate-data.csv`: **The primary dataset** for this project, containing Toronto real estate data. Ensure this file is placed in the same directory as `datathon.py` or update the file path within the script.

## Methodology in Detail

The `datathon.py` script executes the following steps:

1.  **Exploratory Data Analysis (EDA):**
    *   Initial dataset inspection using `df.head()`.
    *   Visualization of house price distribution via histograms (`plt.hist(df['price'])`).
    *   Pairwise correlation analysis of predictors using `sns.pairplot(df)`.
    *   Identification of missing values using `df.isnull().sum()`.
    *   Geographic visualization of house prices on a Toronto map using `plotly.express`, categorized by price quantiles and sized by price, using latitude (`lt`) and longitude (`lg`) data.

2.  **Data Preprocessing Pipeline:**
    *   Implementation of a `data_pipeline` using scikit-learn `Pipeline` for streamlined data cleaning and imputation.
    *   **`RealEstateDataCleaner` Transformer:**
        *   Drops irrelevant `id_` column.
        *   Removes rows with missing `price` values.
        *   Categorical feature encoding for columns like `size`, `exposure`, `DEN`, `parking`, and `ward` using predefined mappings (`size_mapping`, `exposure_mapping`, `den_mapping`, `parking_mapping`, `ward_mapping`).
    *   **`SeparateImputer` Transformer:**
        *   Missing value imputation for categorical features (`beds`, `size`) using `KNNCategoricalImputer` (KNN-based).
        *   Missing value imputation for numerical features (`maint`, `D_mkt`) using `IterativeImputer` (MICE).
    *   Removal of rows with non-integer values in the `D_mkt` column post-imputation to ensure data integrity.

3.  **Feature Engineering:**
    *   **`compute_local_avg_price` Function:**
        *   Calculates `local_avg_price`: For each house, computes the average price of neighboring houses (within a 700-meter radius, `radius_km = 0.7`) with the same number of bedrooms, using a `BallTree` for efficient spatial queries based on latitude (`lt`) and longitude (`lg`).
        *   Calculates `local_neighbor_count`:  Counts the number of neighboring houses used to compute `local_avg_price`.
    *   Latitude (`lt`) and longitude (`lg`) columns are dropped after engineering local price features as they are no longer directly used by the predictive models.

4.  **Model Training and Evaluation:**
    *   Dataset split into training (`train_df`) and testing (`test_df`) sets using `train_test_split` (80/20 split).
    *   Feature scaling using `StandardScaler` applied to numerical features in both training (`X_train_scaled`) and testing (`X_test_scaled`) sets.
    *   Training and evaluation of four regression models:
        *   `rf`: `RandomForestRegressor`
        *   `et`: `ExtraTreesRegressor`
        *   `xgb`: `XGBRegressor`
        *   `nn_model`: Feed Forward Neural Network (Keras `Sequential` model).
    *   Ensemble prediction (`ensemble_pred`) by averaging predictions from all four models.
    *   Model evaluation using Root Mean Squared Error (RMSE) and a custom accuracy metric (predictions within 15% of actual price).
    *   Visualizations of model performance: Scatter plots of Predicted vs Actual prices and histograms of error distributions for each model and the ensemble.
    *   Comparison table (`results`) summarizing RMSE for all models.

## Key Variables in `datathon.py`

*   **`df`**:  Initial Pandas DataFrame loaded from `real-estate-data.csv`.
*   **`cleaned_df`**: Pandas DataFrame after preprocessing and imputation via `data_pipeline`.
*   **`data_pipeline`**: Scikit-learn `Pipeline` for data cleaning and imputation.
*   **`X_train`, `y_train`, `X_test`, `y_test`**: Training and testing feature matrices and target variable (house price).
*   **`X_train_scaled`, `X_test_scaled`**: Scaled feature matrices for training and testing data.
*   **`rf`, `et`, `xgb`, `nn_model`**: Instances of trained regression models (Random Forest, Extra Trees, XGBoost, Neural Network).
*   **`rf_pred`, `et_pred`, `xgb_pred`, `nn_pred`**: Price predictions generated by each model on the test set.
*   **`ensemble_pred`**: Averaged price predictions from the ensemble model.
*   **`rf_rmse`, `et_rmse`, `xgb_rmse`, `loss` (NN Loss), `ensemble_rmse`**: Root Mean Squared Error values for each model's performance.
*   **`accuracy`**: Custom accuracy metric (percentage of Random Forest predictions within 15% of the actual price).
*   **`errors`, `errors_et`, `errors_xgb`, `errors_ensemble`, `errors_rf`**: Arrays of prediction errors (Predicted - Actual price) for each model, used for error distribution visualization.

## Model Performance Summary

The project evaluates and compares the performance of four individual regression models and an ensemble model. Based on the Root Mean Squared Error (RMSE), the ensemble model achieves the lowest error, suggesting superior predictive capability in terms of RMSE. However, when considering a custom accuracy metric that measures predictions within a 15% error margin, the Random Forest Regressor demonstrates strong performance. The generated visualizations offer further insights into each model's prediction accuracy and error patterns.

## Dependencies

*   `pandas`
*   `math`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scipy`
*   `sklearn` (scikit-learn)
*   `xgboost`
*   `keras`

Install dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy xgboost keras plotly_express
