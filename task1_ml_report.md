
# California House Price Prediction Using Machine Learning

**Project:** Artificial Intelligence and Machine Learning — Task 1  
**Author:** Vishalini  
**Technology:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## 1. Introduction and Objective

House price prediction is a regression problem in machine learning. The objective of this project is to build a model that predicts median house values using housing-related features from the California Housing dataset.

The project explores the dataset, visualizes important relationships, trains two regression models, and compares their performance using standard evaluation metrics.

## 2. Dataset and Exploratory Data Analysis

The California Housing dataset contains 20,640 records and nine columns, including eight input features and one target variable, `MedHouseVal`.

The input features include median income, house age, average rooms, average bedrooms, population, average occupancy, latitude, and longitude. The target variable represents median house value in units of $100,000.

The dataset was inspected for missing values and summarized using descriptive statistics. No missing values were found.

Histograms, scatter plots, and a correlation heatmap were used to understand the distribution of house values and relationships between features. The scatter plot showed a positive relationship between median income and median house value.

## 3. Methodology

The dataset was divided into input features (X) and the target variable (y). It was then split into training and testing sets using an 80:20 ratio and `random_state=42`.

Two regression algorithms were trained:

- **Linear Regression:** Used as the baseline model to learn a linear relationship between the input features and house values.
- **Random Forest Regressor:** Used to learn more complex relationships by combining predictions from multiple decision trees.

Both models were evaluated on the same test set using Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and the R² score.

## 4. Results and Comparison

| Metric | Linear Regression | Random Forest |
|---|---:|---:|
| MAE | 0.5332 | 0.3277 |
| MSE | 0.5559 | 0.2560 |
| RMSE | 0.7456 | 0.5060 |
| R² Score | 0.5758 | 0.8046 |

The Random Forest model performed better than Linear Regression on the test dataset. It achieved lower MAE, MSE, and RMSE values and a higher R² score.

The Random Forest MAE of approximately 0.3277 corresponds to about $32,773 in the dataset's dollar scale. Its RMSE of approximately 0.5060 corresponds to about $50,599. These values describe errors on this test set and do not guarantee accuracy for individual real-world properties.

## 5. Conclusion and Future Improvements

This project demonstrated the process of developing and evaluating machine learning regression models for house price prediction. Exploratory data analysis helped reveal patterns in the dataset, while model evaluation provided a quantitative comparison of the two algorithms.

The Random Forest Regressor was the better-performing model among the two tested. It was saved using Pickle and reloaded to test a prediction.

Future work could include hyperparameter tuning, cross-validation, feature engineering, and testing with newer housing data. The target values in this dataset are capped, which limits the model's ability to represent some higher-priced properties. Therefore, predictions should be treated as estimates rather than guaranteed market prices.

## 6. Tools and Libraries

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Pickle
- Visual Studio Code and Jupyter Notebook
