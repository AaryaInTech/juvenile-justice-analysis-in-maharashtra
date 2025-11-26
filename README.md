# juvenile-justice-analysis-in-maharashtra

# Data_Science_Lab

# Juvenile Crime Analysis in Maharashtra (2017–2022)

A comprehensive **Data Science Mini-Project** analyzing juvenile crime patterns across districts in Maharashtra, India, using **Machine Learning**, **Deep Learning**, and **Time-Series Forecasting** techniques.

---

## Overview

This project performs **Exploratory Data Analysis (EDA)** and applies multiple **predictive modeling** approaches to uncover patterns, trends, and insights into **juvenile crimes** in Maharashtra between **2017 and 2022**.

**Dataset:** [`districtwise-sll-crime-by-juveniles-2017-onwards.csv`](https://indiadataportal.com/p/crime-statistics/r/ncrb-cii_sll_crime_by_juveniles-dt-yr-aaa)
**Source:** India Data Portal (NCRB Crime Statistics)

---

## Objectives

* Analyze juvenile crime distribution across Maharashtra’s districts
* Identify temporal trends, anomalies, and correlations
* Build regression and forecasting models for predictive insights
* Compare multiple ML/DL models for performance
* Develop an **interactive Streamlit dashboard** for visualization

---

## Key Findings

* Major crime contributors: **POCSO**, **Arms Act**, and **Liquor/Narcotics violations**
* **Chandrapur** and **Mumbai** recorded highest juvenile crime rates
* Crime patterns show high variance between urban and rural districts
* Forecast models predict a **slight upward trend** in future years
* **SVM and Prophet** provided best predictive performance

---

## 🛠️ Tech Stack

| Category                    | Tools / Libraries                         |
| --------------------------- | ----------------------------------------- |
| **Language**                | Python 3.x                                |
| **Data Processing**         | pandas, numpy                             |
| **Visualization**           | matplotlib, seaborn, plotly               |
| **Machine Learning**        | scikit-learn, xgboost, lightgbm, catboost |
| **Time-Series Forecasting** | statsmodels, prophet                      |
| **Deep Learning**           | tensorflow, keras                         |
| **App / Deployment**        | streamlit                                 |

##

```
```

---

## Data Cleaning

```bash
```

* Filled missing values (`NaN`) with **0** for numeric crime columns
* Dropped unnecessary columns (`id`, `registration_circles`)
* Created a `total_crime` column (sum of all crime types)
* Filtered data to include **Maharashtra** only
* Removed outliers using **Z-score method**

---

## Exploratory Data Analysis (EDA)

Includes:

* District-wise total crime distribution
* Year-wise crime trend (2017–2022)
* Correlation heatmaps and pairplots
* Outlier visualization using boxplots
* Top and bottom districts by total crimes

---

## Machine Learning Models Used

### **Supervised Learning**

* Linear Regression
* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* Gradient Boosting
* XGBoost
* LightGBM
* CatBoost

### **Unsupervised Learning**

* K-Means Clustering
* Hierarchical Clustering
* PCA (Principal Component Analysis)
* t-SNE

### **Time-Series Forecasting**

* ARIMA
* SARIMA
* LSTM
* Prophet

### **Deep Learning**

* Artificial Neural Networks (ANN)
* Recurrent Neural Networks (RNN)
* Autoencoders

### **Other Methods**

* Feature Engineering
* Dimensionality Reduction
* Ensemble Models

---

## 📈 Model Performance Summary

| Model             | R²    | MAE  | RMSE | Performance  |
| ----------------- | ----- | ---- | ---- | ------------ |
| Linear Regression | 0.989 | 0.51 | 0.79 | Excellent* |
| SVM               | 0.904 | 1.15 | 2.37 | Excellent  |
| CatBoost          | 0.757 | 1.80 | 3.76 | Very Good  |
| XGBoost           | 0.754 | 1.72 | 3.79 | Very Good  |
| Gradient Boosting | 0.684 | 1.86 | 4.29 | Good       |
| Random Forest     | 0.583 | 2.10 | 4.93 | Good       |
| LightGBM          | 0.551 | 2.68 | 5.11 | Good       |
| Decision Tree     | 0.391 | 2.71 | 5.95 | Fair      |

**Forecasting Models:**

* Prophet: R² = 0.95 (Best Forecaster)
* ARIMA/SARIMA: Handled seasonality well
* LSTM: Captured long-term temporal patterns

---

## Visual Insights

* Interactive charts for yearly and district-wise analysis
* Correlation heatmap of top crime types
* Cluster visualization using K-Means and PCA
* Forecast visualization for 2023–2025
* Model performance comparison charts

---

## Statistical Summary

| Metric                 | Value                 |
| ---------------------- | --------------------- |
| Total Records          | 289                   |
| Years Covered          | 2017–2022             |
| Districts              | 36                    |
| Crime Categories       | 70+                   |
| Outliers Removed       | 10 (3.46%)            |
| Best Regression Model  | SVM (R² = 0.9037)     |
| Best Forecasting Model | Prophet (R² = 0.9525) |
| PCA Variance Retained  | 95%                   |
| Optimal Clusters       | 3–5                   |

---

## Streamlit Dashboard

### Features:

* District-wise crime visualization
* Yearly trends and comparisons
* Forecasting insights
* Interactive filters and charts
* Model performance summary

Run the app:

```bash
streamlit run app.py
```

---

## Conclusion

* **Geographic disparity:** Major differences across districts
* **Predictive strength:** SVM and ensemble models performed best
* **Time-series forecasting:** Prophet produced accurate forecasts
* **Feature insights:** POCSO crimes had strongest correlation with total crimes
* **Dashboard:** Provides clear and accessible insights for stakeholders

---

## Author

**Aarya Gourkar**
PRN: 22070521038
Semester VII – Section A
Course Code: TE7253 – Data Science
