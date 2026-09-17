# Porter Delivery Time Prediction

## Project Overview

The objective of this project is to build a regression model that predicts food delivery time in minutes.

The project uses order information, restaurant-related information, and delivery-partner availability to estimate how long an order may take to reach the customer.

The model can provide a baseline for understanding the factors associated with food delivery time and support more accurate delivery-time estimation.

---

## Business Objective

The main objectives of this project are:

- Predict food delivery time in minutes.
- Understand the factors associated with delivery delays.
- Analyze order, restaurant, and delivery-partner information.
- Identify patterns affecting delivery time.
- Build an interpretable regression model.
- Provide a baseline that can be improved with more advanced machine learning models.

---

## Machine Learning Problem

This is a **Regression Problem**.

The target variable is:

- `delivery_time` — total delivery time in minutes.

The target variable was calculated using:

```python
df['delivery_time'] = (
    df['actual_delivery_time'] - df['created_at']
).dt.total_seconds() / 60
```

---

## Dataset Overview

The original dataset contains:

- **197,428 records**
- **14 original columns**

The dataset includes:

- Market ID
- Order Creation Time
- Actual Delivery Time
- Store ID
- Store Primary Category
- Order Protocol
- Total Items
- Subtotal
- Number of Distinct Items
- Minimum Item Price
- Maximum Item Price
- Total On-Shift Delivery Partners
- Total Busy Delivery Partners
- Total Outstanding Orders

---

## Tools and Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## Project Workflow

### 1. Data Understanding

The dataset was initially explored using:

- `head()`
- `tail()`
- `shape`
- `columns`
- `info()`
- `describe()`
- Missing-value analysis
- Duplicate-value analysis

This helped understand the dataset structure, data types, distributions, and missing values.

---

## Data Preprocessing

### Date Conversion

The following columns were converted into datetime format:

- `created_at`
- `actual_delivery_time`

### Target Variable Creation

The target variable `delivery_time` was calculated as the difference between actual delivery time and order creation time.

The result was converted from seconds to minutes.

---

## Time-Based Feature Engineering

Two additional time-based features were created:

### Order Hour

```python
df['order_hour'] = df['created_at'].dt.hour
```

This represents the hour of the day when the order was placed.

### Order Day

```python
df['order_day'] = df['created_at'].dt.day_name()
```

This represents the day of the week when the order was placed.

---

## Missing Value Handling

Missing values were present in several columns including:

- Market ID
- Store Primary Category
- Order Protocol
- Total On-Shift Partners
- Total Busy Partners
- Total Outstanding Orders
- Actual Delivery Time

The missing values were handled using:

- Mode for categorical/discrete features.
- Median for delivery-partner-related numerical features.
- Records with missing `actual_delivery_time` were removed because the target delivery time could not be calculated without it.

After preprocessing, the dataset contained no missing values in the features used for analysis.

---

## Categorical Encoding

Categorical variables such as:

- Store Primary Category
- Order Day

were converted into numerical format using `LabelEncoder`.

This allowed the variables to be used in the regression model.

---

## Exploratory Data Analysis

Several visualizations were created to understand the data and identify patterns.

### Restaurant Category Distribution

A count plot was used to understand the frequency of orders across different restaurant categories.

### Delivery Time Distribution

A histogram was used to understand the distribution of delivery times.

### Subtotal vs Delivery Time

A scatter plot was used to examine the relationship between order subtotal and delivery time.

### Total Items vs Delivery Time

A scatter plot was used to check whether orders containing more items tend to take longer to deliver.

### Outlier Analysis

Box plots were used to detect unusual values in:

- Delivery Time
- Subtotal
- Total Items

---

## Outlier Treatment

Significant outliers were found in the delivery-time variable.

The **Interquartile Range (IQR)** method was used to identify and remove extreme delivery-time observations.

The lower and upper limits were calculated using:

```text
Lower Limit = Q1 - 1.5 × IQR
Upper Limit = Q3 + 1.5 × IQR
```

Removing extreme target values helped reduce the effect of unrealistic delivery-time observations on the regression model.

---

## Correlation Analysis

A correlation heatmap was created to examine relationships among numerical features.

This helped understand how order information and delivery-partner availability were associated with delivery time.

---

## Data Preparation for Modeling

The following columns were removed because they were not used directly by the regression model:

- `store_id`
- `created_at`
- `actual_delivery_time`

The target variable was:

```text
delivery_time
```

The remaining variables were used as model features.

---

## Train-Test Split

The dataset was divided into:

- **80% Training Data**
- **20% Testing Data**

A random state of `42` was used to make the split reproducible.

---

## Machine Learning Model

### Linear Regression

Linear Regression was selected as the baseline regression algorithm.

The model was trained on the training dataset and evaluated on unseen testing data.

---

## Model Performance

The Linear Regression model produced the following results:

| Metric | Result |
|---|---:|
| MAE | 10.63 minutes |
| MSE | 178.49 |
| RMSE | 13.36 minutes |
| R² Score | 0.1803 |
| Train R² | 0.1835 |
| Test R² | 0.1803 |

---

## Model Interpretation

### Mean Absolute Error

The **MAE of 10.63 minutes** means that the model's predictions differ from the actual delivery time by approximately 10.6 minutes on average.

### Root Mean Squared Error

The **RMSE of 13.36 minutes** indicates that some larger prediction errors are present.

### R² Score

The model achieved an **R² score of 0.1803**.

This means that the current model explains approximately 18% of the variation in delivery time.

The relatively low R² indicates that many factors affecting delivery time are not captured by the available features or by a simple linear relationship.

---

## Train vs Test Performance

The model achieved:

```text
Train R² = 0.1835
Test R²  = 0.1803
```

The training and testing scores are very similar.

This indicates that the model does not show significant overfitting.

However, both scores are relatively low, suggesting that the Linear Regression model is underfitting the data.

More advanced features or non-linear machine learning models may improve performance.

---

## Actual vs Predicted Analysis

The Actual vs Predicted plot showed that the model captures the general relationship between actual and predicted delivery times.

However, predictions were widely scattered around the ideal prediction line.

The model also showed a tendency to:

- Underestimate some longer delivery times.
- Overestimate some shorter delivery times.

This is consistent with the relatively low R² score.

---

## Residual Analysis

Residuals were calculated using:

```text
Residual = Actual Delivery Time - Predicted Delivery Time
```

The residual plot showed that prediction errors were generally distributed around zero.

However, some structure and spread remained in the residuals, suggesting that Linear Regression does not capture all patterns present in the data.

The residual distribution was approximately centered around zero with a slight right skew.

---

## Model Coefficient Analysis

Some of the largest positive coefficients in the fitted Linear Regression model were:

| Feature | Coefficient |
|---|---:|
| Number of Distinct Items | 0.423 |
| Total Outstanding Orders | 0.284 |

Positive coefficients indicate that increases in these variables are associated with increases in predicted delivery time when other model variables are held constant.

Some negative coefficients included:

| Feature | Coefficient |
|---|---:|
| Order Hour | -0.191 |
| Total On-Shift Partners | -0.301 |
| Order Protocol | -0.698 |

These coefficients describe how the fitted Linear Regression model responds to the encoded input variables.

Because the features use different units and some categorical variables were numerically encoded, raw coefficient magnitudes should be interpreted carefully rather than treated as direct feature-importance rankings.

---

## Key Insights

1. Orders containing more distinct items are associated with longer predicted delivery times.
2. A higher number of outstanding orders is associated with increased predicted delivery time.
3. Delivery-partner availability is associated with delivery-time predictions.
4. Order timing also contributes to delivery-time variation.
5. Several price-related variables have relatively small coefficients in the fitted Linear Regression model.
6. A large portion of delivery-time variation remains unexplained by the current baseline model.

---

## Business Recommendations

### 1. Monitor Outstanding Orders

A high number of outstanding orders can indicate operational load.

Porter can monitor this metric when estimating expected delivery times.

### 2. Consider Delivery Partner Availability

Delivery-partner availability should be considered while estimating expected delivery time.

Periods with fewer available partners may require adjusted delivery-time estimates.

### 3. Consider Order Complexity

Orders with more distinct items may require additional preparation time and should be considered when generating delivery estimates.

### 4. Include Additional Features

Future models could benefit from additional information such as:

- Delivery distance
- Traffic conditions
- Weather
- Restaurant preparation time
- Geographic location
- Peak-hour indicators
- Historical restaurant performance

These variables may help explain additional variation in delivery time.

### 5. Test More Advanced Models

Linear Regression provides an interpretable baseline.

Future work can compare the baseline with models such as:

- Random Forest Regression
- Gradient Boosting
- XGBoost
- LightGBM

These models may capture non-linear relationships more effectively.

---

## Limitations

The Linear Regression model explains only about **18% of the variation in delivery time**.

Therefore, this model should be treated as a baseline rather than a production-ready delivery-time prediction system.

Additional features and more advanced modeling techniques would likely be required to achieve higher predictive accuracy.

---

## Conclusion

This project demonstrates the complete workflow of building a regression model for food delivery-time prediction.

The project included:

- Data understanding
- Missing-value handling
- Date preprocessing
- Feature engineering
- Categorical encoding
- Exploratory Data Analysis
- Outlier treatment
- Train-test splitting
- Linear Regression
- Model evaluation
- Residual analysis
- Model coefficient interpretation

The Linear Regression model achieved:

- **MAE: 10.63 minutes**
- **RMSE: 13.36 minutes**
- **R²: 0.1803**
- **Train R²: 0.1835**
- **Test R²: 0.1803**

The similar training and testing scores indicate that the model generalizes consistently without significant overfitting.

However, the low R² indicates underfitting and suggests that additional features and more advanced models could improve delivery-time predictions.

Overall, this project provides an interpretable baseline and demonstrates how regression analysis can be applied to a real-world delivery-time prediction problem.

---

## Repository Structure

```text
Porter-Delivery-Time-Prediction/
│
├── Porter_Delivery_Time_Prediction.ipynb
├── dataset.csv
├── README.md
└── requirements.txt
```

---

## How to Run the Project

1. Download or clone this repository.
2. Install the required Python libraries.
3. Open the Jupyter Notebook.
4. Keep `dataset.csv` in the project folder.
5. Run the notebook cells from top to bottom.

---

## Author

**Barla Charan**

MBA – Systems with Business Analytics
