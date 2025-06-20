# 🏡 Boston Housing Price Prediction using Decision Tree

This project uses a **Decision Tree Regressor** to predict housing prices from the classic **Boston Housing Dataset**. It includes complete data preprocessing, model training, evaluation, and performance visualization.

---

## 📊 Dataset

- **Source:** [OpenML Boston Housing Dataset](https://www.openml.org/d/531)
- **Features:** 13 input features (e.g., crime rate, number of rooms, tax rate)
- **Target:** Median value of owner-occupied homes (in $1000s)

---

## 🧪 Model Used

- **Model:** Decision Tree Regressor  
- **Hyperparameters:**
  - `criterion='squared_error'`
  - `splitter='best'`
  - `max_depth=8`
  - `max_features='sqrt'`
  - `random_state=26`

---

## 🧼 Preprocessing Steps

- Renamed target column `MEDV` to `PRICE`
- Converted `RAD` and `CHAS` columns to integers
- Checked for missing values
- Performed train-test split (`test_size=0.2`, `random_state=21`)

---

## 📈 Model Evaluation

| Metric                   | Value    |
|--------------------------|----------|
| R² Score                 | 0.8851   |
| Mean Absolute Error (MAE)| 2.4365   |
| Mean Squared Error (MSE) | 10.8156  |
| Root Mean Squared Error  | 3.2887   |

✅ The model explains about **88.5%** of the variance in housing prices with reasonably low prediction error.

---

## 📉 Actual vs Predicted Plot

A scatter plot compares actual vs predicted values, showing how close the model’s predictions are to the real values.
