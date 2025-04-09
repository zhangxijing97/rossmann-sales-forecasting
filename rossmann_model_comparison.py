# =============================================================================
# 📦 1. Imports & Utility Functions
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import lightgbm as lgb
from tensorflow import keras
from tensorflow.keras import layers

def rmspe(y_true, y_pred):
    """Root Mean Squared Percentage Error."""
    mask = (y_true != 0)
    y_true_nonzero = y_true[mask]
    y_pred_nonzero = y_pred[mask]
    return np.sqrt(np.mean(((y_true_nonzero - y_pred_nonzero) / y_true_nonzero) ** 2))

# =============================================================================
# 📂 2. Load & Prepare Dataset
# =============================================================================
df = pd.read_csv("train.csv", parse_dates=["Date"])
df.sort_values(["Store", "Date"], inplace=True)

# Filter for Store #1, open days only
store_id = 1
df_store = df[(df["Store"] == store_id) & (df["Open"] == 1)].copy()
df_store = df_store[["Date", "Sales", "Promo", "SchoolHoliday", "StateHoliday"]]
df_store.sort_values("Date", inplace=True)
df_store.reset_index(drop=True, inplace=True)

# =============================================================================
# 🔀 3. Time-Based Train/Test Split
# =============================================================================
split_date = pd.to_datetime("2015-07-01")
train_df = df_store[df_store["Date"] < split_date].copy()
test_df = df_store[df_store["Date"] >= split_date].copy()

print(f"Train range: {train_df['Date'].min()} to {train_df['Date'].max()} (rows={len(train_df)})")
print(f"Test range:  {test_df['Date'].min()} to {test_df['Date'].max()}   (rows={len(test_df)})")

# =============================================================================
# 🧠 4. Feature Preparation
# =============================================================================

# SARIMAX
y_train_arima = train_df["Sales"].values
y_test_arima = test_df["Sales"].values

# Prophet
prophet_train = train_df.rename(columns={"Date": "ds", "Sales": "y"})
prophet_test = test_df.rename(columns={"Date": "ds", "Sales": "y"})

# LightGBM / NN
def make_features(df_in):
    df_out = df_in.copy()
    df_out["day_of_week"] = df_out["Date"].dt.dayofweek
    df_out["day"] = df_out["Date"].dt.day
    df_out["month"] = df_out["Date"].dt.month
    df_out["year"] = df_out["Date"].dt.year
    df_out["StateHoliday"] = df_out["StateHoliday"].replace({0: "0", "a": "1", "b": "2", "c": "3"}).astype(int)
    return df_out

train_ml = make_features(train_df)
test_ml = make_features(test_df)

FEATURES = ["Promo", "SchoolHoliday", "StateHoliday", "day_of_week", "day", "month", "year"]
X_train_ml = train_ml[FEATURES].values
y_train_ml = train_ml["Sales"].values
X_test_ml = test_ml[FEATURES].values
y_test_ml = test_ml["Sales"].values

# =============================================================================
# 🔧 5. Model Training
# =============================================================================

# SARIMAX
print("Fitting SARIMAX...")
arima_model = SARIMAX(
    y_train_arima,
    order=(2, 1, 2),
    seasonal_order=(1, 1, 1, 7),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)
arima_forecast = arima_model.forecast(steps=len(y_test_arima))

# Prophet
print("Fitting Prophet...")
prophet_m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
prophet_m.fit(prophet_train)
prophet_forecast = prophet_m.predict(prophet_test[["ds"]])["yhat"].values

# LightGBM
print("Fitting LightGBM...")
lgb_model = lgb.train(
    {"objective": "regression", "metric": "rmse", "verbosity": -1, "random_state": 42},
    train_set=lgb.Dataset(X_train_ml, label=y_train_ml),
    num_boost_round=500,
    valid_sets=[lgb.Dataset(X_test_ml, label=y_test_ml)]
)
lgb_forecast = lgb_model.predict(X_test_ml)

# Neural Net
print("Fitting Neural Network...")
nn_model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train_ml.shape[1],)),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train_ml, y_train_ml, validation_data=(X_test_ml, y_test_ml), epochs=20, batch_size=32, verbose=0)
nn_forecast = nn_model.predict(X_test_ml).ravel()

# =============================================================================
# 📊 6. Evaluation & Comparison
# =============================================================================
arima_rmspe = rmspe(y_test_arima, arima_forecast)
prophet_rmspe = rmspe(y_test_arima, prophet_forecast)
lgb_rmspe = rmspe(y_test_ml, lgb_forecast)
nn_rmspe = rmspe(y_test_ml, nn_forecast)

print("\nRMSPE Results:")
print(f"  SARIMAX:    {arima_rmspe:.4f}")
print(f"  Prophet:    {prophet_rmspe:.4f}")
print(f"  LightGBM:   {lgb_rmspe:.4f}")
print(f"  Neural Net: {nn_rmspe:.4f}")

results = {
    "SARIMAX": arima_rmspe,
    "Prophet": prophet_rmspe,
    "LightGBM": lgb_rmspe,
    "NeuralNet": nn_rmspe
}
best_model = min(results, key=results.get)
print(f"\nBest model is {best_model} with RMSPE of {results[best_model]:.4f}")

# =============================================================================
# 📈 7. Visualization
# =============================================================================
test_df = test_df.copy()
test_df["SARIMAX_Pred"] = arima_forecast
test_df["Prophet_Pred"] = prophet_forecast
test_df["LightGBM_Pred"] = lgb_forecast
test_df["NN_Pred"] = nn_forecast

plt.figure(figsize=(10, 6))
plt.plot(test_df["Date"], test_df["Sales"], label="Actual")
plt.plot(test_df["Date"], test_df["SARIMAX_Pred"], label="SARIMAX")
plt.plot(test_df["Date"], test_df["Prophet_Pred"], label="Prophet")
plt.plot(test_df["Date"], test_df["LightGBM_Pred"], label="LightGBM")
plt.plot(test_df["Date"], test_df["NN_Pred"], label="Neural Net")
plt.title("Actual vs. Predicted Sales (Test Set)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.xticks(rotation=45, fontsize=8)
plt.legend()
plt.tight_layout()
plt.show()