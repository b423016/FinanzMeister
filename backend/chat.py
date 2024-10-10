import yfinance as yf
import matplotlib.pyplot as plt


# Download historical data for a stock (e.g., Apple)
stock_data = yf.download("AAPL", start="2020-01-01", end="2023-03-01")
print(stock_data.head(10))  # Display the first few rows
# Calculate moving averages


# Calculate moving averages
stock_data['1_day_MA'] = stock_data['Close'].rolling(window=1).mean()
stock_data['7_day_MA'] = stock_data['Close'].rolling(window=7).mean()
stock_data['30_day_MA'] = stock_data['Close'].rolling(window=30).mean()

# Print the resulting DataFrame with moving averages
print(stock_data['1_day_MA'].head(5))
print(stock_data['7_day_MA'].tail(5))
print(stock_data['30_day_MA'].tail(5))

# Calculate the exponential moving average
stock_data['6_day_EMA'] = stock_data['Close'].ewm(span=6, adjust=False).mean()
stock_data['12_day_EMA'] = stock_data['Close'].ewm(span=12, adjust=False).mean()

# print ema
print(stock_data['12_day_EMA'].tail(5))

stock_data['volatility_10'] = stock_data['Close'].rolling(window=10).std()
stock_data['returns'] = stock_data['Close'].pct_change()
print(stock_data['volatility_10'].head(5))

# calculate relative Strength index \
# RSI above 70 indicates overbought conditions (potential sell).
# RSI below 30 indicates oversold conditions (potential buy).


def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


stock_data['RSI'] = calculate_rsi(stock_data)


# print
print(stock_data['RSI'].tail(5))

# calculate the bollinger bands  measures market
stock_data['Middle_Band'] = stock_data['Close'].rolling(window=20).mean()
stock_data['Upper_Band'] = stock_data['Middle_Band'] + 2*stock_data['Close'].rolling(window=20).std()
stock_data['Lower_Band'] = stock_data['Middle_Band'] - 2*stock_data['Close'].rolling(window=20).std()

# print bollinger band
print(stock_data['Middle_Band'].tail(5))

# calculating th eon balance volume uses price and volume to predict price movements
stock_data['OBV'] = (stock_data['Volume'] * ((stock_data['Close'] - stock_data['Close'].shift(1)) > 0)).cumsum()

# print
print(stock_data['OBV'].tail(5))

# performing average true range mesures market volatility
# more atr more volatile

high_low = stock_data['High'] - stock_data['Low']
high_close = abs(stock_data['High'] - stock_data['Close'].shift(1))
low_close = abs(stock_data['Low'] - stock_data['Close'].shift(1))

true_range = high_low.combine(high_close, max).combine(low_close, max)
stock_data['ATR'] = true_range.rolling(window=14).mean()

# print atr
print(stock_data['ATR'].tail(5))

# visualising features

# Plot closing price and moving averages
plt.figure(figsize=(10,5))
plt.plot(stock_data['Close'], label='Close Price')
plt.plot(stock_data['1_day_MA'], label='10-Day MA')
plt.plot(stock_data['7_day_MA'], label='50-Day MA')
plt.legend()
plt.show()

# Plot RSI
plt.figure(figsize=(10,5))
plt.plot(stock_data['RSI'], label='RSI')
plt.axhline(30, linestyle='--', alpha=0.5)
plt.axhline(70, linestyle='--', alpha=0.5)
plt.legend()
plt.show()


# Plot Bollinger Bands
plt.figure(figsize=(10,5))
plt.plot(stock_data['Close'], label='Close Price')
plt.plot(stock_data['Upper_Band'], label='Upper Band')
plt.plot(stock_data['Middle_Band'], label='Middle Band')
plt.plot(stock_data['Lower_Band'], label='Lower Band')
plt.fill_between(stock_data.index, stock_data['Lower_Band'], stock_data['Upper_Band'], color='gray', alpha=0.3)
plt.legend()
plt.show()

# applying power transformation for scaling
from sklearn.preprocessing import PowerTransformer

# Selecting the columns that need transformation
features_to_transform = ['Close', 'Volume', '1_day_MA', '7_day_MA', '30_day_MA',
                         '6_day_EMA', '12_day_EMA', 'RSI', 'OBV', 'ATR']

# Apply Yeo-Johnson Transformation
transformer = PowerTransformer(method='yeo-johnson')

# Fit and transform the selected features
stock_data_transformed = stock_data.copy()  # To keep the original data intact
stock_data_transformed[features_to_transform] = transformer.fit_transform(stock_data[features_to_transform])

# Check the transformation result
print(stock_data_transformed.head(5))


# plotting original vs transformed data
import matplotlib.pyplot as plt
import seaborn as sns

# Plot the original and transformed 'Close' prices for comparison
plt.figure(figsize=(14, 6))

# Original Close Price
plt.subplot(1, 2, 1)
sns.histplot(stock_data['Close'], kde=True, color='blue')
plt.title('Original Close Price Distribution')

# Transformed Close Price
plt.subplot(1, 2, 2)
sns.histplot(stock_data_transformed['Close'], kde=True, color='green')
plt.title('Transformed Close Price Distribution (Yeo-Johnson)')

plt.tight_layout()
plt.show()


# splitting the data in train and test split  80,20 to maintain the temporal order for timeseries nature fo data

from sklearn.model_selection import train_test_split

# Define feature columns and target
feature_cols = ['Volume', '1_day_MA', '7_day_MA', '30_day_MA',
               '6_day_EMA', '12_day_EMA', 'RSI', 'OBV', 'ATR']
target = 'Close'

# Drop rows with NaN values resulting from feature engineering
stock_data_clean = stock_data_transformed.dropna()

# Features and target
X = stock_data_clean[feature_cols]
y = stock_data_clean[target]

# Train-Test Split (80% train, 20% test) without shuffling
split_index = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

print(f"Training samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")


# model selection xgboost and joblib
import xgboost as xgb

from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Initialize the XGBoost Regressor with hyperparameters optimized for performance
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
early_stopping_rounds = 50,
model = XGBRegressor(early_stopping_rounds=early_stopping_rounds)
# Train the model with early stopping
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# Save the trained XGBoost model
joblib.dump(xgb_model, 'xgb_stock_model.joblib')
print("XGBoost model saved as 'xgb_stock_model.joblib'")


# using the lighgtbgm machines

import lightgbm as lgb

# Initialize the LightGBM Regressor
lgb_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds = 50,
    force_col_wise=True,
    num_threads=4,
    verbose=100
)
# Train the model with early stopping
lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
)

# Save the trained LightGBM model
joblib.dump(lgb_model, 'lgbm_stock_model.joblib')
print("LightGBM model saved as 'lgbm_stock_model.joblib'")

# using catboost

from catboost import CatBoostRegressor

# Initialize the CatBoost Regressor
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.01,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=100
)

# Train the model with early stopping
cat_model.fit(
    X_train, y_train,
    eval_set=(X_test, y_test),
    early_stopping_rounds=50
)

# Save the trained CatBoost model
joblib.dump(cat_model, 'catboost_stock_model.joblib')
print("CatBoost model saved as 'catboost_stock_model.joblib'")

#  we will be suing prophet for  the job

from prophet import hProphet

# Prepare the data in the required format for Prophet
df = stock_data[['Close']].reset_index()
df.columns = ['ds', 'y']  # Rename columns for Prophet
print(df.head(10))
print(df.tail(10))

# Create the Prophet model
model = Prophet()

# Fit the model to the data
model.fit(df)


# Define the period for future prediction (e.g., 30 days)
future = model.make_future_dataframe(periods=30)

# Generate predictions
forecast = model.predict(future)
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

# Plot the forecasted stock prices
model.plot(forecast)
plt.title('Stock Price Prediction using Prophet')
plt.show()

# Plot the components (trend, seasonality)
model.plot_components(forecast)
plt.show()

# Adding weekly seasonality to the model
model = Prophet(weekly_seasonality=True)
model.fit(df)

# Forecasting with tuned model
forecast = model.predict(future)

# Experiment with different changepoint scales
model = Prophet(changepoint_prior_scale=0.1)  # Default is 0.05, trying larger values like 0.1, 0.2
model.fit(df)
forecast = model.predict(future)
print(forecast[['ds', 'yhat']].tail(5))

model = Prophet(seasonality_mode='additive')
model.fit(df)
forecast = model.predict(future)
print(forecast[['ds', 'yhat']].tail(5))


from prophet.diagnostics import cross_validation, performance_metrics

# Cross-validation
df_cv = cross_validation(model, initial='366 days', period='180 days', horizon='30 days')

# Performance metrics
df_p = performance_metrics(df_cv)
print(df_p.head())

import pandas as pd
# Define a dataframe for holidays or special events
holidays = pd.DataFrame({
  'holiday': 'earnings_release',
  'ds': pd.to_datetime(['2024-07-01', '2024-10-01']),  # example dates
  'lower_window': 0,
  'upper_window': 1,
})

# Add holidays to the model
model = Prophet(holidays=holidays)
model.fit(df)
forecast = model.predict(future)
print(forecast[['ds', 'yhat']].tail(5))

# Tuning weekly seasonality (stock markets have strong weekly patterns)
model = Prophet(weekly_seasonality=0.1)

# Manually add other seasonal titties if necessary (like quarterly patterns)
model.add_seasonality(name='quarterly', period=90, fourier_order=5)
model.fit(df)
forecast = model.predict(future)
print(forecast[['ds', 'yhat']].tail(5))
plt.plot(df_cv[['ds', 'yhat']].tail(5))

# from sklearn.model_selection import ParameterGrid

# best parameter from grid search
# Best params: {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 0.1}, Best MAE: 3.9550510694997096

from prophet.diagnostics import cross_validation

# Perform cross-validation: initial training period is 366 days, and validation for 180 days with 30-day horizon
df_cv = cross_validation(model, initial='366 days', period='180 days', horizon='30 days')

# Output the cross-validation result
print(df_cv.head())

from prophet.diagnostics import performance_metrics

# Calculate performance metrics
df_p = performance_metrics(df_cv)
print(df_p[['horizon', 'mae', 'rmse', 'mape']].head())

from prophet.plot import plot_cross_validation_metric

# Plot the cross-validation metric (Mean Absolute Error in this case)
fig = plot_cross_validation_metric(df_cv, metric='mae')
plt.show()

fig = plot_cross_validation_metric(df_cv, metric='rmse')

# Add title and labels
fig.suptitle("Prophet Cross-Validation: RMSE Metric", fontsize=16)
plt.xlabel("Horizon (Days)", fontsize=12)
plt.ylabel("RMSE", fontsize=12)

# Show plot
plt.show()

import math
df['y_pred_naive'] = df['y'].shift(1)  # Naive prediction: the next value is the same as the previous

# Calculate performance of the naive model
mae_naive = mean_absolute_error(df['y'][1:], df['y_pred_naive'][1:])
RMSE = math.sqrt(mae_naive)
print(RMSE)
print(f'Naive Forecast MAE: {mae_naive}')


# Example: Tuning changepoint_prior_scale and seasonality_prior_scale
model = Prophet(changepoint_prior_scale=0.1, seasonality_prior_scale=0.1)
model.fit(df)


# Perform the same evaluation steps as above after tuning

# Split the data into train and test sets
train_size = int(0.8 * len(df))
train = df[:train_size]
test = df[train_size:]
model = Prophet()
# Train the Prophet model on the train set
model.fit(train)

# Make future predictions and compare with the test set
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

# Calculate MAE on test set
mae_test = mean_absolute_error(test['y'], forecast['yhat'][-len(test):])
print(f'Test Set MAE: {mae_test}')

df['yhat'] = forecast['yhat']
# Calculate residuals
df['residuals'] = df['y'] - df['yhat']

# Plot the residuals
plt.figure(figsize=(10,5))
plt.plot(df['ds'], df['residuals'], label='Residuals')
plt.axhline(0, color='red', linestyle='--')
plt.title('Residual Plot')
plt.legend()
plt.show()

# ######################################## Sentiment analysis ##################################### #