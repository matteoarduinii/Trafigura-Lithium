import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

file_path = '/Users/matteoarduini/Downloads/lithium_price_data.csv'
lithium_data = pd.read_csv(file_path)

lithium_data['Date'] = pd.to_datetime(lithium_data['Date'], format='%d/%m/%Y')
lithium_data.set_index('Date', inplace=True)

lithium_data['Price'] = lithium_data['Price'].replace({',': ''}, regex=True).astype(float)

# Sort and drop na
lithium_data = lithium_data.sort_index()
lithium_data = lithium_data[['Price']].dropna()

# fit model
model = ARIMA(lithium_data['Price'], order=(1, 1, 1))
fitted_model = model.fit()

# forecasting
forecast_steps = 300
forecast = fitted_model.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(lithium_data.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')
forecast_df = pd.DataFrame(forecast.predicted_mean, index=forecast_index, columns=['Forecast'])

# Plot
plt.figure(figsize=(12, 6))
plt.plot(lithium_data['Price'], label='Historical Price')
plt.title('Lithium Price Over 5 Years')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
