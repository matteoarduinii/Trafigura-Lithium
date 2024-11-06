import numpy as np
import matplotlib.pyplot as plt

initial_price = 10450

# Monte Carlo parameters
time_horizon = 12  # forecast horizon (12 months)
num_simulations = 10000

# Parameters
trump_demand_growth = 0.015
harris_demand_growth = 0.05
trump_supply_increase = 0.025
harris_supply_increase = 0.01
volatility = 0.05  # conservative monthly volatility

# Monte Carlo simulation function
def monte_carlo_simulation(demand_growth, supply_increase, volatility):
    price_paths = []
    for _ in range(num_simulations):
        prices = [initial_price]
        for _ in range(time_horizon):
            random_volatility = np.random.normal(0, volatility)
            # Calculate price change based on demand and supply impacts
            price_change = (1 + 10*(demand_growth) - supply_increase + random_volatility)
            # Apply a cap on the price change to limit extremes
            price_change = max(min(price_change, 1.1), 0.9)
            new_price = prices[-1] * price_change
            prices.append(new_price)
        price_paths.append(prices)
    return np.array(price_paths)

# Run simulations for Trump and Harris scenarios
trump_prices = monte_carlo_simulation(trump_demand_growth / 12, trump_supply_increase / 12, volatility)
harris_prices = monte_carlo_simulation(harris_demand_growth / 12, harris_supply_increase / 12, volatility)

# Graphing
plt.figure(figsize=(12, 6))
plt.hist(trump_prices[:, -1], bins=50, alpha=0.5, label="Trump Price Distribution")
plt.hist(harris_prices[:, -1], bins=50, alpha=0.5, color='orange', label="Harris Price Distribution")
plt.xlabel("Final Lithium Price (USD)")
plt.ylabel("Frequency")
plt.title("Monte Carlo Simulation: Final Lithium Price (12 Month Forecast) Distribution for Trump vs. Harris Scenarios")
plt.legend()
plt.show()
