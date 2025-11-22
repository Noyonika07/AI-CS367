import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# -----------------------------
# Download S&P500 Historical Data
# -----------------------------
stock_ticker = '^GSPC'
start_date = '2014-01-01'
end_date = '2025-12-31'

stock_data = yf.download(stock_ticker, start=start_date, end=end_date)

print("Sample Data:")
print(stock_data.head())

# ---------------------------------------
# Fix for the new Yahoo Finance structure
# ---------------------------------------
# Extract close prices from multi-index columns
close_prices = stock_data[('Price', 'Close')]

# Compute returns
stock_data['Returns'] = close_prices.pct_change()
stock_data.dropna(inplace=True)

print("Returns Data:")
print(stock_data[[('Price', 'Close'), 'Returns']].head())

# -----------------------------
# Plot Close Prices & Returns
# -----------------------------
plt.figure(figsize=(10, 6))

# Close price plot
plt.subplot(2, 1, 1)
plt.plot(stock_data.index, close_prices.loc[stock_data.index], color='blue')
plt.title('S&P 500 Close Prices (2014–2025)')
plt.xlabel('Date')
plt.ylabel('Price')

# Returns plot
plt.subplot(2, 1, 2)
plt.plot(stock_data.index, stock_data['Returns'], color='green')
plt.title('S&P 500 Daily Returns (2014–2025)')
plt.xlabel('Date')
plt.ylabel('Returns')

plt.tight_layout()
plt.show()

# -----------------------------
# Fit Gaussian HMM to returns
# -----------------------------
returns_data = stock_data[['Returns']].values
n_states = 2

hmm_model = GaussianHMM(
    n_components=n_states,
    covariance_type="full",
    n_iter=1000
)

hmm_model.fit(returns_data)

def forward_algorithm(hmm_model, returns):
    transition_matrix = hmm_model.transmat_
    start_prob = hmm_model.startprob_

    T = returns.shape[0]
    N = hmm_model.n_components

    alpha = np.zeros((T, N))

    # emission prob for first observation
    log_likelihood = hmm_model._compute_log_likelihood(returns[:1])[0]

    for i in range(N):
        alpha[0, i] = start_prob[i] * np.exp(log_likelihood[i])

    # recursive steps
    for t in range(1, T):
        log_likelihood = hmm_model._compute_log_likelihood(returns[t:t+1])[0]
        for j in range(N):
            alpha[t, j] = np.sum(alpha[t-1, :] * transition_matrix[:, j]) * np.exp(log_likelihood[j])

    P_O_given_lambda = np.sum(alpha[-1, :])
    return P_O_given_lambda, alpha

P_O_given_lambda, alpha = forward_algorithm(hmm_model, returns_data)
print(f"P(O | λ) = {P_O_given_lambda}")

hidden_states = hmm_model.predict(returns_data)
stock_data['Hidden_State'] = hidden_states

plt.figure(figsize=(12, 8))
plt.plot(stock_data.index, close_prices.loc[stock_data.index], label='Close Price')

for i in range(n_states):
    state_mask = (hidden_states == i)
    plt.fill_between(
        stock_data.index,
        close_prices.loc[stock_data.index].min(),
        close_prices.loc[stock_data.index],
        where=state_mask,
        alpha=0.3,
        label=f"State {i} — {'High Volatility' if i == 1 else 'Low Volatility'}"
    )

plt.title('S&P 500 Close Prices with Hidden Markov States')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

print("\nMeans of Hidden States:")
print(hmm_model.means_)

print("\nCovariances of Hidden States:")
print(hmm_model.covars_)

print("\nTransition Matrix:")
print(hmm_model.transmat_)

plt.matshow(hmm_model.transmat_, cmap='viridis')
plt.title("HMM Transition Matrix")
plt.colorbar()
plt.show()
