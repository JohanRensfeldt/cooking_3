from FDM import OptionPricingImplicit 
from FDM import OptionPricingExplicit
from Montecarlo import Montecarlo

K = 15.0      # Strike price
s0 = 14.0      # Initial stock price
S_max =  K * 4 # Maximum stock price
T = 0.5        # Time to expiration
r = 0.1       # Risk-free rate
sigma = 0.25    # Volatility
gamma = 1    # Elasticity parameter for CEV
M = 100        # Number of stock price steps
N = 100     # Number of time steps

if __name__== 'main':

    explicit = OptionPricingExplicit(K, s0, S_max, T, r, sigma, M, N)
    implicit = OptionPricingImplicit(K, s0, S_max, T, r, sigma, gamma, M, N)
    montecarlo = Montecarlo(K, s0, 'milstein', T, r, sigma, gamma, M, N)