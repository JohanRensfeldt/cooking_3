from FDM import OptionPricingImplicit 
from FDM import OptionPricingExplicit
from Montecarlo import Montecarlo
import numpy as np

K = 15.0      # Strike price
s0 = 14.0      # Initial stock price
S_max =  K * 4 # Maximum stock price
T = 0.5        # Time to expiration
r = 0.1       # Risk-free rate
sigma = 0.25    # Volatility
gamma = 1    # Elasticity parameter for CEV
M = 100        # Number of stock price steps
N = 100     # Number of time steps

if __name__== "__main__":

    montecarlo = Montecarlo(N, K, T, gamma, r, sigma, anti = True, disc_model = 'Milstein')
    price = montecarlo.V_calc(N ,M,s0)
    print("European Call Option Price montecarlo: ", price)
    error = montecarlo.model_calc_error_diff(N, M, s0)
    print("Error montecarlo: ", error)

    option_pricer = OptionPricingImplicit(K, s0, S_max, T, r, sigma, gamma, M, N)
    option_price = option_pricer.FDM()
    print("European Call Option Price CEV model:", option_price)
    print("European Call Option Price BS model:", option_pricer.bsexact())
    
    N_array = np.linspace(10, 1000, 10).astype(int)
    option_pricer.plot_time_vs_N(N_array)

    op = OptionPricingExplicit(K, s0, S_max, T, r, sigma, M, N)
    op.plot_error()

    option_price = op.explicit(op.M, op.N)
    print("European Call Option Price: ", option_price)


