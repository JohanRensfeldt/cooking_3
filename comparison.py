from FDM import OptionPricingImplicit 
from FDM import OptionPricingExplicit
from Montecarlo import MontecarloMultiAsset
from Montecarlo import Montecarlo
from FDM import FDM_Multi_Asset_Option
import numpy as np



multi_asset_montecarlo = False
multi_asset_FDM = True
montecarlo = False
montecarloGreeks = False
FDMgreeks = False
FDM = False

if __name__== "__main__":

    if multi_asset_montecarlo:
        nr_steps = 1000
        K = 100
        T = 1.0
        gamma = 1.0
        r = 0.05
        sigma = [0.2, 0.3, 0.4, 0.5]  # For two assets
        anti = False
        disc_model = 'Euler_Murayama'
        S_0 = [100, 100, 100, 100]  # Initial values for two assets
        corr_matrix = np.array([
        [1, 0.5, 0.3, 0.1],
        [0.5, 1, 0.2, 0.4],
        [0.3, 0.2, 1, 0.5],
        [0.1, 0.4, 0.5, 1]
        ])
        num_runs = 1000

        # Creating an instance of the Monte Carlo simulator for multi-asset
        mc_multi = MontecarloMultiAsset(nr_steps, K, T, [0.5, 0.5, 0.5, 0.5], [0.05, 0.05, 0.05, 0.05], [0.2, 0.2, 0.2, 0.2], anti, disc_model, corr_matrix)

        # Calculating the option value
        V = mc_multi.V_calc(nr_steps, num_runs, S_0)
        print(f"The estimated value of the multi-asset option is: {V}")

    elif montecarlo:

        K = 15.0      # Strike price
        s0 = 14.0      # Initial stock price
        S_max =  K * 4 # Maximum stock price
        T = 0.5        # Time to expiration
        r = 0.1       # Risk-free rate
        sigma = 0.25    # Volatility
        gamma = 1    # Elasticity parameter for CEV
        M = 100        # Number of stock price steps
        N = 100     # Number of time steps

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
    
    elif montecarloGreeks:

        K = 15.0      # Strike price
        s0 = 14.0      # Initial stock price
        S_max =  K * 4 # Maximum stock price
        T = 0.5        # Time to expiration
        r = 0.1       # Risk-free rate
        sigma = 0.25    # Volatility
        gamma = 1    # Elasticity parameter for CEV
        M = 100        # Number of stock price steps
        N = 100     # Number of time steps
        
        # Initialize Montecarlo object
        mc = Montecarlo(1000, K, T, gamma, r, sigma, True, 'Milstein')

        # Number of Monte Carlo simulations
        num_runs = 1000
        # Initial stock price
        S_0 = 14.0

        # Calculate Greeks
        delta_value = mc.delta(S_0, num_runs)
        gamma_value = mc.gamma_greek(S_0, num_runs)
        vega_value = mc.vega(S_0, num_runs)
        theta_value = mc.theta(S_0, num_runs)
        rho_value = mc.rho(S_0, num_runs)

        print(f"Delta: {delta_value}")
        print(f"Gamma: {gamma_value}")
        print(f"Vega: {vega_value}")
        print(f"Theta: {theta_value}")
        print(f"Rho: {rho_value}")
    
    elif FDMgreeks:
        
        K = 15.0
        s0 = 14.0
        S_max = K * 4
        T = 0.5
        r = 0.1
        sigma = 0.25
        gamma = 1
        M = 500
        N = 500

        option = OptionPricingImplicit(K, s0, S_max, T, r, sigma, gamma, M, N)
        option.calculate_greeks()

    elif FDM:
        K = 15.0      # Strike price
        s0 = 14.0      # Initial stock price
        S_max =  K * 4 # Maximum stock price
        T = 0.5        # Time to expiration
        r = 0.1       # Risk-free rate
        sigma = 0.25    # Volatility
        gamma = 1    # Elasticity parameter for CEV
        M = 100       # Number of stock price steps
        N = 100     # Number of time steps


        option = OptionPricingImplicit(K, s0, S_max, T, r, sigma, gamma, M, N)
        price_FDM = option.FDM()
        price_exact = option.bsexact()
        print(f'price FDM: {price_FDM}')
        print(f'price Exact: {price_exact}')
        print(f'error: {np.abs(price_FDM-price_exact)}')
        print('************************************')
        exact_price_Greeks = option.bsexact_Greeks()
        for key, greek in exact_price_Greeks.items():
            print(f'Exact price Greeks: {key} {greek}')

    elif multi_asset_FDM:
        option = FDM_Multi_Asset_Option(K=100, s0_1=50, s0_2=50, S_max_1=100, S_max_2=100, T=1, r_1=0.05, r_2=0.05, sigma_1=0.2, sigma_2=0.2, M1=100, M2=100, N=1000)
        price = option.implicit()
        print(f"The option price is: {price}")
