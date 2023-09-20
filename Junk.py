class RunSimulations:
    
    def __init__(self):
        self.K = 15.0
        self.T = 0.5
        self.gamma = 1.0
        self.r = 0.1
        self.sigma = 0.25
        self.anti = True
        self.disc_model = 'Milstein'
        self.S_0 = [100, 100, 100, 100]
        self.corr_matrix =  np.array([
                [1, 0.5, 0.3, 0.1],
                [0.5, 1, 0.2, 0.4],
                [0.3, 0.2, 1, 0.5],
                [0.1, 0.4, 0.5, 1]
                ])
        self.s0 = 14.0
        self.num_runs = 1000
        self.nr_steps = 1000
        self.N = 1000
        self.M = 1000
        self.S_max = self.K * 4

    
    def run_montecarlo(self):
        montecarlo = Montecarlo(self.N, self.K, self.T, self.gamma, self.r, self.sigma, self.anti, self.disc_model)
        price = montecarlo.V_calc(self.N ,self.M ,self.s0)
        print("European Call Option Price montecarlo: ", price)
        error = montecarlo.model_calc_error_diff(self.N, self.M, self.s0)
        print("Error montecarlo: ", error)

    def run_FDM_Multi_Asset(self):
        option = FDM_Multi_Asset_Option(K=100, s0_1=50, s0_2=50, S_max_1=100, S_max_2=100, T=1, r_1=0.05, r_2=0.05, sigma_1=0.2, sigma_2=0.2, M1=100, M2=100, N=1000)
        price = option.implicit()
        print(f"The option price is: {price}")
    
    def run_Montecarlo_Multi_Asset(self):
        mc_multi = MontecarloMultiAsset(self.nr_steps, self.K, self.T, [0.5, 0.5, 0.5, 0.5], [0.05, 0.05, 0.05, 0.05], [0.2, 0.2, 0.2, 0.2], self.anti, self.disc_model, self.corr_matrix)
        V = mc_multi.V_calc(self.nr_steps, self.num_runs, self.S_0)
        print(f"The estimated value of the multi-asset option is: {V}")
    
    def run_FDM(self):
        option = OptionPricingImplicit(self.K, self.s0, self.S_max, self.T, self.r, self.sigma, self.gamma,self.M, self.N)
        price_FDM = option.FDM()
        price_exact = option.bsexact()
        print(f'price FDM: {price_FDM}')
        print(f'price Exact: {price_exact}')
        print(f'error: {np.abs(price_FDM-price_exact)}')
        print('************************************')
        exact_price_Greeks = option.bsexact_Greeks()
        for key, greek in exact_price_Greeks.items():
            print(f'Exact price Greeks: {key} {greek}')

    def run_Montecarlo_Greeks(self):
        mc = Montecarlo(self.nr_steps,self. K, self.T, self.gamma, self.r, self.sigma, self.anti, 'Milstein')

        delta_value = mc.delta(self.s0, self.num_runs)
        gamma_value = mc.gamma_greek(self.s0, self.num_runs)
        vega_value = mc.vega(self.s0, self.num_runs)
        theta_value = mc.theta(self.s0, self.num_runs)
        rho_value = mc.rho(self.s0, self.num_runs)

        print(f"Delta: {delta_value}")
        print(f"Gamma: {gamma_value}")
        print(f"Vega: {vega_value}")
        print(f"Theta: {theta_value}")
        print(f"Rho: {rho_value}")
    
    def run_FDM_Greeks(self):
        option = OptionPricingImplicit(self.K, self.s0, self.S_max, self.T, self.r, self.sigma, self.gamma, self.M, self.N)
        option.calculate_greeks()
        
    def run_all(self):
        self.run_montecarlo()
        self.run_FDM_Multi_Asset()
        self.run_Montecarlo_Multi_Asset()
        self.run_FDM()
        self.run_Montecarlo_Greeks()
        self.run_FDM_Greeks()        

if __name__== "__main__":
    run = RunSimulations()
    if Run_all:
        run.run_all()
    else:
        for key, value in simulations.items():
            if value:
                getattr(run, f"run_{key}")()
                print("************************************")