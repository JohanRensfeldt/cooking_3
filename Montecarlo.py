import numpy as np
import matplotlib.pyplot as plt
from scipy import special

class Montecarlo:

    def __init__(self, nr_steps, K, T, gamma, r, sigma, anti, disc_model):
        self.anti = anti
        self.nr_steps = nr_steps
        self.disc_model = disc_model
        self.K = K
        self.T = T
        self.gamma = gamma
        self.r = r
        self.sigma = sigma

    def dW(self, dt):
        return np.random.normal(scale=np.sqrt(dt))
    
    def eu_call_val(self, S):
        if S < self.K:
            return 0
        else:
            return S-self.K

    def run_simulation(self, S_0,dt,nr_steps):
        S = np.zeros(nr_steps)
        S[0] = S_0
        if self.disc_model == 'Euler_Murayama':
            if self.anti:
                S2 = S
                Z = S
                for i in range(1,nr_steps):
                    S[i] = self.Euler_Murayama(S[i-1],dt,self.dW(dt))
                    S2[i] = self.Euler_Murayama(S2[i-1],dt,-self.dW(dt))
                    Z[i] = (S[i]+S2[i])/2
                return Z
            else:
                for i in range(1,nr_steps):
                    S[i] = self.Euler_Murayama(S[i-1],dt,self.dW(dt))
                return S
        elif self.disc_model == 'Milstein':
            if self.anti:
                S2 = S
                Z = S
                for i in range(1,nr_steps):
                    dW = self.dW(dt)
                    S[i] = self.Milstein(S[i-1],dt, dW)
                    S2[i] = self.Milstein(S2[i-1],dt,-dW)
                    Z[i] = (S[i]+S2[i])/2
                return Z
            else:
                for i in range(1,nr_steps):
                    S[i] = self.Milstein(S[i-1],dt,self.dW(dt))
                return S

        else:
            raise Exception('Unsupported discretization model')
        
    def Euler_Murayama(self, S_prev, dt, dw):
        S = S_prev + self.r * S_prev * dt + self.sigma * (S_prev**self.gamma) * dw
        return S
    
    def Milstein(self, S_prev, dt, dw):
        S = S_prev + self.r * S_prev * dt + self.sigma * (S_prev ** self.gamma) * dw + 0.5 * self.sigma**2 * (S_prev ** self.gamma) * (dw ** 2 - dt)
        return S

    
    def V_calc(self, nr_steps,num_runs, S_0):
        #Calculate value of stock
        dt = self.T/nr_steps
        S_vals = np.zeros((num_runs,nr_steps))
        for i in range(num_runs):
            S_vals[i,:] = self.run_simulation(S_0,dt,nr_steps)

        #Calculate mean option value on final stock value
        V_vals = np.zeros(num_runs)
        for i in range(num_runs):
            V_vals[i] = self.eu_call_val(S_vals[i,-1])
        V = np.exp(-self.r*self.T)*np.mean(V_vals)

        return V
    
    def bsexact(self, sigma: float, R: float, K: float, T: float, s: float):
        d1 = (np.log(s/K)+(R+0.5*sigma**2)*(T))/(sigma*np.sqrt(T))
        d2 = d1-sigma*np.sqrt(T)
        F = 0.5*s*(1+special.erf(d1/np.sqrt(2)))-np.exp(-R*T)*K*0.5*(1+special.erf(d2/np.sqrt(2)))
        return F
    
    def model_calc_error(self, nr_steps,num_runs,S_0):
        dt = self.T/nr_steps
        V_calc = self.V_calc(nr_steps,num_runs,dt,S_0)
        V_exact = self.bsexact(self.sigma,self.r,self.K,self.T,S_0)

        error = np.abs(V_calc-V_exact)/V_exact*100
        return error
    
    def model_calc_error_diff(self, nr_steps,num_runs,S_0):
        dt = self.T/nr_steps
        V_calc = self.V_calc(nr_steps,num_runs ,S_0)
        V_exact = self.bsexact(self.sigma,self.r,self.K,self.T,S_0)

        error = V_calc-V_exact
        return error
    
    def delta(self, S_0, num_runs, bump=0.01):
        # Calculate option value for bumped up and down initial stock prices
        V_up = self.V_calc(self.nr_steps, num_runs, S_0 + bump)
        print(f'V_up: {V_up}')
        V_down = self.V_calc(self.nr_steps, num_runs, S_0 - bump)
        print(f'V_down: {V_down}')
        return (V_up - V_down) / (2 * bump)

    def gamma_greek(self, S_0, num_runs, bump=0.01):
        delta_up = self.delta(S_0 + bump, num_runs)
        delta_down = self.delta(S_0 - bump, num_runs)
        return (delta_up - delta_down) / (2 * bump)

    def vega(self, S_0, num_runs, bump=0.01):
        original_sigma = self.sigma
        self.sigma += bump
        V_up = self.V_calc(self.nr_steps, num_runs, S_0)
        self.sigma = original_sigma - bump
        V_down = self.V_calc(self.nr_steps, num_runs, S_0)
        self.sigma = original_sigma  # Resetting to original sigma
        return (V_up - V_down) / (2 * bump)

    def theta(self, S_0, num_runs, time_bump=0.01):
        original_T = self.T
        self.T += time_bump
        V_up = self.V_calc(self.nr_steps, num_runs, S_0)
        self.T = original_T - time_bump
        V_down = self.V_calc(self.nr_steps, num_runs, S_0)
        self.T = original_T  # Resetting to original T
        return (V_down - V_up) / (2 * time_bump)

    def rho(self, S_0, num_runs, bump=0.01):
        original_r = self.r
        self.r += bump
        V_up = self.V_calc(self.nr_steps, num_runs, S_0)
        self.r = original_r - bump
        V_down = self.V_calc(self.nr_steps, num_runs, S_0)
        self.r = original_r  # Resetting to original r
        return (V_up - V_down) / (2 * bump)


class MontecarloMultiAsset:

    def __init__(self, nr_steps, K, T, gamma, r, sigma, anti, disc_model, corr_matrix):
        self.anti = anti
        self.nr_steps = nr_steps
        self.disc_model = disc_model
        self.K = K
        self.T = T
        self.gamma = np.array(gamma)
        self.r = np.array(r)
        self.sigma = sigma
        self.corr_matrix = corr_matrix

    def dW(self, dt):
        return np.random.normal(scale=np.sqrt(dt), size=len(self.sigma))

    def eu_basket_val(self, S):
        basket_value = np.sum(S)
        return max(basket_value - self.K, 0)

    def run_simulation(self, S_0, dt, nr_steps):
        d = len(S_0)  # Number of assets
        S = np.zeros((nr_steps, d))
        S[0, :] = S_0
        chol = np.linalg.cholesky(self.corr_matrix)

        if self.disc_model == 'Euler_Murayama':
            for i in range(1, nr_steps):
                dW_corr = np.dot(chol, self.dW(dt))
                for j in range(d):
                    S[i, j] = self.Euler_Murayama(S[i-1, j], dt, dW_corr[j], j)


        elif self.disc_model == 'Milstein':
            for i in range(1, nr_steps):
                dW_corr = np.dot(chol, self.dW(dt))
                for j in range(d):
                    S[i, j] = self.Milstein(S[i-1, j], dt, dW_corr[j])

        else:
            raise Exception('Unsupported discretization model')

        return S

    def Euler_Murayama(self, S_prev, dt, dw, j):
        return S_prev + self.r[j] * S_prev * dt + self.sigma[j] * (S_prev ** self.gamma[j]) * dw


    def Milstein(self, S_prev, dt, dw):
        return S_prev + self.r * S_prev * dt + self.sigma * (S_prev ** self.gamma) * dw + \
               0.5 * self.sigma ** 2 * (dw ** 2 - dt)

    def V_calc(self, nr_steps, num_runs, S_0):
        dt = self.T / nr_steps
        S_vals = np.zeros((num_runs, nr_steps, len(S_0)))

        for i in range(num_runs):
            S_vals[i, :, :] = self.run_simulation(S_0, dt, nr_steps)

        V_vals = np.zeros(num_runs)
        for i in range(num_runs):
            V_vals[i] = self.eu_basket_val(S_vals[i, -1, :])

        V = np.exp(-self.r[0] * self.T) * np.mean(V_vals)
        return V
    
    def bsexact(self, sigma: float, R: float, K: float, T: float, s: float):
        d1 = (np.log(s/K)+(R+0.5*sigma**2)*(T))/(sigma*np.sqrt(T))
        d2 = d1-sigma*np.sqrt(T)
        F = 0.5*s*(1+special.erf(d1/np.sqrt(2)))-np.exp(-R*T)*K*0.5*(1+special.erf(d2/np.sqrt(2)))
        return F