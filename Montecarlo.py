import numpy as np
import matplotlib.pyplot as plt
from scipy import special

class model:

    def __init__(self, nr_steps, anti, disc_model, K, T, gamma, r, sigma):
        self.anti = anti
        self.nr_steps = nr_steps
        self.disc_model = disc_model
        self.K = K
        self.T = T
        self.gamma = gamma
        self.r = r
        self.sigma = sigma

    #Global model constants

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
                    S[i] = self.Milstein(S[i-1],dt,self.dW(dt))
                    S2[i] = self.Milstein(S2[i-1],dt,-self.dW(dt))
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
        S = S_prev + self.r * S_prev * dt + self.sigma * (S_prev**self.gamma) * dw + 0.5 * self.sigma**2 * S_prev * (dw**2 - dt)
        return S
    
    def bsexact(self, sigma: float, R: float, K: float, T: float, s: float):
        d1 = (np.log(s/K)+(R+0.5*sigma**2)*(T))/(sigma*np.sqrt(T))
        d2 = d1-sigma*np.sqrt(T)
        F = 0.5*s*(1+special.erf(d1/np.sqrt(2)))-np.exp(-R*T)*K*0.5*(1+special.erf(d2/np.sqrt(2)))
        return F

    def V_calc(self, nr_steps,num_runs,dt,S_0):
        #Calculate value of stock
        S_vals = np.zeros((num_runs,nr_steps))
        for i in range(num_runs):
            S_vals[i,:] = self.run_simulation(S_0,dt,nr_steps)

        #Calculate mean option value on final stock value
        V_vals = np.zeros(num_runs)
        for i in range(num_runs):
            V_vals[i] = self.eu_call_val(S_vals[i,-1])
        V = np.exp(-self.r*self.T)*np.mean(V_vals)

        return V

    def model_calc_error(self, nr_steps,num_runs,S_0):
        dt = self.T/nr_steps
        V_calc = self.V_calc(nr_steps,num_runs,dt,S_0)
        V_exact = self.bsexact(self.sigma,self.r,self.K,self.T,S_0)

        error = np.abs(V_calc-V_exact)/V_exact*100
        return error
    
    def model_calc_error_diff(self, nr_steps,num_runs,S_0):
        dt = self.T/nr_steps
        V_calc = self.V_calc(nr_steps,num_runs,dt,S_0)
        V_exact = self.bsexact(self.sigma,self.r,self.K,self.T,S_0)

        error = V_calc-V_exact
        return error