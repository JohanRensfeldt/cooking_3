import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import time as time

class OptionPricingExplicit:
    def __init__(self, K, s0, S_max, T, r, sigma, M, N):
        self.K = K
        self.s0 = s0
        self.S_max = S_max
        self.T = T
        self.r = r
        self.sigma = sigma
        self.M = M
        self.N = N

    def bsexact(self):
        d1 = (np.log(self.s0/self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        F = 0.5 * self.s0 * (1 + sp.erf(d1 / np.sqrt(2))) - np.exp(-self.r * self.T) * self.K * 0.5 * (1 + sp.erf(d2 / np.sqrt(2)))
        return F

    def explicit(self, M, N):
        dS = self.S_max / float(M)
        dt = self.T / float(N)
        grid = np.zeros((M + 1, N + 1))
        grid[:, -1] = np.maximum(0, np.linspace(0, self.S_max, M + 1) - self.K)

        for j in reversed(range(N)):
            for i in range(1, M):
                S = i * dS
                alpha = 0.5 * dt * (self.sigma ** 2 * S ** 2 / dS ** 2 - self.r * S / dS)
                beta = 1.0 - dt * (self.sigma ** 2 * S ** 2 / dS ** 2 + self.r)
                gamma = 0.5 * dt * (self.sigma ** 2 * S ** 2 / dS ** 2 + self.r * S / dS)

                grid[i, j] = alpha * grid[i - 1, j + 1] + beta * grid[i, j + 1] + gamma * grid[i + 1, j + 1]

        return grid[int(self.s0 / dS), 0]

    def plot_error(self):
        N_array = np.logspace(1, 3, num=3, base=10).astype(int)
        M_array = np.logspace(1, 3, num=3, base=10).astype(int)
        
        error_array = np.zeros(len(N_array))
        option_price_array = np.zeros(len(N_array))
        
        b_exact = self.bsexact()

        for i in range(len(N_array)):
            option_price_array[i] = self.explicit(M_array[i], N_array[i])
            error_array[i] = np.abs(option_price_array[i] - b_exact)

        plt.plot(N_array, error_array)
        plt.show()

class OptionPricingImplicit:
    def __init__(self, K, s0, S_max, T, r, sigma, gamma, M, N):
        self.K = K
        self.s0 = s0
        self.S_max = S_max
        self.T = T
        self.r = r
        self.sigma = sigma
        self.gamma = gamma
        self.M = M
        self.N = N
        self.dS = S_max / float(M)
        self.dt = T / float(N)
        
    def grid_setup(self):
        i_values = np.linspace(0, self.S_max, self.M+1)
        j_values = np.linspace(0, self.T, self.N+1)
        grid = np.zeros((self.M+1, self.N+1))
        grid[:, -1] = np.maximum(0, i_values - self.K)
        return grid, i_values, j_values
    
    def bsexact(self):
        d1 = (np.log(self.s0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        F = 0.5 * self.s0 * (1 + sp.erf(d1 / np.sqrt(2))) - np.exp(-self.r * self.T) * self.K * 0.5 * (1 + sp.erf(d2 / np.sqrt(2)))
        return F

    def FDM(self):
        grid, i_values, j_values = self.grid_setup()
        for j in reversed(range(self.N)):
            diag = np.zeros(self.M-1)
            sub_diag = np.zeros(self.M-2)
            super_diag = np.zeros(self.M-2)
            
            for i in range(1, self.M):
                S = i * self.dS
                S_alpha = S ** self.gamma
                diag[i-1] = 1 + self.dt * (self.sigma ** 2 * S_alpha ** 2 + self.r)
                
                if i > 1:
                    sub_diag[i-2] = -0.5 * self.dt * (self.sigma ** 2 * S_alpha ** 2 - self.r * S)
                
                if i < self.M-1:
                    super_diag[i-1] = -0.5 * self.dt * (self.sigma ** 2 * S_alpha ** 2 + self.r * S)
                
            A = np.diag(diag) + np.diag(sub_diag, k=-1) + np.diag(super_diag, k=1)
            B = grid[1:self.M, j+1]
            B[-1] += 0.5 * self.dt * (self.sigma ** 2 * self.S_max ** (2 * self.gamma) + self.r * self.S_max) * (self.S_max - self.K * np.exp(-self.r * self.dt * (self.N-j)))
            grid[1:self.M, j] = np.linalg.solve(A, B)
        
        return grid[int(self.M * self.K / self.S_max), 0]
    
    def plot_time_vs_N(self, N_array):
        time_array = np.zeros(len(N_array))
        for i, n in enumerate(N_array):
            self.N = n
            self.dt = self.T / float(n)
            start_time = time.time()
            self.FDM()
            end_time = time.time()
            time_array[i] = end_time - start_time
        
        plt.figure()
        plt.plot(N_array, time_array)
        plt.xlabel("N")
        plt.ylabel("Time")
        plt.title("Time vs N implicit")
        plt.show()

if __name__ == "__main__":
    option_pricer = OptionPricingImplicit(K=15.0, s0=14.0, S_max=60, T=0.5, r=0.1, sigma=0.25, gamma=1, M=100, N=100)
    option_price = option_pricer.FDM()
    print("European Call Option Price CEV model:", option_price)
    print("European Call Option Price BS model:", option_pricer.bsexact())
    
    N_array = np.linspace(10, 1000, 10).astype(int)
    option_pricer.plot_time_vs_N(N_array)

    op = OptionPricingExplicit(K=15.0, s0=14.0, S_max=60.0, T=0.5, r=0.1, sigma=0.25, M=100, N=100)
    op.plot_error()

    option_price = op.explicit(op.M, op.N)
    print("European Call Option Price: ", option_price)