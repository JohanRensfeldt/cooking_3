import numpy as np
import scipy.special as sp
from scipy.stats import norm
import matplotlib.pyplot as plt
import time as time
from tqdm import tqdm
from scipy.linalg import solve

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
    
    def bsexact_Greeks(self):
        d1 = (np.log(self.s0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        # Calculate option price
        F = self.s0 * norm.cdf(d1) - np.exp(-self.r * self.T) * self.K * norm.cdf(d2)
        
        # Calculate Delta
        delta = norm.cdf(d1)

        # Calculate Gamma
        gamma = norm.pdf(d1) / (self.s0 * self.sigma * np.sqrt(self.T))

        # Calculate Vega
        vega = self.s0 * np.sqrt(self.T) * norm.pdf(d1)

        # Calculate Theta
        theta = - (self.s0 * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

        # Calculate Rho
        rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)

        return delta, gamma, vega, theta, rho

    def explicit(self, M, N):
        dS = self.S_max / float(M)
        dt = self.T / float(N)
        grid = np.zeros((M + 1, N + 1))
        grid[:, -1] = np.maximum(0, np.linspace(0, self.S_max, M + 1) - self.K)
     
        for j in tqdm(reversed(range(N)), desc="Processing", total=N, leave=True):
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

    def bsexact_Greeks(self):
        d1 = (np.log(self.s0 / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)

        F = self.s0 * norm.cdf(d1) - np.exp(-self.r * self.T) * self.K * norm.cdf(d2)
        
        # Calculate Delta
        delta = norm.cdf(d1)

        # Calculate Gamma
        gamma = norm.pdf(d1) / (self.s0 * self.sigma * np.sqrt(self.T))

        # Calculate Vega
        vega = self.s0 * np.sqrt(self.T) * norm.pdf(d1)

        # Calculate Theta
        theta = - (self.s0 * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d2)

        # Calculate Rho
        rho = self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d2)

        greeks = {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}

        return greeks

    def FDM(self, calculateGreeks=True, perturbed_r=None):
        self.r = perturbed_r if perturbed_r is not None else self.r
        grid = self.grid_setup()
        
        for j in tqdm(reversed(range(self.N)), desc="Processing", total=self.N, leave=True):
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
        
        if calculateGreeks:
            delta, gamma, theta = self.calculate_greeks(grid)
            original_price = self.FDM(calculateGreeks=False)  # Original FDM without perturbation

            # Perturb interest rate slightly to compute Rho
            perterbation = 0.001
            perturbed_r = self.r + perterbation # Add 0.001 to the original interest rate
            perturbed_price = self.FDM(calculateGreeks=False, perturbed_r=perturbed_r)
            print(f"perturbed_price: {perturbed_price}")
            print(f"original_price: {original_price}")
            rho = (perturbed_price - original_price) / perterbation  # Use the perturbation value here

            print(f"delta: {delta}")
            print(f"gamma: {gamma}")
            print(f"theta: {theta}")
            print(f"rho: {rho}")
            

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

    def calculate_error(self):
        N_array = np.logspace(1, 3, num=3, base=10).astype(int)
        M_array = np.logspace(1, 3, num=3, base=10).astype(int)
        
        error_array = np.zeros(len(N_array))
        option_price_array = np.zeros(len(N_array))
        
        b_exact = self.bsexact()

        for i in range(len(N_array)):
            option_price_array[i] = self.FDM(M_array[i], N_array[i])
            error_array[i] = np.abs(option_price_array[i] - b_exact)

        plt.plot(N_array, error_array)
        plt.show()

    def calculate_greeks(self, grid):
        # Locate the index for s0 in the grid
        index_s0 = int(self.M * self.s0 / self.S_max)

        # Obtain the option prices at s0, s0 + dS, and s0 - dS
        V_minus = grid[index_s0 - 1, 0]
        V_plus = grid[index_s0 + 1, 0]
        V_exact = grid[index_s0, 0]
        V_next_time = grid[index_s0, 1]  # Assume grid is M x N with N being the time dimension
        
        # Calculate Delta, Gamma and Theta
        delta = (V_plus - V_minus) / (2 * self.dS)
        gamma = (V_plus - 2 * V_exact + V_minus) / (self.dS ** 2)
        theta = (V_next_time - V_exact) / self.dt
        
        # Additional calculations for Vega, Theta, Rho can be done here
        
        return delta, gamma, theta

class FDM_Multi_Asset_Option:

    def __init__(self, K, s0_1, s0_2, S_max_1, S_max_2, T, r_1, r_2, sigma_1, sigma_2, M1, M2, N):
        self.K = K
        self.s0_1 = s0_1
        self.s0_2 = s0_2
        self.S_max_1 = S_max_1
        self.S_max_2 = S_max_2
        self.T = T
        self.r_1 = r_1
        self.r_2 = r_2
        self.sigma_1 = sigma_1
        self.sigma_2 = sigma_2
        self.M1 = M1
        self.M2 = M2
        self.N = N
    
    def implicit(self):
        # Define variables
        M1, M2, N = self.M1, self.M2, self.N
        dS1 = self.S_max_1 / M1
        dS2 = self.S_max_2 / M2
        dt = self.T / N

        # Initialize the grid
        grid = np.zeros((M1 + 1, M2 + 1, N + 1))

        # Terminal condition
        for i in range(M1 + 1):
            for j in range(M2 + 1):
                grid[i, j, -1] = max(0, i * dS1 + j * dS2 - self.K)

        # Initialize matrix to store coefficients (for simplicity, assuming M1 = M2)
        A = np.zeros((M1 - 1, M1 - 1))
        B = np.zeros((M2 - 1, M2 - 1))

        iterator = tqdm(reversed(range(N)), desc="Processing", total=N, leave=True)

        # Time-stepping
        for n in iterator:
            # Create the A and B matrices
            for i in range(1, M1):
                S1 = i * dS1
                alpha1 = 0.25 * dt * (self.sigma_1 ** 2 * S1 ** 2 / dS1 ** 2 - self.r_1 * S1 / dS1)
                beta1 = 0.5 * dt * (self.sigma_1 ** 2 * S1 ** 2 / dS1 ** 2 + self.r_1)
                gamma1 = alpha1

                if i > 1:
                    A[i - 1, i - 2] = -alpha1
                A[i - 1, i - 1] = 1 + beta1
                if i < M1 - 1:
                    A[i - 1, i] = -gamma1

            # Same for B (assuming M1 = M2 for simplicity)
            for j in range(1, M2):
                S2 = j * dS2
                alpha2 = alpha1
                beta2 = beta1
                gamma2 = alpha2

                if j > 1:
                    B[j - 1, j - 2] = -alpha2
                B[j - 1, j - 1] = 1 + beta2
                if j < M2 - 1:
                    B[j - 1, j] = -gamma2

            # Solve the linear system at each time step for each asset
            for i in range(1, M1):
                vec = grid[i, 1:M2, n + 1]
                grid[i, 1:M2, n] = solve(B, vec)

            for j in range(1, M2):
                vec = grid[1:M1, j, n + 1]
                grid[1:M1, j, n] = solve(A, vec)

        return grid[int(self.s0_1 / dS1), int(self.s0_2 / dS2), 0]

