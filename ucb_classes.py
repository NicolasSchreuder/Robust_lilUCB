import numpy as np

import bisect

class lilUCB():
    def __init__(self, K, ν=.1, ɛ=.01, β=1., σ=1.):
        """
        Initialize parameters
        """
        self.ɛ = ɛ
        self.β = β
        self.λ = ((2+self.β)/self.β)**2
        self.K = K # number of arms
        self.σ = σ
        
        # upper bound constant
        self.UCB_cst = (1+β)*(1+np.sqrt(self.ɛ))*self.σ*np.sqrt(2*(1+self.ɛ))
        
        # confidence for given confidence δ
        self.c_ɛ = (2+ɛ)*(1/np.log(1+ɛ))**(1+ɛ)/ɛ
        self.δ = (np.sqrt(1+ν)-1)**2/(4*self.c_ɛ)
        
    def stopping_criterion(self, T, sum_T):
        return ((1+self.λ)*T >= 1 + self.λ*sum_T).any()
    
    def UCB(self, T):
        """
        Compute upper confidence bounds for the arms
        """
        lil = np.log(np.log((1+self.ɛ)*T)/self.δ)
        return self.UCB_cst*np.sqrt(lil/T)
    
    def run(self, mab): 
                
        # initialization
        T = np.zeros(self.K) # number of pulls per arm
        sum_T = 0 # total number of pulls
        theta_hat = np.zeros(self.K) # estimation of parameter theta
        
        # initialize by pulling each arm once
        for k in range(self.K):
            theta_hat[k]  = mab.pull_arm(k) 
            T[k] += 1
            sum_T += 1
        
        # compute upper confidence bounds
        UCB_values = theta_hat + self.UCB(T)     
        
        while not self.stopping_criterion(T, sum_T):
            # find and pull arm with maximum upper confidence bound
            I = np.argmax(UCB_values)
            
            reward = mab.pull_arm(I)
            
            # update estimation of paper for chosen arm
            theta_hat[I] = (T[I]*theta_hat[I] + reward)/(T[I]+1)    
            
            # update counters
            T[I] += 1
            sum_T += 1
            UCB_values[I] = theta_hat[I] + self.UCB(T[I])
            
            
        return T
    

class robust_lilUCB():
    def __init__(self, K, ν=.1, β=1., σ=1.):
        self.β = β
        self.λ = ((2+self.β)/self.β)**2
        self.K = K
        self.σ = σ
        
        # strong convexity radius
        self.r = .5
        
        # strong convexity constant
        self.α = 0.97
        
        # algorithm confidence
        self.δ = ((np.sqrt(16*ν+9)-3)/16)**2
                
        self.c_δ = np.log(1/self.δ)
        self.ucb_constant = (1+self.β)*4*self.σ/self.α
        
        # warm-up index
        self.n_0 = 790
                
    def stopping_criterion(self, T, sum_T):
        return ((1+self.λ)*T >= self.n_0 + self.λ*sum_T).any()
        
    def upper_bound(self, T):
        """
        Compute upper confidence bound for each arm
        """        
        return self.ucb_constant*np.sqrt((np.log(np.log(T)) + self.c_δ)/T)
    
    def run(self, mab):
        
        # initialization
        T = np.zeros(self.K, dtype=int)
        sum_T = 0 # total number of pulls
        theta_hat = [[] for k in range(self.K)]
        for k in range(self.K):
            theta_hat[k]  = [mab.pull_arm(k)]
            T[k] += 1
            sum_T += 1
        
        # pull each arm n_0 times
        for l in range(self.n_0):
            for k in range(self.K):
                reward = mab.pull_arm(k)
                bisect.insort(theta_hat[k], reward)
                T[k] +=1
                sum_T+=1
                
        medians = np.array([theta_hat[k][T[k]//2] for k in range(self.K)])
        upper_bounds = self.upper_bound(T)
        
        while not self.stopping_criterion(T, sum_T):
            
            # find max UCB index
            I = np.argmax(medians + (1+self.β)*upper_bounds)
            reward = mab.pull_arm(I)
            bisect.insort(theta_hat[I], reward)
            
            # update counters
            T[I] += 1
            sum_T += 1
            medians[I] = theta_hat[I][T[I]//2]
            upper_bounds[I] = self.upper_bound(T[I])
            
        return T
    