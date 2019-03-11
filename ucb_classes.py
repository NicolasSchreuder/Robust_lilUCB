import numpy as np

import bisect

class lilUCB():
    def __init__(self, K, δ=.1, ϵ=.01, β=1., σ=.5):
        """
        Initialize parameters
        """
        self.ϵ = ϵ
        self.β = β
        self.λ = ((2+self.β)/self.β)**2
        self.K = K # number of arms
        self.σ = σ
        
        # upper bound constant
        self.UCB_cst = 2*(self.σ**2)*(1+self.ϵ)
        
        # confidence for given confidence δ
        self.δ = np.log(1+self.ϵ)*(δ*self.ϵ/(2+self.ϵ))**(1/(1+self.ϵ))
        
    def stopping_criterion(self, T, sum_T):
        return ((1+self.λ)*T >= 1 + self.λ*sum_T).any()
    
    def UCB(self, T):
        """
        Compute upper confidence bounds for the arms
        """
        lil = np.log(np.log((1+self.ϵ)*T)/self.δ)
        return np.sqrt(self.UCB_cst*lil/T)
    
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
        UCB_values = theta_hat + (1+self.β)*(1+np.sqrt(self.ϵ))*self.UCB(T)     
        
        while not self.stopping_criterion(T, sum_T) and sum_T < 1e8:
            # find and pull arm with maximum upper confidence bound
            I = np.argmax(UCB_values)
            
            reward = mab.pull_arm(I)
            
            # update estimation of paper for chosen arm
            theta_hat[I] = (T[I]*theta_hat[I] + reward)/(T[I]+1)    
            
            # update counters
            T[I] += 1
            sum_T += 1
            UCB_values[I] = theta_hat[I] + (1+self.β)*(1+np.sqrt(self.ϵ))*self.UCB(T[I])
            
        return T
    
#a = MAB(20, 'sparse', False, 0.3)
#lilUCB_ = lilUCB(δ=0.1, ϵ=0.01, β=1, K=20)

class robust_lilUCB():
    def __init__(self, K, δ=.1, ϵ=.01, β=1., σ=.5, theoretical_cst=False, n0=1):
        self.ϵ = ϵ
        self.β = β
        self.λ = ((2+self.β)/self.β)**2
        self.K = K
        self.σ = σ
        
        # strong convexity radius
        self.r = 0.5
        
        # strong convexity constant
        self.α = 1
        
        # warm-up index
        #self.n_0 = ((32*(1+self.ϵ)*self.σ**2)/(self.r**2 * self.α**2)
        #            *np.log(2*np.log((32*(1+self.ϵ)*self.σ**2)/(self.r**2 * self.α**2 * self.δ))/self.δ))
        self.n0 = n0
        
        # algorithm confidence for given final confidence
        self.δ = np.log(1+ϵ)**(1+ϵ)/(4*(1+1/ϵ))*δ
        
        self.log_δ = np.log(δ)
        
        # True if bounds are computed with theoretical constants
        self.theoretical_cst = theoretical_cst
        
    def stopping_criterion(self, T, sum_T):
        return ((1+self.λ)*T >= 1 + self.λ*sum_T).any()
    
    def upper_bound(self, T):
        """
        Compute upper confidence bound for each arm
        """
        lil = 2*((1+self.ϵ)*np.log(np.log(T)) - self.log_δ)
        
        if self.theoretical_cst==True:
            cst = 4*np.sqrt(2)*self.σ/self.α
        else:
            cst = 0.3
        return (1+self.β)*cst*np.sqrt(lil/T)
    
    def run(self, mab):
        # initialization
        T = np.zeros(self.K, dtype=int)
        sum_T = 0 # total number of pulls
        theta_hat = [[] for k in range(self.K)]
        for k in range(self.K):
            theta_hat[k]  = [mab.pull_arm(k)]
            T[k] +=1
            sum_T += 1
        
        # pull each arm n0 times
        for l in range(self.n0):
            for k in range(self.K):
                reward = mab.pull_arm(k)
                bisect.insort(theta_hat[k], reward)
                T[k] +=1
                sum_T+=1
                
        medians = np.array([theta_hat[k][T[k]//2] for k in range(self.K)])
        upper_bounds = self.upper_bound(T)
        
        while (not self.stopping_criterion(T, sum_T)) and (sum_T < 1e8):
            
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
    
#a = MAB(20, 'sparse', False, 0.3)
#lilUCB_ = lilUCB(δ=0.1, ϵ=0.01, β=1, K=20)


class medianUCB():
    def __init__(self, δ, ϵ, K):
        self.δ = δ
        self.ϵ = ϵ
        self.K = K
        self.σ = 1.
        self.ρ = 0.3
        
    def stopping_criterion(self, T, medians):
        BAI = np.argmax(medians)
        B = self.UCB(T)
        for k in range(self.K):
            if k==BAI:
                continue
            if medians[BAI] - B[BAI] < medians[k]  + B[k]:
                return False
        return True
    
    def UCB(self, T):
        """
        Compute upper confidence bound for each arm
        """
        return np.sqrt(np.log(T/self.δ)/(2*T))
    
    def lil_upper_bound(self, T):
        """
        Compute upper confidence bound for each arm
        """
        lil = np.log(np.log((1+self.ϵ)*T/self.δ)/self.δ)
        cst = 2*(self.σ**2)*(1+self.ϵ)
        return np.sqrt(cst*lil/T)

    
    def run(self, mab):
        # initialization
        n = 0
        T = np.zeros(self.K, dtype=int)
        theta_hat = [[] for k in range(self.K)]
        for k in range(self.K):
            theta_hat[k]  += [mab.pull_arm(k)]
            T[k] +=1
            n+=1
        
        medians = np.array([theta_hat[k][T[k]//2] for k in range(self.K)])
        
        while not self.stopping_criterion(T, medians):
            if n>1e5*(4*self.K):
                return T
            # find max UCB index
            medians = np.array([theta_hat[k][T[k]//2] for k in range(self.K)])
            I = np.argmax(medians + self.ρ*self.UCB(T))
            reward = mab.pull_arm(I)
            bisect.insort(theta_hat[I], reward)
            
            # update counters
            T[I] += 1
            n+=1
        return T
    
class nonadaptive():
    def __init__(self, K, δ, ϵ):
        self.K = K
        self.δ = δ
        self.ϵ = ϵ
        self.σ = .3
        
    def UCB(self, T):
        # self.K is a typo ?
        LIL = np.log(2*np.log((1+self.ϵ)*T + 2)/self.δ)
        sqrt_cst = 2*self.σ**2*(1+self.ϵ)
        cst = 1+np.sqrt(self.ϵ)
        return cst*np.sqrt(sqrt_cst*LIL/T)
    
    def stopping_criterion(self, theta_hat, T):
        BAI = np.argmax(theta_hat)
        B = self.UCB(T)
        for k in range(self.K):
            if k==BAI:
                continue
            
            if theta_hat[BAI] - B[BAI] < theta_hat[k]  + B[k]:
                return False
        return True
    
    def run(self, mab):
        
        self.cycle = np.random.permutation(np.arange(10))
        
        # initialization
        n = 0
        T = np.ones(self.K)
        theta_hat = np.zeros(self.K)
        for k in self.cycle:
            theta_hat[k]  = mab.pull_arm(k)
            n+=1
        while not self.stopping_criterion(theta_hat, T):
            for k in self.cycle:
                reward = mab.pull_arm(k)
                theta_hat[k]  = (T[k]*theta_hat[k] + reward)/(T[k] + 1)
                T[k] += 1
                n+=1
        
        return T, theta_hat
    
# nonadaptive_proc = nonadaptive(K=10, δ=0.2, ϵ=0.01)
# T = nonadaptive_proc.run(a)