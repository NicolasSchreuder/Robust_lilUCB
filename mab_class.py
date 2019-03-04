import numpy as np

class MAB():
    """
    Stochastic multi armed bandit framework
    """
    
    def __init__(self, K, scenario, distrib, σ=0.5, α=0):
        """
        Initialize multi-armed bandit game
        K : int, number of arms
        scenario : str, "sparse", "alpha"
        distrib : str, reward generating process
        """
        self.K = K
        self.scenario = scenario
        self.distrib = distrib
        self.α = α
        self.σ = σ
        
        # initialize parameter theta
        if scenario == 'sparse':
            # sparse model
            self.θ = np.zeros(K)
            self.θ[0] = 0.5
            
        elif scenario == 'alpha':
            # exponential decrease model
            assert α != 0
            self.θ = np.ones(K)
            for k in range(1, K):
                self.θ[k] = self.θ[k] - (k/K)**self.α
    
    def pull_arm(self, k):
        """
        Pull arm k and receive reward
        k : int, pulled arm index
        """
        #assert k < self.K # valid arm
        
        if self.distrib == "cauchy":
            reward =  self.θ[k] + np.random.standard_cauchy()
            
        elif self.distrib == "gaussian":
            reward = np.random.normal(loc=self.θ[k], scale=self.σ)
        
        elif self.distrib == "huber":
            binomial_draw = np.random.binomial(n=1, p=0.1)
            reward = (1-binomial_draw)*np.random.normal(loc=self.θ[k], scale=self.σ) + binomial_draw*1e2
        return reward

# a = MAB(10, 'sparse', False, 0.3)
# a.θ, a.pull_arm(2)