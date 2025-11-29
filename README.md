# bayesian-markov-switching-sp500

**Methodology**
1. Classify log returns of S&P 500 into two regimes: **crisis** and **calm**
    ---> Each regime has their own distribution for returns and hence there are 4 parameters: 
         - Mean return (calm)
         - Stdv return (calm)
         - Mean return (crisis)
         - Stdv return (crisis)

2. Each regime's parameters have their own priors
    ---> Each parameter has own distribution. ie:
        - Mean (calm) ~ Normal(0,1)
        - Stdv (calm) ~ HalfNormal(1)
        - Mean (crisis) ~ Normal(0,1)
        - Stdv (calm) ~ HalfNormal(2)

3. Plug into STAN model to run HMC (Hamiltonian Monte Carlo) simulations

        