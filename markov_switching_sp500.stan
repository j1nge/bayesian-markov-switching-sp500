// Two-regime Markov Switching Model for S&P 500 Returns
// Regimes differ in mean and volatility

data {
  int<lower=1> T;              // number of observations
  vector[T] y;                 // log returns
}

parameters {
  // Regime-specific parameters
  ordered[2] mu;               // means (ordered for identification)
  vector<lower=0>[2] sigma;    // standard deviations
  
  // Transition probabilities (staying in same regime)
  real<lower=0, upper=1> p11;  // P(regime 1 -> regime 1)
  real<lower=0, upper=1> p22;  // P(regime 2 -> regime 2)
}

transformed parameters {
  // Transition probability matrix
  matrix[2, 2] Gamma;
  Gamma[1, 1] = p11;
  Gamma[1, 2] = 1 - p11;
  Gamma[2, 1] = 1 - p22;
  Gamma[2, 2] = p22;
  
  // Log-likelihood contributions for each regime at each time
  matrix[T, 2] log_lik;
  for (t in 1:T) {
    for (k in 1:2) {
      log_lik[t, k] = normal_lpdf(y[t] | mu[k], sigma[k]);
    }
  }
}

model {
  // Priors
  mu ~ normal(0, 0.01);              // small mean returns
  sigma ~ inv_gamma(2, 0.01);        // volatility priors
  p11 ~ beta(10, 2);                 // favor staying in regime 1
  p22 ~ beta(10, 2);                 // favor staying in regime 2
  
  // Forward algorithm for marginal likelihood
  vector[2] log_alpha[T];
  vector[2] log_Gamma_trans[2];
  
  // Precompute log transition probabilities
  for (i in 1:2) {
    for (j in 1:2) {
      log_Gamma_trans[i, j] = log(Gamma[i, j]);
    }
  }
  
  // Initialize with stationary distribution
  {
    vector[2] pi;
    pi[1] = (1 - p22) / (2 - p11 - p22);
    pi[2] = (1 - p11) / (2 - p11 - p22);
    log_alpha[1] = log(pi) + log_lik[1]';
  }
  
  // Forward recursion
  for (t in 2:T) {
    for (j in 1:2) {
      vector[2] log_contributions;
      for (i in 1:2) {
        log_contributions[i] = log_alpha[t-1, i] + log_Gamma_trans[i, j];
      }
      log_alpha[t, j] = log_sum_exp(log_contributions) + log_lik[t, j];
    }
  }
  
  // Add log-likelihood
  target += log_sum_exp(log_alpha[T]);
}

generated quantities {
  // Viterbi algorithm for most likely state sequence
  int<lower=1, upper=2> state[T];
  matrix[T, 2] delta;
  
  // Forward pass
  delta[1, 1] = log_lik[1, 1];
  delta[1, 2] = log_lik[1, 2];
  
  for (t in 2:T) {
    for (j in 1:2) {
      vector[2] vals;
      for (i in 1:2) {
        vals[i] = delta[t-1, i] + log(Gamma[i, j]);
      }
      delta[t, j] = max(vals) + log_lik[t, j];
    }
  }
  
  // Backward pass
  state[T] = delta[T, 1] > delta[T, 2] ? 1 : 2;
  for (t in 1:(T-1)) {
    int t_rev = T - t;
    vector[2] vals;
    for (i in 1:2) {
      vals[i] = delta[t_rev, i] + log(Gamma[i, state[t_rev + 1]]);
    }
    state[t_rev] = vals[1] > vals[2] ? 1 : 2;
  }
}
