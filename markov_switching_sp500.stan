
data {
  int<lower=1> T;           // nO, of days
  vector[T] r;              // daily log returns
}

// transformed data
transformed data {
  vector[2] alpha_calm   = [10.0, 1.0]';  
  vector[2] alpha_crisis = [1.0, 10.0]';  
}


parameters {
  vector[2] mu;                     

  vector<lower=0>[2] sigma;         // regime volatilities (std devs)
                                    // sigma[1]=calm, sigma[2]=crisis

  simplex[2] pi;                    // t = 0 regime probabilities

  simplex[2] P[2];                  // transition matrix rows:
                                    // p[1] = calm row  (calm -> calm, crisis)
                                    // P[2] = crisis row(crisis -> calm, crisis)
}

// parameters 
// log_alpha[t, k] = log p(z_t = k, r_1,...,r_t | theta)
transformed parameters {
  matrix[T, 2] log_alpha;

  // t = 1
  for (k in 1:2) {
    log_alpha[1, k] =
      log(pi[k]) +
      normal_lpdf(r[1] | mu[k], sigma[k]);
  }


  // compute likelihood w/ recursion for t = 2,...,T
  for (t in 2:T) {
    for (j in 1:2) {
      vector[2] log_terms;
      // sum over previous states 
      for (i in 1:2) {
        log_terms[i] =
          log_alpha[t - 1, i] +
          log(P[i, j]);
      }
      log_alpha[t, j] =
        log_sum_exp(log_terms) +
        normal_lpdf(r[t] | mu[j], sigma[j]);
    }
  }
}

// model parameters we impose priors on 
model {
  mu[1] ~ normal(0, 0.01);    // Calm mean
  mu[2] ~ normal(0, 0.02);    // crisis mean 

  // half-normal priors on volatilities
  sigma[1] ~ normal(0, 0.01); // calm volatility
  sigma[2] ~ normal(0, 0.03); // crisis volatility 

  // vague prior on regime probabilities
  pi ~ dirichlet(rep_vector(1.0, 2));

  // transition probs; encode persistenc in mx rows
  P[1] ~ dirichlet(alpha_calm);    // calm row
  P[2] ~ dirichlet(alpha_crisis);  // crisis row


  // log p(r_1..r_T | theta) = log_sum_exp over final states
  
  // log posterior 
  target += log_sum_exp(row(log_alpha, T));
}


generated quantities {
  real log_lik;

  // log-likelihood of observed series
  log_lik = log_sum_exp(row(log_alpha, T));
}
