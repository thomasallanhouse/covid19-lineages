// Consider Bayesian approach to logistic data with GP prior on f
// Code based on examples given at link below (accessed 16 July 2025)
// https://mc-stan.org/docs/stan-users-guide/gaussian-processes.html

// The input data
data {
  int<lower=1> N; // Length of data
  array[N] real t; // Time points
  array[N] int<lower=0> m; // Trials
  array[N] int<lower=0> y; // Number of positive trials
}

transformed data {
  vector[N] mu = rep_vector(0, N); // Prior is zero-mean
}

parameters {
  real log_ell; // GP RBF length scale
  real log_sigma; // GP RBF noise
  vector[N] f; // The process on transformed space
}

model {
  matrix[N, N] K;
  matrix[N, N] L;
  K = gp_exp_quad_cov(t, exp(log_sigma), exp(log_ell));
  for (n in 1:N) {
    K[n, n] = K[n, n] + 1e-3;
    // This is to ensure inversion works and should ideally be as small as possible
    }
  L = cholesky_decompose(K);
  f ~ multi_normal_cholesky(mu, L);
  
  // Uninformative priors on GP RBF hyperparameters
  log_ell ~ cauchy(0,10);
  log_sigma ~ cauchy(0,10);
 
  y ~ binomial_logit(m, f);
}
