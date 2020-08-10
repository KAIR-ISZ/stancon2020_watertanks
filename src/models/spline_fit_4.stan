data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  int<lower=0> L;   // number of simulated samples
  matrix[N, K] x;   // predictor matrix
  vector[N] y;      // outcome vector
}
transformed data {
  matrix[N, K] Q_ast;
  matrix[K, K] R_ast;
  matrix[K, K] R_ast_inverse;
  vector[L] ones_L = rep_vector(1, L);
  matrix[L,K] sub_X = block(x,1,1,L,K);  
  // thin and scale the QR decomposition
  Q_ast = qr_thin_Q(x) * sqrt(N - 1);
  R_ast = qr_thin_R(x) / sqrt(N - 1);
  R_ast_inverse = inverse(R_ast);
}
parameters {
  vector[K] theta;      // coefficients on Q_ast
  real<lower=0> sigma;  // error scale
}
model {
  theta ~ normal(0,0.2);
  sigma ~ exponential(1);
  y ~ normal(Q_ast * theta, sigma);  // likelihood
}
generated quantities {
  vector[K] beta;
  real y_pred[L];
  vector[L] mu;
  beta = R_ast_inverse * theta; // coefficients on x
  mu = sub_X*beta;
  y_pred = normal_rng(mu,ones_L*sigma);
  
}
