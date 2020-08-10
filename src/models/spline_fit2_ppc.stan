data {
  int<lower=0> N;   // number of data items
  int<lower=0> K;   // number of predictors
  int<lower=0> L;   // number of simulated samples
  matrix[N, K] x;   // predictor matrix
}
transformed data {
  matrix[N, K] Q_ast;
  matrix[K, K] R_ast;
  matrix[K, K] R_ast_inverse;
  vector[K] ones_K = rep_vector(1, K);

  // thin and scale the QR decomposition
  Q_ast = qr_thin_Q(x) * sqrt(N - 1);
  R_ast = qr_thin_R(x) / sqrt(N - 1);
  R_ast_inverse = inverse(R_ast);

}
generated quantities {
  real theta[K] = normal_rng(0,0.2*ones_K);
  real sigma = exponential_rng(1);
  vector[K] beta;
  real y_pred[L];
  beta = R_ast_inverse * to_vector(theta); // coefficients on x
  for (i in 1:L) {
    y_pred[i] = normal_rng(Q_ast[i,:]*to_vector(theta),sigma);
  }
}
