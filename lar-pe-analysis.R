# reproduction of Efron et al. 2004 results on diabetes dataset

require(FWDselect)
require(lars)
data(diabetes)

# see https://search.r-project.org/CRAN/refmans/FWDselect/html/diabetes.html
y = diabetes$y
x = as.data.frame(unclass(diabetes$x))

### reproducing fig 5 of Efron et al.

# Create new covariate matrix with interactions and squared terms
base_vars <- names(x)
sq_vars  <- base_vars[-2]   # don't square binary sex variable

# add squared terms
x_sq <- cbind(
  x,
  setNames(lapply(x[sq_vars], function(z) z^2),
           paste0(sq_vars, "_sq"))
)

# interaction terms among original variables
x_inter <- model.matrix(~ .^2 - 1, data = x)

x_large <- cbind(x_inter, x_sq[setdiff(names(x_sq), base_vars)])

# compute 'true' means and define simulator
lars_true <- lars(as.matrix(x), y, type='lar', max.steps=10, trace=TRUE)
beta_true <- lars_true$beta[length(x)+1, ]

true_model_mu <- as.matrix(x) %*% beta_true
residuals <- (y - mean(y)) - true_model_mu

simulator <- function(residuals, mu){
  resampled_residuals <- sample(residuals, replace = TRUE, size = length(residuals))
  return(mu + resampled_residuals)
}

# proportion of variance explained
prop_explained <- function(mu_est, mu){
  return(1 - (norm(mu - mu_est, type='2')**2 /norm(mu, type='2')**2))
}

# simulating
num_sims = 100
results = data.frame(
  lar_k = numeric(),
  lar_pe = numeric(),
  fs_k = numeric(),
  fs_pe = numeric()
)

for (k in 1:40){
  lar_pe_list = c()
  fs_pe_list = c()
  for (i in 1:num_sims){
    y_sim = mean(y) + simulator(residuals, true_model_mu) # simulated data
    lar_model = lars(as.matrix(x_large), y_sim, type='lar', max.steps=k)
    fs_model = lars(as.matrix(x_large), y_sim, type='stepwise', max.steps=k)
    
    lar_mu = as.matrix(x_large) %*% lar_model$beta[k+1,]
    fs_mu = as.matrix(x_large) %*% fs_model$beta[k+1,]
    
    lar_pe_list = c(lar_pe_list, prop_explained(lar_mu, true_model_mu))
    fs_pe_list = c(fs_pe_list, prop_explained(fs_mu, true_model_mu))
    
  }
  
  lar_pe = mean(lar_pe_list)
  fs_pe = mean(fs_pe_list)
  
  results[dim(results)[1]+1,] <- c(k, lar_pe, k, fs_pe)
}

plot(1:40, results$fs_pe, ylim = c(0.75, 1.0), type='l', col='blue', lwd=2,
     ylab = 'Proportion explained', xlab='Number of components')
lines(1:40, results$lar_pe, ylim = c(0.75, 1.0), type='l', col='red', lwd=2)
legend(x=23, y=0.78, legend=c('Forward selection', 'LARS'), col = c('blue', 'red'), lwd = 2)

