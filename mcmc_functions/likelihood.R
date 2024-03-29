library(mvtnorm)

### LOG LIKELIHOOD ###
logLik <- function(Sigma, beta) {
  sum(sapply(subjs, function(i) {
    dmvnorm(as.vector(newY[[i]]), as.vector(newX[[i]] %*% beta), Sigma, log = TRUE)
  })) * (n / m)
}

