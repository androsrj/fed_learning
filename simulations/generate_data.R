library(mvtnorm)

n <- 4000
K <- 4
beta <- 5
sigma.sq <- 2
theta <- 3

tau.sq <- rep(c(0.1, 0.2, 0.3, 0.4), each = n/K)

coords <- cbind(runif(n,0,1), runif(n,0,1))
X <- as.matrix(cbind(1, rnorm(n)))
B <- as.matrix(c(1, beta))
p <- length(B)
D <- as.matrix(dist(coords))
R <- exp(-theta*D)
W <- t(rmvnorm(1, rep(0,n), sigma.sq*R))
Y <- rnorm(n, X%*%B + W, sqrt(tau.sq))

indices <- lapply(1:K, \(i) ((i-1)*(n/K)+1):(n/K *i))
X <- lapply(indices, \(i) X[i,])
Y <- lapply(indices, \(i) Y[i])
coords <- lapply(indices, \(i) coords[i,])

save(X, Y, D, W, coords, file = "data/train.RData")
