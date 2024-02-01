# Clear environment and free unused memory
rm(list = ls())
gc()

# SOURCES
source("../mcmc_functions/mcmc.R") # Metropolis-Gibbs Sampler
source("../mcmc_functions/priors.R")
source("../mcmc_functions/jacobians.R")
source("../mcmc_functions/likelihood.R")
source("../mcmc_functions/posterior.R")
source("../other_functions/parallel_functions.R") # Parallel wrapper functions
source("../other_functions/helper_functions.R") # Other misc functions (not part of MCMC)

# Libraries
library(parallel) # For parallel computation
library(doParallel) # For parallel computation
library(foreach) # For parallel computation
library(spBayes) # For spatial GPR
library(spNNGP) # For NNGP
library(mvtnorm)
library(fields)

# Read in
load("data/train.RData")
#load("data/test.RData")

# Clusters and seed
nCores <- 4
mySeed <- 856021
p <- ncol(X[[1]])
n <- nrow(X[[1]])
K <- nCores

nIter <- 1000
checkEvery <- 5
nComm <- nIter / checkEvery
alpha <- 0

cov.model <- "exponential"
starting <- rep(list(list("phi"=4, "sigma.sq"=1, "tau.sq"=0.25)), nCores)
tuning <- list("phi"=0.1, "sigma.sq"=0.05, "tau.sq"=0.01)
priors <- list("beta.Norm"=list(rep(0,p), diag(n/K,p)),
               "phi.Unif"=c(1, 10), 
               "sigma.sq.IG"=c(1, 1),
               "tau.sq.IG"=c(1, 1))

#basic <- spLM(Y[[i]] ~ X[[i]], coords = coords, starting = starting,
#              tuning = tuning, priors = priors, cov.model = cov.model, n.samples = nIter)
#apply(basic$p.theta.samples, 2, mean)

samples <- rep(list(matrix(0, nrow = nIter, ncol = 3)), nCores)
for (i in 1:nComm) {
  cl <- makeCluster(nCores)
  registerDoParallel(cl)
  strt <- Sys.time()
  set.seed(mySeed)
  obj <- foreach(i = 1:nCores, .packages = c("mvtnorm", "spBayes")) %dopar% spLM(Y[[i]] ~ X[[i]] - 1, 
                                                                                 coords = coords[[i]], 
                                                                                 starting = starting[[i]],
                                                                                 tuning = tuning, 
                                                                                 priors = priors, 
                                                                                 cov.model = cov.model, 
                                                                                 n.samples = checkEvery)
  final.time <- Sys.time() - strt 
  stopCluster(cl)
  
  for (j in 1:nCores) {
    samples[[j]][(checkEvery * (i-1) + 1):(checkEvery * i),] <- as.matrix(obj[[j]]$p.theta.samples)
  }

  param_ests <- sapply(1:nCores, \(i) apply(obj[[i]]$p.theta.samples, 2, mean))
  starting <- lapply(1:nCores, function(i) {
    list("phi" = alpha * param_ests["phi", i] + (1 - alpha) * mean(param_ests["phi", ]), 
         "sigma.sq" = alpha * param_ests["sigma.sq", i] + (1 - alpha) * mean(param_ests["sigma.sq", ]),  
         "tau.sq" = alpha * param_ests["tau.sq", i] + (1 - alpha) * mean(param_ests["tau.sq", ]))
    })
}



sapply(1:nCores, \(i) apply(samples[[i]], 2, mean))[2,]
sapply(starting, \(x) x$tau.sq)

#rm(list = ls())
#if (file.exists(".RData")) {
#  file.remove(".RData")
#}
gc()
