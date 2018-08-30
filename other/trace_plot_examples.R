trueA <- 5
trueB <- 0
trueSd <- 10
sampleSize <- 31
set.seed(123)

# create independent x-values 
x <- (-(sampleSize-1)/2):((sampleSize-1)/2)
# create dependent values according to ax + b + N(0,sd)
y <-  trueA * x + trueB + rnorm(n=sampleSize,mean=0,sd=trueSd)

likelihood <- function(param){
  a = param[1]
  b = param[2]
  sd = param[3]
  
  pred = a*x + b
  singlelikelihoods = dnorm(y, mean = pred, sd = sd, log = T)
  sumll = sum(singlelikelihoods)
  return(sumll)   
}

# Example: plot the likelihood profile of the slope a
slopevalues <- function(x){return(likelihood(c(x, trueB, trueSd)))}
slopelikelihoods <- lapply(seq(3, 7, by=.05), slopevalues )

# Prior distribution
prior <- function(param){
  a = param[1]
  b = param[2]
  sd = param[3]
  aprior = dunif(a, min=0, max=10, log = T)
  bprior = dnorm(b, sd = 5, log = T)
  sdprior = dunif(sd, min=0, max=30, log = T)
  return(aprior+bprior+sdprior)
}

posterior <- function(param){
  return (likelihood(param) + prior(param))
}

proposalfunction <- function(param, variance){
  return(rnorm(3,mean = param, sd= variance))
}

run_metropolis_MCMC <- function(startvalue, iterations, variance){
  chain = array(dim = c(iterations+1,3))
  chain[1,] = startvalue
  for (i in 1:iterations){
    proposal = proposalfunction(chain[i,], variance)
    
    probab = exp(posterior(proposal) - posterior(chain[i,]))
    if (runif(1) < probab){
      chain[i+1,] = proposal
    }else{
      chain[i+1,] = chain[i,]
    }
  }
  return(chain)
}

startvalue = c(4,0,10)

chain_good = run_metropolis_MCMC(startvalue, 10000, variance =  c(0.2,0.6,0.4))
chain_small = run_metropolis_MCMC(startvalue, 10000, variance =  c(0.00001,0.00003,0.00002))
chain_large = run_metropolis_MCMC(startvalue, 10000, variance =  c(20,2,1.5))

burnIn = 1000
acceptance = 1-mean(duplicated(chain[-(1:burnIn),]))

par(mfrow=c(1,3))
plot(chain_good[-(1:burnIn),1], type = "l", xlab="Sample Index" , main = "Chain Values With Suitable Variance", ylab = 'Parameter Value', cex.main=2.5, cex.axis=2.5, cex.lab=2.5)
plot(chain_small[-(1:burnIn),1], type = "l", lwd=1.5, xlab="Sample Index" , main = "Chain Values With Variance Too Small", ylab = 'Parameter Value', cex.main=2.5, cex.axis=2.5, cex.lab=2.5)
plot(chain_large[-(1:burnIn),1], type = "l", lwd =1.5, xlab="Sample Index" , main = "Chain Values With Variance Too Large", ylab = 'Parameter Value', cex.main=2.5, cex.axis=2.5, cex.lab=2.5)
