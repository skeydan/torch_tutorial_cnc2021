library(torch)

# This is a lookahead to our final topic, torch for optimization.
# Let's use autograd to implement our own minimization routine.

# 1
# Fill out the missing pieces (TBD) to make this work. 

# 2
# Can you find a learning rate that speeds up the process?

# function to minimize
f <- function(x) x^2 - 7

# we start from x = 11
param <- torch_tensor(11, requires_grad = TRUE)

# learning rate: fraction of gradient to subtract
lr <- 0.1

for (i in 1:10) {
  
  cat("Iteration: ", i, "\n")
  
  # call function with current parameter
  # value <- TBD
  cat("Value is: ", as.numeric(value), "\n")
  
  # compute gradient of value w.r.t. param
  # TBD
  cat("Gradient is: ", as.matrix(param$grad), "\n")
  
  # update
  # wrap in with_no_grad
  with_no_grad({
    # subtract a fraction of gradient from param
    # TBD
    
    # zero out on every iteration (would accumulate otherwise)
    param$grad$zero_()
  })
  
  cat("After update: Param is: ", as.matrix(param), "\n\n")
  
  if (abs(-7 - as.numeric(value)) < 0.00005) break
}

