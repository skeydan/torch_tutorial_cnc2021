# Here again is the neural network from the demo.

# 1
# Modify the network to use a different optimizer, e.g., optim_sgd (stochastic gradient descent).
# What happens to performance? Can you change optimizer parameters to improve things?
   
# 2
# Think of other modifications you could make to improve performance on this given "dataset".
# (That is, we're not thinking about generalization at this point - but we will in the next section).

library(torch)


### generate training data -----------------------------------------------------

# input dimensionality (number of input features)
d_in <- 3
# output dimensionality (number of predicted features)
d_out <- 1
# number of observations in training set
n <- 100

# create random data
x <- torch_randn(n, d_in)
y <- x[, 1, NULL] * 0.2 - x[, 2, NULL] * 1.3 - x[, 3, NULL] * 0.5 + torch_randn(n, 1)


### define the network ---------------------------------------------------------

# dimensionality of hidden layer
d_hidden <- 32

model <- nn_sequential(
  nn_linear(d_in, d_hidden),
  nn_relu(),
  nn_linear(d_hidden, d_out)
)

### network parameters ---------------------------------------------------------

learning_rate <- 0.08

# optimizer applies gradient updates for us
optimizer <- optim_adam(model$parameters, lr = learning_rate)

### training loop --------------------------------------------------------------

for (t in 1:200) {
  
  ### -------- Forward pass -------- 
  y_pred <- model(x)
  
  ### -------- compute loss -------- 
  # mean squared error loss
  loss <- nnf_mse_loss(y_pred, y, reduction = "sum")
  if (t %% 10 == 0)
    cat("Epoch: ", t, "   Loss: ", loss$item(), "\n")
  
  ### -------- Backpropagation -------- 
  
  # Still need to zero out the gradients before the backward pass, only this time,
  # on the optimizer object
  optimizer$zero_grad()
  
  # gradients are still computed on the loss tensor (no change here)
  loss$backward()
  
  ### -------- Update weights -------- 
  # use the optimizer to update model parameters
  optimizer$step()
}





