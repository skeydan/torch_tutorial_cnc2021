library(torch)

x <- torch_randn(c(7,2))

# A linear module done manually ---------------------------------------------------------

w <- torch_tensor(c(0.1, 0.1), requires_grad = TRUE)
b <- torch_tensor(0.5, requires_grad = TRUE)
  
x$matmul(w) + b  
  
  

# Linear module with torch ------------------------------------------------

l <- nn_linear(in_features = 2, out_features = 1)
l(x)

l$weight # default is uniform(-sqrt(num_features), sqrt(num_features))
nn_init_constant_(l$weight, 0.1)
l$bias # default same as for weights
nn_init_constant_(l$bias, 0.5)

l(x)


# we get autograd for free!

# before backward
loss <- l(x)$sum() # assume we wanted to minimize sum of outputs
l$weight$grad
l$bias$grad

out$grad_fn

# compute gradients
out$backward()
l$weight$grad
l$bias$grad


# Other modules -----------------------------------------------------------

# single RGB 32x32 image 
img <- torch_rand(c(1, 3, 32, 32))

# convolutional layer
conv <- nn_conv2d(in_channels = 3, out_channels = 1, kernel_size = 3, padding = 1)
conv(img)

# spatial pooling
pool <- nn_max_pool2d(kernel_size = 2)
conv(img) %>% pool()

# and many, many more



# Composing modules -------------------------------------------------------

model <- nn_sequential(
  nn_linear(2, 16),
  nn_relu(),
  nn_linear(16, 1)
)

model$parameters

model(x)



# A simple neural network -------------------------------------------------


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




