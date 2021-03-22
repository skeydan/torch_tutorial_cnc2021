
# Neural network walkthrough ----------------------------------------------

# Steps:
# Data prep & data loading
# Model definition
# Training 
# Evaluation

# see https://torch.mlverse.org/start/guess_the_correlation/ for detailed explanations
# see also: https://torch.mlverse.org/start/what_if/ for ideas how to play with this
# and https://torch.mlverse.org/start/custom_dataset/ for how to create a dataset class 


library(torch)
library(torchvision)
library(torchdatasets)


# Dataset -----------------------------------------------------------------
# We use one of the datasets available in torchdatasets.
# As we download the data, we tell torch what to do for pre-processing, and how we want 
# it split up.


# training-validation-test split
train_indices <- 1:10000
val_indices <- 10001:15000
test_indices <- 15001:20000

# custom image transformations
add_channel_dim <- function(img) img$unsqueeze(1)
crop_axes <- function(img) transform_crop(img, top = 0, left = 21, height = 131, width = 130)

# where to store
root <- file.path(tempdir(), "correlation")

train_ds <- guess_the_correlation_dataset(
  # where to unpack
  root = root,
  # additional preprocessing 
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
  # don't take all data, but just the indices we pass in
  indexes = train_indices,
  download = TRUE
)

valid_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
  indexes = val_indices,
  download = FALSE
)

test_ds <- guess_the_correlation_dataset(
  root = root,
  transform = function(img) crop_axes(img) %>% add_channel_dim(),
  indexes = test_indices,
  download = FALSE
)

train_ds[1]


# Dataloader --------------------------------------------------------------
# Purpose: feed data to torch in batches.

train_dl <- dataloader(train_ds, batch_size = 8, shuffle = TRUE)
valid_dl <- dataloader(valid_ds, batch_size = 8)
test_dl <- dataloader(test_ds, batch_size = 8)

batch <- dataloader_make_iter(train_dl) %>% dataloader_next()

par(mfrow = c(8,8), mar = rep(0, 4))

images <- as.array(batch$x$squeeze(2))

images %>%
  purrr::array_tree(1) %>%
  purrr::map(as.raster) %>%
  purrr::iwalk(~{plot(.x)})



# Model -------------------------------------------------------------------
# A torch module is a container that can hold an arbitrary number of sub-modules.
# This module ("model") contains convolutional as well as linear submodules ("layers").

net <- nn_module(
  
  "corr-cnn",
  
  initialize = function() {
    
    # submodules (layers)
    self$conv1 <- nn_conv2d(in_channels = 1, out_channels = 32, kernel_size = 3)
    self$conv2 <- nn_conv2d(in_channels = 32, out_channels = 64, kernel_size = 3)
    self$conv3 <- nn_conv2d(in_channels = 64, out_channels = 128, kernel_size = 3)
    
    self$fc1 <- nn_linear(in_features = 14 * 14 * 128, out_features = 128)
    self$fc2 <- nn_linear(in_features = 128, out_features = 1)
    
  },
  
  forward = function(x) {
    
    x %>% 
      self$conv1() %>%
      nnf_relu() %>% # ReLU activation
      nnf_avg_pool2d(2) %>% # average pooling (downsizing)
      
      self$conv2() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      
      self$conv3() %>%
      nnf_relu() %>%
      nnf_avg_pool2d(2) %>%
      
      torch_flatten(start_dim = 2) %>%
      self$fc1() %>%
      nnf_relu() %>%
      
      self$fc2() # no activation function for regression 
  }
)

model <- net()

device <- torch_device(if (cuda_is_available()) "cuda" else "cpu")

model <- model$to(device = device)


# Training ----------------------------------------------------------------
# For training, we need to decide on an adequate loss function, as well as 
# the optimization algorithm to use.

loss <- nnf_mse_loss(output, b$y$unsqueeze(2))

optimizer <- optim_adam(model$parameters)


# actions to be performed while training
train_batch <- function(b) {
  
  optimizer$zero_grad()
  
  output <- model(b$x$to(device = device))
  
  # calculate loss
  loss <- nnf_mse_loss(output, b$y$unsqueeze(2)$to(device = device))
  
  # have gradients get calculated        
  loss$backward()
  
  # have gradients get applied
  optimizer$step()
  
  loss$item()
  
}

# actions to be performed for evaluation
valid_batch <- function(b) {
  
  output <- model(b$x$to(device = device))
  # calculate loss
  loss <- nnf_mse_loss(output, b$y$unsqueeze(2)$to(device = device))
  loss$item()
  
}

# iterate over 10 epochs
num_epochs <- 10

for (epoch in 1:num_epochs) {
  
  # put model in training mode (backpropagation)
  model$train()
  
  train_losses <- c()
  
  for (b in enumerate(train_dl)) {
    
    loss <- train_batch(b)
    train_losses <- c(train_losses, loss)
    
  }
  
  # put model in evaluation mode (no backpropagation)
  model$eval()
  
  valid_losses <- c()
  
  for (b in enumerate(valid_dl)) {
    
    loss <- valid_batch(b)
    valid_losses <- c(valid_losses, loss)
    
  }
  
  cat(sprintf("\nLoss at epoch %d: training: %1.5f, validation: %1.5f\n", epoch, mean(train_losses), mean(valid_losses)))
  
}



# Evaluation --------------------------------------------------------------
# on the test set.

model$eval()

test_batch <- function(b) {
  
  output <- model(b$x$to(device = device))
  loss <- nnf_mse_loss(output, b$y$unsqueeze(2)$to(device = device))
  
  preds <<- c(preds, output %>% as.numeric())
  targets <<- c(targets, b$y %>% as.numeric())
  test_losses <<- c(test_losses, loss$item())
  
}

test_losses <- c()
preds <- c()
targets <- c()

for (b in enumerate(test_dl)) {
  test_batch(b)
}

mean(test_losses)

df <- data.frame(preds = preds, targets = targets)

library(ggplot2)

ggplot(df, aes(x = targets, y = preds)) +
  geom_point(size = 0.1) +
  theme_classic() +
  xlab("true correlations") +
  ylab("model predictions")


