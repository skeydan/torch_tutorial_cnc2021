library(torch)
library(tidyverse)
library(tsibble)
library(tsibbledata)
library(lubridate)
library(feasts)


# Scope -------------------------------------------------------------------

# In this first exploration of torch RNNs, we aim to forecast the next observation,
# given a configurable sequence of prior measurements.

# Data --------------------------------------------------------------------

vic_elec

vic_elec_daily <- vic_elec %>%
  select(Time, Demand) %>%
  index_by(Date = date(Time)) %>%
  summarise(
    Demand = sum(Demand) / 1e3) 

vic_elec_2014 <-  vic_elec_daily %>%
  filter(year(Date) == 2014) 

cmp <- vic_elec_2014 %>%
  model(STL(Demand)) %>% 
  components()

cmp %>% autoplot()



# Train-valid-test split --------------------------------------------------

elec_train <- vic_elec_daily %>% 
  filter(year(Date) %in% c(2012, 2013)) %>%
  as_tibble() %>%
  select(Demand) %>%
  as.matrix()

elec_valid <- vic_elec_daily %>% 
  filter(year(Date) == 2014) %>%
  as_tibble() %>%
  select(Demand) %>%
  as.matrix()

elec_test <- vic_elec_daily %>% 
  filter(year(Date) %in% c(2014), month(Date) %in% 1:4) %>%
  as_tibble() %>%
  select(Demand) %>%
  as.matrix()

train_mean <- mean(elec_train)
train_sd <- sd(elec_train)


# Create torch dataset ----------------------------------------------------

elec_dataset <- dataset(
  name = "elec_dataset",
  
  initialize = function(x, n_timesteps) {
    
    self$n_timesteps <- n_timesteps
    self$x <- torch_tensor((x - train_mean) / train_sd)

  },
  
  .getitem = function(i) {
    
    start <- i
    end <- start + self$n_timesteps - 1
    
    list(
      x = self$x[start:end],
      y = self$x[end + 1]
    )
    
  },
  
  .length = function() {
    length(self$x) - self$n_timesteps
  }
)

n_timesteps <- 7 * 2

train_ds <- elec_dataset(elec_train, n_timesteps)
length(train_ds)
train_ds[1]

valid_ds <- elec_dataset(elec_valid, n_timesteps)
length(valid_ds)

test_ds <- elec_dataset(elec_test, n_timesteps)
length(test_ds)


# Dataloaders -------------------------------------------------------------

batch_size <- 32
train_dl <- train_ds %>% dataloader(batch_size = batch_size, shuffle = TRUE)

valid_dl <- valid_ds %>% dataloader(batch_size = batch_size)

test_dl <- test_ds %>% dataloader(batch_size = 1)

b <- dataloader_make_iter(train_dl) %>% dataloader_next()
b



# Model -------------------------------------------------------------------

model <- nn_module(
  
  initialize = function(input_size, hidden_size) {

    self$rnn <- nn_gru(
        input_size = input_size,
        hidden_size = hidden_size,
        batch_first = TRUE
      )
    
    self$output <- nn_linear(hidden_size, 1)
    
  },
  
  forward = function(x) {
    
    # list of [output, hidden]
    # we use the output, which is of size (batch_size, n_timesteps, hidden_size)
    x <- self$rnn(x)[[1]]
    
    # from the output, we only want the final timestep
    # shape now is (batch_size, hidden_size)
    x <- x[ , dim(x)[2], ]
    
    # feed this to a single output neuron
    # final shape then is (batch_size, 1)
    x %>% self$output() 
  }
  
)

net <- model(input_size = 1, hidden_size = 64)
net(b$x)


# Train -------------------------------------------------------------------

optimizer <- optim_adam(net$parameters, lr = 0.001)

num_epochs <- 100

train_batch <- function(b) {
  
  optimizer$zero_grad()
  output <- net(b$x)
  target <- b$y
  
  loss <- nnf_mse_loss(output, target)
  loss$backward()
  optimizer$step()
  
  loss$item()
}

valid_batch <- function(b) {
  
  output <- net(b$x)
  target <- b$y
  
  loss <- nnf_mse_loss(output, target)
  loss$item()
  
}

for (epoch in 1:num_epochs) {
  
  net$train()
  train_loss <- c()
  
  coro::loop(for (b in train_dl) {
    loss <-train_batch(b)
    train_loss <- c(train_loss, loss)
  })
  
  cat(sprintf("\nEpoch %d, training: loss: %3.5f \n", epoch, mean(train_loss)))
  
  net$eval()
  valid_loss <- c()
  
  coro::loop(for (b in valid_dl) {
    loss <- valid_batch(b)
    valid_loss <- c(valid_loss, loss)
  })
  
  cat(sprintf("\nEpoch %d, validation: loss: %3.5f \n", epoch, mean(valid_loss)))
}


# Evaluation --------------------------------------------------------------

net$eval()

preds <- rep(NA, n_timesteps)

coro::loop(for (b in test_dl) {
  output <- net(b$x)
  preds <- c(preds, output %>% as.numeric())
})

preds_ts <- vic_elec_daily %>% 
  filter(year(Date) %in% c(2014), month(Date) %in% 1:4) %>%
  add_column(forecast = preds * train_sd + train_mean) %>%
  pivot_longer(-Date) %>%
  update_tsibble(key = name)

preds_ts %>%
  autoplot() +
  scale_colour_manual(values = c("#08c5d1", "#00353f")) +
  theme_minimal()




