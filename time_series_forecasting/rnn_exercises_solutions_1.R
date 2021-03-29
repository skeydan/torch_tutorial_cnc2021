# 1
# Try modifying the multi-step network to use temperature as an additional predictor (not as an outcome).
# Does this help with learning?


# Solution ---------------------------------------------------------------

library(torch)
library(tidyverse)
library(tsibble)
library(tsibbledata)
library(lubridate)
library(fable)

# Data --------------------------------------------------------------------

vic_elec

vic_elec_daily <- vic_elec %>%
  select(Time, Demand, Temperature) %>%
  index_by(Date = date(Time)) %>%
  summarise(
    Demand = sum(Demand) / 1e3,
    Temperature = max(Temperature)) 

# Train-valid-test split --------------------------------------------------

elec_train <- vic_elec_daily %>% 
  filter(year(Date) %in% c(2012, 2013)) %>%
  as_tibble() %>%
  select(Demand, Temperature) %>%
  as.matrix()

elec_valid <- vic_elec_daily %>% 
  filter(year(Date) == 2014) %>%
  as_tibble() %>%
  select(Demand, Temperature) %>%
  as.matrix()

elec_test <- vic_elec_daily %>% 
  filter(year(Date) %in% c(2014), month(Date) %in% 1:4) %>%
  as_tibble() %>%
  select(Demand, Temperature) %>%
  as.matrix()

train_mean_demand <- colMeans(elec_train)[1]
train_sd_demand <- sd(elec_train[ , 1])

train_mean_temp <- colMeans(elec_train)[2]
train_sd_temp <- sd(elec_train[ , 2])

# Create torch dataset ----------------------------------------------------

elec_dataset <- dataset(
  name = "elec_dataset",
  
  initialize = function(x, n_timesteps, n_forecast) {
    
    self$n_timesteps <- n_timesteps
    self$n_forecast <- n_forecast
    
    demand <- (x[ , 1] - train_mean_demand) / train_sd_demand
    temp <- (x[ , 2] - train_mean_temp) / train_sd_temp
    self$x <- torch_tensor(cbind(demand, temp))

  },
  
  .getitem = function(i) {
    
    start <- i
    end <- start + self$n_timesteps - 1
    pred_length <- self$n_forecast
    
    list(
      x = self$x[start:end, ],
      y = self$x[(end + 1):(end + pred_length), 1]
    )
    
  },
  
  .length = function() {
    nrow(self$x) - self$n_timesteps - self$n_forecast + 1
  }
)

n_timesteps <- 7 * 2
n_forecast <- 7 * 2

train_ds <- elec_dataset(elec_train, n_timesteps, n_forecast)
length(train_ds)
train_ds[1]

valid_ds <- elec_dataset(elec_valid, n_timesteps, n_forecast)
length(valid_ds)

test_ds <- elec_dataset(elec_test, n_timesteps, n_forecast)
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
  
  initialize = function(input_size, hidden_size, linear_size, output_size, dropout = 0) {
    
    self$rnn <- nn_gru(
        input_size = input_size,
        hidden_size = hidden_size,
        batch_first = TRUE
      )
    
    self$mlp <- nn_sequential(
      nn_linear(hidden_size, linear_size),
      nn_relu(),
      nn_dropout(dropout),
      nn_linear(linear_size, output_size)
    )
    
  },
  
  forward = function(x) {
    
    x <- self$rnn(x)
    # pass last timestep of RNN output to MLP
    x[[1]][ ,-1, ..] %>% 
      self$mlp()
    
  }
  
)

net <- model(input_size = 2, hidden_size = 64, linear_size = 128,
             output_size = n_forecast, dropout = 0.5)
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

test_preds <- vector(mode = "list", length = length(test_dl))

i <- 1

coro::loop(for (b in test_dl) {
  
  output <- net(b$x)
  preds <- as.numeric(output)
  
  test_preds[[i]] <- preds
  i <<- i + 1
  
})

vic_elec_test <- vic_elec_daily %>% 
  filter(year(Date) %in% c(2014), month(Date) %in% 1:4) %>%
  select(Demand)

test_pred1 <- test_preds[[1]]
test_pred1 <- c(rep(NA, n_timesteps), test_pred1, rep(NA, nrow(vic_elec_test) - n_timesteps - n_forecast))

test_pred2 <- test_preds[[21]]
test_pred2 <- c(rep(NA, n_timesteps + 20), test_pred2, rep(NA, nrow(vic_elec_test) - 20 - n_timesteps - n_forecast))

test_pred3 <- test_preds[[41]]
test_pred3 <- c(rep(NA, n_timesteps + 40), test_pred3, rep(NA, nrow(vic_elec_test) - 40 - n_timesteps - n_forecast))

test_pred4 <- test_preds[[61]]
test_pred4 <- c(rep(NA, n_timesteps + 60), test_pred4, rep(NA, nrow(vic_elec_test) - 60 - n_timesteps - n_forecast))

test_pred5 <- test_preds[[81]]
test_pred5 <- c(rep(NA, n_timesteps + 80), test_pred5, rep(NA, nrow(vic_elec_test) - 80 - n_timesteps - n_forecast))


preds_ts <- vic_elec_test %>% 
  add_column(
    ex_1 = test_pred1 * train_sd_demand + train_mean_demand,
    ex_2 = test_pred2 * train_sd_demand + train_mean_demand,
    ex_3 = test_pred3 * train_sd_demand + train_mean_demand,
    ex_4 = test_pred4 * train_sd_demand + train_mean_demand,
    ex_5 = test_pred5 * train_sd_demand + train_mean_demand) %>%
  pivot_longer(-Date) %>%
  update_tsibble(key = name)


preds_ts %>%
  autoplot() +
  scale_color_hue(h = c(80, 300), l = 70) +
  theme_minimal()



