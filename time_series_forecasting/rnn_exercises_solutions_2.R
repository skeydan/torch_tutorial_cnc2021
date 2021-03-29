# 2
# In the following subset of nycflights13::weather, use pressure to forecast wind speed
# over the next 24 hours, choosing a suitable length for the input sequence.
# Use the first half of 2013 for training, and the second for validation.
# To display sample predictions, use a small subset of the validation data.

# Note: The dataset contains missing values, which the model won't know how to handle. 
# Think of a suitable way to replace them.

library(torch)
library(tidyverse)
library(tsibble)
library(tsibbledata)
library(lubridate)
library(fable)

# Zoom in on year 2013, and a subset of the variables.
weather <- nycflights13::weather %>% 
  filter(origin == "JFK", year == 2013) %>%
  select(time_hour, temp, humid, pressure, wind_speed)

# For plotting, just use a single month.
weather_tsbl <- weather %>% 
  filter(month(time_hour) == 2) %>%
  as_tsibble(index = time_hour)

weather_tsbl <- weather_tsbl %>%
  mutate(temp = scale(temp), humid = scale(humid), pressure = scale(pressure), wind_speed = scale(wind_speed)) %>%
  pivot_longer(-time_hour, names_to = "variable") %>%
  update_tsibble(key = variable)

weather_tsbl %>% 
  autoplot() +
  scale_color_hue(h = c(80, 300), l = 70) +
  theme_minimal()

# Solution ---------------------------------------------------------------

weather_train <- weather %>% 
  filter(month(time_hour) %in% 1:6) %>%
  select(pressure, wind_speed) %>%
  as.matrix()

weather_valid <-  weather %>% 
  filter(month(time_hour) %in% 7:12) %>%
  select(pressure, wind_speed) %>%
  as.matrix()

weather_test <- weather %>% 
  filter(month(time_hour) == 7, day(time_hour) < 15) %>%
  select(pressure, wind_speed) %>%
  as.matrix()

train_mean_pressure <- mean(na.omit(weather_train[ , 1]))
train_sd_pressure <- sd(na.omit(weather_train[ , 1]))

train_mean_wind_speed <- mean(na.omit(weather_train[ , 2]))
train_sd_wind_speed <- sd(na.omit(weather_train[ , 2]))

weather_train[, 1] <- replace_na(weather_train[, 1], train_mean_pressure)
weather_train[, 2] <- replace_na(weather_train[, 2], train_mean_wind_speed)

weather_valid[, 1] <- replace_na(weather_valid[, 1], train_mean_pressure)
weather_valid[, 2] <- replace_na(weather_valid[, 2], train_mean_wind_speed)

weather_test[, 1] <- replace_na(weather_test[, 1], train_mean_pressure)
weather_test[, 2] <- replace_na(weather_test[, 2], train_mean_wind_speed)

# Create torch dataset ----------------------------------------------------

weather_dataset <- dataset(
  name = "weather_dataset",
  
  initialize = function(x, y, n_timesteps, n_forecast) {
    
    self$n_timesteps <- n_timesteps
    self$n_forecast <- n_forecast
    self$x <- torch_tensor((x - train_mean_pressure) / train_sd_pressure)
    self$y <- torch_tensor((y - train_mean_wind_speed) / train_sd_wind_speed)
    
  },
  
  .getitem = function(i) {
    
    start <- i
    end <- start + self$n_timesteps - 1
    pred_length <- self$n_forecast
    
    list(
      x = self$x[start:end]$unsqueeze(2),
      y = self$y[(end + 1):(end + pred_length)]
    )
    
  },
  
  .length = function() {
    length(self$x) - self$n_timesteps - self$n_forecast + 1
  }
)

n_timesteps <- 24 * 3
n_forecast <- 24

train_ds <- weather_dataset(weather_train[ , 1], weather_train[ , 2], n_timesteps, n_forecast)
length(train_ds)
train_ds[1]

valid_ds <- weather_dataset(weather_valid[ ,1], weather_valid[ ,2], n_timesteps, n_forecast)
length(valid_ds)

test_ds <- weather_dataset(weather_test[ ,1], weather_test[ ,2], n_timesteps, n_forecast)
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

net <- model(input_size = 1, hidden_size = 64, linear_size = 128,
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

weather_test <- weather %>% 
  filter(month(time_hour) == 7, day(time_hour) < 15) %>%
  select(time_hour, pressure, wind_speed)

test_pred1 <- test_preds[[1]]
test_pred1 <- c(rep(NA, n_timesteps), test_pred1, rep(NA, nrow(weather_test) - n_timesteps - n_forecast))

test_pred2 <- test_preds[[71]]
test_pred2 <- c(rep(NA, n_timesteps + 70), test_pred2, rep(NA, nrow(weather_test) - 70 - n_timesteps - n_forecast))

test_pred3 <- test_preds[[141]]
test_pred3 <- c(rep(NA, n_timesteps + 140), test_pred3, rep(NA, nrow(weather_test) - 140 - n_timesteps - n_forecast))

test_pred4 <- test_preds[[201]]
test_pred4 <- c(rep(NA, n_timesteps + 200), test_pred4, rep(NA, nrow(weather_test) - 200 - n_timesteps - n_forecast))


preds_ts <- weather_test %>% 
  select(-pressure) %>%
  add_column(
    ex_1 = test_pred1 * train_sd_wind_speed + train_mean_wind_speed,
    ex_2 = test_pred2 * train_sd_wind_speed + train_mean_wind_speed,
    ex_3 = test_pred3 * train_sd_wind_speed + train_mean_wind_speed,
    ex_4 = test_pred4 * train_sd_wind_speed + train_mean_wind_speed) %>%
  pivot_longer(-time_hour) %>%
  as_tsibble(key = name)


preds_ts %>%
  autoplot() +
  scale_color_hue(h = c(80, 300), l = 70) +
  theme_minimal()

