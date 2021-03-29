# 1
# Try modifying the multi-step network to use temperature as an additional predictor (not as an outcome).
# Does this help with learning?



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
