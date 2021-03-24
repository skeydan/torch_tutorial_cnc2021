library(dplyr)
library(torch)

# 1
# "Translate" the following R code to torch:

m1 <- matrix(1:32, ncol = 8, byrow = TRUE)
m2 <- matrix(1:8, ncol = 1)

(m1 %*% m2)^2 %>% sum() %>% sqrt()

m1 * rbind(t(m2), t(m2), t(m2), t(m2))

t(m1) %>% apply(2, sum)

(m1 - mean(m1)) / sd(m1)

# 2 
# Can you devise a clever way to compute an outer product between these two tensors?
t1 <- torch_tensor(c(0, 10, 20, 30))
t2 <- torch_tensor(c(1, 2, 3))

