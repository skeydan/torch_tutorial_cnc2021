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



# Solutions ---------------------------------------------------------------

# 1
t1 <- matrix(1:32, ncol = 8, byrow = TRUE) %>% torch_tensor()
t2 <- 1:8 %>% torch_tensor()

# need to cast to float in order to be able to call torch_sqrt()
t1$matmul(t2)$square()$sum()$to(dtype = torch_float())$sqrt()

# broadcasting takes care of duplication for us 
# also, no transposition needed
t1 * t2

# dimension 1 (not 2) collapses the rows, giving us one value per "feature"
# given an index, in R, think "group by", in torch, think "collapse"
t1$t()$sum(dim = 1)

# torch_mean() and torch_std() need float
t1 <- t1$to(dtype = torch_float())
(t1 - t1$mean()) / t1$std()


# 2
# given
t1 <- torch_tensor(c(0, 10, 20, 30))
t2 <- torch_tensor(c(1, 2, 3))

# way 1 
# unsqueeze to have a column vector and a row vector, and matrix multiply
t1 <- t1$unsqueeze(2)
t1

t2 <- t2$unsqueeze(1)
t2

# (4,1) * (1,3)
t1$matmul(t2)


# way 2
# make use of broadcasting

t1 <- t1$unsqueeze(2)
t1

# (4,1) * (3) ---> (4,1) * (1,3) ---> (4,3) * (4,3)
t1 * t2
