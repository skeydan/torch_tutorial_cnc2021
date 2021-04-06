library(torch)

# More details here:
# https://torch.mlverse.org/technical/tensors/

# Creating tensors --------------------------------------------------------

torch_tensor(1)
torch_tensor(1, dtype = torch_int())
torch_tensor(1, device = "cuda")

torch_tensor(c(1, 2, 3)) # float tensor

torch_tensor(matrix(1:9, ncol = 3)) # integer tensor
torch_tensor(matrix(1:9, ncol = 3))$to(dtype = torch_float()) # cast to float
torch_tensor(matrix(1:9, ncol = 3, byrow = TRUE))

torch_zeros(c(3, 3))
torch_rand(c(3, 3))
torch_arange(1, 9)
torch_logspace(start = 0.1, end = 1.0, steps = 5)

# Tensor operations -------------------------------------------------------

t1 <- torch_tensor(c(1, 2, 3))
t2 <- torch_tensor(c(1, 2, 3))

t1$sub(t2)
t1
t1$sub_(t2)
t1

t1 <- t2$add(1)
t1

t1$mul(t2)

t1$matmul(t2)
t1$t()$matmul(t2)
t1$dot(t2)



# Converting back to R ----------------------------------------------------

as.numeric(t1)

torch_ones(c(2, 2)) %>% as.matrix() 

torch_ones(c(2, 2, 2)) %>% as.array() 



# Reshaping tensors -------------------------------------------------------

t1 <- torch_randn(c(1, 2, 3, 4))
t1

t1$unsqueeze(4)

t1$squeeze(1)

t1$view(c(6, 4))
t1$view(24)



# Indexing and slicing ----------------------------------------------------

t1

t1[ , 1, , ]
t1[ , 1, , , drop = FALSE]

t1[1, 1, 1:2, ]
t1[1, 1, 1:2, , drop = FALSE]

t2 <- torch_tensor(1:17)
t2[-1] 
t2[2:10:2] 


# Broadcasting ------------------------------------------------------------

t1 <- torch_randn(c(3,5))
t2 <- torch_randn(c(1,5))
t1
t2

# 1 
t1 + 1

# 2
t1$add(t2)

# 3
t2 <- torch_randn(c(5))
t2
t1$add(t2)

# 2 + 3
t1 <- torch_randn(c(5))
t2 <- torch_randn(c(3,1))
t1
t2
t1$add(t2)


# Broadcasting rules in a nutshell ----------------------------------------

# 1 We align array shapes, starting from the right.
  
  # Example

  # t1, shape:     8  1  6  1
  # t2, shape:        7  1  5
  

# 2 Starting to look from the right, the sizes along aligned axes either have to match exactly,
#   or one of them has to be equal to 1.
#   In the latter case, the 1-dimensional tensor is broadcast to the larger one.

  # Example: this happens in the last (for t1) as well as the second-from-last dimension (for t2)

  # t1, shape:     8  1  6  5
  # t2, shape:        7  6  5


# 3 If on the left, one of the arrays has an additional axis (or more than one),
#   the other is virtually expanded to have a size of 1 in that place.
#   Then, broadcasting will happen as stated in (2).

  # Example: this happens in t1’s leftmost dimension. First, there is a virtual expansion

  # t1, shape:     8  1  6  1
  # t2, shape:     1  7  1  5

  # and then, broadcasting happens:
  
  # t1, shape:     8  1  6  1
  # t2, shape:     8  7  1  5

