library(torch)

##
a <- torch_tensor(matrix(1:4, ncol = 2, byrow = TRUE))$to(dtype = torch_float())
b <- a$mul(2)
c <- b$sum()

c$backward() #! element 0 of tensors does not require grad and does not have a grad_fn

##
a <- torch_tensor(matrix(1:4, ncol = 2, byrow = TRUE), dtype = torch_float(), requires_grad = TRUE)
b <- a$mul(2)
c <- b$sum()

c$backward()

c$grad_fn
b$grad_fn

a$grad

##
a <- torch_tensor(matrix(1:4, ncol = 2, byrow = TRUE), dtype = torch_float(), requires_grad = TRUE)
b <- a$mul(2)
b$retain_grad()
c <- b$sum()

c$backward()

b$grad # how does c change as b changes?
a$grad # how does b change as a changes?

# say we want to update a (for example, in optimization)
# we need to exempt that operation from "recording"
a$sub_(0.1 * a$grad) #! a leaf Variable that requires grad is being used in an in-place operation.
                         
with_no_grad( {
  a$sub_(0.1 * a$grad)
})

a
