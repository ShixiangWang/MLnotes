# source: https://pytorch.apachecn.org/docs/1.4/blitz/tensor_tutorial.html

# Run in R start
reticulate::use_python("/Users/wsx/Library/r-miniconda/bin/python", required = T)
# reticulate::repl_python()
# Run in R end

from __future__ import print_function
import torch

# TENSOR ----------------------

# Empty tensor
x = torch.empty(5, 3)
print(x)

# Random tensor
x = torch.rand(5, 3)
print(x)


# Zero matrix with long data type
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# Construct tensor from data
x = torch.tensor([5, 3])
print(x)

# Construct tensor based on tensor
x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # 重载 dtype!
print(x)                                      # 结果size一致

# Shape of tensor
print(x.size())

# OPERATIONS ---------------

# +
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)  # The output tensor is given
print(result)

y.add_(x) # Add in place
print(y)
# More in-place modification functions are end with '_'

# Index
print(x[:, 1])

# Change shape
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# Transform one-element tensor to scalar
x = torch.randn(1)
print(x)
print(x.item())

# More available at <https://pytorch.org/docs/stable/torch.html>

# CONNECT to NumPy -----------------
a = torch.ones(5)
print(a)
a.numpy()

# b is a reference
b = a.numpy()
print(b)
a.add_(1)
print(a)
print(b)

# ndarray to tensor
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# CPU上的所有张量(CharTensor除外)都支持与Numpy的相互转换。

# TENSOR in CUDA -----------------
# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    x = x.to(device)                       # 或者使用`.to("cuda")`方法
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype
    
# PyTorch中，所有神经网络的核心是 autograd 包。
# 先简单介绍一下这个包，然后训练我们的第一个的神经网络。
# 
# autograd 包为张量上的所有操作提供了自动求导机制。
# 它是一个在运行时定义(define-by-run）的框架，
# 这意味着反向传播是根据代码如何运行来决定的，并且每次迭代可以是不同的。

