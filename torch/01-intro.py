# source: https://pytorch.apachecn.org/docs/1.4/blitz/tensor_tutorial.html

# Run in R start
reticulate::use_python("/Users/wsx/Library/r-miniconda/bin/python", required = T)
reticulate::repl_python()
# Run in R end

from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)
