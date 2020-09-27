# reticulate::use_python("/Users/wsx/Library/r-miniconda/bin/python", required = T)
# reticulate::repl_python()
from __future__ import print_function
import torch

x = torch.empty(5, 3)
print(x)
