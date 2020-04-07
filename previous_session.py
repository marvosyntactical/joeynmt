# coding: utf-8
get_ipython().run_line_magic('clear', '')
import numpy as np
import torch
output = np.random.rand(3,12,40000)
idxr = np.random.randin(0,40000,(3,12,52))
idxr = np.random.randint(0,40000,(3,12,52))
vals = np.random.rand(3,12,52)
b,s,k = vals.shape
get_ipython().run_line_magic('pinfo', 'np.tile')
output
x
b
x = np.arange(b)
y = np.arange(s)
z = np.arange(k)
x
x.tile((2,2))
np.tile(x,(2,2))
x.repeat(2,2)
x.repeat(1,1)
x_ = torch.from_numpy(x)
x_
x_.repeat(2,2)
torch.tile
i1
int(kb/batch)+1
int(k/b)+1
k
b
17.3333333*3
17.3333333333333333333333333333333*3
clera
clera
cle
get_ipython().run_line_magic('clear', '')
int(k/s)
int(k/s)+1
s
x
np.tile(x,3)
np.tile(x,(1,3))
np.tile(x,(3,1))
idx1 = np.tile(x,int(k/b)+1)
idx1
idx1 = np.tile(x,int(k/b)+1)[0:k]
idx1
get_ipython().run_line_magic('pinfo', 'torch.roll')
get_ipython().run_line_magic('pinfo', 'torch.repeat')
get_ipython().run_line_magic('pinfo', 'torch.tensor.repeat')
get_ipython().run_line_magic('pinfo', 'torch.Tensor.repeat')
get_ipython().run_line_magic('pinfo', 'torch.repeat_interleave')
x
x_
i1 = torch.repeat(x_,int(k/b)+1)[0:k]
i1 = x_.repeat(int(k/b)+1)[0:k]
i1
idx1
len(i1)
len(idx1)
y_ = torch.from_numpy(y)
i2 = y_.repeat(int(k/s)+1)[0:k]
outputs
v
V
dir()
