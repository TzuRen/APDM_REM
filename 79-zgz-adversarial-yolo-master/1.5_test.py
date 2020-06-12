import torch

def L2_dist(x, y):
    return reduce_sum((x - y) ** 2)

def reduce_sum(x, keepdim=True):
    for a in reversed(range(1, x.dim())):
        x = x.sum(a, keepdim=keepdim)
    return x

a=torch.rand(2,3,2,3)

b=torch.rand(2,3,2,3)

c=L2_dist(a,b)

print('a,b,c',a,b,c)
