import torch as th

def normalize(x : th.Tensor, eps : float = 0.001) -> th.Tensor:
    return x/ (x.norm(dim=-1, keepdim=True)+eps)

def dot(a, b, keepdim=True):
    return th.sum(a*b, dim=-1, keepdim=keepdim)
