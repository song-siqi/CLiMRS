import torch
a=torch.rand(2)
a=a.to("cuda:0")
# b=a.to("cuda:1")
# print(b)
# tensor([0.0, 0.0], device='cuda:1')
print(a)
# tensor([0.9285, 0.3294], device='cuda:0')
