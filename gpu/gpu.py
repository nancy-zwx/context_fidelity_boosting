import torch

torch.cuda.empty_cache()

while True:
    x = torch.randn(200, 200).cuda()
    y = x * x
    del x, y
