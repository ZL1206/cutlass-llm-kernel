import torch
import wgmma_rs_ops

torch.manual_seed(42)

m = 256
n = 192
k = 256

dtype = torch.float16
device = "cuda"

input = torch.randn(m, k, dtype=dtype, device=device)
weight = torch.randn(n, k, dtype=dtype, device=device)
out = torch.randn(m, n, dtype=dtype, device=device)

wgmma_rs_ops.wgmma_matmul(input, weight, out)

torch.cuda.synchronize()

ref_out = torch.matmul(input, weight.t())

print("ref_out:")
print("m = 0:")
for n in range(24):
    for r in range(2):
        for e in range(2):
            row = r * 8
            col = n * 8 + e
            print(ref_out[row, col])
print("m = 1:")
offset = 128
for n in range(24):
    for r in range(2):
        for e in range(2):
            row = r * 8 + offset
            col = n * 8 + e
            print(ref_out[row, col])