import torch
import qk_matmul_ops

torch.manual_seed(42)

seqlen_q = 128
seqlen_k = 64
head_size = 128
query = torch.randn((seqlen_q, head_size), dtype=torch.float16, device="cuda")
key = torch.randn((seqlen_k, head_size), dtype=torch.float16, device="cuda")
out = torch.empty((seqlen_q, seqlen_k), dtype=torch.float16, device="cuda")
out_ref = out.clone()

qk_matmul_ops.qk_matmul(query, key, out)

out_ref = torch.matmul(query, key.t())


torch.cuda.synchronize()

print("out: ")
print(out)
print("out_ref: ")
print(out_ref)

print("mean diff: ", (out_ref - out).abs().mean())
print("max diff ", (out_ref - out).abs().max())