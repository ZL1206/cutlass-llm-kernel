import torch
import qk_matmul_ops

torch.manual_seed(42)

seqlen_q = 111
seqlen_k = 56
head_size = 128
query = torch.randn((seqlen_q, head_size), dtype=torch.float16, device="cuda")
key = torch.randn((seqlen_k, head_size), dtype=torch.float16, device="cuda")
out = torch.empty((seqlen_q, seqlen_k), dtype=torch.float16, device="cuda")


softmax_scale = head_size ** (-0.5) 
qk_matmul_ops.qk_matmul(query, key, out, softmax_scale, False)

out_ref = torch.matmul(query, key.t())
out_ref = out_ref * softmax_scale
out_ref = torch.softmax(out_ref, dim=-1)

torch.cuda.synchronize()

print("out: ")
print(out)
print(out.size())
print("out_ref: ")
print(out_ref)
print(out_ref.size())

print("mean diff: ", (out_ref - out).abs().mean())
print("max diff ", (out_ref - out).abs().max())