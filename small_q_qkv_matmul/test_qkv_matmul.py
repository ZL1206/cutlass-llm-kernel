import torch
import small_q_qkv_matmul_ops

torch.manual_seed(42)

seqlen_q = 32
seqlen_k = 64
head_size = 128
query = torch.randn((seqlen_q, head_size), dtype=torch.float16, device="cuda")
key = torch.randn((seqlen_k, head_size), dtype=torch.float16, device="cuda")
value = torch.randn((seqlen_k, head_size), dtype=torch.float16, device="cuda")
out = torch.empty((seqlen_q, head_size), dtype=torch.float16, device="cuda")

softmax_scale = head_size ** (-0.5) 
small_q_qkv_matmul_ops.small_q_qkv_matmul(query, key, value, out, softmax_scale, False)
"""
groups = out.chunk(4, dim=0)
result = torch.zeros_like(groups[0])  # 初始化 [16, 128] 的全零张量
for group in groups:
    result += group
out = result
"""

q, k, v = query.float(), key.float(), value.float()
out_ref = torch.matmul(q, k.t())
out_ref = out_ref * softmax_scale
lse = torch.logsumexp(out_ref, dim=1)
print("lse: ")
print(lse)
out_ref = torch.softmax(out_ref, dim=-1)
out_ref = torch.matmul(out_ref, v)


torch.cuda.synchronize()

print("out: ")
print(out)
print(out.size())
print("out_ref: ")
print(out_ref)
print(out_ref.size())

print("mean diff: ", (out_ref - out).abs().mean())
print("max diff ", (out_ref - out).abs().max())