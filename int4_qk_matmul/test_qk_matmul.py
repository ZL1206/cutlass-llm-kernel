import torch
import int4_qk_matmul_ops

torch.manual_seed(42)

seqlen_q = 16
seqlen_k = 64
head_size = 128
query = torch.randn((seqlen_q, head_size), dtype=torch.float16, device="cuda")
key = torch.randn((seqlen_k, head_size), dtype=torch.float16, device="cuda")
key = torch.randint(0, 256, (seqlen_k, head_size // 2), dtype=torch.uint8, device="cuda")

high_bits = (key.clone() >> 4) & 0x0F  # 取高 4 位
low_bits = key.clone() & 0x0F
uint4_tensor = torch.cat([high_bits.unsqueeze(-1), low_bits.unsqueeze(-1)], dim=-1).view(seqlen_k, head_size)
torch.set_printoptions(profile="full")
print(key)
print(uint4_tensor)
torch.set_printoptions(profile="default")
out = torch.empty((seqlen_q, seqlen_k), dtype=torch.float16, device="cuda")


softmax_scale = head_size ** (-0.5) 
int4_qk_matmul_ops.small_qk_matmul(query, key, out, softmax_scale, False)

"""
out_ref = torch.matmul(query, key.t())
#out_ref = out_ref * softmax_scale
#out_ref = torch.softmax(out_ref, dim=-1)

torch.cuda.synchronize()

print("out: ")
print(out)
print(out.size())
print("out_ref: ")
print(out_ref)
print(out_ref.size())

print("mean diff: ", (out_ref - out).abs().mean())
print("max diff ", (out_ref - out).abs().max())"
"""