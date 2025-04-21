import torch
import int4_qkv_matmul_ops
from rearrange_kv import rerange_k, rerange_v, pack_uint4_to_uint8

torch.manual_seed(42)

dtype = torch.float16
seqlen_q = 16
seqlen_k = 64
head_size = 128
query = torch.randn((seqlen_q, head_size), dtype=dtype, device="cuda")

key = torch.randint(0, 16, (seqlen_k, head_size), dtype=torch.uint8, device="cuda")
value = torch.randint(0, 16, (seqlen_k, head_size), dtype=torch.uint8, device="cuda")
key_scale = torch.randn((2, 64), dtype=dtype, device="cuda")
value_scale = torch.randn((2, 64), dtype=dtype, device="cuda")
ref_key = key.to(dtype)
ref_value = value.to(dtype)

torch.set_printoptions(threshold=64*128, linewidth=1000)

print("key:\n")
print(key)
print("------------------------")
print("key_scale:\n")
print(key_scale)
print("------------------------")
# rearrange
key = rerange_k(key)
value = rerange_v(value)

#pack
key = pack_uint4_to_uint8(key)
value = pack_uint4_to_uint8(value)

torch.set_printoptions(profile="default")

out = torch.empty((seqlen_q, head_size), dtype=dtype, device="cuda")

torch.cuda.synchronize()

softmax_scale = head_size ** (-0.5) 
out, fa_lse = int4_qkv_matmul_ops.int4_qkv_matmul(query, key, value, key_scale, value_scale, out, softmax_scale, False)

# torch
q, k, v = query.float(), ref_key.float(), ref_value.float()
k = k * key_scale[0, :].unsqueeze(1) + key_scale[1, :].unsqueeze(1)
v = v * value_scale[0, :].unsqueeze(1) + value_scale[1, :].unsqueeze(1)
scores = torch.matmul(q, k.t())
scores = scores * softmax_scale
torch_lse = torch.logsumexp(scores, dim=1)
attention = torch.softmax(scores, dim=-1)
out_ref = torch.matmul(attention, v)

torch.cuda.synchronize()

print("out: ")
print(out)
print(out.size())
print("out_ref: ")
print(out_ref)
print(out_ref.size())

print("mean diff: ", (out_ref - out).abs().mean())
print("max diff ", (out_ref - out).abs().max())


print("lse: ")
print(fa_lse)
print(torch_lse)