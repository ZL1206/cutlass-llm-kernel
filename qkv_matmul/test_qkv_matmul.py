import torch
import qkv_matmul_ops

torch.manual_seed(42)

seqlen_q = 110
seqlen_k = 64
head_size = 128
query = torch.randn((seqlen_q, head_size), dtype=torch.float16, device="cuda")
key = torch.randn((seqlen_k, head_size), dtype=torch.float16, device="cuda")
value = torch.randn((seqlen_k, head_size), dtype=torch.float16, device="cuda")
out = torch.empty((seqlen_q, head_size), dtype=torch.float16, device="cuda")


softmax_scale = head_size ** (-0.5) 
qkv_matmul_ops.qkv_matmul(query, key, value, out, softmax_scale, False)

out_ref = torch.matmul(query, key.t())
out_ref = out_ref * softmax_scale
out_ref = torch.softmax(out_ref, dim=-1)
out_ref = torch.matmul(out_ref, value)

torch.cuda.synchronize()

print("out: ")
print(out)
print(out.size())
print("out_ref: ")
print(out_ref)
print(out_ref.size())


print("mean diff: ", (out_ref - out).abs().mean())
diff = (out_ref - out).abs()
max_value, max_index = diff.max(), diff.argmax()
max_index_2d = (max_index // diff.size(1), max_index % diff.size(1))
print("max diff ", max_value, max_index, max_index_2d, out_ref[max_index_2d[0], max_index_2d[1]], out[max_index_2d[0], max_index_2d[1]])