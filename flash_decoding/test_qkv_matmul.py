import torch
import math
import argparse
import int4_attention_ops
from einops import rearrange, repeat
from rearrange_kv import rerange_k, rerange_v, pack_uint4_to_uint8

torch.manual_seed(42)

def test(causal, varlen):

    print(f"causal: {causal}, varlen: {varlen}")

    dtype = torch.bfloat16
    page_size = 64
    max_seqlen_k = 128 * 1024
    num_blocks = 10
    
    num_heads = 16
    num_heads_k = 1
    head_size = 128
    softmax_scale = head_size ** (-0.5) 

    k_cache = torch.randint(0, 16, (num_blocks, num_heads_k, page_size, 128), dtype=torch.uint8, device="cuda")
    k_cache_pack = torch.randint(0, 16, (num_blocks, num_heads_k, page_size, 64), dtype=torch.uint8, device="cuda")
    k_params = torch.randn((num_blocks, num_heads_k, 2, 64), dtype=dtype, device="cuda")

    v_cache = torch.randint(0, 16, (num_blocks, num_heads_k, page_size, 128), dtype=torch.uint8, device="cuda")
    v_cache_pack = torch.randint(0, 16, (num_blocks, num_heads_k, page_size, 64), dtype=torch.uint8, device="cuda")
    v_params = torch.randn((num_blocks, num_heads_k, 2, 64), dtype=dtype, device="cuda")

    ref_k_cache = k_cache.to(dtype)
    ref_v_cache = v_cache.to(dtype)

    print("k_cache: ")
    print(k_cache)
    print("ref_k_cache: ")
    print(ref_k_cache)

    # rearrange and pack
    for i in range(num_blocks):
        for j in range(num_heads_k):
            k = k_cache[i, j, :]
            k = rerange_k(k)
            k = pack_uint4_to_uint8(k)
            k_cache_pack[i, j, :] = k

            v = v_cache[i, j, :]
            v = rerange_v(v)
            v = pack_uint4_to_uint8(v)
            v_cache_pack[i, j, :] = v

    batch_size = 1
    seqlen_q = 1
    seqlen_k = 64
    max_seqlen_q = seqlen_q
    max_seqlen_k = seqlen_k
    cu_seqlen_q = [0, 1]
    seqlens_k = [64]

    q = torch.randn(batch_size, seqlen_q, num_heads, 128, device="cuda", dtype=dtype)
    q_unpad = rearrange(q, "b s h d -> (b s) h d")

    
    cu_seqlen_q = torch.tensor(cu_seqlen_q, device="cuda", dtype=torch.int32)
    print("cu_seqlen_q")
    print(cu_seqlen_q)
    seqlens_k = torch.tensor(seqlens_k, device="cuda", dtype=torch.int32)
    
    block_table = rearrange(
        torch.randperm(num_blocks, dtype=torch.int32, device="cuda"),
        "(b nblocks) -> b nblocks",
        b=batch_size,
    )
    
    

    print("block_table:")
    print(block_table)

    print("k_cache_pack: ")
    print(k_cache_pack[6])
    print("k_cache_pack stride 0 is ", k_cache_pack.stride(0))
    out = torch.zeros(q_unpad.shape[0], num_heads, 128, dtype=dtype, device="cuda")
    
    torch.cuda.synchronize()
    
    out, fa_lse = int4_attention_ops.varlen_fwd(q_unpad, k_cache_pack, v_cache_pack, k_params, v_params, cu_seqlen_q, seqlens_k, block_table, out, max_seqlen_q, max_seqlen_k, softmax_scale, False)
    torch.cuda.synchronize()
    
    # pytorch

    for i in range(num_blocks):
        for j in range(num_heads_k):
            k = ref_k_cache[i, j, :]
            k_param = k_params[i, j, :]
            k = k * k_param[0, :].unsqueeze(1) + k_param[1, :].unsqueeze(1)
            ref_k_cache[i, j, :] = k

            v = ref_v_cache[i, j, :]
            v_param = v_params[i, j, :]
            v = v * v_param[0, :].unsqueeze(1) + v_param[1, :].unsqueeze(1) 
            ref_v_cache[i, j, :] = v



    k = rearrange(
        # pytorch 1.12 doesn't have indexing with int32
        ref_k_cache[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) h p d -> b (nblocks p) h d",
        #"(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    
    v = rearrange(
        # pytorch 1.12 doesn't have indexing with int32
        ref_v_cache[block_table.to(dtype=torch.long).flatten()],
        "(b nblocks) h p d -> b (nblocks p) h d",
        #"(b nblocks) block_size ... -> b (nblocks block_size) ...",
        b=batch_size,
    )[:, :seqlen_k]
    

    q, k, v = q.float(), k.float(), v.float()

    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    if causal:
        row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
        col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
        sk = (seqlen_k)
        sq = (seqlen_q)
        mask = col_idx > row_idx + sk - sq
        print("mask: ", mask)
        scores.masked_fill_(mask, float("-inf"))
    lse_ref = torch.logsumexp(scores, dim=-1)
    attention = torch.softmax(scores, dim=-1)
    out_ref = torch.einsum("bhts,bshd->bthd", attention, v)
    
    print(out_ref)
    print(out)
    print("mean diff: ", (out_ref - out).abs().mean())
    print("max diff ", (out_ref - out).abs().max())
    print(lse_ref)
    print(fa_lse)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal", action="store_true", default=False,
                    help="if causal")
    parser.add_argument("--varlen", action="store_true", default=False,
                    help="if varlen")
    args = parser.parse_args()
    test(args.causal, args.varlen)