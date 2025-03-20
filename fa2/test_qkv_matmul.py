import torch
import math
import argparse
import flash_attn_ops
from flash_attn import (
    flash_attn_func,
    flash_attn_varlen_func,
)
from einops import rearrange, repeat

torch.manual_seed(42)

def test(causal, varlen):
    print(f"causal: {causal}, varlen: {varlen}")
    batch = 1
    num_heads = 8
    num_heads_k = 2
    seqlen = [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (128, 256),
        (113, 211),
        (108, 256),
        (256, 512),
        (1023, 1024),
        (256, 129),
    ]
    seqlen = [(1023, 1024)]

    for i in range(len(seqlen)):
        seqlen_q = seqlen[i][0]
        seqlen_k = seqlen[i][1]
        head_size = 128
        device = "cuda"
        query = torch.randn((batch, seqlen_q, num_heads, head_size), dtype=torch.float16, device="cuda")
        key = torch.randn((batch, seqlen_k, num_heads_k, head_size), dtype=torch.float16, device="cuda")
        value = torch.randn((batch, seqlen_k, num_heads_k, head_size), dtype=torch.float16, device="cuda")
        out = torch.empty((batch, seqlen_q, num_heads, head_size), dtype=torch.float16, device="cuda")
        q_fa = query.clone()
        k_fa = key.clone()
        v_fa = value.clone()

        softmax_scale = head_size ** (-0.5) 
        flash_attn_ops.fwd(query, key, value, out, softmax_scale, causal)
    
        torch.cuda.synchronize()
        # pytorch
        q, k, v = query.float(), key.float(), value.float()
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
            attention = torch.softmax(scores, dim=-1)
        else:
            attention = torch.softmax(scores, dim=-1)
        out_ref = torch.einsum("bhts,bshd->bthd", attention, v)

        out_fa = flash_attn_func(q_fa, k_fa, v_fa, causal=causal)

        torch.cuda.synchronize()

        print("out: ")
        print(out)
        print(out.size())
        print("out_ref: ")
        print(out_ref)
        print(out_ref.size())
        print(out_fa)
        print(out_fa.size())


        print("mean diff: ", (out_ref - out).abs().mean())
        print("max diff ",  (out_ref - out).abs().max())
        print("mean diff: ", (out - out_fa).abs().mean())
        print("max diff ", (out - out_fa).abs().max())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal", action="store_true", default=False,
                    help="if causal")
    parser.add_argument("--varlen", action="store_true", default=False,
                    help="if varlen")
    args = parser.parse_args()
    test(args.causal, args.varlen)