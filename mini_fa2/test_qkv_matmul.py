import torch
import argparse
import mini_flash_attn_ops
from flash_attn import (
    flash_attn_func,
    flash_attn_varlen_func,
)
from einops import rearrange, repeat

torch.manual_seed(42)

def test(causal, varlen):
    print(f"causal: {causal}, varlen: {varlen}")
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
    seqlen = [(256, 1024)]
    for i in range(len(seqlen)):
        seqlen_q = seqlen[i][0]
        seqlen_k = seqlen[i][1]
        head_size = 128
        device = "cuda"
        query = torch.randn((seqlen_q, head_size), dtype=torch.float16, device="cuda")
        key = torch.randn((seqlen_k, head_size), dtype=torch.float16, device="cuda")
        value = torch.randn((seqlen_k, head_size), dtype=torch.float16, device="cuda")
        out = torch.empty((seqlen_q, head_size), dtype=torch.float16, device="cuda")


        softmax_scale = head_size ** (-0.5) 
        mini_flash_attn_ops.mini_fwd(query, key, value, out, softmax_scale, causal)

        out_ref = torch.matmul(query, key.t())
        scores = out_ref * softmax_scale
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
        out_ref = torch.matmul(attention, value)

        query = query.unsqueeze(0).unsqueeze(2)
        key = key.unsqueeze(0).unsqueeze(2)
        value = value.unsqueeze(0).unsqueeze(2)
        out_fa = flash_attn_func(query, key, value, causal=causal)

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
        diff = (out_ref - out).abs()
        max_value, max_index = diff.max(), diff.argmax()
        max_index_2d = (max_index // diff.size(1), max_index % diff.size(1))
        print("max diff ", max_value, max_index, max_index_2d, out_ref[max_index_2d[0], max_index_2d[1]], out[max_index_2d[0], max_index_2d[1]])
        print("mean diff: ", (out - out_fa[0,:,0,:]).abs().mean())
        print("max diff ", (out - out_fa[0,:,0,:]).abs().max())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--causal", action="store_true", default=False,
                    help="if causal")
    parser.add_argument("--varlen", action="store_true", default=False,
                    help="if varlen")
    args = parser.parse_args()
    test(args.causal, args.varlen)