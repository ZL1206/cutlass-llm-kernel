import torch


def mask_step(m_block, seqlen_q, seqlen_k, causal, even_mn, kblockm, kblockn) :

    n_block_max = (seqlen_k + kblockn - 1) // kblockn
    if causal:
        n_block_max = min(n_block_max, ((m_block + 1) * kblockm + seqlen_k - seqlen_q + kblockn - 1) // kblockn)
    print(f"n_block_max: {n_block_max}")
    n_masking_steps = 0
    if not causal and not even_mn:
        n_masking_steps = 1
    elif causal:
        no_masking_steps = max((m_block * kblockm + seqlen_k - seqlen_q) // kblockn, 0)
        n_masking_steps = n_block_max - no_masking_steps
    return n_masking_steps


seqlen = [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (1023, 1024),
        (256, 129),
    ]

kblockm = 128
kblockn = 64
for i in range(len(seqlen)):
    seqlen_q = seqlen[i][0]
    seqlen_k = seqlen[i][1]
    print("seqlen_q, seqlen_k: ", seqlen_q, seqlen_k)
    even_mn = seqlen_q % kblockm == 0 and seqlen_k % kblockn == 0
    for causal in [False, True]:
        num_m_block = (seqlen_q + kblockm - 1) // kblockm
        m_blocks = [i for i in range(num_m_block)]
        for m_blocks in m_blocks:
            n_masking_steps = mask_step(m_blocks, seqlen_q, seqlen_k, causal, even_mn, kblockm, kblockn)
            print(f"m_blocks: {m_blocks}, seqlen_q: {seqlen_q}, seqlen_k: {seqlen_k}, causal: {causal}, even_mn: {even_mn}, kblockm: {kblockm}, kblockn: {kblockn}, n_masking_steps: {n_masking_steps}")


