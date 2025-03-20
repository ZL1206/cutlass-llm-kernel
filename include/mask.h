#pragma once
#include <cute/tensor.hpp>

namespace flash {

using namespace cute;



struct Mask {

    const int max_seqlen_k, max_seqlen_q;

    __forceinline__ __device__ Mask(const int max_seqlen_k, const int max_seqlen_q)
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q) {
    };

    // Causal_mask: whether this particular iteration needs causal masking
    template <bool Causal_mask=false, bool Is_even_MN=true, typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask(Tensor<Engine, Layout> &tensor_,
                                               const int col_idx_offset_,
                                               const int row_idx_offset_,
                                               const int tid,
                                               const int mma_m) {
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");
        static constexpr bool Need_masking = Causal_mask || !Is_even_MN;
        // if (cute::thread0()) { printf("Has_alibi = %d, Causal_mask=%d, Is_local=%d, Is_even_MN = %d, Need_masking = %d\n", Has_alibi, Causal_mask, Is_local, Is_even_MN, Need_masking); }
        if constexpr (Need_masking) {
            // Reshape tensor_ from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
            Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));
            // Do we need both row and column indices, or just column incides?
            static constexpr bool Col_idx_only = !Causal_mask;
            const int lane_id = threadIdx.x % 32;
            const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
            if constexpr (Col_idx_only) {
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        #pragma unroll
                        for (int mi = 0; mi < size<0>(tensor); ++mi) {
                            if constexpr (!Is_even_MN) {
                                if (col_idx >= max_seqlen_k) { tensor(mi, make_coord(j, nj)) = -INFINITY; }
                            }
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int mi = 0; mi < size<0 ,1>(tensor); mi++) {
                    const int row_idx_base = row_idx_offset_ + mi * mma_m + (tid / 32) * 16 + (tid % 32) / 4;
                    #pragma unroll
                    for (int i = 0; i < size<0, 0>(tensor); i++) {
                        const int row_idx = row_idx_base + i * 8;
                        const int col_idx_limit_right = std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q);
                        #pragma unroll
                        for (int nj = 0; nj < size<1, 1>(tensor); nj++) {
                            const int col_idx_base = col_idx_offset + nj * 8;
                            #pragma unroll
                            for (int j = 0; j < size<1, 0>(tensor); j++) {
                                const int col_idx = col_idx_base + j;
                                if constexpr (Causal_mask) {
                                    if (col_idx >= col_idx_limit_right) {
                                        tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

};

} // namespace FLASH_NAMESPACE
