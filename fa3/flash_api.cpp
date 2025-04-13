#include <torch/python.h>
#include <torch/nn/functional.h>
#include <torch/version.h>  // For TORCH_VERSION* macros
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "static_switch.h"
#include "tile_size.h"
#include "heuristics.h"
#include "cuda_check.h"


// b: batch_size
// b_k: batch_size_k
// s_q: seqlen_q
// s_k: seqlen_k
// s_k_new: seqlen_k_new
// h: num_heads
// h_k: num_heads_k
// d: head_size



void run_mha_fwd(Flash_fwd_params &params, cudaStream_t stream) {


    SPLIT_SWITCH(params.num_splits > 1, Split, [&] {
        FP16_SWITCH(!params.is_bf16, [&] {
            BOOL_SWITCH(params.is_causal, Is_causal, [&] {
                run_mha_fwd_<elem_type, 128, Is_causal, Split>(params, stream);
            });
        });
    });


}




std::vector<at::Tensor>
mha_fwd(
    at::Tensor& q, // (b, s_q, h, d) or (total_q, h, d) if there is cu_seqlens_q
    const at::Tensor& k, // (num_pages, page_size, h_k, d)
    const at::Tensor& v, // (num_pages, page_size, h_k, d)
    std::optional<at::Tensor> &out_,  // (b, s_q, h, dv) or (total_q, h, dv) if there is cu_seqlens_q
    std::optional<const at::Tensor> &cu_seqlens_q_,  // b+1
    std::optional<const at::Tensor> &cu_seqlens_k_,  // b+1
    std::optional<int> max_seqlen_q_,
    std::optional<int> max_seqlen_k_,
    std::optional<const at::Tensor> &page_table_, // (b_k, max_num_pages_per_seq)
    float const softmax_scale,
    bool is_causal,
    std::optional<at::Tensor> &scheduler_metadata_,  // (b + 1)
    int num_splits,
    std::optional<bool> pack_gqa_,
    int const sm_margin
) {
    auto dprops = at::cuda::getCurrentDeviceProperties();
    bool is_sm8x = dprops->major >= 8;
    TORCH_CHECK(is_sm8x, "FlashAttention only supports Ampere GPUs or newer.");

    auto q_type = q.scalar_type();
    TORCH_CHECK(q_type == at::ScalarType::Half || q_type == at::ScalarType::BFloat16,
                "FlashAttention only supports fp16, bf16, and fp8_e4m3 data type");
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    at::Tensor page_table;
    const bool paged_KV = page_table_.has_value();

    if (paged_KV) {
        page_table = page_table_.value();
        CHECK_DEVICE(page_table);
        TORCH_CHECK(page_table.dtype() == torch::kInt32, "page_table must have dtype torch.int32");
        TORCH_CHECK(page_table.stride(-1) == 1, "page_table must have contiguous last dimension");
    }

    at::Tensor cu_seqlens_q;
    bool const is_varlen_q = cu_seqlens_q_.has_value();
    if (is_varlen_q) {
        cu_seqlens_q = cu_seqlens_q_.value();
        CHECK_DEVICE(cu_seqlens_q); CHECK_CONTIGUOUS(cu_seqlens_q);
        TORCH_CHECK(cu_seqlens_q.dtype() == torch::kInt32, "cu_seqlens_q must have dtype torch.int32");
        TORCH_CHECK(max_seqlen_q_.has_value(), "max_seqlen_q must be provided if cu_seqlens_q is provided");
    }

    at::Tensor cu_seqlens_k;
    bool const is_varlen_k = cu_seqlens_k_.has_value();
    if (is_varlen_k) {
        cu_seqlens_k = cu_seqlens_k_.value();
        CHECK_DEVICE(cu_seqlens_k); CHECK_CONTIGUOUS(cu_seqlens_k);
        TORCH_CHECK(cu_seqlens_k.dtype() == torch::kInt32, "cu_seqlens_k must have dtype torch.int32");
        TORCH_CHECK(max_seqlen_k_.has_value(), "max_seqlen_k must be provided if cu_seqlens_k is provided");
        TORCH_CHECK(!paged_KV, "If cu_seqlens_k is passed in, then page table is not supported");
        TORCH_CHECK(!kv_batch_idx_.has_value(), "If cu_seqlens_k is passed in, then page table is not supported");
    }

    auto const sizes = q.sizes();
    const int batch_size = !is_varlen_q ? sizes[0] : cu_seqlens_q.size(0) - 1;

    int seqlen_q = !is_varlen_q ? sizes[1] : max_seqlen_q_.value();

    int total_q = !is_varlen_q ? batch_size * sizes[1] : sizes[0];

    int num_heads = q.size(-2);

    int const head_size = q.size(-1);

    int const max_num_pages_per_seq = page_table.size(1);

    int const num_pages = k.size(0);

    int const page_size = k.size(1);

    if (seqlen_q == 1) {
       is_causal = false; 
    }

    TORCH_CHECK(head_size % 32 == 0, "head_size should be a multiple of " + std::to_string(32));

    auto opts = q.options();

    at::Tensor out;

    if (out_.has_value()) {
        out = out_.value();
        TORCH_CHECK(out.scalar_type() == out_type, "For FP16/BF16 input, output must have the same dtype as inputs. For FP8 input, output must have dtype BF16");
        CHECK_DEVICE(out);
        TORCH_CHECK(out.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        if (!is_varlen_q) {
            CHECK_SHAPE(out, batch_size, seqlen_q, num_heads, head_size_v);
        } else {
            CHECK_SHAPE(out, total_q, num_heads, head_size_v);
        }
    } else {
        out = !is_varlen_q
            ? torch::empty({batch_size, seqlen_q, num_heads, head_size_v}, opts.dtype(out_type))
            : torch::empty({total_q, num_heads, head_size_v}, opts.dtype(out_type));
    }


    at::cuda::CUDAGuard device_guard{(char)q.get_device()};

    at::Tensor softmax_lse;
    if (!is_varlen_q) {
        softmax_lse = torch::empty({batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
    } else {
        softmax_lse = torch::empty({num_heads, total_q}, opts.dtype(at::kFloat));
    }

    Flash_fwd_params params;

    set_params_fprop(params,
                     batch_size,
                     seqlen_q, seqlen_k,
                     seqlen_q_rounded, seqlen_k_rounded,
                     num_heads, num_heads_k,
                     head_size, head_size_rounded,
                     q, k, v, out,
                     !is_varlen_q ? nullptr : cu_seqlens_q.data_ptr(),
                     !is_varlen_k ? nullptr : cu_seqlens_k.data_ptr(),
                     seqused_q_.has_value() ? seqused_q_.value().data_ptr() : nullptr,
                     seqused_k_.has_value() ? seqused_k_.value().data_ptr() : nullptr,
                     softmax_lse.data_ptr(),
                     /*p_dropout=*/0.f,
                     softmax_scale,
                     window_size_left,
                     window_size_right,
                     softcap,
                     sm_margin);
    
    if (paged_KV) {
        params.page_table = page_table.data_ptr<int>();
        params.page_table_batch_stride = page_table.stride(0);
    }

    params.page_size = page_size;
    params.num_pages = num_pages;

    params.pack_gqa = true;


    at::Tensor out_accum, softmax_lse_accum;
    auto outaccum_type = at::ScalarType::Float;
    if (params.num_splits > 1) {
        TORCH_CHECK(params.num_splits <= 256, "num_splits > 256 not supported");
        if (!is_varlen_q) {
            out_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen_q, head_size_v}, opts.dtype(outaccum_type));
            softmax_lse_accum = torch::empty({params.num_splits, batch_size, num_heads, seqlen_q}, opts.dtype(at::kFloat));
            params.oaccum_batch_stride = out_accum.stride(1);
            params.lseaccum_batch_stride = softmax_lse_accum.stride(1);
        } else {
            out_accum = torch::empty({params.num_splits, num_heads, total_q, head_size_v}, opts.dtype(outaccum_type));
            softmax_lse_accum = torch::empty({params.num_splits, num_heads, total_q}, opts.dtype(at::kFloat));
        }
        params.is_fp32 = false;
        params.oaccum_ptr = out_accum.data_ptr();
        params.softmax_lseaccum_ptr = softmax_lse_accum.data_ptr();
        params.oaccum_split_stride = out_accum.stride(0);
        params.oaccum_row_stride = out_accum.stride(-2);
        params.oaccum_head_stride = out_accum.stride(-3);
        params.lseaccum_split_stride = softmax_lse_accum.stride(0);
        params.lseaccum_head_stride = softmax_lse_accum.stride(-2);
    }

    if (total_q > 0 && (total_k + params.total_knew) > 0 && num_heads_k > 0) {
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        run_mha_fwd(params, stream);
        if (params.num_splits > 1) {
            if (out_type == at::ScalarType::BFloat16) {
                // Since we want output in BF16. Otherwise fwd_combine will output to FP16
                params.is_bf16 = true;
            }
            // Unless there's seqused_q, for the purpose of attn_combine, we can just treat it as batch=1
            // and seqlen = total_q, and don't need to dispatch to Varlen there.
            // However, with dynamic split, each row needs to know which batch it belongs to
            // to read the number of splits, so we just use the varlen version of combine kernel.
            // if (is_varlen_q && !seqused_q_.has_value()) {
            // if (is_varlen_q) {
            //     params.b = 1;
            //     params.seqlen_q = total_q;
            // }
            run_mha_fwd_combine(params, stream, true /*enable_pdl*/);
        }
    } else if (total_q > 0 && num_heads_k > 0) {
        // If seqlen_k == 0, then we have an empty tensor. We need to set the output to 0.
        out.zero_();
        softmax_lse.fill_(std::numeric_limits<float>::infinity());
    }

    // return {out, softmax_lse};
    return {out, softmax_lse, out_accum, softmax_lse_accum};


}

