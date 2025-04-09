#pragma once

#include <cmath>

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include "utils.h"


namespace flash {

using namespace cute;


template <
  class SrcType,
  class DstType
>
struct ConvertKvCache;


template <>
struct ConvertKvCache<
  cutlass::int4b_t,
  cutlass::bfloat16_t
> {
  template<class EngineIn, class EngineOut>
  CUTLASS_DEVICE
  static void convert(
    cute::Tensor<EngineIn,
                cute::Layout<Shape<_2>, Stride<_1>>
                > const& src, 
    cute::Tensor<EngineOut,
                cute::Layout<Shape<_8>, Stride<_1>>
                >& dst) {

    using DstArray = cutlass::Array<cutlass::bfloat16_t, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, 4, sizeof(DstArray)>;

    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r       = cute::recast<RegArray>(dst)(0);
    CUTLASS_PRAGMA_UNROLL
    for (size_t ii = 0; ii < RegArray::kElements; ++ii) {
      r[ii] = src_reg >> (4 * (ii));
      static constexpr uint32_t xor_mask = 0x43084308;
      static constexpr uint32_t lo_mask  = 0x000F000F;
      static constexpr uint32_t immLut   = (0xf0 & 0xcc) ^ 0xaa;
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(lo_mask), "n"(xor_mask), "n"(immLut));
      static constexpr uint32_t lo_bias = xor_mask; // 0x43084308, {136, 136}
      {
        __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
        bf16x2_val = __hsub2(bf16x2_val,
                              reinterpret_cast<const __nv_bfloat162&>(lo_bias));
      }
    }
  }
};




template <>
struct ConvertKvCache<
  cutlass::uint4b_t,
  cutlass::bfloat16_t
> {
  template<class EngineIn, class EngineOut>
  CUTLASS_DEVICE
  static void convert(
    cute::Tensor<EngineIn,
                cute::Layout<Shape<_2>, Stride<_1>>
                > const& src, 
    cute::Tensor<EngineOut,
                cute::Layout<Shape<_8>, Stride<_1>>
                >& dst) {

    using DstArray = cutlass::Array<cutlass::bfloat16_t, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, 4, sizeof(DstArray)>;

    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r       = cute::recast<RegArray>(dst)(0);
    CUTLASS_PRAGMA_UNROLL
    for (size_t ii = 0; ii < RegArray::kElements; ++ii) {
      r[ii] = src_reg >> (4 * (ii));
      static constexpr uint32_t or_mask = 0x43004300;
      static constexpr uint32_t lo_mask = 0x000F000F;
      static constexpr uint32_t immLut  = (0xf0 & 0xcc) | 0xaa;
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(lo_mask), "n"(or_mask), "n"(immLut));
      static constexpr uint32_t lo_bias = or_mask; // 0x43004300, {128, 128}
      {
        __nv_bfloat162& bf16x2_val = reinterpret_cast<__nv_bfloat162&>(r[ii]);
        bf16x2_val = __hsub2(bf16x2_val,
                             reinterpret_cast<const __nv_bfloat162&>(lo_bias));
      }
    }
  }
};


template <>
struct ConvertKvCache<
  cutlass::int4b_t,
  cutlass::half_t
> {
  template<class EngineIn, class EngineOut>
  CUTLASS_DEVICE
  static void convert(
    cute::Tensor<EngineIn,
                cute::Layout<Shape<_2>, Stride<_1>>
                > const& src, 
    cute::Tensor<EngineOut,
                cute::Layout<Shape<_8>, Stride<_1>>
                >& dst) {

    using DstArray = cutlass::Array<cutlass::half_t, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, 4, sizeof(DstArray)>;

    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r       = cute::recast<RegArray>(dst)(0);
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ii += 2) {
      auto src_ = src_reg >> (4 * (ii));
      r[ii + 0] = src_;
      r[ii + 1] = src_;
      static constexpr uint32_t lo_xor_mask = 0x64086408;
      static constexpr uint32_t hi_xor_mask = 0x64806480;
      static constexpr uint32_t lo_mask     = 0x000F000F;
      static constexpr uint32_t hi_mask     = 0x00F000F0;
      static constexpr uint32_t immLut      = (0xf0 & 0xcc) ^ 0xaa;
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii + 0])
          : "n"(lo_mask), "n"(lo_xor_mask), "n"(immLut));
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii + 1])
          : "n"(hi_mask), "n"(hi_xor_mask), "n"(immLut));
      static constexpr uint32_t lo_bias  = 0x64086408; // {1032, 1032}
      static constexpr uint32_t hi_bias  = 0xD480D480; // {-72, -72}
      static constexpr uint32_t hi_scale = 0x2C002C00; // {1/16, 1/16}
      {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 0]);
        fp16x2_val = __hsub2(fp16x2_val,
                             reinterpret_cast<const half2&>(lo_bias));
      }
      {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 1]);
        fp16x2_val = __hfma2(fp16x2_val,
                              reinterpret_cast<const half2&>(hi_scale),
                              reinterpret_cast<const half2&>(hi_bias));
      }
    }
  }
};



template <>
struct ConvertKvCache<
  cutlass::uint4b_t,
  cutlass::half_t
> {
  template<class EngineIn, class EngineOut>
  CUTLASS_DEVICE
  static void convert(
    cute::Tensor<EngineIn,
                cute::Layout<Shape<_2>, Stride<_1>>
                > const& src, 
    cute::Tensor<EngineOut,
                cute::Layout<Shape<_8>, Stride<_1>>
                >& dst) {

    using DstArray = cutlass::Array<cutlass::half_t, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, 4, sizeof(DstArray)>;

    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r       = cute::recast<RegArray>(dst)(0);
    CUTLASS_PRAGMA_UNROLL
    for (int ii = 0; ii < RegArray::kElements; ii += 2) {
      auto src_ = src_reg >> (4 * (ii));
      r[ii + 0] = src_;
      r[ii + 1] = src_;
      static constexpr uint32_t or_mask = 0x64006400;
      static constexpr uint32_t lo_mask = 0x000F000F;
      static constexpr uint32_t hi_mask = 0x00F000F0;
      static constexpr uint32_t immLut  = (0xf0 & 0xcc) | 0xaa;
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii])
          : "n"(lo_mask), "n"(or_mask), "n"(immLut));
      asm volatile(
          "{\n"
          "  lop3.b32 %0, %0, %1, %2, %3;\n"
          "}\n"
          : "+r"(r[ii + 1])
          : "n"(hi_mask), "n"(or_mask), "n"(immLut));
      static constexpr uint32_t lo_bias  = or_mask;    // 0x64006400, {1024, 1024}
      static constexpr uint32_t hi_bias  = 0xD400D400; // {-64, -64}
      static constexpr uint32_t hi_scale = 0x2C002C00; // {1/16, 1/16}
      {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 0]);
        fp16x2_val = __hsub2(fp16x2_val,
                             reinterpret_cast<const half2&>(lo_bias));
      }
      {
        half2& fp16x2_val = reinterpret_cast<__half2&>(r[ii + 1]);
        fp16x2_val = __hfma2(fp16x2_val,
                             reinterpret_cast<const half2&>(hi_scale),
                             reinterpret_cast<const half2&>(hi_bias));
      }
    }
  }
};



}



