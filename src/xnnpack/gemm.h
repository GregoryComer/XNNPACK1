// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  void fn_name(                                            \
      size_t mr,                                           \
      size_t nc,                                           \
      size_t kc,                                           \
      const void* a,                                       \
      size_t a_stride,                                     \
      const void* w,                                       \
      void* c,                                             \
      size_t cm_stride,                                    \
      size_t cn_stride,                                    \
      const union xnn_bf16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_shland)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_shland)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_shland)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_shland)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_shland)

DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonfma_zip)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonfma_zip)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonfma_zip)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonfma_zip)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonfma_zip)

DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_1x8c2__neonbf16_bfdot_lane_ld128)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_4x8c2__neonbf16_bfdot_lane_ld128)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_5x8c2__neonbf16_bfdot_lane_ld128)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_6x8c2__neonbf16_bfdot_lane_ld128)

DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfdot)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfdot)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfdot)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfdot)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfdot)

DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_1x4c8__neonbf16_bfmlal)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_2x4c8__neonbf16_bfmlal)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_3x4c8__neonbf16_bfmlal)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_4x4c8__neonbf16_bfmlal)
DECLARE_BF16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_bf16_gemm_minmax_ukernel_5x4c8__neonbf16_bfmlal)


#define DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  void fn_name(                                           \
      size_t mr,                                          \
      size_t nc,                                          \
      size_t kc,                                          \
      const void* a,                                      \
      size_t a_stride,                                    \
      const void* w,                                      \
      void* c,                                            \
      size_t cm_stride,                                   \
      size_t cn_stride,                                   \
      const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_1x8__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_1x8__neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld32)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_1x16__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_1x16__neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_4x8__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_4x8__neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld32)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_4x16__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_4x16__neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_6x8__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_6x8__neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a55r0)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_cortex_a75)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld32)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_6x16__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_6x16__neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_8x8__asm_aarch64_neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_8x8__neonfp16arith_ld64)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_8x16__neonfp16arith_ld64)

DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_1x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_1x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_3x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_4x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_4x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_5x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_5x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_6x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_gemm_minmax_ukernel_7x8__avx2_broadcast)

DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_gemm_minmax_ukernel_1x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_gemm_minmax_ukernel_1x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_gemm_minmax_ukernel_3x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_gemm_minmax_ukernel_4x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_gemm_minmax_ukernel_4x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_gemm_minmax_ukernel_5x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_gemm_minmax_ukernel_5x16__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_gemm_minmax_ukernel_6x8__avx2_broadcast)
DECLARE_F16_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_f32acc_gemm_minmax_ukernel_7x8__avx2_broadcast)


#define DECLARE_F32_GEMM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t mr,                                   \
      size_t nc,                                   \
      size_t kc,                                   \
      const float* a,                              \
      size_t a_stride,                             \
      const float* w,                              \
      float* c,                                    \
      size_t cm_stride,                            \
      size_t cn_stride,                            \
      const union xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#define DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                            \
      size_t mr,                                        \
      size_t nc,                                        \
      size_t kc,                                        \
      const float* a,                                   \
      size_t a_stride,                                  \
      const float* w,                                   \
      float* c,                                         \
      size_t cm_stride,                                 \
      size_t cn_stride,                                 \
      const union xnn_f32_relu_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#define DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                              \
      size_t mr,                                          \
      size_t nc,                                          \
      size_t kc,                                          \
      const float* a,                                     \
      size_t a_stride,                                    \
      const float* w,                                     \
      float* c,                                           \
      size_t cm_stride,                                   \
      size_t cn_stride,                                   \
      const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_goi_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_goi_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__neon_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x2__neon_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__neon_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__neon_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x2__neon_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__neon_lane_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x16__neon_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_2x16__neon_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x16__neon_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x16__neon_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x16__neon_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x16__neon_lane_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_2x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x16__aarch64_neonfma_lane_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__neon_dup_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__neon_dup_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__neon_dup_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__neon_dup_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__neon_dup_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__neonfma_dup_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__neonfma_dup_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__neonfma_dup_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8s4__neon)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8s4__neon)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8s4__neon)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_8x8s4__neon)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8s4__neonfma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8s4__neonfma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8s4__neonfma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_8x8s4__neonfma)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x4__asm_aarch32_vfp_ld64)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch32_neon_cortex_a53_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x4__asm_aarch32_vfp_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a7)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a53_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a55)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_cortex_a75_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch32_neon_ld64)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__sse_load1)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8__sse_load1)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__sse_load1)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__sse_load1)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__sse_load1)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__sse_dup)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8__sse_dup)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__sse_dup)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__sse_dup)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__sse_dup)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8s4__sse)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8s4__sse)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8s4__sse)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8s4__sse)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8s4__sse)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x2c4__sse)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x2c4__sse)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__sse2_dup)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8__sse2_dup)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__sse2_dup)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__sse2_dup)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__sse2_dup)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x16__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x16__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x16__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x16__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x16__avx_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_7x8__avx_broadcast)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x16__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x16__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x16__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x16__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_7x8__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_8x8__fma3_broadcast)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x16s4__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x16s4__fma3_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x16s4__fma3_broadcast)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_7x16__avx512f_broadcast)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_8x16__avx512f_broadcast)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_3x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__wasmsimd_loadsplat)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_3x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_6x8__wasmsimd_loadsplat)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__wasmsimd_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_3x8__wasmsimd_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__wasmsimd_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8__wasmsimd_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__wasmsimd_splat)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_1x8__wasmsimd_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_3x8__wasmsimd_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_4x8__wasmsimd_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_5x8__wasmsimd_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_6x8__wasmsimd_splat)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_arm_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_arm_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_arm_splat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8__wasmsimd_x86_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__wasmsimd_x86_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__wasmsimd_x86_splat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8s4__wasmsimd)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_3x8s4__wasmsimd)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8s4__wasmsimd)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8s4__wasmsimd)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8s4__wasmsimd)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_1x8s4__wasmsimd)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_3x8s4__wasmsimd)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_4x8s4__wasmsimd)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_5x8s4__wasmsimd)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_6x8s4__wasmsimd)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_arm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_arm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_arm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_arm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_arm)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8s4__wasmsimd_x86)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8s4__wasmsimd_x86)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8s4__wasmsimd_x86)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8s4__wasmsimd_x86)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8s4__wasmsimd_x86)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x2c4__wasmsimd)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_4x2c4__wasmsimd)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x2c4__wasmsimd_arm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x2c4__wasmsimd_x86)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_1x4__wasm)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_2x4__wasm)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_4x2__wasm)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_4x4__wasm)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x4__wasm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_2x4__wasm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x2__wasm)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x4__wasm)

DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_1x4__scalar)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_2x4__scalar)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x2__scalar)
DECLARE_F32_GEMM_UKERNEL_FUNCTION(xnn_f32_gemm_ukernel_4x4__scalar)

DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_1x4__scalar)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_2x4__scalar)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_4x2__scalar)
DECLARE_F32_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_gemm_relu_ukernel_4x4__scalar)

DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_1x4__scalar)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_2x4__scalar)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x2__scalar)
DECLARE_F32_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemm_minmax_ukernel_4x4__scalar)

#define DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                   \
      size_t mr,                                               \
      size_t nr,                                               \
      size_t k,                                                \
      const float* a,                                          \
      size_t a_stride,                                         \
      const void* w,                                           \
      float* c,                                                \
      size_t cm_stride,                                        \
      size_t cn_stride,                                        \
      const union xnn_f32_qc4w_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__neon_dup_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__neon_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x8__neon_dup_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x8__neon_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_5x8__neon_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_6x8__neon_dup_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_6x8__neon_lane_ld64)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x16__avx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_2x16__avx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_3x16__avx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x16__avx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_5x16__avx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_6x16__avx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_7x16__avx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_8x16__avx_broadcast)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x16__fma3_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_2x16__fma3_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_3x16__fma3_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x16__fma3_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_5x16__fma3_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_6x16__fma3_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_7x16__fma3_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_8x16__fma3_broadcast)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x16__avx2_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_2x16__avx2_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_3x16__avx2_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x16__avx2_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_5x16__avx2_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_6x16__avx2_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_7x16__avx2_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_8x16__avx2_broadcast)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x32__avx512skx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_2x32__avx512skx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_3x32__avx512skx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x32__avx512skx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_5x32__avx512skx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_6x32__avx512skx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_7x32__avx512skx_broadcast)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_8x32__avx512skx_broadcast)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__sse2_dup)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_3x8__sse2_dup)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x8__sse2_dup)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_5x8__sse2_dup)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_6x8__sse2_dup)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x8__sse41_dup)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_3x8__sse41_dup)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x8__sse41_dup)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_5x8__sse41_dup)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_6x8__sse41_dup)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x4__wasm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_2x4__wasm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x2__wasm)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x4__wasm)

DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_1x4__scalar)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_2x4__scalar)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x2__scalar)
DECLARE_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc4w_gemm_minmax_ukernel_4x4__scalar)

#define DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                            \
      size_t mr,                                        \
      size_t nc,                                        \
      size_t kc,                                        \
      const float* a,                                   \
      size_t a_stride,                                  \
      const void* w,                                    \
      float* c,                                         \
      size_t cm_stride,                                 \
      size_t cn_stride,                                 \
      const union xnn_f32_default_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#define DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
      size_t mr,                                             \
      size_t nc,                                             \
      size_t kc,                                             \
      const float* a,                                        \
      size_t a_stride,                                       \
      const void* w,                                         \
      float* c,                                              \
      size_t cm_stride,                                      \
      size_t cn_stride,                                      \
      const union xnn_f32_relu_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#define DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                   \
      size_t mr,                                               \
      size_t nc,                                               \
      size_t kc,                                               \
      const float* a,                                          \
      size_t a_stride,                                         \
      const void* w,                                           \
      float* c,                                                \
      size_t cm_stride,                                        \
      size_t cn_stride,                                        \
      const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__aarch64_neonfma_lane_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__neon_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x2__neon_lane_ld64)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neon_ld128_acc2_prfm)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc2_prfm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_acc4_prfm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64_prfm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc2_prfm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_acc4_prfm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__asm_aarch64_neonfma_ld128_prfm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x1__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__neonfma)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__neonfma)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__neonfma)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_dup_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neon_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__neonfma_dup_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_dup_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neon_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__neonfma_dup_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__neon_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_dup_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neon_lane_ld64)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__neonfma_dup_ld64)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__sse2_load1)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__sse41_load1)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__sse2_load1)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__sse41_load1)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__sse2_load1)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__sse41_load1)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__sse2_load1)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__sse41_load1)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__sse2_load1)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__sse41_load1)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__sse2_dup)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__sse2_dup)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__sse2_dup)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__sse2_dup)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__sse2_dup)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__sse41_dup)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__sse41_dup)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__sse41_dup)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__sse41_dup)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__sse41_dup)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__sse2)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__sse41)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__sse2)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__sse41)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__sse2)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__sse41)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__sse2)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__sse41)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__sse2)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__sse41)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__sse2)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__sse41)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x2c4__sse2)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x2c4__sse41)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_2x16__avx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x16__avx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__avx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x16__avx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_7x16__avx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_8x16__avx_broadcast)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__fma3_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_2x16__fma3_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x16__fma3_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__fma3_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x16__fma3_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x16__fma3_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_7x16__fma3_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_8x16__fma3_broadcast)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_7x8__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_8x8__avx2_broadcast)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_2x16__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x16__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x16__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_7x16__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_8x16__avx2_broadcast)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x16s4__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_2x16s4__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x16s4__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x16s4__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x16s4__avx2_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x16s4__avx2_broadcast)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x16__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_2x16__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x16__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x16__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x16__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x16__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_7x16__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_8x16__avx512skx_broadcast)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x32__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_2x32__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x32__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x32__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x32__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x32__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_7x32__avx512skx_broadcast)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_8x32__avx512skx_broadcast)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_1x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_3x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_4x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_5x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_6x8__wasmsimd_loadsplat)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_3x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_4x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_5x8__wasmsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_6x8__wasmsimd_loadsplat)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_arm_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_arm_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_arm_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_arm_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_arm_loadsplat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_x86_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_x86_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_x86_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_x86_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_x86_loadsplat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_1x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_3x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_4x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_5x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_6x8__wasmsimd_splat)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_3x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_4x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_5x8__wasmsimd_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_6x8__wasmsimd_splat)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_arm_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_arm_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_arm_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_arm_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_arm_splat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmsimd_x86_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmsimd_x86_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmsimd_x86_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmsimd_x86_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmsimd_x86_splat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_splat)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_1x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_3x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_4x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_5x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_6x8s4__wasmsimd)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_1x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_3x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_4x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_5x8s4__wasmsimd)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_6x8s4__wasmsimd)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmsimd_arm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmsimd_arm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmsimd_arm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmsimd_arm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmsimd_arm)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmsimd_x86)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmsimd_x86)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmsimd_x86)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmsimd_x86)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmsimd_x86)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_4x2c4__wasmsimd)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_4x2c4__wasmsimd)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmrelaxedsimd_fma)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmsimd_arm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x2c4__wasmsimd_x86)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_1x4__wasm)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_2x4__wasm)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_4x2__wasm)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_4x4__wasm)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__wasm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__wasm)

DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_1x4__scalar)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_2x4__scalar)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_4x2__scalar)
DECLARE_F32_QC8W_GEMM_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_ukernel_4x4__scalar)

DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_1x4__scalar)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_2x4__scalar)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_4x2__scalar)
DECLARE_F32_QC8W_GEMM_RELU_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_relu_ukernel_4x4__scalar)

DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_2x4__scalar)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x2__scalar)
DECLARE_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_qc8w_gemm_minmax_ukernel_4x4__scalar)

#define DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
      size_t mr,                                             \
      size_t nc,                                             \
      size_t kc,                                             \
      const float* a,                                        \
      size_t a_stride,                                       \
      const float* w,                                        \
      float* c,                                              \
      size_t cm_stride,                                      \
      size_t cn_stride,                                      \
      const float* acc,                                      \
      const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__neon_lane_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__neon_lane_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__neon_lane_ld128)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__neon_lane_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__neon_lane_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__neon_lane_ld128)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__aarch64_neonfma_lane_ld128)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__aarch64_neonfma_lane_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__aarch64_neonfma_lane_ld128)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__neon_dup_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__neon_dup_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__neon_dup_ld128)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__neon_dup_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__neon_dup_ld128)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__neonfma_dup_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__neonfma_dup_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__neonfma_dup_ld128)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__neonfma_dup_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__neonfma_dup_ld128)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8s4__neon)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8s4__neon)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8s4__neon)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_8x8s4__neon)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8s4__neonfma)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8s4__neonfma)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8s4__neonfma)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_8x8s4__neonfma)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x12__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a55)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__asm_aarch64_neonfma_ld128)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x12__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a53)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a55)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a73)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__asm_aarch64_neonfma_cortex_a75_prfm)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__asm_aarch64_neonfma_ld64)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__asm_aarch64_neonfma_ld128)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__sse_load1)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8__sse_load1)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__sse_load1)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__sse_load1)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__sse_dup)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8__sse_dup)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__sse_dup)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__sse_dup)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__sse2_dup)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8__sse2_dup)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__sse2_dup)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__sse2_dup)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8s4__sse)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8s4__sse)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8s4__sse)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8s4__sse)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__avx_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x16__avx_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x16__avx_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__avx_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x16__avx_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__avx_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x16__avx_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__avx_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_7x8__avx_broadcast)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__fma3_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x16__fma3_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x16__fma3_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__fma3_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x16__fma3_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__fma3_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x16__fma3_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__fma3_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_7x8__fma3_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_8x8__fma3_broadcast)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x16s4__fma3_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x16s4__fma3_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x16s4__fma3_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x16s4__fma3_broadcast)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x16__avx512f_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x16__avx512f_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x16__avx512f_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x16__avx512f_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_7x16__avx512f_broadcast)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_8x16__avx512f_broadcast)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__wasmsimd_arm_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8__wasmsimd_arm_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__wasmsimd_arm_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__wasmsimd_arm_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__wasmsimd_arm_loadsplat)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__wasmsimd_x86_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8__wasmsimd_x86_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__wasmsimd_x86_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__wasmsimd_x86_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__wasmsimd_x86_loadsplat)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__wasmrelaxedsimd_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__wasmrelaxedsimd_loadsplat)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__wasmrelaxedsimd_fma_loadsplat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__wasmrelaxedsimd_fma_loadsplat)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__wasmsimd_arm_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8__wasmsimd_arm_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__wasmsimd_arm_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__wasmsimd_arm_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__wasmsimd_arm_splat)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__wasmsimd_x86_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8__wasmsimd_x86_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__wasmsimd_x86_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__wasmsimd_x86_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__wasmsimd_x86_splat)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__wasmrelaxedsimd_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8__wasmrelaxedsimd_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__wasmrelaxedsimd_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__wasmrelaxedsimd_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__wasmrelaxedsimd_splat)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8__wasmrelaxedsimd_fma_splat)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8__wasmrelaxedsimd_fma_splat)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8s4__wasmsimd_arm)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8s4__wasmsimd_arm)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8s4__wasmsimd_arm)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8s4__wasmsimd_arm)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8s4__wasmsimd_arm)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8s4__wasmsimd_x86)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8s4__wasmsimd_x86)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8s4__wasmsimd_x86)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8s4__wasmsimd_x86)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8s4__wasmsimd_x86)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8s4__wasmrelaxedsimd)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8s4__wasmrelaxedsimd)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8s4__wasmrelaxedsimd)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8s4__wasmrelaxedsimd)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8s4__wasmrelaxedsimd)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_3x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_5x8s4__wasmrelaxedsimd_fma)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_6x8s4__wasmrelaxedsimd_fma)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x4__wasm)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_2x4__wasm)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x2__wasm)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x4__wasm)

DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_1x4__scalar)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_2x4__scalar)
DECLARE_F32_GEMMINC_MINMAX_UKERNEL_FUNCTION(xnn_f32_gemminc_minmax_ukernel_4x4__scalar)


#define DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                              \
      size_t mr,                                          \
      size_t nc,                                          \
      size_t kc,                                          \
      const uint8_t* a,                                   \
      size_t a_stride,                                    \
      const void* w,                                      \
      uint8_t* c,                                         \
      size_t cm_stride,                                   \
      size_t cn_stride,                                   \
      const union xnn_qu8_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_2x8__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_3x8__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16__neon_mlal_lane)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a75_prfm)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__asm_aarch64_neondot_cortex_a55)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__asm_aarch64_neondot_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__asm_aarch64_neondot_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_1x32c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_2x8c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_2x32c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_3x8c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_3x32c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_5x8c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_5x16c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c4__neondot)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x16c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x16c4__neondot)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x16c4__neondot)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__xop_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__xop_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__xop_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__xop_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__xop_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__xop_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__xop_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__xop_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__xop_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__xop_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__xop_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx2)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x8c8__avx2)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x8c8__avx2)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x8c8__avx2)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__avx512skx)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x8c8__avx512skx)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x8c8__avx512skx)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x8c8__avx512skx)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_6x16c8__avx512skx)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x16c8__avx512skx_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x16c8__avx512skx_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x16c8__avx512skx_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_6x16c8__avx512skx_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4__wasm_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x2__wasm_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4__wasm_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x2__wasm_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x1c4__armsimd32)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x2c4__armsimd32)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x1c4__armsimd32)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x2c4__armsimd32)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_1x2__scalar)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_1x4__scalar)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_2x2__scalar)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_2x4__scalar)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_3x2__scalar)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_3x4__scalar)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x2__scalar)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x4__scalar)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_imagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_imagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_imagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_imagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_imagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_imagic)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_imagic)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x16c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x16c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_6x16c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_8x16c8__neoni8mm)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_1x8c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_2x8c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_3x8c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_4x8c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_6x8c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_fp32_ukernel_8x8c8__neoni8mm)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_1x8c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_2x8c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_3x8c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x8c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_6x8c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_8x8c8__neoni8mm)

DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_1x16c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_2x16c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_3x16c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_4x16c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_6x16c8__neoni8mm)
DECLARE_QU8_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qu8_gemm_minmax_rndnu_ukernel_8x16c8__neoni8mm)

#define DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)                                     \
  XNN_INTERNAL void fn_name(                                                                           \
      size_t mr,                                                                                       \
      size_t nr,                                                                                       \
      size_t k,                                                                                        \
      const int8_t* a,                                                                                 \
      size_t a_stride,                                                                                 \
      const void* w,                                                                                   \
      void* c,                                                                                         \
      size_t cm_stride,                                                                                \
      size_t cn_stride,                                                                                \
      const union xnn_f16_qc4w_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],                 \
      const struct xnn_qd8_quantization_params quantization_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

#define DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)                                     \
  XNN_INTERNAL void fn_name(                                                                           \
      size_t mr,                                                                                       \
      size_t nr,                                                                                       \
      size_t k,                                                                                        \
      const int8_t* a,                                                                                 \
      size_t a_stride,                                                                                 \
      const void* w,                                                                                   \
      float* c,                                                                                        \
      size_t cm_stride,                                                                                \
      size_t cn_stride,                                                                                \
      const union xnn_f32_qc4w_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],                 \
      const struct xnn_qd8_quantization_params quantization_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x16c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x16c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x16c8__neoni8mm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x32c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x32c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x32c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x32c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x32c8__neoni8mm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c4__neondotfp16arith)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16c4__neondotfp16arith)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x16__neonfp16arith_mlal_lane_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x16__neonfp16arith_mlal_lane_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x16__neonfp16arith_mlal_lane_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x16__neonfp16arith_mlal_lane_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x16__neonfp16arith_mlal_lane_prfm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512vnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512vnni)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512vnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512vnni_prfm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnnigfni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512vnnigfni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnnigfni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnnigfni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512vnnigfni)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnnigfni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512vnnigfni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnnigfni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnnigfni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni_prfm)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512vnnigfni_prfm)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx2)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx2)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx2)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx2)

DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_1x8c8__avx512skx)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_3x8c8__avx512skx)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_5x8c8__avx512skx)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_6x8c8__avx512skx)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_7x8c8__avx512skx)
DECLARE_QD8_F16_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__neoni8mm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__neoni8mm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x32c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x32c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x32c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x32c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x32c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x32c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x32c8__neoni8mm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x32c8__neoni8mm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c4__neondot)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__neondot)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c4__neondot)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnni)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnni_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnnigfni)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c4__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c4__avx512vnnigfni_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnni)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512vnnigfni_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx512vnni)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx512vnni_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx512vnnigfni)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx512vnnigfni_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx512vnnigfni_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x16c8__avx512skx_prfm)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx2)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx2)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x8c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x8c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_5x8c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_6x8c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_7x8c8__avx512skx)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_8x8c8__avx512skx)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__xop_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__xop_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__xop_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__xop_ld128)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__avx_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__avx_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__avx_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__avx_ld128)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse41_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse41_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse41_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse41_ld128)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse2_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse2_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse2_ld128)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse2_ld128)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__xop_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__xop_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__xop_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__xop_ld64)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__avx_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__avx_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__avx_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__avx_ld64)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse41_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse41_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse41_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse41_ld64)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__sse2_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__sse2_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__sse2_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__sse2_ld64)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x1__scalar)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__scalar)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__scalar)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__scalar)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__scalar)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__scalar)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__scalar)

DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x2__wasm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x4__wasm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_1x8__wasm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x2__wasm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x4__wasm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__wasm)
DECLARE_QD8_F32_QC4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_gemm_minmax_ukernel_4x4__wasm)

#define DECLARE_QD8_F32_QC4W_BL_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)                                  \
  XNN_INTERNAL void fn_name(                                                                           \
      size_t mr,                                                                                       \
      size_t nr,                                                                                       \
      size_t k,                                                                                        \
      size_t bl,                                                                                       \
      const int8_t* a,                                                                                 \
      size_t a_stride,                                                                                 \
      const void* w,                                                                                   \
      float* c,                                                                                        \
      size_t cm_stride,                                                                                \
      size_t cn_stride,                                                                                \
      const union xnn_f32_qc4w_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],                 \
      const struct xnn_qd8_quantization_params quantization_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QD8_F32_QC4W_BL_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_bl_gemm_minmax_ukernel_1x2__scalar)
DECLARE_QD8_F32_QC4W_BL_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_bl_gemm_minmax_ukernel_1x16c4__neondot)
DECLARE_QD8_F32_QC4W_BL_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_bl_gemm_minmax_ukernel_1x8c4__neondot)
DECLARE_QD8_F32_QC4W_BL_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc4w_bl_gemm_minmax_ukernel_4x8c4__neondot)

#define DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)                                     \
  XNN_INTERNAL void fn_name(                                                                           \
      size_t mr,                                                                                       \
      size_t nr,                                                                                       \
      size_t k,                                                                                        \
      size_t bl,                                                                                       \
      const int8_t* a,                                                                                 \
      size_t a_stride,                                                                                 \
      const void* w,                                                                                   \
      void* c,                                                                                         \
      size_t cm_stride,                                                                                \
      size_t cn_stride,                                                                                \
      const union xnn_f16_qc4w_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],                 \
      const struct xnn_qd8_quantization_params quantization_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x2__scalar)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x4__scalar)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8c4__neondot)
DECLARE_QD8_F16_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x8c4__neondot)

#define DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)                                     \
  XNN_INTERNAL void fn_name(                                                                           \
      size_t mr,                                                                                       \
      size_t nr,                                                                                       \
      size_t k,                                                                                        \
      size_t bl,                                                                                       \
      const int8_t* a,                                                                                 \
      size_t a_stride,                                                                                 \
      const void* w,                                                                                   \
      void* c,                                                                                         \
      size_t cm_stride,                                                                                \
      size_t cn_stride,                                                                                \
      const union xnn_f32_qc4w_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],                 \
      const struct xnn_qd8_quantization_params quantization_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QD8_F32_QB4W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qb4w_gemm_minmax_ukernel_1x2__scalar)


#define DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)                                     \
  XNN_INTERNAL void fn_name(                                                                           \
      size_t mr,                                                                                       \
      size_t nr,                                                                                       \
      size_t k,                                                                                        \
      const int8_t* a,                                                                                 \
      size_t a_stride,                                                                                 \
      const void* w,                                                                                   \
      void* c,                                                                                         \
      size_t cm_stride,                                                                                \
      size_t cn_stride,                                                                                \
      const union xnn_f16_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],                      \
      const struct xnn_qd8_quantization_params quantization_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__neoni8mm)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x16c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x16c8__neoni8mm)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x32c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x32c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x32c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x32c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x32c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x32c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x32c8__neoni8mm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x32c8__neoni8mm)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c4__neondotfp16arith)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x16c4__neondotfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x16c4__neondotfp16arith)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c2s4__neonfp16arith)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c2s4__neonfp16arith)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondotfp16arith_cortex_a55)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_cortex_a55)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondotfp16arith_ld128)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx512vnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avx512vnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avx512vnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avx512vnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx512vnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__avx512vnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx512vnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx512vnni)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx512vnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avx512vnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avx512vnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avx512vnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx512vnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__avx512vnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx512vnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx512vnni_prfm)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx2)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avx2)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avx2)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avx2)

DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_1x8c8__avx512skx)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_2x8c8__avx512skx)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_3x8c8__avx512skx)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_4x8c8__avx512skx)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_5x8c8__avx512skx)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_6x8c8__avx512skx)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_7x8c8__avx512skx)
DECLARE_QD8_F16_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f16_qc8w_gemm_minmax_ukernel_8x8c8__avx512skx)

#define DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)                                     \
  XNN_INTERNAL void fn_name(                                                                           \
      size_t mr,                                                                                       \
      size_t nr,                                                                                       \
      size_t k,                                                                                        \
      const int8_t* a,                                                                                 \
      size_t a_stride,                                                                                 \
      const void* w,                                                                                   \
      float* c,                                                                                        \
      size_t cm_stride,                                                                                \
      size_t cn_stride,                                                                                \
      const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],                      \
      const struct xnn_qd8_quantization_params quantization_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__asm_aarch64_neondot_ld128)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__neondot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c4__neondot)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__aarch64_neondot_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__aarch64_neondot_ld128)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__neondot_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neondot_ld64)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x32c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x32c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x32c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x32c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x32c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x32c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x32c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x32c8__neoni8mm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__neoni8mm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__neoni8mm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__neoni8mm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8__neon_mlal_lane_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16__neon_mlal_lane)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neon_mlal)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c2s4__neonv8_mlal)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neon_mlal)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx2)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx512skx)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__avx_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__avx_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__sse41_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__xop_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__xop_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__avx_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__avx_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse41_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__sse41_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__xop_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__xop_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c2s4__neonv8_mlal)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__avx_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__avx_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse41_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__sse41_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__xop_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__xop_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__avx_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__avx_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse41_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__sse41_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__xop_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__xop_ld128)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512skx)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512skx)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512skx_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512skx_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c4__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c4__avx512vnni)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c4__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c4__avx512vnni_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512vnni)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x16c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x16c8__avx512vnni_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx512vnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx512vnni)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avx512vnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avx512vnni_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c8__avxvnni_prfm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c8__avxvnni_prfm)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__avxvnni_u2_acc2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__avxvnni_u2_acc2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__avxvnni_u2_acc2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__avxvnni_u2_acc2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__avxvnni_u2_acc2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__avxvnni_u2_acc2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c4__avxvnni_u2_acc2)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c4__avxvnni_u2_acc2)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8c4__avxvnni_u4_acc4)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8c4__avxvnni_u4_acc4)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x8c4__avxvnni_u4_acc4)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x8c4__avxvnni_u4_acc4)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_5x8c4__avxvnni_u4_acc4)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_6x8c4__avxvnni_u4_acc4)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_7x8c4__avxvnni_u4_acc4)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_8x8c4__avxvnni_u4_acc4)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c8__wasmsimd_dot16x2_ld128)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4c16__wasmsdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4c16__wasmsdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_3x4c16__wasmsdot)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4c16__wasmsdot)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__scalar)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__scalar)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__scalar)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x2__scalar)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4__scalar)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__scalar)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__scalar)

DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x2__wasm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__wasm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x8__wasm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x2__wasm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x4__wasm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_2x8__wasm)
DECLARE_QD8_F32_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_gemm_minmax_ukernel_4x4__wasm)

#define DECLARE_QD8_F32_QC8W_BL_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name)                                  \
  XNN_INTERNAL void fn_name(                                                                           \
      size_t mr,                                                                                       \
      size_t nr,                                                                                       \
      size_t k,                                                                                        \
      size_t bl,                                                                                       \
      const int8_t* a,                                                                                 \
      size_t a_stride,                                                                                 \
      const void* w,                                                                                   \
      float* c,                                                                                        \
      size_t cm_stride,                                                                                \
      size_t cn_stride,                                                                                \
      const union xnn_f32_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)],                      \
      const struct xnn_qd8_quantization_params quantization_params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QD8_F32_QC8W_BL_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qd8_f32_qc8w_bl_gemm_minmax_ukernel_1x2__scalar)

#define DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                   \
      size_t mr,                                               \
      size_t nc,                                               \
      size_t kc,                                               \
      const int8_t* a,                                         \
      size_t a_stride,                                         \
      const void* w,                                           \
      int8_t* c,                                               \
      size_t cm_stride,                                        \
      size_t cn_stride,                                        \
      const union xnn_qs8_qc8w_conv_minmax_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neon_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neon_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neon_mlal_lane_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8__neonv8_mlal_lane_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neonv8_mlal_lane)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16__neonv8_mlal_lane_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_dup)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_dup)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_dup)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_dup)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld1r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld1r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_ld1r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld1r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neon_mlal_ld2r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neon_mlal_ld2r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neonv8_mlal_ld2r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4__neonv8_mlal_ld2r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4s2__neon_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4s2__neon_mlal)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4s2__neonv8_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c4s2__neonv8_mlal)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_dup)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_dup)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_dup)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_dup)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld1r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld1r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld1r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld1r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld2r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld2r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld2r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld2r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neon_mlal_ld4r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neon_mlal_ld4r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2__neonv8_mlal_ld4r)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2__neonv8_mlal_ld4r)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neon_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neon_mlal)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c2s4__neonv8_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c2s4__neonv8_mlal)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neon_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neon_mlal)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neonv8_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neonv8_mlal)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c4__neondot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__neondot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c4__neondot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c4__neondot)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__neondot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__neondot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c4__neondot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__neondot)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__aarch64_neondot_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neondot_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__aarch64_neondot_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neondot_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__asm_aarch64_neon_mlal_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_cortex_a53_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mlal_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__asm_aarch64_neon_mull)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c16__asm_aarch64_neon_mlal)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_cortex_a53_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16__asm_aarch64_neon_mlal_lane_ld64_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld32)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__asm_aarch64_neondot_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_cortex_a55)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld32)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__asm_aarch64_neondot_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_cortex_a55)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c4__asm_aarch32_neondot_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a7_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_cortex_a53_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neon_mlal_lane_ld64_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a35_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_cortex_a53_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8__asm_aarch32_neonv8_mlal_lane_ld64_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__avx_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__xop_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__xop_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse2_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__sse41_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__avx_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__xop_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__xop_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__xop_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__xop_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__xop_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__xop_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse2_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__sse41_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__avx_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__xop_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__xop_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__xop_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__xop_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse2_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__sse41_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__sse41_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__avx_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__avx_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__xop_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__xop_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__xop_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx2)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx2)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avx512skx)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512skx)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512skx)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512skx_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__avx512skx_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512skx_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512skx_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512skx_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__avx512skx_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512skx_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512skx_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__avx512vnni)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c4__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c4__avx512vnni_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avx512vnni)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avx512vnni_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avxvnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avxvnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avxvnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avxvnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avxvnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avxvnni)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x8c8__avxvnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__avxvnni_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512vnni)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512vnni)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_5x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_7x16c8__avx512vnni_prfm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__avx512vnni_prfm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c16__wasmsdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c16__wasmsdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c16__wasmsdot)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c16__wasmsdot)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2__wasmsimd_dot16x2_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c2s4__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c2s4__wasmsimd_dot16x2_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld64)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld64)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4c8__wasmsimd_dot16x2_ld128)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4c8__wasmsimd_dot16x2_ld128)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__wasm_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__wasm_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__wasm_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__wasm_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__wasm_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__wasm_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__wasm_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__wasm_fmagic)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x1c4__armsimd32)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2c4__armsimd32)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x1c4__armsimd32)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2c4__armsimd32)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_fmagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_fmagic)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_imagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_imagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_imagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_imagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_imagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_imagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_imagic)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_imagic)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x2__scalar_lrintf)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x4__scalar_lrintf)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x2__scalar_lrintf)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x4__scalar_lrintf)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x2__scalar_lrintf)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_lrintf)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x4__scalar_lrintf)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_1x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_2x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_3x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_4x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_6x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_8x16c8__neoni8mm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_1x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_2x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_3x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_4x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_6x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_rndnu_ukernel_8x8c8__neoni8mm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x16c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x16c8__neoni8mm)

DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_1x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_2x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_6x8c8__neoni8mm)
DECLARE_QS8_QC8W_GEMM_MINMAX_UKERNEL_FUNCTION(xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_8x8c8__neoni8mm)


#define DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL xnn_status_t fn_name(                    \
      struct xnn_code_buffer* code,                     \
      size_t max_mr,                                    \
      size_t nc_mod_nr,                                 \
      size_t kc,                                        \
      const void* params);

DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_1x8__aarch32_neon_cortex_a53_prfm)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a7)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a53_prfm)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a55)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_cortex_a75_prfm)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_4x8__aarch32_neon_ld64)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_qc8_gemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_qc8_gemm_fp32_ukernel_4x8__aarch32_neonv8_mlal_lane_ld64_prfm)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_qc8_gemm_fp32_ukernel_4x8c4__aarch32_neondot_ld64)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_qs8_gemm_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_qs8_gemm_rndnu_ukernel_4x8__aarch32_neon_mlal_lane_ld64_prfm)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_qs8_gemm_rndnu_ukernel_4x8c4__aarch32_neondot_ld64)

DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f16_gemm_ukernel_1x16__aarch64_neonfp16arith_ld64)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f16_gemm_ukernel_4x16__aarch64_neonfp16arith_ld64)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a55r0)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_cortex_a75)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f16_gemm_ukernel_6x16__aarch64_neonfp16arith_ld64)

DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a53_prfm)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_cortex_a75_prfm)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_1x8__aarch64_neonfma_ld64)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a53_prfm)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a55)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_cortex_a75_prfm)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_4x8__aarch64_neonfma_ld128)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a53)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a53_prfm)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a55)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_cortex_a75_prfm)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__aarch64_neonfma_ld128)

DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_xinf)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x1)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x2)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x4)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_loadsplat_x8)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_xinf)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x1)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x2)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x4)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_loadsplat_x8)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_xinf)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x1)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x2)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x4)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_loadsplat_x8)

DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_xinf)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x1)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x2)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_fma_x4)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_xinf)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x1)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x2)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8s4__wasmrelaxedsimd32_x86_x4)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_xinf)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x1)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x2)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8s4__wasmsimd32_x86_x4)

DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_xinf)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x1)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x2)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_fma_splat_x4)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_xinf)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x1)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x2)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmrelaxedsimd32_x86_splat_x4)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_xinf)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x1)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x2)
DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION(xnn_generate_f32_gemm_ukernel_6x8__wasmsimd32_x86_splat_x4)
#undef DECLARE_GENERATE_GEMM_UKERNEL_FUNCTION

#ifdef __cplusplus
}  // extern "C"
#endif
