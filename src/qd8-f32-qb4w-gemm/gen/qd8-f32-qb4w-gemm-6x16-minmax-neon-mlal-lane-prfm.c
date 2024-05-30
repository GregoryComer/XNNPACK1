// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/neon-mlal-lane.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/gemm.h>
#include <xnnpack/math.h>
#include <xnnpack/prefetch.h>


void xnn_qd8_f32_qb4w_gemm_minmax_ukernel_6x16__neon_mlal_lane_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 6);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(int8_t) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  size_t bl = params->scalar.blocksize;
  assert(bl <= round_up_po2(kc, 2));
  assert(bl != 0);
  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const int8_t* a4 = (const int8_t*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }
  const int8_t* a5 = (const int8_t*) ((uintptr_t) a4 + a_stride);
  float* c5 = (float*) ((uintptr_t) c4 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 6) {
    a5 = a4;
    c5 = c4;
  }

  const int8x8_t vmask = vmov_n_s8(INT8_C(0xF0));
  kc = round_up_po2(kc, 2);
  do {
    float32x4_t vksum0123 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vksum4567 = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vksum89AB = vld1q_f32(w); w = (const float*) w + 4;
    float32x4_t vksumCDEF = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vzp01 = vcvtq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    float32x4_t vout0x0123 = vmulq_lane_f32(vksum0123, vget_low_f32(vzp01), 0);
    float32x4_t vout1x0123 = vmulq_lane_f32(vksum0123, vget_high_f32(vzp01), 0);
    float32x4_t vout0x4567 = vmulq_lane_f32(vksum4567, vget_low_f32(vzp01), 0);
    float32x4_t vout1x4567 = vmulq_lane_f32(vksum4567, vget_high_f32(vzp01), 0);
    float32x4_t vout0x89AB = vmulq_lane_f32(vksum89AB, vget_low_f32(vzp01), 0);
    float32x4_t vout1x89AB = vmulq_lane_f32(vksum89AB, vget_high_f32(vzp01), 0);
    float32x4_t vout0xCDEF = vmulq_lane_f32(vksumCDEF, vget_low_f32(vzp01), 0);
    float32x4_t vout1xCDEF = vmulq_lane_f32(vksumCDEF, vget_high_f32(vzp01), 0);
    const float32x4_t vzp23 = vcvtq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    float32x4_t vout2x0123 = vmulq_lane_f32(vksum0123, vget_low_f32(vzp23), 0);
    float32x4_t vout3x0123 = vmulq_lane_f32(vksum0123, vget_high_f32(vzp23), 0);
    float32x4_t vout2x4567 = vmulq_lane_f32(vksum4567, vget_low_f32(vzp23), 0);
    float32x4_t vout3x4567 = vmulq_lane_f32(vksum4567, vget_high_f32(vzp23), 0);
    float32x4_t vout2x89AB = vmulq_lane_f32(vksum89AB, vget_low_f32(vzp23), 0);
    float32x4_t vout3x89AB = vmulq_lane_f32(vksum89AB, vget_high_f32(vzp23), 0);
    float32x4_t vout2xCDEF = vmulq_lane_f32(vksumCDEF, vget_low_f32(vzp23), 0);
    float32x4_t vout3xCDEF = vmulq_lane_f32(vksumCDEF, vget_high_f32(vzp23), 0);
    const float32x4_t vzp45 = vcvtq_f32_s32(vld1q_s32(&quantization_params[4].zero_point));
    float32x4_t vout4x0123 = vmulq_lane_f32(vksum0123, vget_low_f32(vzp45), 0);
    float32x4_t vout5x0123 = vmulq_lane_f32(vksum0123, vget_high_f32(vzp45), 0);
    float32x4_t vout4x4567 = vmulq_lane_f32(vksum4567, vget_low_f32(vzp45), 0);
    float32x4_t vout5x4567 = vmulq_lane_f32(vksum4567, vget_high_f32(vzp45), 0);
    float32x4_t vout4x89AB = vmulq_lane_f32(vksum89AB, vget_low_f32(vzp45), 0);
    float32x4_t vout5x89AB = vmulq_lane_f32(vksum89AB, vget_high_f32(vzp45), 0);
    float32x4_t vout4xCDEF = vmulq_lane_f32(vksumCDEF, vget_low_f32(vzp45), 0);
    float32x4_t vout5xCDEF = vmulq_lane_f32(vksumCDEF, vget_high_f32(vzp45), 0);

    size_t n_blocks = kc / bl;
    for (size_t nb=0; nb<n_blocks; ++nb) {
      float32x4_t vacc0x0123 = vdupq_n_s32(0);
      float32x4_t vacc0x4567 = vdupq_n_s32(0);
      float32x4_t vacc0x89AB = vdupq_n_s32(0);
      float32x4_t vacc0xCDEF = vdupq_n_s32(0);
      float32x4_t vacc1x0123 = vdupq_n_s32(0);
      float32x4_t vacc1x4567 = vdupq_n_s32(0);
      float32x4_t vacc1x89AB = vdupq_n_s32(0);
      float32x4_t vacc1xCDEF = vdupq_n_s32(0);
      float32x4_t vacc2x0123 = vdupq_n_s32(0);
      float32x4_t vacc2x4567 = vdupq_n_s32(0);
      float32x4_t vacc2x89AB = vdupq_n_s32(0);
      float32x4_t vacc2xCDEF = vdupq_n_s32(0);
      float32x4_t vacc3x0123 = vdupq_n_s32(0);
      float32x4_t vacc3x4567 = vdupq_n_s32(0);
      float32x4_t vacc3x89AB = vdupq_n_s32(0);
      float32x4_t vacc3xCDEF = vdupq_n_s32(0);
      float32x4_t vacc4x0123 = vdupq_n_s32(0);
      float32x4_t vacc4x4567 = vdupq_n_s32(0);
      float32x4_t vacc4x89AB = vdupq_n_s32(0);
      float32x4_t vacc4xCDEF = vdupq_n_s32(0);
      float32x4_t vacc5x0123 = vdupq_n_s32(0);
      float32x4_t vacc5x4567 = vdupq_n_s32(0);
      float32x4_t vacc5x89AB = vdupq_n_s32(0);
      float32x4_t vacc5xCDEF = vdupq_n_s32(0);

      size_t k = bl;
    while (k >= 8 * sizeof(int8_t)) {
      const int8x8_t va0 = vld1_s8(a0); a0 += 8;
      const int16x8_t vxa0 = vmovl_s8(va0);
      const int8x8_t va1 = vld1_s8(a1); a1 += 8;
      const int16x8_t vxa1 = vmovl_s8(va1);
      const int8x8_t va2 = vld1_s8(a2); a2 += 8;
      const int16x8_t vxa2 = vmovl_s8(va2);
      const int8x8_t va3 = vld1_s8(a3); a3 += 8;
      const int16x8_t vxa3 = vmovl_s8(va3);
      const int8x8_t va4 = vld1_s8(a4); a4 += 8;
      const int16x8_t vxa4 = vmovl_s8(va4);
      const int8x8_t va5 = vld1_s8(a5); a5 += 8;
      const int16x8_t vxa5 = vmovl_s8(va5);

      const int8x8_t vb01234567c01 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb89ABCDEFc01 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb01234567c23 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb89ABCDEFc23 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb01234567c45 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb89ABCDEFc45 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb01234567c67 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb89ABCDEFc67 = vld1_s8(w); w = (const uint8_t*) w + 8;
      const int8x8_t vb01234567c0 = vshl_n_s8(vb01234567c01, 4);
      const int8x8_t vb01234567c1 = vand_s8(vb01234567c01, vmask);
      const int8x8_t vb89ABCDEFc0 = vshl_n_s8(vb89ABCDEFc01, 4);
      const int8x8_t vb89ABCDEFc1 = vand_s8(vb89ABCDEFc01, vmask);
      const int8x8_t vb01234567c2 = vshl_n_s8(vb01234567c23, 4);
      const int8x8_t vb01234567c3 = vand_s8(vb01234567c23, vmask);
      const int8x8_t vb89ABCDEFc2 = vshl_n_s8(vb89ABCDEFc23, 4);
      const int8x8_t vb89ABCDEFc3 = vand_s8(vb89ABCDEFc23, vmask);
      const int8x8_t vb01234567c4 = vshl_n_s8(vb01234567c45, 4);
      const int8x8_t vb01234567c5 = vand_s8(vb01234567c45, vmask);
      const int8x8_t vb89ABCDEFc4 = vshl_n_s8(vb89ABCDEFc45, 4);
      const int8x8_t vb89ABCDEFc5 = vand_s8(vb89ABCDEFc45, vmask);
      const int8x8_t vb01234567c6 = vshl_n_s8(vb01234567c67, 4);
      const int8x8_t vb01234567c7 = vand_s8(vb01234567c67, vmask);
      const int8x8_t vb89ABCDEFc6 = vshl_n_s8(vb89ABCDEFc67, 4);
      const int8x8_t vb89ABCDEFc7 = vand_s8(vb89ABCDEFc67, vmask);

      const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
      const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);
      const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
      const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);
      const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
      const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);
      const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
      const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);
      const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
      const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);
      const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
      const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);
      const int16x8_t vxb01234567c6 = vmovl_s8(vb01234567c6);
      const int16x8_t vxb89ABCDEFc6 = vmovl_s8(vb89ABCDEFc6);
      const int16x8_t vxb01234567c7 = vmovl_s8(vb01234567c7);
      const int16x8_t vxb89ABCDEFc7 = vmovl_s8(vb89ABCDEFc7);

      xnn_prefetch_to_l1((const int8_t*) w + 448);
      xnn_prefetch_to_l1((const int8_t*) w + 512);

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa4), 0);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa4), 0);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa5), 0);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa5), 0);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa4), 1);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa4), 1);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa5), 1);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa5), 1);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa4), 2);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa4), 2);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa5), 2);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa5), 2);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa4), 3);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa4), 3);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa5), 3);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa5), 3);

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa4), 0);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa4), 0);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa5), 0);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa5), 0);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa4), 1);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa4), 1);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa5), 1);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa5), 1);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa0), 2);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa1), 2);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa2), 2);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa3), 2);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa4), 2);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c6), vget_high_s16(vxa5), 2);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa0), 2);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa1), 2);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa1), 2);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa2), 2);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa2), 2);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa3), 2);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa3), 2);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa4), 2);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa4), 2);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc6), vget_high_s16(vxa5), 2);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc6), vget_high_s16(vxa5), 2);
      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa0), 3);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa1), 3);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa2), 3);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa3), 3);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa4), 3);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa4), 3);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c7), vget_high_s16(vxa5), 3);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c7), vget_high_s16(vxa5), 3);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa0), 3);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa1), 3);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa1), 3);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa2), 3);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa2), 3);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa3), 3);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa3), 3);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa4), 3);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa4), 3);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc7), vget_high_s16(vxa5), 3);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc7), vget_high_s16(vxa5), 3);

      k -= 8 * sizeof(int8_t);
    }
    if XNN_UNLIKELY(k != 0) {
      const int8x8_t va0 = vld1_s8(a0); a0 = (const int8_t*) ((uintptr_t) a0 + k);
      const int16x8_t vxa0 = vmovl_s8(va0);
      const int8x8_t va1 = vld1_s8(a1); a1 = (const int8_t*) ((uintptr_t) a1 + k);
      const int16x8_t vxa1 = vmovl_s8(va1);
      const int8x8_t va2 = vld1_s8(a2); a2 = (const int8_t*) ((uintptr_t) a2 + k);
      const int16x8_t vxa2 = vmovl_s8(va2);
      const int8x8_t va3 = vld1_s8(a3); a3 = (const int8_t*) ((uintptr_t) a3 + k);
      const int16x8_t vxa3 = vmovl_s8(va3);
      const int8x8_t va4 = vld1_s8(a4); a4 = (const int8_t*) ((uintptr_t) a4 + k);
      const int16x8_t vxa4 = vmovl_s8(va4);
      const int8x8_t va5 = vld1_s8(a5); a5 = (const int8_t*) ((uintptr_t) a5 + k);
      const int16x8_t vxa5 = vmovl_s8(va5);

      const int8x8_t vb01234567c01 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb89ABCDEFc01 = vld1_s8(w); w = (const int8_t*) w + 8;
      const int8x8_t vb01234567c0 = vshl_n_s8(vb01234567c01, 4);
      const int8x8_t vb01234567c1 = vand_s8(vb01234567c01, vmask);
      const int8x8_t vb89ABCDEFc0 = vshl_n_s8(vb89ABCDEFc01, 4);
      const int8x8_t vb89ABCDEFc1 = vand_s8(vb89ABCDEFc01, vmask);

      const int16x8_t vxb01234567c0 = vmovl_s8(vb01234567c0);
      const int16x8_t vxb89ABCDEFc0 = vmovl_s8(vb89ABCDEFc0);

      vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa0), 0);
      vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa0), 0);
      vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
      vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa1), 0);
      vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
      vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa1), 0);
      vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
      vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa2), 0);
      vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
      vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa2), 0);
      vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
      vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa3), 0);
      vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
      vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa3), 0);
      vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
      vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa4), 0);
      vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa4), 0);
      vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa4), 0);
      vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
      vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c0), vget_low_s16(vxa5), 0);
      vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc0), vget_low_s16(vxa5), 0);
      vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc0), vget_low_s16(vxa5), 0);

      if (k >= 2 * sizeof(int8_t)) {
        const int16x8_t vxb01234567c1 = vmovl_s8(vb01234567c1);
        const int16x8_t vxb89ABCDEFc1 = vmovl_s8(vb89ABCDEFc1);

        vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa0), 1);
        vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa0), 1);
        vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
        vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa1), 1);
        vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
        vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa1), 1);
        vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
        vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa2), 1);
        vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
        vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa2), 1);
        vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
        vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa3), 1);
        vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
        vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa3), 1);
        vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
        vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa4), 1);
        vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa4), 1);
        vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa4), 1);
        vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
        vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c1), vget_low_s16(vxa5), 1);
        vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc1), vget_low_s16(vxa5), 1);
        vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc1), vget_low_s16(vxa5), 1);

        if (k > 2 * sizeof(int8_t)) {
          const int8x8_t vb01234567c23 = vld1_s8(w); w = (const int8_t*) w + 8;
          const int8x8_t vb89ABCDEFc23 = vld1_s8(w); w = (const int8_t*) w + 8;
          const int8x8_t vb01234567c2 = vshl_n_s8(vb01234567c23, 4);
          const int8x8_t vb01234567c3 = vand_s8(vb01234567c23, vmask);
          const int8x8_t vb89ABCDEFc2 = vshl_n_s8(vb89ABCDEFc23, 4);
          const int8x8_t vb89ABCDEFc3 = vand_s8(vb89ABCDEFc23, vmask);
          const int16x8_t vxb01234567c2 = vmovl_s8(vb01234567c2);
          const int16x8_t vxb89ABCDEFc2 = vmovl_s8(vb89ABCDEFc2);

          vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
          vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa0), 2);
          vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
          vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa0), 2);
          vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
          vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa1), 2);
          vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
          vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa1), 2);
          vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
          vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa2), 2);
          vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
          vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa2), 2);
          vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
          vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa3), 2);
          vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
          vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa3), 2);
          vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
          vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa4), 2);
          vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa4), 2);
          vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa4), 2);
          vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
          vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c2), vget_low_s16(vxa5), 2);
          vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc2), vget_low_s16(vxa5), 2);
          vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc2), vget_low_s16(vxa5), 2);

          if (k >= 4 * sizeof(int8_t)) {
            const int16x8_t vxb01234567c3 = vmovl_s8(vb01234567c3);
            const int16x8_t vxb89ABCDEFc3 = vmovl_s8(vb89ABCDEFc3);

            vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
            vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa0), 3);
            vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
            vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa0), 3);
            vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
            vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa1), 3);
            vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
            vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa1), 3);
            vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
            vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa2), 3);
            vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
            vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa2), 3);
            vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
            vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa3), 3);
            vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
            vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa3), 3);
            vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
            vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa4), 3);
            vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa4), 3);
            vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa4), 3);
            vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
            vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c3), vget_low_s16(vxa5), 3);
            vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc3), vget_low_s16(vxa5), 3);
            vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc3), vget_low_s16(vxa5), 3);

            if (k > 4 * sizeof(int8_t)) {
              const int8x8_t vb01234567c45 = vld1_s8(w); w = (const int8_t*) w + 8;
              const int8x8_t vb89ABCDEFc45 = vld1_s8(w); w = (const int8_t*) w + 8;
              const int8x8_t vb01234567c4 = vshl_n_s8(vb01234567c45, 4);
              const int8x8_t vb01234567c5 = vand_s8(vb01234567c45, vmask);
              const int8x8_t vb89ABCDEFc4 = vshl_n_s8(vb89ABCDEFc45, 4);
              const int8x8_t vb89ABCDEFc5 = vand_s8(vb89ABCDEFc45, vmask);
              const int16x8_t vxb01234567c4 = vmovl_s8(vb01234567c4);
              const int16x8_t vxb89ABCDEFc4 = vmovl_s8(vb89ABCDEFc4);

              vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
              vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa0), 0);
              vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
              vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa0), 0);
              vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
              vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa1), 0);
              vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
              vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa1), 0);
              vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
              vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa2), 0);
              vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
              vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa2), 0);
              vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
              vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa3), 0);
              vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
              vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa3), 0);
              vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
              vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa4), 0);
              vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa4), 0);
              vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa4), 0);
              vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
              vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c4), vget_high_s16(vxa5), 0);
              vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc4), vget_high_s16(vxa5), 0);
              vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc4), vget_high_s16(vxa5), 0);

              if (k >= 6 * sizeof(int8_t)) {
                const int16x8_t vxb01234567c5 = vmovl_s8(vb01234567c5);
                const int16x8_t vxb89ABCDEFc5 = vmovl_s8(vb89ABCDEFc5);

                vacc0x0123 = vmlal_lane_s16(vacc0x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                vacc0x4567 = vmlal_lane_s16(vacc0x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa0), 1);
                vacc0x89AB = vmlal_lane_s16(vacc0x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                vacc0xCDEF = vmlal_lane_s16(vacc0xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa0), 1);
                vacc1x0123 = vmlal_lane_s16(vacc1x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
                vacc1x4567 = vmlal_lane_s16(vacc1x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa1), 1);
                vacc1x89AB = vmlal_lane_s16(vacc1x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
                vacc1xCDEF = vmlal_lane_s16(vacc1xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa1), 1);
                vacc2x0123 = vmlal_lane_s16(vacc2x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
                vacc2x4567 = vmlal_lane_s16(vacc2x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa2), 1);
                vacc2x89AB = vmlal_lane_s16(vacc2x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
                vacc2xCDEF = vmlal_lane_s16(vacc2xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa2), 1);
                vacc3x0123 = vmlal_lane_s16(vacc3x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
                vacc3x4567 = vmlal_lane_s16(vacc3x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa3), 1);
                vacc3x89AB = vmlal_lane_s16(vacc3x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
                vacc3xCDEF = vmlal_lane_s16(vacc3xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa3), 1);
                vacc4x0123 = vmlal_lane_s16(vacc4x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
                vacc4x4567 = vmlal_lane_s16(vacc4x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa4), 1);
                vacc4x89AB = vmlal_lane_s16(vacc4x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa4), 1);
                vacc4xCDEF = vmlal_lane_s16(vacc4xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa4), 1);
                vacc5x0123 = vmlal_lane_s16(vacc5x0123, vget_low_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
                vacc5x4567 = vmlal_lane_s16(vacc5x4567, vget_high_s16(vxb01234567c5), vget_high_s16(vxa5), 1);
                vacc5x89AB = vmlal_lane_s16(vacc5x89AB, vget_low_s16(vxb89ABCDEFc5), vget_high_s16(vxa5), 1);
                vacc5xCDEF = vmlal_lane_s16(vacc5xCDEF, vget_high_s16(vxb89ABCDEFc5), vget_high_s16(vxa5), 1);

              }
            }
          }
        }
      }
    }

    const float32x4_t vfilter_output_scale0123 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale4567 = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scale89AB = vld1q_f32(w); w = (const float*) w + 4;
    const float32x4_t vfilter_output_scaleCDEF = vld1q_f32(w); w = (const float*) w + 4;

    const float32x4_t vf0x0123 = vcvtq_f32_s32(vacc0x0123);
    const float32x4_t vf0x4567 = vcvtq_f32_s32(vacc0x4567);
    const float32x4_t vf0x89AB = vcvtq_f32_s32(vacc0x89AB);
    const float32x4_t vf0xCDEF = vcvtq_f32_s32(vacc0xCDEF);

    vout0x0123 = vfmaq_f32(vout0x0123, vf0x0123, vfilter_output_scale0123);
    vout0x4567 = vfmaq_f32(vout0x4567, vf0x4567, vfilter_output_scale4567);
    vout0x89AB = vfmaq_f32(vout0x89AB, vf0x89AB, vfilter_output_scale89AB);
    vout0xCDEF = vfmaq_f32(vout0xCDEF, vf0xCDEF, vfilter_output_scaleCDEF);
    const float32x4_t vf1x0123 = vcvtq_f32_s32(vacc1x0123);
    const float32x4_t vf1x4567 = vcvtq_f32_s32(vacc1x4567);
    const float32x4_t vf1x89AB = vcvtq_f32_s32(vacc1x89AB);
    const float32x4_t vf1xCDEF = vcvtq_f32_s32(vacc1xCDEF);

    vout1x0123 = vfmaq_f32(vout1x0123, vf1x0123, vfilter_output_scale0123);
    vout1x4567 = vfmaq_f32(vout1x4567, vf1x4567, vfilter_output_scale4567);
    vout1x89AB = vfmaq_f32(vout1x89AB, vf1x89AB, vfilter_output_scale89AB);
    vout1xCDEF = vfmaq_f32(vout1xCDEF, vf1xCDEF, vfilter_output_scaleCDEF);
    const float32x4_t vf2x0123 = vcvtq_f32_s32(vacc2x0123);
    const float32x4_t vf2x4567 = vcvtq_f32_s32(vacc2x4567);
    const float32x4_t vf2x89AB = vcvtq_f32_s32(vacc2x89AB);
    const float32x4_t vf2xCDEF = vcvtq_f32_s32(vacc2xCDEF);

    vout2x0123 = vfmaq_f32(vout2x0123, vf2x0123, vfilter_output_scale0123);
    vout2x4567 = vfmaq_f32(vout2x4567, vf2x4567, vfilter_output_scale4567);
    vout2x89AB = vfmaq_f32(vout2x89AB, vf2x89AB, vfilter_output_scale89AB);
    vout2xCDEF = vfmaq_f32(vout2xCDEF, vf2xCDEF, vfilter_output_scaleCDEF);
    const float32x4_t vf3x0123 = vcvtq_f32_s32(vacc3x0123);
    const float32x4_t vf3x4567 = vcvtq_f32_s32(vacc3x4567);
    const float32x4_t vf3x89AB = vcvtq_f32_s32(vacc3x89AB);
    const float32x4_t vf3xCDEF = vcvtq_f32_s32(vacc3xCDEF);

    vout3x0123 = vfmaq_f32(vout3x0123, vf3x0123, vfilter_output_scale0123);
    vout3x4567 = vfmaq_f32(vout3x4567, vf3x4567, vfilter_output_scale4567);
    vout3x89AB = vfmaq_f32(vout3x89AB, vf3x89AB, vfilter_output_scale89AB);
    vout3xCDEF = vfmaq_f32(vout3xCDEF, vf3xCDEF, vfilter_output_scaleCDEF);
    const float32x4_t vf4x0123 = vcvtq_f32_s32(vacc4x0123);
    const float32x4_t vf4x4567 = vcvtq_f32_s32(vacc4x4567);
    const float32x4_t vf4x89AB = vcvtq_f32_s32(vacc4x89AB);
    const float32x4_t vf4xCDEF = vcvtq_f32_s32(vacc4xCDEF);

    vout4x0123 = vfmaq_f32(vout4x0123, vf4x0123, vfilter_output_scale0123);
    vout4x4567 = vfmaq_f32(vout4x4567, vf4x4567, vfilter_output_scale4567);
    vout4x89AB = vfmaq_f32(vout4x89AB, vf4x89AB, vfilter_output_scale89AB);
    vout4xCDEF = vfmaq_f32(vout4xCDEF, vf4xCDEF, vfilter_output_scaleCDEF);
    const float32x4_t vf5x0123 = vcvtq_f32_s32(vacc5x0123);
    const float32x4_t vf5x4567 = vcvtq_f32_s32(vacc5x4567);
    const float32x4_t vf5x89AB = vcvtq_f32_s32(vacc5x89AB);
    const float32x4_t vf5xCDEF = vcvtq_f32_s32(vacc5xCDEF);

    vout5x0123 = vfmaq_f32(vout5x0123, vf5x0123, vfilter_output_scale0123);
    vout5x4567 = vfmaq_f32(vout5x4567, vf5x4567, vfilter_output_scale4567);
    vout5x89AB = vfmaq_f32(vout5x89AB, vf5x89AB, vfilter_output_scale89AB);
    vout5xCDEF = vfmaq_f32(vout5xCDEF, vf5xCDEF, vfilter_output_scaleCDEF);
    }
    float32x4_t one_sixteenth = vdupq_n_f32(1/16.0);
    vout0x0123 = vmulq_f32(vout0x0123, one_sixteenth);
    vout0x4567 = vmulq_f32(vout0x4567, one_sixteenth);
    vout0x89AB = vmulq_f32(vout0x89AB, one_sixteenth);
    vout0xCDEF = vmulq_f32(vout0xCDEF, one_sixteenth);
    vout1x0123 = vmulq_f32(vout1x0123, one_sixteenth);
    vout1x4567 = vmulq_f32(vout1x4567, one_sixteenth);
    vout1x89AB = vmulq_f32(vout1x89AB, one_sixteenth);
    vout1xCDEF = vmulq_f32(vout1xCDEF, one_sixteenth);
    vout2x0123 = vmulq_f32(vout2x0123, one_sixteenth);
    vout2x4567 = vmulq_f32(vout2x4567, one_sixteenth);
    vout2x89AB = vmulq_f32(vout2x89AB, one_sixteenth);
    vout2xCDEF = vmulq_f32(vout2xCDEF, one_sixteenth);
    vout3x0123 = vmulq_f32(vout3x0123, one_sixteenth);
    vout3x4567 = vmulq_f32(vout3x4567, one_sixteenth);
    vout3x89AB = vmulq_f32(vout3x89AB, one_sixteenth);
    vout3xCDEF = vmulq_f32(vout3xCDEF, one_sixteenth);
    vout4x0123 = vmulq_f32(vout4x0123, one_sixteenth);
    vout4x4567 = vmulq_f32(vout4x4567, one_sixteenth);
    vout4x89AB = vmulq_f32(vout4x89AB, one_sixteenth);
    vout4xCDEF = vmulq_f32(vout4xCDEF, one_sixteenth);
    vout5x0123 = vmulq_f32(vout5x0123, one_sixteenth);
    vout5x4567 = vmulq_f32(vout5x4567, one_sixteenth);
    vout5x89AB = vmulq_f32(vout5x89AB, one_sixteenth);
    vout5xCDEF = vmulq_f32(vout5xCDEF, one_sixteenth);

    const float32x4_t vinput_scale01 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[0].zero_point));
    const float32x4_t vinput_scale23 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[2].zero_point));
    const float32x4_t vinput_scale45 = vreinterpretq_f32_s32(vld1q_s32(&quantization_params[4].zero_point));

    const float32x4_t vbias0123 = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0x0123 = vfmaq_lane_f32(vbias0123, vout0x0123, vget_low_f32(vinput_scale01), 1);
      vout1x0123 = vfmaq_lane_f32(vbias0123, vout1x0123, vget_high_f32(vinput_scale01), 1);
      vout2x0123 = vfmaq_lane_f32(vbias0123, vout2x0123, vget_low_f32(vinput_scale23), 1);
      vout3x0123 = vfmaq_lane_f32(vbias0123, vout3x0123, vget_high_f32(vinput_scale23), 1);
      vout4x0123 = vfmaq_lane_f32(vbias0123, vout4x0123, vget_low_f32(vinput_scale45), 1);
      vout5x0123 = vfmaq_lane_f32(vbias0123, vout5x0123, vget_high_f32(vinput_scale45), 1);
    #else
      vout0x0123 = vmlaq_lane_f32(vbias0123, vout0x0123, vget_low_f32(vinput_scale01), 1);
      vout1x0123 = vmlaq_lane_f32(vbias0123, vout1x0123, vget_high_f32(vinput_scale01), 1);
      vout2x0123 = vmlaq_lane_f32(vbias0123, vout2x0123, vget_low_f32(vinput_scale23), 1);
      vout3x0123 = vmlaq_lane_f32(vbias0123, vout3x0123, vget_high_f32(vinput_scale23), 1);
      vout4x0123 = vmlaq_lane_f32(vbias0123, vout4x0123, vget_low_f32(vinput_scale45), 1);
      vout5x0123 = vmlaq_lane_f32(vbias0123, vout5x0123, vget_high_f32(vinput_scale45), 1);
    #endif
    const float32x4_t vbias4567 = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0x4567 = vfmaq_lane_f32(vbias4567, vout0x4567, vget_low_f32(vinput_scale01), 1);
      vout1x4567 = vfmaq_lane_f32(vbias4567, vout1x4567, vget_high_f32(vinput_scale01), 1);
      vout2x4567 = vfmaq_lane_f32(vbias4567, vout2x4567, vget_low_f32(vinput_scale23), 1);
      vout3x4567 = vfmaq_lane_f32(vbias4567, vout3x4567, vget_high_f32(vinput_scale23), 1);
      vout4x4567 = vfmaq_lane_f32(vbias4567, vout4x4567, vget_low_f32(vinput_scale45), 1);
      vout5x4567 = vfmaq_lane_f32(vbias4567, vout5x4567, vget_high_f32(vinput_scale45), 1);
    #else
      vout0x4567 = vmlaq_lane_f32(vbias4567, vout0x4567, vget_low_f32(vinput_scale01), 1);
      vout1x4567 = vmlaq_lane_f32(vbias4567, vout1x4567, vget_high_f32(vinput_scale01), 1);
      vout2x4567 = vmlaq_lane_f32(vbias4567, vout2x4567, vget_low_f32(vinput_scale23), 1);
      vout3x4567 = vmlaq_lane_f32(vbias4567, vout3x4567, vget_high_f32(vinput_scale23), 1);
      vout4x4567 = vmlaq_lane_f32(vbias4567, vout4x4567, vget_low_f32(vinput_scale45), 1);
      vout5x4567 = vmlaq_lane_f32(vbias4567, vout5x4567, vget_high_f32(vinput_scale45), 1);
    #endif
    const float32x4_t vbias89AB = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0x89AB = vfmaq_lane_f32(vbias89AB, vout0x89AB, vget_low_f32(vinput_scale01), 1);
      vout1x89AB = vfmaq_lane_f32(vbias89AB, vout1x89AB, vget_high_f32(vinput_scale01), 1);
      vout2x89AB = vfmaq_lane_f32(vbias89AB, vout2x89AB, vget_low_f32(vinput_scale23), 1);
      vout3x89AB = vfmaq_lane_f32(vbias89AB, vout3x89AB, vget_high_f32(vinput_scale23), 1);
      vout4x89AB = vfmaq_lane_f32(vbias89AB, vout4x89AB, vget_low_f32(vinput_scale45), 1);
      vout5x89AB = vfmaq_lane_f32(vbias89AB, vout5x89AB, vget_high_f32(vinput_scale45), 1);
    #else
      vout0x89AB = vmlaq_lane_f32(vbias89AB, vout0x89AB, vget_low_f32(vinput_scale01), 1);
      vout1x89AB = vmlaq_lane_f32(vbias89AB, vout1x89AB, vget_high_f32(vinput_scale01), 1);
      vout2x89AB = vmlaq_lane_f32(vbias89AB, vout2x89AB, vget_low_f32(vinput_scale23), 1);
      vout3x89AB = vmlaq_lane_f32(vbias89AB, vout3x89AB, vget_high_f32(vinput_scale23), 1);
      vout4x89AB = vmlaq_lane_f32(vbias89AB, vout4x89AB, vget_low_f32(vinput_scale45), 1);
      vout5x89AB = vmlaq_lane_f32(vbias89AB, vout5x89AB, vget_high_f32(vinput_scale45), 1);
    #endif
    const float32x4_t vbiasCDEF = vld1q_f32(w); w = (const float*) w + 4;

    #if XNN_ARCH_ARM64
      vout0xCDEF = vfmaq_lane_f32(vbiasCDEF, vout0xCDEF, vget_low_f32(vinput_scale01), 1);
      vout1xCDEF = vfmaq_lane_f32(vbiasCDEF, vout1xCDEF, vget_high_f32(vinput_scale01), 1);
      vout2xCDEF = vfmaq_lane_f32(vbiasCDEF, vout2xCDEF, vget_low_f32(vinput_scale23), 1);
      vout3xCDEF = vfmaq_lane_f32(vbiasCDEF, vout3xCDEF, vget_high_f32(vinput_scale23), 1);
      vout4xCDEF = vfmaq_lane_f32(vbiasCDEF, vout4xCDEF, vget_low_f32(vinput_scale45), 1);
      vout5xCDEF = vfmaq_lane_f32(vbiasCDEF, vout5xCDEF, vget_high_f32(vinput_scale45), 1);
    #else
      vout0xCDEF = vmlaq_lane_f32(vbiasCDEF, vout0xCDEF, vget_low_f32(vinput_scale01), 1);
      vout1xCDEF = vmlaq_lane_f32(vbiasCDEF, vout1xCDEF, vget_high_f32(vinput_scale01), 1);
      vout2xCDEF = vmlaq_lane_f32(vbiasCDEF, vout2xCDEF, vget_low_f32(vinput_scale23), 1);
      vout3xCDEF = vmlaq_lane_f32(vbiasCDEF, vout3xCDEF, vget_high_f32(vinput_scale23), 1);
      vout4xCDEF = vmlaq_lane_f32(vbiasCDEF, vout4xCDEF, vget_low_f32(vinput_scale45), 1);
      vout5xCDEF = vmlaq_lane_f32(vbiasCDEF, vout5xCDEF, vget_high_f32(vinput_scale45), 1);
    #endif

    const float32x4_t voutput_min = vld1q_dup_f32(&params->scalar.min);
    vout0x0123 = vmaxq_f32(vout0x0123, voutput_min);
    vout0x4567 = vmaxq_f32(vout0x4567, voutput_min);
    vout0x89AB = vmaxq_f32(vout0x89AB, voutput_min);
    vout0xCDEF = vmaxq_f32(vout0xCDEF, voutput_min);
    vout1x0123 = vmaxq_f32(vout1x0123, voutput_min);
    vout1x4567 = vmaxq_f32(vout1x4567, voutput_min);
    vout1x89AB = vmaxq_f32(vout1x89AB, voutput_min);
    vout1xCDEF = vmaxq_f32(vout1xCDEF, voutput_min);
    vout2x0123 = vmaxq_f32(vout2x0123, voutput_min);
    vout2x4567 = vmaxq_f32(vout2x4567, voutput_min);
    vout2x89AB = vmaxq_f32(vout2x89AB, voutput_min);
    vout2xCDEF = vmaxq_f32(vout2xCDEF, voutput_min);
    vout3x0123 = vmaxq_f32(vout3x0123, voutput_min);
    vout3x4567 = vmaxq_f32(vout3x4567, voutput_min);
    vout3x89AB = vmaxq_f32(vout3x89AB, voutput_min);
    vout3xCDEF = vmaxq_f32(vout3xCDEF, voutput_min);
    vout4x0123 = vmaxq_f32(vout4x0123, voutput_min);
    vout4x4567 = vmaxq_f32(vout4x4567, voutput_min);
    vout4x89AB = vmaxq_f32(vout4x89AB, voutput_min);
    vout4xCDEF = vmaxq_f32(vout4xCDEF, voutput_min);
    vout5x0123 = vmaxq_f32(vout5x0123, voutput_min);
    vout5x4567 = vmaxq_f32(vout5x4567, voutput_min);
    vout5x89AB = vmaxq_f32(vout5x89AB, voutput_min);
    vout5xCDEF = vmaxq_f32(vout5xCDEF, voutput_min);

    const float32x4_t voutput_max = vld1q_dup_f32(&params->scalar.max);
    vout0x0123 = vminq_f32(vout0x0123, voutput_max);
    vout0x4567 = vminq_f32(vout0x4567, voutput_max);
    vout0x89AB = vminq_f32(vout0x89AB, voutput_max);
    vout0xCDEF = vminq_f32(vout0xCDEF, voutput_max);
    vout1x0123 = vminq_f32(vout1x0123, voutput_max);
    vout1x4567 = vminq_f32(vout1x4567, voutput_max);
    vout1x89AB = vminq_f32(vout1x89AB, voutput_max);
    vout1xCDEF = vminq_f32(vout1xCDEF, voutput_max);
    vout2x0123 = vminq_f32(vout2x0123, voutput_max);
    vout2x4567 = vminq_f32(vout2x4567, voutput_max);
    vout2x89AB = vminq_f32(vout2x89AB, voutput_max);
    vout2xCDEF = vminq_f32(vout2xCDEF, voutput_max);
    vout3x0123 = vminq_f32(vout3x0123, voutput_max);
    vout3x4567 = vminq_f32(vout3x4567, voutput_max);
    vout3x89AB = vminq_f32(vout3x89AB, voutput_max);
    vout3xCDEF = vminq_f32(vout3xCDEF, voutput_max);
    vout4x0123 = vminq_f32(vout4x0123, voutput_max);
    vout4x4567 = vminq_f32(vout4x4567, voutput_max);
    vout4x89AB = vminq_f32(vout4x89AB, voutput_max);
    vout4xCDEF = vminq_f32(vout4xCDEF, voutput_max);
    vout5x0123 = vminq_f32(vout5x0123, voutput_max);
    vout5x4567 = vminq_f32(vout5x4567, voutput_max);
    vout5x89AB = vminq_f32(vout5x89AB, voutput_max);
    vout5xCDEF = vminq_f32(vout5xCDEF, voutput_max);

    if XNN_LIKELY(nc >= 16) {
      vst1q_f32(&c0[0], vout0x0123);
      vst1q_f32(&c0[4], vout0x4567);
      vst1q_f32(&c0[8], vout0x89AB);
      vst1q_f32(&c0[12], vout0xCDEF);
      vst1q_f32(&c1[0], vout1x0123);
      vst1q_f32(&c1[4], vout1x4567);
      vst1q_f32(&c1[8], vout1x89AB);
      vst1q_f32(&c1[12], vout1xCDEF);
      vst1q_f32(&c2[0], vout2x0123);
      vst1q_f32(&c2[4], vout2x4567);
      vst1q_f32(&c2[8], vout2x89AB);
      vst1q_f32(&c2[12], vout2xCDEF);
      vst1q_f32(&c3[0], vout3x0123);
      vst1q_f32(&c3[4], vout3x4567);
      vst1q_f32(&c3[8], vout3x89AB);
      vst1q_f32(&c3[12], vout3xCDEF);
      vst1q_f32(&c4[0], vout4x0123);
      vst1q_f32(&c4[4], vout4x4567);
      vst1q_f32(&c4[8], vout4x89AB);
      vst1q_f32(&c4[12], vout4xCDEF);
      vst1q_f32(&c5[0], vout5x0123);
      vst1q_f32(&c5[4], vout5x4567);
      vst1q_f32(&c5[8], vout5x89AB);
      vst1q_f32(&c5[12], vout5xCDEF);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);
      a4 = (const int8_t*) ((uintptr_t) a4 - kc);
      a5 = (const int8_t*) ((uintptr_t) a5 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      c5 = (float*) ((uintptr_t) c5 + cn_stride);

      nc -= 16;
    } else {
      if (nc & 8) {
        vst1q_f32(c0, vout0x0123); c0 += 4;
        vout0x0123 = vout0x89AB;
        vst1q_f32(c1, vout1x0123); c1 += 4;
        vout1x0123 = vout1x89AB;
        vst1q_f32(c2, vout2x0123); c2 += 4;
        vout2x0123 = vout2x89AB;
        vst1q_f32(c3, vout3x0123); c3 += 4;
        vout3x0123 = vout3x89AB;
        vst1q_f32(c4, vout4x0123); c4 += 4;
        vout4x0123 = vout4x89AB;
        vst1q_f32(c5, vout5x0123); c5 += 4;
        vout5x0123 = vout5x89AB;
        vst1q_f32(c0, vout0x4567); c0 += 4;
        vout0x4567 = vout0xCDEF;
        vst1q_f32(c1, vout1x4567); c1 += 4;
        vout1x4567 = vout1xCDEF;
        vst1q_f32(c2, vout2x4567); c2 += 4;
        vout2x4567 = vout2xCDEF;
        vst1q_f32(c3, vout3x4567); c3 += 4;
        vout3x4567 = vout3xCDEF;
        vst1q_f32(c4, vout4x4567); c4 += 4;
        vout4x4567 = vout4xCDEF;
        vst1q_f32(c5, vout5x4567); c5 += 4;
        vout5x4567 = vout5xCDEF;
      }
      if (nc & 4) {
        vst1q_f32(c0, vout0x0123); c0 += 4;
        vout0x0123 = vout0x4567;
        vst1q_f32(c1, vout1x0123); c1 += 4;
        vout1x0123 = vout1x4567;
        vst1q_f32(c2, vout2x0123); c2 += 4;
        vout2x0123 = vout2x4567;
        vst1q_f32(c3, vout3x0123); c3 += 4;
        vout3x0123 = vout3x4567;
        vst1q_f32(c4, vout4x0123); c4 += 4;
        vout4x0123 = vout4x4567;
        vst1q_f32(c5, vout5x0123); c5 += 4;
        vout5x0123 = vout5x4567;
      }
      float32x2_t vout0x01 = vget_low_f32(vout0x0123);
      float32x2_t vout1x01 = vget_low_f32(vout1x0123);
      float32x2_t vout2x01 = vget_low_f32(vout2x0123);
      float32x2_t vout3x01 = vget_low_f32(vout3x0123);
      float32x2_t vout4x01 = vget_low_f32(vout4x0123);
      float32x2_t vout5x01 = vget_low_f32(vout5x0123);
      if (nc & 2) {
        vst1_f32(c0, vout0x01); c0 += 2;
        vst1_f32(c1, vout1x01); c1 += 2;
        vst1_f32(c2, vout2x01); c2 += 2;
        vst1_f32(c3, vout3x01); c3 += 2;
        vst1_f32(c4, vout4x01); c4 += 2;
        vst1_f32(c5, vout5x01); c5 += 2;
        vout0x01 = vget_high_f32(vout0x0123);
        vout1x01 = vget_high_f32(vout1x0123);
        vout2x01 = vget_high_f32(vout2x0123);
        vout3x01 = vget_high_f32(vout3x0123);
        vout4x01 = vget_high_f32(vout4x0123);
        vout5x01 = vget_high_f32(vout5x0123);
      }
      if (nc & 1) {
        vst1_lane_f32(c0, vout0x01, 0);
        vst1_lane_f32(c1, vout1x01, 0);
        vst1_lane_f32(c2, vout2x01, 0);
        vst1_lane_f32(c3, vout3x01, 0);
        vst1_lane_f32(c4, vout4x01, 0);
        vst1_lane_f32(c5, vout5x01, 0);
      }
      nc = 0;
    }
  } while (nc != 0);
}
