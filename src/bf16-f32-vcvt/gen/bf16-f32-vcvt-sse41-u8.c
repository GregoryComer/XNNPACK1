// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-vcvt/sse41.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <smmintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vcvt.h"


void xnn_bf16_f32_vcvt_ukernel__sse41_u8(
    size_t batch,
    const xnn_bfloat16* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  for (; batch >= 8 * sizeof(xnn_bfloat16); batch -= 8 * sizeof(xnn_bfloat16)) {
    const __m128i vbf0 = _mm_loadu_si64(i + 0);
    const __m128i vbf1 = _mm_loadu_si64(i + 4);

    const __m128 vf0 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(vbf0), 16));
    const __m128 vf1 = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(vbf1), 16));

    _mm_storeu_ps(output + 0, vf0);
    _mm_storeu_ps(output + 4, vf1);
    i += 8;
    output += 8;
  }
  for (; batch >= 4 * sizeof(xnn_bfloat16); batch -= 4 * sizeof(xnn_bfloat16)) {
    const __m128i vbf = _mm_loadu_si64(i);
    i += 4;

    const __m128 vf = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(vbf), 16));

    _mm_storeu_ps(output, vf);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(xnn_bfloat16));
    assert(batch <= 3 * sizeof(xnn_bfloat16));
    const __m128i vbf = _mm_loadu_si64(i);

    __m128 vf = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(vbf), 16));
    if (batch & (2 * sizeof(xnn_bfloat16))) {
      _mm_storel_pi((__m64*) output, vf);
      vf = _mm_movehl_ps(vf, vf);
      output += 2;
    }
    if (batch & (1 * sizeof(xnn_bfloat16))) {
      _mm_store_ss(output, vf);
    }
  }
}
