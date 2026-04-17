// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-bf16-vcvt/sse41.c.in
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


void xnn_f32_bf16_vcvt_ukernel__sse41_u8(
    size_t batch,
    const float* input,
    xnn_bfloat16* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vbias = _mm_set1_epi32(0x7FFFu);
  const __m128i vone = _mm_set1_epi32(1u);
  const __m128i vabs_mask = _mm_set1_epi32(0x7FFFFFFFu);
  const __m128i vexp_mask = _mm_set1_epi32(0x7F800000u);
  const __m128i vquiet = _mm_set1_epi32(0x00400000u);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m128i vi0 = _mm_castps_si128(_mm_loadu_ps(input + 0));
    __m128i vi1 = _mm_castps_si128(_mm_loadu_ps(input + 4));
    input += 8;

    const __m128i vlsb0 = _mm_and_si128(_mm_srli_epi32(vi0, 16), vone);
    const __m128i vlsb1 = _mm_and_si128(_mm_srli_epi32(vi1, 16), vone);

    const __m128i vrounded0 = _mm_add_epi32(_mm_add_epi32(vi0, vbias), vlsb0);
    const __m128i vrounded1 = _mm_add_epi32(_mm_add_epi32(vi1, vbias), vlsb1);

    const __m128i vabsi0 = _mm_and_si128(vi0, vabs_mask);
    const __m128i vabsi1 = _mm_and_si128(vi1, vabs_mask);

    const __m128i vnanmask0 = _mm_cmpgt_epi32(vabsi0, vexp_mask);
    const __m128i vnanmask1 = _mm_cmpgt_epi32(vabsi1, vexp_mask);

    vi0 = _mm_blendv_epi8(vrounded0, _mm_or_si128(vi0, vquiet), vnanmask0);
    vi1 = _mm_blendv_epi8(vrounded1, _mm_or_si128(vi1, vquiet), vnanmask1);

    const __m128i vbf0 = _mm_srli_epi32(vi0, 16);
    const __m128i vbf1 = _mm_srli_epi32(vi1, 16);

    const __m128i vpacked0 = _mm_packus_epi32(vbf0, vbf1);
    _mm_storeu_si128((__m128i*) (o + 0), vpacked0);
    o += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    __m128i vi = _mm_castps_si128(_mm_loadu_ps(input));
    input += 4;

    const __m128i vlsb = _mm_and_si128(_mm_srli_epi32(vi, 16), vone);
    const __m128i vrounded = _mm_add_epi32(_mm_add_epi32(vi, vbias), vlsb);
    const __m128i vabsi = _mm_and_si128(vi, vabs_mask);
    const __m128i vnanmask = _mm_cmpgt_epi32(vabsi, vexp_mask);
    vi = _mm_blendv_epi8(vrounded, _mm_or_si128(vi, vquiet), vnanmask);
    const __m128i vbf = _mm_srli_epi32(vi, 16);
    const __m128i vpacked = _mm_packus_epi32(vbf, vbf);

    _mm_storel_epi64((__m128i*) o, vpacked);
    o += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));
    __m128i vi = _mm_castps_si128(_mm_loadu_ps(input));

    const __m128i vlsb = _mm_and_si128(_mm_srli_epi32(vi, 16), vone);
    const __m128i vrounded = _mm_add_epi32(_mm_add_epi32(vi, vbias), vlsb);
    const __m128i vabsi = _mm_and_si128(vi, vabs_mask);
    const __m128i vnanmask = _mm_cmpgt_epi32(vabsi, vexp_mask);
    vi = _mm_blendv_epi8(vrounded, _mm_or_si128(vi, vquiet), vnanmask);
    const __m128i vbf = _mm_srli_epi32(vi, 16);
    const __m128i vpacked = _mm_packus_epi32(vbf, vbf);

    if (batch & (2 * sizeof(float))) {
      _mm_storeu_si32(o, vpacked);
      o += 2;
      // Shift remaining data to low 32 bits.
      __m128i vremaining = _mm_srli_si128(vpacked, 4);
      if (batch & (1 * sizeof(float))) {
        *o = (uint16_t) _mm_extract_epi16(vremaining, 0);
      }
    } else if (batch & (1 * sizeof(float))) {
      *o = (uint16_t) _mm_extract_epi16(vpacked, 0);
    }
  }
}
