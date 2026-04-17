// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-bf16-vcvt/avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vcvt.h"


void xnn_f32_bf16_vcvt_ukernel__avx2_u16(
    size_t batch,
    const float* input,
    xnn_bfloat16* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256i vbias = _mm256_set1_epi32(0x7FFFu);
  const __m256i vone = _mm256_set1_epi32(1u);
  const __m256i vabs_mask = _mm256_set1_epi32(0x7FFFFFFFu);
  const __m256i vexp_mask = _mm256_set1_epi32(0x7F800000u);
  const __m256i vquiet = _mm256_set1_epi32(0x00400000u);
  const __m256i vperm = _mm256_setr_epi8(
    2, 3, 6, 7, 10, 11, 14, 15,
    -1, -1, -1, -1, -1, -1, -1, -1,
    2, 3, 6, 7, 10, 11, 14, 15,
    -1, -1, -1, -1, -1, -1, -1, -1);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m256i vi0 = _mm256_castps_si256(_mm256_loadu_ps(input + 0));
    __m256i vi1 = _mm256_castps_si256(_mm256_loadu_ps(input + 8));
    input += 16;

    const __m256i vlsb0 = _mm256_and_si256(_mm256_srli_epi32(vi0, 16), vone);
    const __m256i vlsb1 = _mm256_and_si256(_mm256_srli_epi32(vi1, 16), vone);

    const __m256i vrounded0 = _mm256_add_epi32(_mm256_add_epi32(vi0, vbias), vlsb0);
    const __m256i vrounded1 = _mm256_add_epi32(_mm256_add_epi32(vi1, vbias), vlsb1);

    const __m256i vabsi0 = _mm256_and_si256(vi0, vabs_mask);
    const __m256i vabsi1 = _mm256_and_si256(vi1, vabs_mask);

    const __m256i vnanmask0 = _mm256_cmpgt_epi32(vabsi0, vexp_mask);
    const __m256i vnanmask1 = _mm256_cmpgt_epi32(vabsi1, vexp_mask);

    vi0 = _mm256_blendv_epi8(vrounded0, _mm256_or_si256(vi0, vquiet), vnanmask0);
    vi1 = _mm256_blendv_epi8(vrounded1, _mm256_or_si256(vi1, vquiet), vnanmask1);

    // Shuffle to extract upper 16 bits of each 32-bit element, packed into
    // the low 64 bits of each 128-bit lane.
    const __m256i vshuf0 = _mm256_shuffle_epi8(vi0, vperm);
    const __m256i vshuf1 = _mm256_shuffle_epi8(vi1, vperm);

    // Combine low 64 bits of each lane into a single 128-bit result.
    const __m128i vbf0 = _mm_unpacklo_epi64(_mm256_castsi256_si128(vshuf0), _mm256_extracti128_si256(vshuf0, 1));
    const __m128i vbf1 = _mm_unpacklo_epi64(_mm256_castsi256_si128(vshuf1), _mm256_extracti128_si256(vshuf1, 1));

    _mm_storeu_si128((__m128i*) (o + 0), vbf0);
    _mm_storeu_si128((__m128i*) (o + 8), vbf1);
    o += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    __m256i vi = _mm256_castps_si256(_mm256_loadu_ps(input));
    input += 8;

    const __m256i vlsb = _mm256_and_si256(_mm256_srli_epi32(vi, 16), vone);
    const __m256i vrounded = _mm256_add_epi32(_mm256_add_epi32(vi, vbias), vlsb);
    const __m256i vabsi = _mm256_and_si256(vi, vabs_mask);
    const __m256i vnanmask = _mm256_cmpgt_epi32(vabsi, vexp_mask);
    vi = _mm256_blendv_epi8(vrounded, _mm256_or_si256(vi, vquiet), vnanmask);
    const __m256i vshuf = _mm256_shuffle_epi8(vi, vperm);
    const __m128i vbf = _mm_unpacklo_epi64(_mm256_castsi256_si128(vshuf), _mm256_extracti128_si256(vshuf, 1));

    _mm_storeu_si128((__m128i*) o, vbf);
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    __m256i vi = _mm256_castps_si256(_mm256_loadu_ps(input));

    const __m256i vlsb = _mm256_and_si256(_mm256_srli_epi32(vi, 16), vone);
    const __m256i vrounded = _mm256_add_epi32(_mm256_add_epi32(vi, vbias), vlsb);
    const __m256i vabsi = _mm256_and_si256(vi, vabs_mask);
    const __m256i vnanmask = _mm256_cmpgt_epi32(vabsi, vexp_mask);
    vi = _mm256_blendv_epi8(vrounded, _mm256_or_si256(vi, vquiet), vnanmask);
    const __m256i vshuf = _mm256_shuffle_epi8(vi, vperm);
    const __m128i vbf = _mm_unpacklo_epi64(_mm256_castsi256_si128(vshuf), _mm256_extracti128_si256(vshuf, 1));

    if (batch & (4 * sizeof(float))) {
      _mm_storel_epi64((__m128i*) o, vbf);
      o += 4;
      __m128i vremaining = _mm_unpackhi_epi64(vbf, vbf);
      if (batch & (2 * sizeof(float))) {
        _mm_storeu_si32(o, vremaining);
        o += 2;
        vremaining = _mm_srli_si128(vremaining, 4);
      }
      if (batch & (1 * sizeof(float))) {
        *o = (uint16_t) _mm_extract_epi16(vremaining, 0);
      }
    } else {
      if (batch & (2 * sizeof(float))) {
        _mm_storeu_si32(o, vbf);
        o += 2;
        __m128i vremaining = _mm_srli_si128(vbf, 4);
        if (batch & (1 * sizeof(float))) {
          *o = (uint16_t) _mm_extract_epi16(vremaining, 0);
        }
      } else if (batch & (1 * sizeof(float))) {
        *o = (uint16_t) _mm_extract_epi16(vbf, 0);
      }
    }
  }
}
