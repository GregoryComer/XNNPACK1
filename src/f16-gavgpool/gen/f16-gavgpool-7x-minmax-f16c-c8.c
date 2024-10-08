// Auto-generated file. Do not edit!
//   Template: src/f16-gavgpool/unipass-f16c.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/gavgpool.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8(
    size_t rows,
    size_t channels,
    const xnn_float16* input,
    size_t input_stride,
    const xnn_float16* zero,
    xnn_float16* output,
    const struct xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const uint16_t* i0 = (const uint16_t*) input;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = (const uint16_t*) zero;
  }
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = (const uint16_t*) zero;
  }
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = (const uint16_t*) zero;
  }
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = (const uint16_t*) zero;
  }
  const uint16_t* i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = (const uint16_t*) zero;
  }
  const uint16_t* i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = (const uint16_t*) zero;
  }
  uint16_t* o = (uint16_t*) output;

  const __m256 vscale = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.scale));
  const __m256 vmin = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m256 vmax = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.max));
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  for (; channels >= 8; channels -= 8) {
    const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
    i0 += 8;
    const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
    i1 += 8;

    const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
    __m128i vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(vi0x01234567, vi1x01234567), _MM_FROUND_TO_NEAREST_INT);
    i2 += 8;

    const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
    i3 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
    i4 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
    i5 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
    i6 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);

    vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vscale), _MM_FROUND_TO_NEAREST_INT);

    __m256 vout01234567 = _mm256_max_ps(_mm256_cvtph_ps(vacc01234567), vmin);

    vout01234567 = _mm256_min_ps(vout01234567, vmax);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vout01234567, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      __m128i vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(vi0x01234567, vi1x01234567), _MM_FROUND_TO_NEAREST_INT);

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);

      vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vscale), _MM_FROUND_TO_NEAREST_INT);
      __m256 vout01234567 = _mm256_max_ps(_mm256_cvtph_ps(vacc01234567), vmin);
      vout01234567 = _mm256_min_ps(vout01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vout01234567, _MM_FROUND_TO_NEAREST_INT);
      if (channels & 4) {
        _mm_storel_epi64((__m128i*) o, vh01234567);
        o += 4;
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
      }
      if (channels & 2) {
        _mm_storeu_si32(o, vh01234567);
        o += 2;
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
      }
      if (channels & 1) {
        *o = (uint16_t) _mm_extract_epi16(vh01234567, 0);
      }
    }
  }
}
