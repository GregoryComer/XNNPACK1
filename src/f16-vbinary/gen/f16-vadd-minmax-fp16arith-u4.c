// Auto-generated file. Do not edit!
//   Template: src/f16-vbinary/vop-fp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <string.h>

#include <arm_fp16.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vbinary.h"


void xnn_f16_vadd_minmax_ukernel__fp16arith_u4(
    size_t batch,
    const xnn_float16* restrict input_a,
    const xnn_float16* restrict input_b,
    xnn_float16* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float16_t* a = (const float16_t*) input_a;
  const float16_t* b = (const float16_t*) input_b;
  float16_t* o = (float16_t*) output;

  float16_t vy_min, vy_max;
  memcpy(&vy_min, &params->scalar.min, sizeof(vy_min));
  memcpy(&vy_max, &params->scalar.max, sizeof(vy_max));

  for (; batch >= 4 * sizeof(float16_t); batch -= 4 * sizeof(float16_t)) {
    const float16_t va0 = *a++;
    const float16_t va1 = *a++;
    const float16_t va2 = *a++;
    const float16_t va3 = *a++;

    const float16_t vb0 = *b++;
    const float16_t vb1 = *b++;
    const float16_t vb2 = *b++;
    const float16_t vb3 = *b++;

    float16_t vacc0 = vaddh_f16(va0, vb0);
    float16_t vacc1 = vaddh_f16(va1, vb1);
    float16_t vacc2 = vaddh_f16(va2, vb2);
    float16_t vacc3 = vaddh_f16(va3, vb3);


    vacc0 = vmaxnmh_f16(vacc0, vy_min);
    vacc1 = vmaxnmh_f16(vacc1, vy_min);
    vacc2 = vmaxnmh_f16(vacc2, vy_min);
    vacc3 = vmaxnmh_f16(vacc3, vy_min);

    vacc0 = vminnmh_f16(vacc0, vy_max);
    vacc1 = vminnmh_f16(vacc1, vy_max);
    vacc2 = vminnmh_f16(vacc2, vy_max);
    vacc3 = vminnmh_f16(vacc3, vy_max);

    *o++ = vacc0;
    *o++ = vacc1;
    *o++ = vacc2;
    *o++ = vacc3;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float16_t va = *a++;
      const float16_t vb = *b++;
      float16_t vacc = vaddh_f16(va, vb);
      vacc = vmaxnmh_f16(vacc, vy_min);
      vacc = vminnmh_f16(vacc, vy_max);
      *o++ = vacc;
      batch -= sizeof(float16_t);
    } while (batch != 0);
  }
}
