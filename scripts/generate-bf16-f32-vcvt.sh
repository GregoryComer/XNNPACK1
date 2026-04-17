#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### SSE4.1 ###################################
tools/xngen src/bf16-f32-vcvt/sse41.c.in -D BATCH_TILE=4  -o src/bf16-f32-vcvt/gen/bf16-f32-vcvt-sse41-u4.c &
tools/xngen src/bf16-f32-vcvt/sse41.c.in -D BATCH_TILE=8  -o src/bf16-f32-vcvt/gen/bf16-f32-vcvt-sse41-u8.c &

##################################### AVX2 ####################################
tools/xngen src/bf16-f32-vcvt/avx2.c.in -D BATCH_TILE=8  -o src/bf16-f32-vcvt/gen/bf16-f32-vcvt-avx2-u8.c &
tools/xngen src/bf16-f32-vcvt/avx2.c.in -D BATCH_TILE=16 -o src/bf16-f32-vcvt/gen/bf16-f32-vcvt-avx2-u16.c &

################################## AVX512SKX ##################################
tools/xngen src/bf16-f32-vcvt/avx512skx.c.in -D BATCH_TILE=16 -o src/bf16-f32-vcvt/gen/bf16-f32-vcvt-avx512skx-u16.c &
tools/xngen src/bf16-f32-vcvt/avx512skx.c.in -D BATCH_TILE=32 -o src/bf16-f32-vcvt/gen/bf16-f32-vcvt-avx512skx-u32.c &

################################### ARM NEON ##################################
tools/xngen src/bf16-f32-vcvt/neon.c.in -D BATCH_TILE=8  -o src/bf16-f32-vcvt/gen/bf16-f32-vcvt-neon-u8.c &
tools/xngen src/bf16-f32-vcvt/neon.c.in -D BATCH_TILE=16 -o src/bf16-f32-vcvt/gen/bf16-f32-vcvt-neon-u16.c &
tools/xngen src/bf16-f32-vcvt/neon.c.in -D BATCH_TILE=24 -o src/bf16-f32-vcvt/gen/bf16-f32-vcvt-neon-u24.c &
tools/xngen src/bf16-f32-vcvt/neon.c.in -D BATCH_TILE=32 -o src/bf16-f32-vcvt/gen/bf16-f32-vcvt-neon-u32.c &

#################################### Scalar ###################################
tools/xngen src/bf16-f32-vcvt/scalar.c.in -D BATCH_TILE=1 -o src/bf16-f32-vcvt/gen/bf16-f32-vcvt-scalar-u1.c &
tools/xngen src/bf16-f32-vcvt/scalar.c.in -D BATCH_TILE=2 -o src/bf16-f32-vcvt/gen/bf16-f32-vcvt-scalar-u2.c &
tools/xngen src/bf16-f32-vcvt/scalar.c.in -D BATCH_TILE=3 -o src/bf16-f32-vcvt/gen/bf16-f32-vcvt-scalar-u3.c &
tools/xngen src/bf16-f32-vcvt/scalar.c.in -D BATCH_TILE=4 -o src/bf16-f32-vcvt/gen/bf16-f32-vcvt-scalar-u4.c &

wait
