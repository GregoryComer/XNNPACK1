// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#if XNN_ENABLE_CPUINFO
  #include <cpuinfo.h>
#endif  // XNN_ENABLE_CPUINFO
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>
#include <xnnpack/math.h>

#include <fp16/fp16.h>
#include "bench/utils.h"
#include <benchmark/benchmark.h>
#include "pthreadpool.h"

struct ComputeErrorContext {
  const uint16_t* input;
  const uint16_t* output;
  float* error;
};

static void ComputeError(
  struct ComputeErrorContext* context,
  size_t start,
  size_t range)
{
  const uint16_t* input = context->input;
  const uint16_t* output = context->output;
  float* error = context->error;
  for (size_t i = start; i < start + range; i++) {
    const float output_ref = std::sqrt(fp16_ieee_to_fp32_value(input[i]));
    const float abs_error = std::abs(output_ref - fp16_ieee_to_fp32_value(output[i]));
    const float output_abs = std::abs(output_ref);
    const float output_ulp = (uint32_as_float(float_as_uint32(output_abs) + 1) - output_abs);
    error[i] = float(abs_error / (output_ulp * 8192.0f /* difference between HP and SP ULP */));
  }
}

static void SqrtError(benchmark::State& state,
  xnn_f16_unary_math_fn sqrt,
  benchmark::utils::IsaCheckFunction isa_check = nullptr)
{
  if (isa_check != nullptr && !isa_check(state)) {
    return;
  }

  const uint16_t min_input = UINT16_C(0x3C00);
  const uint16_t max_input = UINT16_C(0x4400);
  // Number of elements in one block of inputs/outputs.
  // Combining multiple elements in a block reduce function call overhead.
  const size_t block_size = 2048;
  // Number of elements in one parallelization tile. Worker threads process this many elements in each task.
  const size_t tile_size = 64;

  // Default: as many as logical processors in the system
  size_t num_threads = 0;
  #if XNN_ENABLE_CPUINFO
    if (cpuinfo_initialize()) {
      num_threads = cpuinfo_get_processors_count();
      #if XNN_ARCH_ARM || XNN_ARCH_ARM64
        // Use all cores except for the least performant cluster
        if (cpuinfo_get_clusters_count() > 1) {
          num_threads -= cpuinfo_get_cluster(cpuinfo_get_clusters_count() - 1)->core_count;
        }
      #endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
    }
  #endif  // XNN_ENABLE_CPUINFO

  std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> threadpool(
    pthreadpool_create(num_threads), pthreadpool_destroy);

  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> x(block_size);
  std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> y(block_size);
  std::vector<float> ulp_error(block_size);
  float max_ulp_error = 0.0f;

  ComputeErrorContext context;
  context.input = x.data();
  context.output = y.data();
  context.error = ulp_error.data();
  for (auto _ : state) {
    for (uint16_t n = min_input; n < max_input; n += block_size) {
      for (uint16_t i = 0; i < block_size; i++) {
        x[i] = std::min<uint16_t>(n + i, max_input);
      }
      std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);

      sqrt(block_size * sizeof(uint16_t), x.data(), y.data());

      pthreadpool_parallelize_1d_tile_1d(
          threadpool.get(),
          reinterpret_cast<pthreadpool_task_1d_tile_1d_t>(ComputeError),
          static_cast<void*>(&context),
          block_size, tile_size, 0 /* flags */);

      max_ulp_error = std::accumulate(ulp_error.cbegin(), ulp_error.cend(), max_ulp_error,
        static_cast<const float& (*)(const float&, const float&)>(std::max<float>));
    }
  }

  state.counters["ULPERROR"] = benchmark::Counter(max_ulp_error);
}

#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  BENCHMARK_CAPTURE(SqrtError, neonfp16arith_nr1fma,
                    xnn_math_f16_sqrt__neonfp16arith_nr1fma,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neonfp16arith_nr1fma1adj,
                    xnn_math_f16_sqrt__neonfp16arith_nr1fma1adj,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
  BENCHMARK_CAPTURE(SqrtError, neonfp16arith_nr1rsqrts,
                    xnn_math_f16_sqrt__neonfp16arith_nr1rsqrts,
                    benchmark::utils::CheckNEONFP16ARITH)
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1);
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)

#ifndef XNNPACK_BENCHMARK_NO_MAIN
BENCHMARK_MAIN();
#endif
