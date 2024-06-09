// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/qd8-f16-qb4w-gemm-minmax.yaml
//   Generator: tools/generate-gemm-test.py

#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/gemm.h>
#include <xnnpack/igemm.h>
#include <xnnpack/isa-checks.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/pack.h>
#include <xnnpack/ppmm.h>
#include <xnnpack/requantization.h>

#include "gemm-microkernel-tester.h"
#include <gtest/gtest.h>

namespace {

std::vector<GemmTestParams> CreateTests1(
    size_t k_block, size_t adj_k_block,
    size_t mr, size_t nr, size_t kr, size_t sr,
    bool is_igemm,
    std::function<void(GemmMicrokernelTester& tester)> test_func,
    std::function<void()> isa_check = nullptr) {
  std::string kbs = std::to_string(k_block);
  std::string kb2s = std::to_string(k_block * 2);
  std::string akbs = std::to_string(adj_k_block);
  std::string nrs = std::to_string(nr);

  std::vector<GemmTestParams> gemm_tests;
  gemm_tests.reserve(42);

  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs,
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
          .b_zero_point(8)
          .bl(kr * sr * 2)
      , test_func, isa_check));
  gemm_tests.push_back(GemmTestParams(
      "strided_cn",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
          .cn_stride(NextPrime(nr + 1))
          .b_zero_point(8)
          .bl(kr * sr * 2)
    , test_func, isa_check));
  if (!is_igemm) {
    gemm_tests.push_back(GemmTestParams(
        "k_eq_" + kbs + "_strided_a",
        GemmMicrokernelTester()
            .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block)
            .a_stride(NextPrime(k_block + 1))
            .b_zero_point(8)
            .bl(kr * sr * 2)
        , test_func, isa_check));
  }
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).k(k_block).iterations(1)
          .b_zero_point(8)
          .bl(kr * sr * 2)
      , test_func, isa_check)
      .loop_n(1, nr)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_m",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).n(nr).k(k_block).iterations(1)
          .b_zero_point(8)
          .bl(kr * sr * 2)
      , test_func, isa_check)
      .loop_m(1, mr));
  gemm_tests.push_back(GemmTestParams(
      "k_eq_" + kbs + "_subtile_n",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).k(k_block).iterations(1)
          .b_zero_point(8)
          .bl(kr * sr * 2)
      , test_func, isa_check)
      .loop_n(1, nr));
  gemm_tests.push_back(GemmTestParams(
      "bl",
      GemmMicrokernelTester()
          .mr(mr).nr(nr).kr(kr).sr(sr).m(mr).n(nr).k(k_block * 12)
          .b_zero_point(8)
      , test_func, isa_check)
      .loop_k(k_block, k_block * 12, k_block, LoopStepType::Linear)
      .loop_bl(2 * kr * sr, k_block * 12, 2 * kr * sr));

  return gemm_tests;
}

}  // namespace


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_1X2__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/1, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x2__scalar,
                      xnn_init_f16_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_1X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/1, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x4__scalar,
                      xnn_init_f16_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_1X8__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/1, /*nr=*/8, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8__scalar,
                      xnn_init_f16_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_2X2__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/2, /*nr=*/2, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x2__scalar,
                      xnn_init_f16_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_2X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/2, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x4__scalar,
                      xnn_init_f16_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_2X8__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/2, /*nr=*/8, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8__scalar,
                      xnn_init_f16_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


INSTANTIATE_TEST_SUITE_P(
    QD8_F16_QB4W_GEMM_MINMAX_4X4__SCALAR, GemmTest,
    testing::ValuesIn(CreateTests1(
        /*k_block=*/2,
        /*adj_k_block=*/2,
        /*mr=*/4, /*nr=*/4, /*kr=*/1, /*sr=*/1,
        /*is_igemm=*/false,
        [](GemmMicrokernelTester& tester) {
          tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x4__scalar,
                      xnn_init_f16_qb4w_minmax_scalar_params,
                      xnn_pack_qs8_qb4w_gemm_goi_w);
        })),
    [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
      return info.param.test_name;
    });


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_1X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/1, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_1x8c8__avx2,
                        xnn_init_f16_qb4w_minmax_avx_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_2X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/2, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_2x8c8__avx2,
                        xnn_init_f16_qb4w_minmax_avx_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_3X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/3, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_3x8c8__avx2,
                        xnn_init_f16_qb4w_minmax_avx_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  INSTANTIATE_TEST_SUITE_P(
      QD8_F16_QB4W_GEMM_MINMAX_4X8C8__AVX2, GemmTest,
      testing::ValuesIn(CreateTests1(
          /*k_block=*/16,
          /*adj_k_block=*/16,
          /*mr=*/4, /*nr=*/8, /*kr=*/8, /*sr=*/1,
          /*is_igemm=*/false,
          [](GemmMicrokernelTester& tester) {
            tester.Test(xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x8c8__avx2,
                        xnn_init_f16_qb4w_minmax_avx_params,
                        xnn_pack_qs8_qb4w_gemm_goi_w);
          },
          []() {
            TEST_REQUIRES_X86_AVX2;
          })),
      [](const testing::TestParamInfo<GemmTest::ParamType>& info) {
        return info.param.test_name;
      });
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64