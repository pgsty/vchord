// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License v3 (AGPLv3).
// See <https://www.gnu.org/licenses/agpl-3.0.html> for details.
//
// Copyright (c) 2025-2026 TensorChord Inc.
// Copyright (c) 2026 PGSTY PTE. LTD.
// Maintained by Ruohang Feng @ PGSTY (rh@vonng.com)

#if defined(__clang__)
#if !(__clang_major__ >= 16)
#error "Clang version must be at least 16."
#endif
#elif defined(__GNUC__)
#if !(__GNUC__ >= 14)
#error "GCC version must be at least 14."
#endif
#else
#error "This file requires Clang or GCC."
#endif

#include <arm_neon.h>
#if defined(__AARCH64EL__)
#include <arm_sve.h>
#endif
#include <stddef.h>
#include <stdint.h>

typedef __fp16 f16;
typedef float f32;

// Both reductions accumulate in f32. Each half is loaded into the bottom of a
// 32-bit container (`ld1h` with zero extension) and widened with `fcvt`, so
// neither products nor differences nor partial sums are rounded to f16, which
// would overflow at 65504.

__attribute__((target("+sve"))) static inline svfloat32_t
fp16_load_widen(svbool_t mask, const f16 *p) {
  svuint32_t bits = svld1uh_u32(mask, (const uint16_t *)p);
  return svcvt_f32_f16_x(mask, svreinterpret_f16_u32(bits));
}

__attribute__((target("+sve"))) float
fp16_reduce_sum_of_xy_a3_512(size_t n, f16 *restrict a, f16 *restrict b) {
  const size_t vl = svcntw();
  svfloat32_t _0 = svdup_f32(0.0f);
  svfloat32_t _1 = svdup_f32(0.0f);
  while (n >= vl * 2) {
    svbool_t mask = svptrue_b32();
    svfloat32_t x_0 = fp16_load_widen(mask, a + 0 * vl);
    svfloat32_t y_0 = fp16_load_widen(mask, b + 0 * vl);
    svfloat32_t x_1 = fp16_load_widen(mask, a + 1 * vl);
    svfloat32_t y_1 = fp16_load_widen(mask, b + 1 * vl);
    _0 = svmla_f32_x(mask, _0, x_0, y_0);
    _1 = svmla_f32_x(mask, _1, x_1, y_1);
    n -= vl * 2, a += vl * 2, b += vl * 2;
  }
  if (n >= vl) {
    svbool_t mask = svptrue_b32();
    svfloat32_t x_0 = fp16_load_widen(mask, a);
    svfloat32_t y_0 = fp16_load_widen(mask, b);
    _0 = svmla_f32_x(mask, _0, x_0, y_0);
    n -= vl, a += vl, b += vl;
  }
  if (n > 0) {
    svbool_t mask = svwhilelt_b32((int64_t)0, (int64_t)n);
    svfloat32_t x_1 = fp16_load_widen(mask, a);
    svfloat32_t y_1 = fp16_load_widen(mask, b);
    _1 = svmla_f32_m(mask, _1, x_1, y_1);
  }
  svbool_t mask = svptrue_b32();
  return svaddv_f32(mask, svadd_f32_x(mask, _0, _1));
}

__attribute__((target("+sve"))) float
fp16_reduce_sum_of_d2_a3_512(size_t n, f16 *restrict a, f16 *restrict b) {
  const size_t vl = svcntw();
  svfloat32_t _0 = svdup_f32(0.0f);
  svfloat32_t _1 = svdup_f32(0.0f);
  while (n >= vl * 2) {
    svbool_t mask = svptrue_b32();
    svfloat32_t x_0 = fp16_load_widen(mask, a + 0 * vl);
    svfloat32_t y_0 = fp16_load_widen(mask, b + 0 * vl);
    svfloat32_t x_1 = fp16_load_widen(mask, a + 1 * vl);
    svfloat32_t y_1 = fp16_load_widen(mask, b + 1 * vl);
    svfloat32_t d_0 = svsub_f32_x(mask, x_0, y_0);
    svfloat32_t d_1 = svsub_f32_x(mask, x_1, y_1);
    _0 = svmla_f32_x(mask, _0, d_0, d_0);
    _1 = svmla_f32_x(mask, _1, d_1, d_1);
    n -= vl * 2, a += vl * 2, b += vl * 2;
  }
  if (n >= vl) {
    svbool_t mask = svptrue_b32();
    svfloat32_t x_0 = fp16_load_widen(mask, a);
    svfloat32_t y_0 = fp16_load_widen(mask, b);
    svfloat32_t d_0 = svsub_f32_x(mask, x_0, y_0);
    _0 = svmla_f32_x(mask, _0, d_0, d_0);
    n -= vl, a += vl, b += vl;
  }
  if (n > 0) {
    svbool_t mask = svwhilelt_b32((int64_t)0, (int64_t)n);
    svfloat32_t x_1 = fp16_load_widen(mask, a);
    svfloat32_t y_1 = fp16_load_widen(mask, b);
    svfloat32_t d_1 = svsub_f32_z(mask, x_1, y_1);
    _1 = svmla_f32_m(mask, _1, d_1, d_1);
  }
  svbool_t mask = svptrue_b32();
  return svaddv_f32(mask, svadd_f32_x(mask, _0, _1));
}
