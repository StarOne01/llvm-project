//===-- Half-precision 10^x function ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GENERIC_EXP10F16_IMPL_H
#define LLVM_LIBC_SRC_MATH_GENERIC_EXP10F16_IMPL_H

#include "expxf16.h"
#include "include/llvm-libc-types/float128.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/FPUtil/multiply_add.h"
#include "src/__support/FPUtil/rounding_mode.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/math/generic/explogxf.h"

namespace LIBC_NAMESPACE_DECL {
namespace generic {

LIBC_INLINE float16 exp10f16(float16 x) {
  using FPBits = typename fputil::FPBits<float16>;
  FPBits xbits(x);

  uint16_t x_u = xbits.uintval();
  uint16_t x_abs = x_u & 0x7fff;

  // When |x| >= log10(2^128), or x is nan
  if (LIBC_UNLIKELY(x_abs >= 0x5344)) {
    // When x < log10(2^-150) or nan
    if (x_u > 0xd3a5) {
      // exp(-Inf) = 0
      if (xbits.is_inf())
        return 0.0f16;
      // exp(nan) = nan
      if (xbits.is_nan())
        return x;
      if (fputil::fenv_is_round_up())
        return FPBits::min_subnormal().get_val();
      fputil::set_errno_if_required(ERANGE);
      fputil::raise_except_if_required(FE_UNDERFLOW);
      return 0.0f16;
    }
    // x >= log10(2^128) or nan
    if (xbits.is_pos() && (x_u >= 0x5344)) {
      // x is finite
      if (x_u < 0x7c00) {
        int rounding = fputil::quick_get_round();
        if (rounding == FE_DOWNWARD || rounding == FE_TOWARDZERO)
          return FPBits::max_normal().get_val();

        fputil::set_errno_if_required(ERANGE);
        fputil::raise_except_if_required(FE_OVERFLOW);
      }
      // x is +inf or nan
      return x + FPBits::inf().get_val();
    }
  }

  // When |x| <= log10(2)*2^-6
  if (LIBC_UNLIKELY(x_abs <= 0x2cd1)) {
    if (LIBC_UNLIKELY(x_u == 0x8af3)) { // x = -0x1.bcb7b2p-27f16
      if (fputil::fenv_is_round_to_nearest())
        return 0x3bffp-1f16;
    }
    // |x| < 2^-25
    // 10^x ~ 1 + log(10) * x
    if (LIBC_UNLIKELY(x_abs <= 0x1400)) {
      return fputil::multiply_add(x, 0x2935p-13f16, 1.0f16);
    }

    return static_cast<float16>(Exp10Base::powb_lo(x));
  }

  // Exceptional value.
  if (LIBC_UNLIKELY(x_u == 0x34a7)) { // x = 0x1.29b2acp-5f16
    if (fputil::fenv_is_round_up())
      return 0x38b3p-15f16;
  }
  // Exact outputs when x = 1, 2, ..., 10.
  // Quick check mask: 0x800fU = ~(bits of 1.0f16 | ... | bits of 10.0f16)
  if (LIBC_UNLIKELY((x_u & 0x800fU) == 0)) {
    switch (x_u) {
    case 0x3c00: // x = 1.0f16
      return 10.0f16;
    case 0x4000: // x = 2.0f16
      return 100.0f16;
    case 0x4200: // x = 3.0f16
      return 1'000.0f16;
    case 0x4400: // x = 4.0f16
      return 10'000.0f16;
    }
  }

  // Range reduction: 10^x = 2^(mid + hi) * 10^lo
  //   rr = (2^(mid + hi), lo)
  auto rr = exp_b_range_reduc<Exp10Base>(x);

  // The low part is approximated by a degree-5 minimax polynomial.
  // 10^lo ~ 1 + COEFFS[0] * lo + ... + COEFFS[4] * lo^5
  using fputil::multiply_add;
  float lo2 = float( rr.lo * rr.lo);
  // c0 = 1 + COEFFS[0] * lo
  float c0 = multiply_add(float(rr.lo), float (Exp10Base::COEFFS[0]), 1.0f);
  // c1 = COEFFS[1] + COEFFS[2] * lo
  float c1 = multiply_add(float (rr.lo), float(Exp10Base::COEFFS[2]), float(Exp10Base::COEFFS[1]));
  // c2 = COEFFS[3] + COEFFS[4] * lo
  float c2 = multiply_add(float(rr.lo), float(Exp10Base::COEFFS[4]), float(Exp10Base::COEFFS[3]));
  // p = c1 + c2 * lo^2
  //   = COEFFS[1] + COEFFS[2] * lo + COEFFS[3] * lo^2 + COEFFS[4] * lo^3
  float p = multiply_add(lo2, c2, c1);
  // 10^lo ~ c0 + p * lo^2
  // 10^x = 2^(mid + hi) * 10^lo
  //      ~ mh * (c0 + p * lo^2)
  //      = (mh * c0) + p * (mh * lo^2)
  return static_cast<float>(multiply_add(p, lo2 * rr.mh, c0 * rr.mh));
}

} // namespace generic
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_MATH_GENERIC_EXP10F_IMPL_H
