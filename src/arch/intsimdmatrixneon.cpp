///////////////////////////////////////////////////////////////////////
// File:        intsimdmatrixneon.cpp
// Description: class for 8-bit int SIMD matrix multipliers.
// Author:      Sriram C.
// Created:     Tue Aug 15 08:01:32 PST 2017
//
// (C) Copyright 2019, KauphiHouse
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////

#if !defined(__ARM_NEON__)
#error Implementation only for ARM Neon capable architectures
#endif

#include "intsimdmatrix.h"

#include <cstdint>
#include <vector>
#include <iostream>

#include <arm_neon.h>

namespace tesseract {

// Computes and returns the dot product of the n-vectors u and v.
// Uses ARM NEON intrinsics to access the SIMD instruction set.
static int32_t IntDotProductNEON(const int8_t* u, const int8_t* v
					, int n) {

  const int OFFSET8 = 8;
  int offset = 0;
  int max_offset = n - n%OFFSET8;
  
  int16x8_t result16;  
  int32x4_t result32 = vdupq_n_s32(0);  // Init to 0
  
  while(offset < max_offset) {
    
    // multiply two 8x8b vectors to get a 8x16b vector
    result16 = vmull_s8(vld1_s8 (u+offset),vld1_s8 (v+offset));

    // add the top 4x16b vector and bottom 4x16b vector of
    // results16 vector and accumulate the results to the
    // 4x32b results32 vector
    result32 = vpadalq_s16(result32,result16);
    offset+=OFFSET8;
  }

  int32_t sum = result32[0]+result32[1]+result32[2]+result32[3];
  
  while (offset < n) {
    sum += u[offset] * v[offset];
    ++offset;
  }
  return sum;
}

  
// Computes part of matrix.vector v = Wu. Computes 1 result.
static void PartialMatrixDotVector1(const int8_t* wi,
					   const double* scales,
					   const int8_t* u, int num_in,
					   double* v) {
  double total = IntDotProductNEON(u, wi, num_in);
  // Add in the bias and correct for integer values.
  *v = (total / INT8_MAX + wi[num_in]) * *scales;
}

static void matrixDotVector(int dim1, int dim2, const int8_t* wi,
                            const double* scales, const int8_t* u, double* v) {
  const int num_out = dim1;
  const int num_in = dim2 - 1;
  int output = 0;
  
  for (; output < num_out; output++) {
    PartialMatrixDotVector1(wi, scales, u, num_in, v);
    wi += dim2;
    scales++;
    v++;
  }
}
  
const IntSimdMatrix IntSimdMatrix::intSimdMatrixNEON = {
  // Function.
  matrixDotVector,
  // Number of 32 bit outputs held in each register.
  1,
  // Maximum number of registers that we will use to hold outputs.
  1,
  // Number of 8 bit inputs in the inputs register.
  1,
  // Number of inputs in each weight group.
  1
};

}  // namespace tesseract
