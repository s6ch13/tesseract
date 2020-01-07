///////////////////////////////////////////////////////////////////////
// File:        dotproductneon.cpp
// Description: Architecture-specific dot-product function.
// Author:      Sriram C.
// Created:     Wed Jul 22 10:57:45 PDT 2018
//
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


#include <iostream>
#if !defined(__ARM_NEON__)
#error Implementation only for ARM NEON capable architectures
#endif


#include <cstdint>
#include "dotproduct.h"
#include <arm_neon.h>
namespace tesseract {

#if !defined(__AARCH64__)

double DotProductNEON(const double* u, const double* v, int n) {
  fprintf(stderr, "DotProduct can't be used on Aarch32\n");
  abort();
}
#else
// ARM 64 bit stuff
// Computes and returns the dot product of the n-vectors u and v.
// Uses ARM NEON intrinsics to access the SIMD instruction set.

double DotProductNEON(const double* u, const double* v, int n) {
  // Place holder for dot product neon
}
#endif



}  // namespace tesseract.


